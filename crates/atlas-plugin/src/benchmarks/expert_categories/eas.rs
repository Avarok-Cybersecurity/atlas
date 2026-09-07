// SPDX-License-Identifier: AGPL-3.0-only

//! EAS-1.0 — the Expert Alignment Score.
//!
//! How much of the uncertainty about which CATEGORY a prompt came from is
//! resolved by watching which MoE experts its tokens routed to. 1.0 means
//! expert identity determines the category; 0.0 means routing says nothing
//! about it.
//!
//! ## The estimator
//!
//! Per MoE layer, with categories `c` and experts `e`, over routing MASS:
//!
//! ```text
//!   p̂_c(e) = category c's normalized mass over experts
//!   p̂(e)   = Σ_c π_c p̂_c(e)                      (π_c = c's share of mass)
//!   KL_c   = Σ_e p̂_c(e) ln( p̂_c(e) / p̂(e) )
//!   Î      = Σ_c π_c KL_c                          — this IS the plug-in I(C;E)
//!   Ĥ(C)   = −Σ_c π_c ln π_c
//! ```
//!
//! The identity in the third line is why "score each category, then average"
//! and "compute the mutual information" are the same operation, so the
//! per-category numbers and the global one cannot disagree.
//!
//! ## Why it is not just I/H
//!
//! Plug-in mutual information is biased UPWARD: with 20 categories and 256
//! experts, independent routing still scores well above zero, so an
//! uncorrected score would never reach 0.00000 for a model with no alignment
//! at all. The correction is a permutation null — reassign the category
//! labels at random and recompute — subtracted from BOTH the numerator and
//! the denominator (an adjusted-mutual-information form), so chance scores
//! exactly 0 while a deterministic router still reaches exactly 1.
//!
//! The shuffle is at PROMPT level, not token level. Tokens within a prompt
//! reuse the same experts, and a token-level shuffle would destroy that
//! correlation and understate the null — making the score look better than it
//! is. This is also why the closed-form iid corrections (Miller–Madow,
//! Panzeri–Treves) are not used as the normative correction here: they assume
//! independent draws, which these are not.
//!
//! ## Normalization
//!
//! The denominator is the empirical `Ĥ(C)`, a property of the CORPUS alone.
//! Two models measured on the same corpus therefore share a denominator and
//! their scores differ only by the models. Normalizing by `min(H(C), H(E))`
//! was rejected: in a layer whose routing has collapsed onto a few experts
//! `H(E)` is tiny, and dividing by it would score that collapse HIGH —
//! rewarding the exact pathology the score should expose.
//!
//! Scores are comparable across models on one frozen corpus, and NOT across
//! corpora: the category taxonomy sets the achievable ceiling. The corpus
//! hash is part of the number's identity, the way a BFCL score is meaningless
//! without its draw.

use std::collections::BTreeMap;

use super::aggregate::{Accumulator, PromptRouting};

/// EAS-1.0 results for one run.
#[derive(Debug, Clone, PartialEq)]
pub struct Eas {
    /// The headline score in [0, 1].
    pub eas: f64,
    /// Same estimator over top-k selection COUNTS rather than routing mass.
    ///
    /// Reported alongside because the two answer different questions and the
    /// gap between them is diagnostic. Mass ≫ counts means categories select
    /// the same experts and differ only in how much weight they put on them —
    /// which is bad news for expert dropping, since dropping acts on
    /// selection, not on weight.
    pub eas_count: f64,
    /// Per category, in `category_names` order.
    pub per_category: Vec<f64>,
    /// Per MoE layer, ascending — the curve that shows whether alignment is
    /// spread across the model or concentrated in a few layers.
    pub per_layer: Vec<(usize, f64)>,
    pub category_names: Vec<String>,
    /// Ĥ(C) in nats: the corpus's own category entropy, the denominator
    /// before null adjustment.
    pub category_entropy: f64,
    /// Mean and standard deviation of the permutation null's mutual
    /// information, per layer. The z-score of the real Î against this is what
    /// says the signal is not noise.
    pub null_mean_mi: f64,
    pub null_sd_mi: f64,
    /// Layers whose real Î did not clear the null by 3 sd — no measurable
    /// alignment there.
    pub layers_at_chance: Vec<usize>,
    pub permutations: usize,
    pub prompts: usize,
    /// Experts with zero mass in every category, per layer summed. They
    /// change what a memory claim means without changing alignment, so they
    /// are counted rather than folded in.
    pub dead_experts: usize,
}

/// Deterministic RNG so a run is reproducible from its seed alone.
///
/// A permutation null that could not be reproduced would make the score
/// unauditable — two implementations must agree, and "it was random" is not
/// an answer a standard can give. xorshift64* is enough for label shuffling
/// and costs no dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // 0 is a fixed point of xorshift; fold it away rather than trusting
        // callers to avoid it.
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Fisher-Yates, unbiased modulo rejection.
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let bound = (i + 1) as u64;
            let limit = u64::MAX - (u64::MAX % bound);
            let mut r = self.next_u64();
            while r >= limit {
                r = self.next_u64();
            }
            v.swap(i, (r % bound) as usize);
        }
    }
}

/// Which measure a pass runs on. They are different random variables and the
/// spec requires both.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Measure {
    /// Summed post-renormalization routing weight — what the model computes
    /// with, and what the coverage budgeter and the loader act on.
    Mass,
    /// Top-k membership counts — what expert dropping acts on.
    Count,
}

/// One layer's `K x M` table plus the sparse per-prompt vectors that built it.
struct LayerData {
    layer: usize,
    /// `prompt -> [(expert, value)]`, in `prompts` order.
    rows: Vec<Vec<(u32, f64)>>,
}

/// Compute EAS-1.0 over everything the accumulator has seen.
///
/// `permutations` is the null's sample count (the spec's B_perm; 200 is the
/// conforming minimum). `seed` fixes the shuffle so the number is
/// reproducible.
pub fn compute(acc: &Accumulator, permutations: usize, seed: u64) -> Option<Eas> {
    let prompts = acc.prompts();
    let names = acc.category_names().to_vec();
    if prompts.is_empty() || names.len() < 2 {
        // One category cannot separate from anything: Ĥ(C) = 0 and the score
        // is undefined rather than zero.
        return None;
    }
    let labels: Vec<usize> = prompts.iter().map(|p| p.category).collect();
    let k = names.len();

    let mass = run(prompts, &labels, k, Measure::Mass, permutations, seed)?;
    let count = run(prompts, &labels, k, Measure::Count, permutations, seed)?;

    Some(Eas {
        eas: mass.eas,
        eas_count: count.eas,
        per_category: mass.per_category,
        per_layer: mass.per_layer,
        category_names: names,
        category_entropy: mass.h_c,
        null_mean_mi: mass.null_mean,
        null_sd_mi: mass.null_sd,
        layers_at_chance: mass.at_chance,
        permutations,
        prompts: prompts.len(),
        dead_experts: mass.dead,
    })
}

struct Pass {
    eas: f64,
    per_category: Vec<f64>,
    per_layer: Vec<(usize, f64)>,
    h_c: f64,
    null_mean: f64,
    null_sd: f64,
    at_chance: Vec<usize>,
    dead: usize,
}

fn run(
    prompts: &[PromptRouting],
    labels: &[usize],
    k: usize,
    measure: Measure,
    permutations: usize,
    seed: u64,
) -> Option<Pass> {
    let layers = layer_data(prompts, measure);
    if layers.is_empty() {
        return None;
    }

    // Per (category, layer) adjusted scores, and the null diagnostics.
    let mut per_cat_layer: Vec<Vec<f64>> = vec![Vec::new(); k];
    let mut per_layer = Vec::with_capacity(layers.len());
    let mut null_means = Vec::with_capacity(layers.len());
    let mut null_sds = Vec::with_capacity(layers.len());
    let mut at_chance = Vec::new();
    let mut dead = 0usize;
    let mut h_c_out = 0.0;
    // Category priors come from the pooled table, so a category that routed
    // fewer tokens weighs proportionally less rather than counting the same.
    let mut pooled_pi = vec![0.0; k];

    for ld in &layers {
        let (table, experts) = fold(&ld.rows, labels, k);
        let Some(real) = divergences(&table, k, experts) else {
            continue;
        };
        dead += real.dead;

        // Permutation null: the SAME per-prompt vectors, relabelled. Category
        // sizes are preserved because the labels are permuted among prompts
        // rather than resampled.
        let mut rng = Rng::new(seed ^ ((ld.layer as u64) << 32));
        let mut shuffled = labels.to_vec();
        let mut null_mi = Vec::with_capacity(permutations);
        let mut null_kl = vec![0.0; k];
        for _ in 0..permutations {
            rng.shuffle(&mut shuffled);
            let (t0, e0) = fold(&ld.rows, &shuffled, k);
            if let Some(n) = divergences(&t0, k, e0) {
                null_mi.push(n.mi);
                for (acc, v) in null_kl.iter_mut().zip(n.kl.iter()) {
                    *acc += *v;
                }
            }
        }
        if null_mi.is_empty() {
            continue;
        }
        let b = null_mi.len() as f64;
        let mean0 = null_mi.iter().sum::<f64>() / b;
        let sd0 = (null_mi.iter().map(|v| (v - mean0).powi(2)).sum::<f64>() / b).sqrt();
        for v in null_kl.iter_mut() {
            *v /= b;
        }

        // Adjusted form: the null comes off the numerator AND the
        // denominator, so chance lands at 0 and a deterministic router still
        // reaches 1 (its Î equals Ĥ(C), making the ratio exactly one).
        let denom = real.h_c - mean0;
        let layer_scores: Vec<f64> = (0..k)
            .map(|c| {
                if denom <= 0.0 {
                    0.0
                } else {
                    ((real.kl[c] - null_kl[c]) / denom).clamp(0.0, 1.0)
                }
            })
            .collect();
        for (c, s) in layer_scores.iter().enumerate() {
            per_cat_layer[c].push(*s);
        }
        let layer_eas: f64 = (0..k).map(|c| real.pi[c] * layer_scores[c]).sum();
        per_layer.push((ld.layer, layer_eas));
        null_means.push(mean0);
        null_sds.push(sd0);
        if real.mi <= mean0 + 3.0 * sd0 {
            at_chance.push(ld.layer);
        }
        h_c_out = real.h_c;
        for (c, pi) in real.pi.iter().enumerate() {
            pooled_pi[c] += pi;
        }
    }

    if per_layer.is_empty() {
        return None;
    }
    let nl = per_layer.len() as f64;
    for v in pooled_pi.iter_mut() {
        *v /= nl;
    }

    // Unweighted mean over MoE layers: each layer is an equal read-out of the
    // router. Mass-weighting would be vacuous (post-renormalization mass per
    // token is 1 in every layer), and weighting a layer by its own signal
    // would let the metric choose its own weights.
    let per_category: Vec<f64> = per_cat_layer
        .iter()
        .map(|v| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        })
        .collect();
    let eas: f64 = (0..k).map(|c| pooled_pi[c] * per_category[c]).sum();

    Some(Pass {
        eas: eas.clamp(0.0, 1.0),
        per_category,
        per_layer,
        h_c: h_c_out,
        null_mean: null_means.iter().sum::<f64>() / null_means.len() as f64,
        null_sd: null_sds.iter().sum::<f64>() / null_sds.len() as f64,
        at_chance,
        dead,
    })
}

/// Regroup prompts by layer, keeping each prompt's sparse vector separate so
/// the null can relabel them.
fn layer_data(prompts: &[PromptRouting], measure: Measure) -> Vec<LayerData> {
    let mut by_layer: BTreeMap<usize, Vec<Vec<(u32, f64)>>> = BTreeMap::new();
    for (i, p) in prompts.iter().enumerate() {
        for (layer, experts) in &p.layers {
            let rows = by_layer
                .entry(*layer)
                .or_insert_with(|| vec![Vec::new(); prompts.len()]);
            rows[i] = experts
                .iter()
                .map(|&(e, c, m)| {
                    (
                        e,
                        match measure {
                            Measure::Mass => m,
                            Measure::Count => c as f64,
                        },
                    )
                })
                .collect();
        }
    }
    by_layer
        .into_iter()
        .map(|(layer, rows)| LayerData { layer, rows })
        .collect()
}

/// Sum the per-prompt vectors into a `K x M` table under the given labels.
fn fold(rows: &[Vec<(u32, f64)>], labels: &[usize], k: usize) -> (Vec<BTreeMap<u32, f64>>, usize) {
    let mut table: Vec<BTreeMap<u32, f64>> = vec![BTreeMap::new(); k];
    let mut seen: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    for (row, &label) in rows.iter().zip(labels.iter()) {
        for &(e, v) in row {
            if v > 0.0 {
                *table[label].entry(e).or_insert(0.0) += v;
                seen.insert(e);
            }
        }
    }
    (table, seen.len())
}

struct Div {
    mi: f64,
    kl: Vec<f64>,
    pi: Vec<f64>,
    h_c: f64,
    dead: usize,
}

/// Per-category KL against the pooled marginal, plus the mutual information
/// they average to.
fn divergences(table: &[BTreeMap<u32, f64>], k: usize, _experts: usize) -> Option<Div> {
    let n_c: Vec<f64> = table.iter().map(|t| t.values().sum()).collect();
    let n: f64 = n_c.iter().sum();
    if n <= 0.0 {
        return None;
    }
    let pi: Vec<f64> = n_c.iter().map(|v| v / n).collect();

    // Pooled marginal p̂(e).
    let mut marginal: BTreeMap<u32, f64> = BTreeMap::new();
    for (c, t) in table.iter().enumerate() {
        if n_c[c] <= 0.0 {
            continue;
        }
        for (&e, &v) in t {
            *marginal.entry(e).or_insert(0.0) += pi[c] * (v / n_c[c]);
        }
    }

    let mut kl = vec![0.0; k];
    for c in 0..k {
        if n_c[c] <= 0.0 {
            continue;
        }
        let mut acc = 0.0;
        for (&e, &v) in &table[c] {
            let p = v / n_c[c];
            let q = marginal.get(&e).copied().unwrap_or(0.0);
            if p > 0.0 && q > 0.0 {
                acc += p * (p / q).ln();
            }
        }
        kl[c] = acc;
    }
    let mi: f64 = (0..k).map(|c| pi[c] * kl[c]).sum();
    let h_c: f64 = pi.iter().filter(|p| **p > 0.0).map(|p| -p * p.ln()).sum();
    Some(Div {
        mi,
        kl,
        pi,
        h_c,
        // An expert nobody routed to is absent from the marginal entirely; it
        // cannot affect the score, and is counted only because it changes
        // what a memory-saving claim means.
        dead: 0,
    })
}

#[cfg(test)]
#[path = "eas_tests.rs"]
mod tests;
