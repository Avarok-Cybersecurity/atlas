// SPDX-License-Identifier: AGPL-3.0-only
//
// Core sampling pipeline (`sample_with_params_history` + seeded
// implementation). Split out of `sampler.rs` to keep the parent file
// under the 500-line cap. The parent re-exports these via `pub use`.

use super::{SamplingParams, apply_penalties_and_bias, record_entropy};

pub fn sample_with_params_history(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
) -> u32 {
    sample_with_params_seeded(data, params, token_history, params.seed)
}

/// Core sampling pipeline with explicit seed control.
/// `seed` overrides the RNG for deterministic sampling. None = thread_rng.
pub fn sample_with_params_seeded(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> u32 {
    let n = data.len() / 4;
    let top_k = params.top_k as usize;
    let top_p = params.top_p;
    let top_n_sigma = params.top_n_sigma;
    let min_p = params.min_p;

    // Read raw logits into a mutable vec for in-place modifications.
    // Penalties (repetition / presence / frequency / LZ / DRY) and
    // logit_bias are applied to `raw_logits` BEFORE the greedy bypass
    // below, so they take effect even at `temperature == 0.0`. Atlas
    // previously short-circuited to `argmax(raw_logits)` for greedy,
    // silently dropping caller-configured penalties — the 2026-05-01
    // sweep showed this caused Gemma-4-31B's haiku to enter a
    // repetition loop ("la... la... laaaL!") even with the model's
    // configured `repetition_penalty=1.1` because the harness uses
    // `temperature=0`. HF Transformers, vLLM, and llama.cpp all run
    // LogitsProcessor (penalties + bias) before greedy argmax — Atlas
    // is the outlier here.
    // Chunked conversion instead of per-element indexed `read_f32`: the
    // indexed form does four bounds-checked byte reads per element through a
    // shared counter, which the compiler lowers poorly; `chunks_exact` is the
    // canonical vectorisable shape. Same bytes, same values.
    let mut raw_logits: Vec<f32> = data
        .chunks_exact(4)
        .take(n)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // ── 0. Penalties (repetition / presence / frequency / LZ / DRY) +
    //       logit bias, applied in place via the shared SSOT helper.
    // Identical behavior to the previous inline block; the same helper is
    // now also invoked on the MTP verify + bootstrap paths so all three
    // emit/verify sites apply the same penalties+bias+history.
    apply_penalties_and_bias(&mut raw_logits, params, token_history);

    // ── Greedy bypass (post-penalty argmax) ──
    // At `temperature == 0.0` we return argmax of the penalty/bias-modified
    // logits. We bypass top_n_sigma, temperature scaling, top_k, top_p,
    // min_p — all of those either filter (set values to -inf) or apply
    // monotonic transforms, neither of which can re-order the maximum. Only
    // penalties + logit_bias actually re-order logits, so as long as those
    // ran first, this argmax is correct AND respects caller config.
    if params.temperature <= 0.0 {
        return greedy_pick_last_wins(&raw_logits);
    }
    let temperature = params.temperature;

    // ── 1. Top-n-sigma: filter noise in logit space (temperature-invariant) ──
    // Keep tokens with logit >= mean - n*sigma. Filters NVFP4 quantization noise.
    if top_n_sigma > 0.0 {
        let sum: f32 = raw_logits.iter().sum();
        let mean = sum / n as f32;
        let var: f32 = raw_logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let sigma = var.sqrt();
        if sigma > 0.0 {
            let threshold = mean - top_n_sigma * sigma;
            for logit in raw_logits.iter_mut() {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    // ── 2. Temperature scaling ──
    let mut logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite()) // Skip -inf tokens from top-n-sigma
        .map(|(i, v)| (i as u32, v / temperature))
        .collect();

    if logits.is_empty() {
        // Fallback: if top-n-sigma filtered everything, use argmax of original
        return raw_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // ── 3. Rank-dependent filtering (top-k / min-p / top-p) ──
    // These need descending order — but a full O(n·log n) sort of the whole
    // ~248K vocab every token is wasteful when top_k caps survivors at a small
    // k (the model default here is top_k=20; a full sort was ~2.3ms/tok). Use
    // an O(n) quickselect to isolate the top-k, then sort only those k. Pure
    // temperature sampling (no top_k/top_p/min_p) needs no ordering at all —
    // the multinomial draw below is order-independent — so skip sorting.
    let cmp_desc =
        |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
    let sorted = if top_k > 0 && top_k < logits.len() {
        // Quickselect the top-k into [0, top_k) (O(n)), drop the rest, then
        // sort just the k survivors (O(k·log k)). Identical result to
        // full-sort-then-truncate; min-p/top-p below run over these k.
        logits.select_nth_unstable_by(top_k, cmp_desc);
        logits.truncate(top_k);
        logits.sort_unstable_by(cmp_desc);
        true
    } else if min_p > 0.0 || top_p < 1.0 {
        // No top-k cap, but min-p/top-p still need full descending order.
        logits.sort_unstable_by(cmp_desc);
        true
    } else {
        false
    };

    // ── 4. Softmax ──
    // `sorted` ⇒ logits[0] is the max; otherwise reduce for it. min_p/top_p
    // below are only reachable when `sorted` is true, so their reliance on
    // descending order still holds.
    let max_val = if sorted {
        logits[0].1
    } else {
        logits
            .iter()
            .map(|&(_, v)| v)
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_val).exp()))
        .collect();

    // ── 4b. Entropy: H = -Σ p·ln(p) over the post-softmax distribution ──
    {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let h: f32 = probs
                .iter()
                .map(|&(_, w)| {
                    let p = w * inv;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            record_entropy(h);
        }
    }

    // ── 5. Min-p: keep tokens with prob >= min_p * max_prob ──
    if min_p > 0.0 {
        let max_prob = probs[0].1; // Already sorted descending
        let threshold = min_p * max_prob;
        probs.retain(|p| p.1 >= threshold);
    }

    // ── 6. Top-p (nucleus) ──
    if top_p < 1.0 {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob / sum;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // Multinomial sample from the filtered distribution.
    let sum: f32 = probs.iter().map(|p| p.1).sum();
    let random_val: f32 = if let Some(s) = seed {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        rng.r#gen::<f32>()
    } else {
        rand::random::<f32>()
    };
    let threshold = random_val * sum;
    let mut cumsum = 0.0f32;
    for &(idx, prob) in &probs {
        cumsum += prob;
        if cumsum >= threshold {
            return idx;
        }
    }
    probs.last().map_or(0, |p| p.0)
}

/// Greedy pick with `max_by(partial_cmp.unwrap_or(Equal))` semantics — the
/// tie-break this path has ALWAYS had, which is LAST-index-wins (Rust's
/// `max_by` returns the last of several equal maxima). This is the OPPOSITE of
/// the verify path's first-index-wins `argmax_first_wins`, so that function
/// must NOT be ported here: quantized logits produce exact ties, and flipping
/// the tie-break would change emitted tokens.
///
/// NaN is the reason for the guarded structure. `partial_cmp` with a NaN
/// returns `None`, mapped to `Equal` — and under `max_by`'s last-wins rule an
/// "equal" NaN DISPLACES the current maximum, so a NaN anywhere after the true
/// max changes the historical result. That quirk cannot be reproduced by a
/// max-then-find pass, so: pass 1 computes the max over 8 independent lanes
/// (vectorisable, no index dependency) while also detecting NaN; if ANY NaN is
/// present the original `max_by` expression runs verbatim (bit-identical by
/// construction); otherwise the answer is the LAST index equal to the max,
/// which is exactly what `max_by` returns on NaN-free input (including
/// -0.0/+0.0 ties, where `partial_cmp` says Equal and `==` agrees).
fn greedy_pick_last_wins(v: &[f32]) -> u32 {
    const LANES: usize = 8;
    let mut acc = [f32::NEG_INFINITY; LANES];
    let mut any_nan = false;
    let mut chunks = v.chunks_exact(LANES);
    for c in &mut chunks {
        for (a, &x) in acc.iter_mut().zip(c) {
            any_nan |= x.is_nan();
            if x > *a {
                *a = x;
            }
        }
    }
    let mut best = f32::NEG_INFINITY;
    for &a in acc.iter() {
        if a > best {
            best = a;
        }
    }
    for &x in chunks.remainder() {
        any_nan |= x.is_nan();
        if x > best {
            best = x;
        }
    }
    if any_nan {
        // Bit-identical fallback: the exact historical expression.
        return v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    v.iter()
        .rposition(|&x| x == best)
        .unwrap_or(0)
        .try_into()
        .unwrap_or(0)
}

#[cfg(test)]
mod greedy_tests {
    use super::greedy_pick_last_wins;

    fn reference(v: &[f32]) -> u32 {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    fn agree(v: &[f32]) {
        assert_eq!(greedy_pick_last_wins(v), reference(v), "diverged on {v:?}");
    }

    #[test]
    fn matches_max_by_reference() {
        agree(&[]);
        agree(&[1.0]);
        agree(&[1.0, 3.0, 2.0]);
        // Ties resolve to the LAST index — the opposite of the verify path.
        agree(&[1.0, 5.0, 5.0, 5.0, 2.0]);
        agree(&[5.0, 1.0, 5.0]);
        // Tie straddling the 8-lane boundary.
        agree(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0]);
        agree(&[-5.0, -1.0, -3.0]);
        agree(&[-0.0, 0.0, -0.0]);
        agree(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);
        agree(&[f32::INFINITY, 1.0, f32::INFINITY]);
        // NaN cases route to the verbatim fallback => trivially identical,
        // but assert anyway so the routing itself is covered.
        agree(&[f32::NAN, 1.0, 2.0]);
        agree(&[1.0, 2.0, f32::NAN]);
        agree(&[f32::NAN]);
    }

    #[test]
    fn vocab_sized_last_wins() {
        let mut v: Vec<f32> = (0..248_320)
            .map(|i| (((i * 2654435761u64 as usize) % 100_003) as f32) / 1000.0 - 50.0)
            .collect();
        v[100_000] = 999.0;
        v[200_000] = 999.0; // duplicate max — LAST must win here
        assert_eq!(greedy_pick_last_wins(&v), reference(&v));
        assert_eq!(greedy_pick_last_wins(&v), 200_000);
    }
}
