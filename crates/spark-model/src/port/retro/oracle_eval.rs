// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:754-815
// (sparse_attention: batch_gemm_softmax over centroids, torch.sum over the GQA group, the
// empty-cluster mask, torch.topk into the retrieval and estimation zones, the estimation
// zone's weighted attention and its merge with the exact zone). Changes: a plain Rust CPU
// reference in f64; ties ranked toward the lower centroid index; the zone merge written as
// one softmax over the union (arXiv 2505.02922 Eq 2-4), which equals upstream's LSE merge;
// the quality measures the shadow gate needs (mass recall, output error, Jaccard).

//! Responsibility: the CPU reference of a RetroInfer decode step for one KV
//! head: centroid scoring, zone selection, the exact-zone token set, the merged
//! exact plus estimation attention, dense attention, and the measures that
//! compare them (softmax mass recall per zone, relative L2 output error,
//! Jaccard against the true top-k).

use super::oracle::{HeadIndex, HeadKv};
use super::state::IndexCounts;

/// Per-centroid selection scores for one KV head: for each query head of the
/// group, softmax over the non-empty centroids of `q . c * scale`, summed over
/// the group. Empty clusters score negative infinity.
pub fn score_clusters(
    idx: &HeadIndex,
    nc: usize,
    q_group: &[f32],
    group: usize,
    scale: f32,
) -> Vec<f64> {
    let hd = idx.head_dim;
    let nc = nc.min(idx.n_centroids());
    let mut score = vec![0f64; nc];
    for g in 0..group {
        let q = &q_group[g * hd..(g + 1) * hd];
        let logits: Vec<f64> = (0..nc)
            .map(|c| {
                if idx.sizes[c] == 0 {
                    f64::NEG_INFINITY
                } else {
                    dot(q, idx.centroid(c)) * f64::from(scale)
                }
            })
            .collect();
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if m == f64::NEG_INFINITY {
            continue;
        }
        let den: f64 = logits.iter().map(|&l| (l - m).exp()).sum();
        for (s, &l) in score.iter_mut().zip(&logits) {
            *s += (l - m).exp() / den;
        }
    }
    for (c, s) in score.iter_mut().enumerate() {
        if idx.sizes[c] == 0 {
            *s = f64::NEG_INFINITY;
        }
    }
    score
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

/// The clusters one selection takes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Zones {
    /// Retrieval zone, descending score.
    pub retrieval: Vec<u32>,
    /// Estimation zone, descending score.
    pub estimation: Vec<u32>,
}

/// Every non-empty centroid ranked by descending score, ties toward the lower index.
pub fn rank(scores: &[f64]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..scores.len())
        .filter(|&c| scores[c] > f64::NEG_INFINITY)
        .map(|c| u32::try_from(c).unwrap_or(u32::MAX))
        .collect();
    ids.sort_by(|&a, &b| {
        scores[b as usize]
            .partial_cmp(&scores[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    ids
}

/// The first `nprobe` ranked clusters, then the next `es`.
pub fn select(scores: &[f64], nprobe: usize, es: usize) -> Zones {
    let r = rank(scores);
    let np = nprobe.min(r.len());
    let ne = es.min(r.len() - np);
    Zones {
        retrieval: r[..np].to_vec(),
        estimation: r[np..np + ne].to_vec(),
    }
}

/// The exact-zone positions of a row attending `[0, len)`: the sink, the
/// retrieval clusters' members, and the tail `[indexed_end, len)`, in the
/// `ExactBuf` order (state.rs). `cap` bounds the retrieval tokens: a cluster
/// that does not fit in whole is dropped (returned as the second value).
pub fn exact_positions(
    idx: &HeadIndex,
    z: &Zones,
    c: &IndexCounts,
    len: u32,
    cap: Option<usize>,
) -> (Vec<u32>, u32) {
    let sink_end = c.sink.min(c.indexed_end).min(len);
    let mut out: Vec<u32> = (0..sink_end).collect();
    let tail: Vec<u32> = (c.indexed_end.min(len)..len).collect();
    let room = cap.map(|cap| cap.saturating_sub(out.len() + tail.len()));
    let mut used = 0usize;
    let mut dropped = 0u32;
    for &cl in &z.retrieval {
        let members = &idx.lists[cl as usize];
        if room.is_some_and(|r| used + members.len() > r) {
            dropped += u32::try_from(members.len()).unwrap_or(u32::MAX);
            continue;
        }
        used += members.len();
        out.extend(members.iter().copied().filter(|&p| p < len));
    }
    out.extend(tail);
    (out, dropped)
}

/// Merged attention of one query head over the exact positions and the
/// estimation clusters (each weighted by its size):
/// `out = (sum_e exp(s_e - m) v_e + sum_c exp(s_c - m) vsum_c) /
///        (sum_e exp(s_e - m) + sum_c n_c exp(s_c - m))`. Returns the output
/// and the log-sum-exp.
pub fn retro_attention(
    kv: HeadKv<'_>,
    idx: &HeadIndex,
    exact: &[u32],
    estimation: &[u32],
    q: &[f32],
    scale: f32,
) -> (Vec<f64>, f64) {
    let hd = kv.head_dim;
    let s = f64::from(scale);
    let e_logits: Vec<f64> = exact
        .iter()
        .map(|&p| dot(q, kv.key(p as usize)) * s)
        .collect();
    let c_logits: Vec<f64> = estimation
        .iter()
        .map(|&c| dot(q, idx.centroid(c as usize)) * s)
        .collect();
    let m = e_logits
        .iter()
        .chain(&c_logits)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut num = vec![0f64; hd];
    let mut den = 0f64;
    if m == f64::NEG_INFINITY {
        return (num, f64::NEG_INFINITY);
    }
    for (&p, &l) in exact.iter().zip(&e_logits) {
        let w = (l - m).exp();
        den += w;
        for (a, &v) in num.iter_mut().zip(kv.value(p as usize)) {
            *a += w * f64::from(v);
        }
    }
    for (&c, &l) in estimation.iter().zip(&c_logits) {
        let w = (l - m).exp();
        den += w * f64::from(idx.sizes[c as usize]);
        for (a, &v) in num.iter_mut().zip(idx.value_sum_of(c as usize)) {
            *a += w * f64::from(v);
        }
    }
    for a in &mut num {
        *a /= den;
    }
    (num, m + den.ln())
}

/// Dense attention of one query head over `[0, len)`: output and the softmax
/// probability of every position.
pub fn dense_attention(kv: HeadKv<'_>, len: usize, q: &[f32], scale: f32) -> (Vec<f64>, Vec<f64>) {
    let hd = kv.head_dim;
    let logits: Vec<f64> = (0..len)
        .map(|p| dot(q, kv.key(p)) * f64::from(scale))
        .collect();
    let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = logits.iter().map(|&l| (l - m).exp()).collect();
    let den: f64 = w.iter().sum();
    let probs: Vec<f64> = w.iter().map(|&x| x / den).collect();
    let mut out = vec![0f64; hd];
    for (p, &pr) in probs.iter().enumerate() {
        for (a, &v) in out.iter_mut().zip(kv.value(p)) {
            *a += pr * f64::from(v);
        }
    }
    (out, probs)
}

/// How one query head's retro attention compares with dense attention.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quality {
    /// Dense softmax mass on the exact-zone positions.
    pub mass_exact: f64,
    /// Dense softmax mass on the exact zone plus the estimation clusters' members.
    pub mass_total: f64,
    /// `|retro - dense| / |dense|` (L2).
    pub out_rel_err: f64,
    /// Jaccard of the exact positions against the `|exact|` most probable positions.
    pub jaccard: f64,
}

impl Quality {
    /// Componentwise mean of several measures.
    pub fn mean(all: &[Quality]) -> Quality {
        let n = all.len().max(1) as f64;
        let sum = |f: fn(&Quality) -> f64| all.iter().map(f).sum::<f64>() / n;
        Quality {
            mass_exact: sum(|q| q.mass_exact),
            mass_total: sum(|q| q.mass_total),
            out_rel_err: sum(|q| q.out_rel_err),
            jaccard: sum(|q| q.jaccard),
        }
    }
}

/// Measure one query head: select with the group's scores, run the merged
/// attention, and compare with dense attention over `[0, len)`.
#[allow(clippy::too_many_arguments)]
pub fn measure_head(
    kv: HeadKv<'_>,
    idx: &HeadIndex,
    c: &IndexCounts,
    zones: &Zones,
    q: &[f32],
    len: u32,
    scale: f32,
    cap: Option<usize>,
) -> Quality {
    let (exact, _) = exact_positions(idx, zones, c, len, cap);
    let (retro, _) = retro_attention(kv, idx, &exact, &zones.estimation, q, scale);
    let (dense, probs) = dense_attention(kv, len as usize, q, scale);
    let mass_exact: f64 = exact.iter().map(|&p| probs[p as usize]).sum();
    let mass_est: f64 = zones
        .estimation
        .iter()
        .flat_map(|&cl| idx.lists[cl as usize].iter())
        .filter(|&&p| p < len)
        .map(|&p| probs[p as usize])
        .sum();
    let err: f64 = retro
        .iter()
        .zip(&dense)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = dense.iter().map(|b| b * b).sum::<f64>().sqrt();
    Quality {
        mass_exact,
        mass_total: mass_exact + mass_est,
        out_rel_err: if norm > 0.0 { err / norm } else { err },
        jaccard: jaccard_top(&exact, &probs),
    }
}

/// Jaccard of `chosen` against the `chosen.len()` most probable positions
/// (ties toward the lower position).
pub fn jaccard_top(chosen: &[u32], probs: &[f64]) -> f64 {
    if chosen.is_empty() {
        return 0.0;
    }
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let k = chosen.len().min(order.len());
    let top: std::collections::HashSet<usize> = order[..k].iter().copied().collect();
    let set: std::collections::HashSet<usize> = chosen.iter().map(|&p| p as usize).collect();
    let inter = set.intersection(&top).count() as f64;
    let union = set.union(&top).count() as f64;
    inter / union
}

/// Measure a whole layer for one decode row: every query head, against its
/// KV head's index, averaged. `q` is `[num_q_heads][head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn measure_layer(
    heads: &[HeadKv<'_>],
    index: &[HeadIndex],
    c: &IndexCounts,
    q: &[f32],
    group: usize,
    len: u32,
    scale: f32,
    cap: Option<usize>,
) -> Quality {
    let hd = heads.first().map_or(0, |h| h.head_dim);
    let mut all = Vec::new();
    for (h, (kv, idx)) in heads.iter().zip(index).enumerate() {
        let qg = &q[h * group * hd..(h + 1) * group * hd];
        let scores = score_clusters(idx, c.n_centroids as usize, qg, group, scale);
        let zones = select(&scores, c.nprobe as usize, c.es as usize);
        for g in 0..group {
            all.push(measure_head(
                *kv,
                idx,
                c,
                &zones,
                &qg[g * hd..(g + 1) * hd],
                len,
                scale,
                cap,
            ));
        }
    }
    Quality::mean(&all)
}
