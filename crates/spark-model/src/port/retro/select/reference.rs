// SPDX-License-Identifier: AGPL-3.0-only
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:765-770, commit
// 03f912c. Changes: a CPU reference of Atlas's selection kernels (`retro_select_ada.cu`), with the
// kernel's logit arithmetic, an f64 softmax, and the kernel's exact total order and clamps.

//! Responsibility: the CPU reference of decode-time zone selection, used by the unit tests and by
//! the GPU microtest: the logits exactly as the scoring kernels form them (lane partials and the
//! warp butterfly, bit for bit), each query head's softmax over the non-empty clusters and the GQA
//! group sum in f64, the order key, the zone sizes with the kernel's clamps, and the selection by
//! the kernel's total order (key descending, then cluster id ascending). It agrees with L33's
//! `oracle_eval::{score_clusters, select}` up to the f32 rounding of the scores.

use crate::port::retro::build::reference::butterfly_sum;
use crate::port::retro::select::MAX_SELECTED;

const D: usize = 256;

/// One selection's inputs, all host values (BF16 data as their f32 values).
pub struct Input<'a> {
    /// `[kv_heads][cap][256]`.
    pub centroids: &'a [f32],
    /// `[kv_heads][cap]`.
    pub cluster_size: &'a [i32],
    pub cap: usize,
    /// Clusters of the index (`header[N_CENTROIDS]`).
    pub nc: usize,
    pub kv_heads: usize,
    pub group: usize,
    pub rows: usize,
    /// `[rows][kv_heads * group][256]`.
    pub q: &'a [f32],
    pub scale: f32,
    pub nprobe: u32,
    pub es: u32,
    pub r_cap: u32,
    pub e_cap: u32,
}

/// One KV head's result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Zones {
    /// Rank order.
    pub retrieval: Vec<u32>,
    /// Rank order, after the retrieval zone.
    pub estimation: Vec<u32>,
    /// The order key of every cluster `< nc` (0 = not selectable).
    pub keys: Vec<u32>,
}

/// `scale * q . c` with the scoring kernels' arithmetic: lane `l` sums dims `8l .. 8l + 7` in
/// order (each product rounded), the warp butterfly adds the lanes, then the scale.
pub fn logit(q: &[f32], c: &[f32], scale: f32) -> f32 {
    let mut lanes = [0.0f32; 32];
    for (l, lane) in lanes.iter_mut().enumerate() {
        for j in 0..8 {
            *lane += q[l * 8 + j] * c[l * 8 + j];
        }
    }
    butterfly_sum(lanes) * scale
}

/// The order key of a probability (`bits + 1`; an empty cluster's -1 maps to 0, never chosen).
pub fn order_key(p: f32) -> u32 {
    if p >= 0.0 { p.to_bits() + 1 } else { 0 }
}

/// Zone sizes with the kernel's clamps: `(kr, ks)`, the retrieval count and the total.
pub fn targets(nprobe: u32, es: u32, n_valid: u32, r_cap: u32, e_cap: u32) -> (u32, u32) {
    let kr = nprobe.min(n_valid).min(r_cap).min(MAX_SELECTED);
    let mut ks = (nprobe + es).min(n_valid);
    if ks.saturating_sub(kr) > e_cap {
        ks = kr + e_cap;
    }
    (kr, ks.min(MAX_SELECTED).max(kr))
}

/// The ranked clusters split into the first `kr` and the next `ks - kr`, both in rank order.
/// Keys of 0 are never picked.
pub fn pick(keys: &[u32], kr: u32, ks: u32) -> (Vec<u32>, Vec<u32>) {
    let mut order: Vec<u32> = (0..keys.len() as u32)
        .filter(|&c| keys[c as usize] > 0)
        .collect();
    order.sort_by(|&a, &b| keys[b as usize].cmp(&keys[a as usize]).then(a.cmp(&b)));
    let ks = (ks as usize).min(order.len());
    let kr = (kr as usize).min(ks);
    (order[..kr].to_vec(), order[kr..ks].to_vec())
}

/// The group-summed softmax probability of every cluster `< nc` of head `h` (f64), `None` for an
/// empty cluster. Each query head's softmax runs over the non-empty clusters only.
pub fn probabilities(inp: &Input, h: usize) -> Vec<Option<f64>> {
    let nq = inp.kv_heads * inp.group;
    let live: Vec<bool> = (0..inp.nc)
        .map(|c| inp.cluster_size[h * inp.cap + c] > 0)
        .collect();
    let mut p = vec![0.0f64; inp.nc];
    for r in 0..inp.rows {
        for g in 0..inp.group {
            let qh = h * inp.group + g;
            let q = &inp.q[(r * nq + qh) * D..(r * nq + qh + 1) * D];
            let s: Vec<f64> = (0..inp.nc)
                .map(|c| {
                    if !live[c] {
                        return f64::NEG_INFINITY;
                    }
                    let base = (h * inp.cap + c) * D;
                    f64::from(logit(q, &inp.centroids[base..base + D], inp.scale))
                })
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if m == f64::NEG_INFINITY {
                continue;
            }
            let l: f64 = s.iter().map(|&x| (x - m).exp()).sum();
            for c in 0..inp.nc {
                p[c] += (s[c] - m).exp() / l;
            }
        }
    }
    (0..inp.nc).map(|c| live[c].then_some(p[c])).collect()
}

/// The selection of every KV head.
pub fn select(inp: &Input) -> Vec<Zones> {
    (0..inp.kv_heads)
        .map(|h| {
            let probs = probabilities(inp, h);
            let keys: Vec<u32> = probs
                .iter()
                .map(|p| p.map_or(0, |v| order_key(v as f32)))
                .collect();
            let n_valid = keys.iter().filter(|&&k| k > 0).count() as u32;
            let (kr, ks) = targets(inp.nprobe, inp.es, n_valid, inp.r_cap, inp.e_cap);
            let (retrieval, estimation) = pick(&keys, kr, ks);
            Zones {
                retrieval,
                estimation,
                keys,
            }
        })
        .collect()
}
