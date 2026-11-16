// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after Quest (page min/max upper-bound selection). The bound is the Quest page upper
// bound: score_p = sum_c max(q_c * kmin_c, q_c * kmax_c). The
// cross-layer combination (a per-layer z-score over the candidate pages, averaged over layers) is
// Atlas's own. No prototype source was read.

//! Responsibility: the pure arithmetic of the window desk's Quest page scorer (port lane L30, request
//! R3 of lane L36): the Quest bound of one page record against one layer's captured query rows, the
//! combination of the layers into one ranking, and the expansion of ranked pages into the logical
//! blocks the desk asks for. No GPU and no state, so `scorer_tests.rs` checks it against the CPU
//! reference of the kernels.
//!
//! # Page records
//! A record is one page's summary as the page-summary arena stores it, for every attention layer in
//! turn: per layer `kv_heads x [kmin; head_dim] [kmax; head_dim]` as f16 bits
//! (`LayerGeom::slot_halves` halves). It is copied from the arena when the page's leading block
//! leaves the GPU, because the arena is keyed by physical block and the block is reused afterwards.
//!
//! # Ranking
//! One selection serves every attention layer (the desk has one block table), so the per-layer
//! bounds are combined: each layer's bounds over the candidate pages become z-scores (raw bounds
//! differ in scale from layer to layer) and a page's rank value is the mean of its z-scores over the
//! layers that had a query. Ties go to the newer page. A page no layer could score is not ranked.

use std::collections::BTreeMap;
use std::sync::Arc;

use spark_runtime::kv_cache::page_summary::LayerGeom;

use super::reference::f16_val;

/// One page's summary for every attention layer, f16 bits (see the module doc).
pub type PageRecord = Arc<[u16]>;

/// One attention layer's captured query rows (post-RoPE, one row per verify token, usually one).
#[derive(Clone, Debug, PartialEq)]
pub struct LayerQuery {
    /// Attention-layer index (the KV pool index).
    pub layer: usize,
    pub num_q_heads: usize,
    /// Each row is `[num_q_heads * head_dim]`.
    pub rows: Vec<Vec<f32>>,
}

/// Halves of a whole page record.
pub fn record_halves(layers: &[LayerGeom]) -> usize {
    layers.iter().map(|l| l.slot_halves() as usize).sum()
}

/// Where each layer's halves start inside a page record.
pub fn layer_offsets(layers: &[LayerGeom]) -> Vec<usize> {
    let mut off = 0usize;
    layers
        .iter()
        .map(|l| {
            let at = off;
            off += l.slot_halves() as usize;
            at
        })
        .collect()
}

/// The Quest bound of one query head against one `(kmin, kmax)` pair: `sum_c max(q_c * kmin_c,
/// q_c * kmax_c)`, an upper bound of `q . k` for every key of the page. Zero query channels are
/// skipped, as the kernel does.
pub fn bound(q: &[f32], kmin: &[u16], kmax: &[u16]) -> f32 {
    let mut sum = 0.0f32;
    for ((&qc, &lo), &hi) in q.iter().zip(kmin).zip(kmax) {
        if qc == 0.0 {
            continue;
        }
        sum += (qc * f16_val(lo)).max(qc * f16_val(hi));
    }
    sum
}

/// One page's score on one layer: per query head the max over the captured rows, then the sum
/// (`gqa_sum`) or the max over every query head of every KV head (the kernel's `page_score`).
/// `None` when the layer's geometry does not match the record or the score is not finite.
pub fn layer_score(
    layers: &[LayerGeom],
    offsets: &[usize],
    record: &[u16],
    lq: &LayerQuery,
    gqa_sum: bool,
) -> Option<f32> {
    let g = layers.get(lq.layer)?;
    let (kvh, hd) = (g.kv_heads as usize, g.head_dim as usize);
    if kvh == 0
        || hd == 0
        || lq.num_q_heads == 0
        || !lq.num_q_heads.is_multiple_of(kvh)
        || lq.rows.is_empty()
    {
        return None;
    }
    let group = lq.num_q_heads / kvh;
    let start = *offsets.get(lq.layer)?;
    let rec = record.get(start..start + kvh * 2 * hd)?;
    let mut page = if gqa_sum { 0.0f32 } else { f32::NEG_INFINITY };
    for h in 0..kvh {
        let kmin = &rec[h * 2 * hd..h * 2 * hd + hd];
        let kmax = &rec[h * 2 * hd + hd..(h + 1) * 2 * hd];
        for j in 0..group {
            let qh = h * group + j;
            let mut head = f32::NEG_INFINITY;
            for row in &lq.rows {
                let q = row.get(qh * hd..(qh + 1) * hd)?;
                head = head.max(bound(q, kmin, kmax));
            }
            page = if gqa_sum { page + head } else { page.max(head) };
        }
    }
    page.is_finite().then_some(page)
}

/// Rank the pages `(page index, record)`, best first, as the module doc describes. Pages no layer
/// could score are left out.
pub fn rank_pages(
    layers: &[LayerGeom],
    queries: &[LayerQuery],
    pages: &[(u32, PageRecord)],
    gqa_sum: bool,
) -> Vec<u32> {
    let offsets = layer_offsets(layers);
    let n = pages.len();
    let mut total = vec![0.0f64; n];
    let mut used = vec![0u32; n];
    for lq in queries {
        let raw: Vec<Option<f32>> = pages
            .iter()
            .map(|(_, rec)| layer_score(layers, &offsets, rec, lq, gqa_sum))
            .collect();
        let vals: Vec<f64> = raw.iter().flatten().map(|&v| f64::from(v)).collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
        let std = var.sqrt().max(1e-9);
        for (i, r) in raw.iter().enumerate() {
            if let Some(v) = r {
                total[i] += (f64::from(*v) - mean) / std;
                used[i] += 1;
            }
        }
    }
    let value = |i: usize| total[i] / f64::from(used[i].max(1));
    let mut order: Vec<usize> = (0..n).filter(|&i| used[i] > 0).collect();
    order.sort_by(|&a, &b| {
        value(b)
            .total_cmp(&value(a))
            .then(pages[b].0.cmp(&pages[a].0))
    });
    order.into_iter().map(|i| pages[i].0).collect()
}

/// The logical blocks to recall: the candidate blocks of each ranked page, pages best first and
/// blocks ascending inside a page, at most `k` in all. `candidates` are the desk's demoted logical
/// block indices; a page is `page_blocks` consecutive blocks.
pub fn blocks_of(ranked_pages: &[u32], candidates: &[u32], page_blocks: u32, k: usize) -> Vec<u32> {
    let pb = page_blocks.max(1);
    let mut by_page: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &b in candidates {
        by_page.entry(b / pb).or_default().push(b);
    }
    for blocks in by_page.values_mut() {
        blocks.sort_unstable();
    }
    let mut out = Vec::new();
    for page in ranked_pages {
        let Some(blocks) = by_page.get(page) else {
            continue;
        };
        for &b in blocks {
            if out.len() >= k {
                return out;
            }
            out.push(b);
        }
    }
    out
}
