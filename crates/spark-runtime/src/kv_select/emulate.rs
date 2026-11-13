// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: an f32 host model of the scorer kernels (`kv_page_score_ada.cu`) that walks
//! the same [`ScorePlan`] tiles in the same order and reduces in the same tree shapes (a
//! lane-strided dot product folded by a warp xor tree, 256-thread block reductions folded
//! warp by warp). It proves on the CPU that the tiling (chunked merges, row tiles, the
//! stats / group-norm / accumulate split) computes the f64 oracle's scores to FP32
//! accuracy, so the GPU check only has to find kernel bugs, not plan bugs. Host `exp` may
//! differ from CUDA `expf` in the last place, so this is close to, not equal to, the device.

use anyhow::{Result, bail};

use super::GqaAgg;
use super::plan::{ScoreDims, ScorePlan};

/// A host copy of the sketch buffers plus one sequence's page list.
pub struct HostSketch<'a> {
    pub num_blocks: usize,
    pub layers: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// `[num_blocks, layers, kv_heads, head_dim]` BF16 bits.
    pub means: &'a [u16],
    /// `[num_blocks]`.
    pub counts: &'a [u8],
    /// The sequence's physical page ids, chronological.
    pub pages: &'a [u32],
}

/// A host copy of the captured query rows.
pub struct HostQuery<'a> {
    pub rows: usize,
    pub rows_cap: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub head_dim: usize,
    /// `[layers, rows_cap, q_heads, head_dim]` BF16 bits.
    pub data: &'a [u16],
}

fn bf(bits: u16) -> f32 {
    half::bf16::from_bits(bits).to_f32()
}

/// The warp xor-tree fold of 32 lane values; lane 0's result.
fn warp_fold(mut v: [f32; 32]) -> f32 {
    for off in [16usize, 8, 4, 2, 1] {
        let prev = v;
        for (l, x) in v.iter_mut().enumerate() {
            *x = prev[l] + prev[l ^ off];
        }
    }
    v[0]
}

/// A 256-thread block sum: each warp folds its lanes, warp 0 folds the eight warp sums.
fn block_fold(partials: &[f32; 256]) -> f32 {
    let mut warps = [0.0f32; 32];
    for (w, slot) in warps.iter_mut().take(8).enumerate() {
        let mut lanes = [0.0f32; 32];
        lanes.copy_from_slice(&partials[w * 32..w * 32 + 32]);
        *slot = warp_fold(lanes);
    }
    warp_fold(warps)
}

/// The device scores of one call, in f32, through the device's tiles and reductions.
pub fn score_blocks_f32(
    sk: &HostSketch,
    q: &HostQuery,
    pages_per_block: usize,
    gqa: GqaAgg,
    scratch_bytes: usize,
) -> Result<Vec<f32>> {
    if sk.layers != q.layers || sk.head_dim != q.head_dim || q.rows > q.rows_cap {
        bail!("emulate: sketch and query shapes disagree");
    }
    let dims = ScoreDims {
        n_pages: sk.pages.len(),
        pages_per_block,
        rows: q.rows,
        layers: sk.layers,
        q_heads: q.q_heads,
        kv_heads: sk.kv_heads,
        head_dim: sk.head_dim,
    };
    let plan = ScorePlan::new(dims, scratch_bytes)?;
    let (nb, kvh, d, qh) = (plan.n_blocks(), dims.kv_heads, dims.head_dim, dims.q_heads);
    let group = dims.group();
    let scale = 1.0f32 / (d as f32).sqrt();
    let heads_avg = if gqa == GqaAgg::Max { kvh } else { qh };
    let weight = 1.0f32 / (dims.layers * heads_avg) as f32;
    let mut scores = vec![0.0f32; nb];
    let mut merged = vec![0.0f32; plan.chunk_blocks * kvh * d];
    let mut wsum = vec![0.0f32; plan.chunk_blocks];
    let mut logits = vec![0.0f32; plan.rows_tile * qh * nb];

    let merge = |layer: usize, (c0, clen): (usize, usize), merged: &mut [f32], wsum: &mut [f32]| {
        for lb in 0..clen {
            let b = c0 + lb;
            let p0 = b * pages_per_block;
            let p1 = (p0 + pages_per_block).min(sk.pages.len());
            let mut tot = 0.0f32;
            for p in p0..p1 {
                tot += f32::from(sk.counts[sk.pages[p] as usize]);
            }
            for g in 0..kvh {
                for e in 0..d {
                    let mut acc = 0.0f32;
                    for p in p0..p1 {
                        let page = sk.pages[p] as usize;
                        let w = sk.counts[page];
                        if w != 0 {
                            let at = ((page * sk.layers + layer) * kvh + g) * d + e;
                            acc += f32::from(w) * bf(sk.means[at]);
                        }
                    }
                    merged[(lb * kvh + g) * d + e] = if tot > 0.0 { acc / tot } else { 0.0 };
                }
            }
            wsum[lb] = tot;
        }
    };

    for layer in 0..dims.layers {
        if plan.n_chunks() == 1 {
            merge(layer, plan.chunk(0), &mut merged, &mut wsum);
        }
        for t in 0..plan.n_row_tiles() {
            let (row0, rows_t) = plan.row_tile(t);
            for c in 0..plan.n_chunks() {
                if plan.n_chunks() > 1 {
                    merge(layer, plan.chunk(c), &mut merged, &mut wsum);
                }
                let (c0, clen) = plan.chunk(c);
                for lb in 0..clen {
                    for r in 0..rows_t {
                        for h in 0..qh {
                            let g = h / group;
                            let qa = ((layer * q.rows_cap + row0 + r) * qh + h) * d;
                            let ma = (lb * kvh + g) * d;
                            let mut lanes = [0.0f32; 32];
                            for (lane, slot) in lanes.iter_mut().enumerate() {
                                let mut e = lane;
                                while e < d {
                                    *slot += bf(q.data[qa + e]) * merged[ma + e];
                                    e += 32;
                                }
                            }
                            let v = warp_fold(lanes);
                            logits[(r * qh + h) * nb + c0 + lb] = if wsum[lb] > 0.0 {
                                v * scale
                            } else {
                                f32::NEG_INFINITY
                            };
                        }
                    }
                }
            }
            let stats: Vec<(f32, f32)> = (0..rows_t * qh)
                .map(|rh| {
                    let l = &logits[rh * nb..(rh + 1) * nb];
                    let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if mx == f32::NEG_INFINITY {
                        return (mx, 0.0);
                    }
                    let mut part = [0.0f32; 256];
                    for (tid, slot) in part.iter_mut().enumerate() {
                        let mut b = tid;
                        while b < nb {
                            *slot += (l[b] - mx).exp();
                            b += 256;
                        }
                    }
                    (mx, block_fold(&part))
                })
                .collect();
            let p = |r: usize, h: usize, b: usize| {
                let (m, s) = stats[r * qh + h];
                if s > 0.0 {
                    (logits[(r * qh + h) * nb + b] - m).exp() / s
                } else {
                    0.0
                }
            };
            let gnorm: Vec<f32> = if gqa == GqaAgg::Max {
                (0..rows_t * kvh)
                    .map(|rg| {
                        let (r, g) = (rg / kvh, rg % kvh);
                        let mut part = [0.0f32; 256];
                        for (tid, slot) in part.iter_mut().enumerate() {
                            let mut b = tid;
                            while b < nb {
                                let mut mx = 0.0f32;
                                for i in 0..group {
                                    mx = mx.max(p(r, g * group + i, b));
                                }
                                *slot += mx;
                                b += 256;
                            }
                        }
                        block_fold(&part)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            for (b, score) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for r in 0..rows_t {
                    if gqa == GqaAgg::Mean {
                        for h in 0..qh {
                            acc += p(r, h, b);
                        }
                    } else {
                        for g in 0..kvh {
                            let gn = gnorm[r * kvh + g];
                            if gn > 0.0 {
                                let mut mx = 0.0f32;
                                for i in 0..group {
                                    mx = mx.max(p(r, g * group + i, b));
                                }
                                acc += mx / gn;
                            }
                        }
                    }
                }
                *score += acc * weight;
            }
        }
    }
    Ok(scores)
}
