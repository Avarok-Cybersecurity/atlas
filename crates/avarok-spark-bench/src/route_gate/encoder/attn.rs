// SPDX-License-Identifier: AGPL-3.0-only
// Portions follow the ARC-1.0 release documentation and the public Qwen3.5 full-attention definition (per-head
// q/k RMSNorm, partial rotary, grouped-query attention, output gate). By Atlas contributors. See
// RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the gated full-attention layer of the Qwen3.5 text stack on the CPU: `q_proj` carries query and gate
//! per head, q and k are RMS-normalized per head, a partial rotary embedding is applied, attention is causal softmax
//! over grouped KV heads, and the output is multiplied by `sigmoid(gate)` before `o_proj`.

use rayon::prelude::*;

use super::TextConfig;
use super::ops::{Lin, maybe_round, rms_norm_1p};
use super::rope::RopeTables;

/// Weights of one full-attention layer.
pub struct AttnWeights {
    pub q: Lin,
    pub k: Lin,
    pub v: Lin,
    pub o: Lin,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Causal softmax attention of one head: `q` is `[t, hd]`, `k` and `v` are `[t, hd]` of the head's KV group.
/// Returns `[t, hd]`. The softmax runs in f32.
pub fn causal_attention(q: &[f32], k: &[f32], v: &[f32], t: usize, hd: usize) -> Vec<f32> {
    let scale = (hd as f32).powf(-0.5);
    let mut out = vec![0f32; t * hd];
    for i in 0..t {
        let qi = &q[i * hd..(i + 1) * hd];
        let mut scores: Vec<f32> = (0..=i)
            .map(|s| super::ops::dot_f32(qi, &k[s * hd..(s + 1) * hd]) * scale)
            .collect();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            total += *s;
        }
        let o = &mut out[i * hd..(i + 1) * hd];
        for (s, p) in scores.iter().enumerate() {
            let w = p / total;
            for (a, b) in o.iter_mut().zip(&v[s * hd..(s + 1) * hd]) {
                *a += w * b;
            }
        }
    }
    out
}

/// Forward of one full-attention layer: `x` is `[t, hidden]`, the result `[t, hidden]`.
pub fn attn_forward(
    c: &TextConfig,
    w: &AttnWeights,
    x: &[f32],
    t: usize,
    rope: &RopeTables,
    bf16: bool,
) -> Vec<f32> {
    let (heads, kvh, hd) = (c.heads, c.kv_heads, c.head_dim);
    let per_head = if c.attn_gate { 2 * hd } else { hd };
    let qg = w.q.forward(x, t, bf16);
    let kp = w.k.forward(x, t, bf16);
    let vp = w.v.forward(x, t, bf16);

    let mut queries = vec![0f32; t * heads * hd];
    let mut gates = vec![0f32; t * heads * hd];
    for i in 0..t {
        for h in 0..heads {
            let src = (i * heads + h) * per_head;
            queries[(i * heads + h) * hd..(i * heads + h + 1) * hd]
                .copy_from_slice(&qg[src..src + hd]);
            if c.attn_gate {
                gates[(i * heads + h) * hd..(i * heads + h + 1) * hd]
                    .copy_from_slice(&qg[src + hd..src + 2 * hd]);
            }
        }
    }
    let mut queries = rms_norm_1p(&queries, &w.q_norm, c.eps, bf16);
    let mut keys = rms_norm_1p(&kp, &w.k_norm, c.eps, bf16);
    for i in 0..t {
        for h in 0..heads {
            rope.apply(
                &mut queries[(i * heads + h) * hd..(i * heads + h + 1) * hd],
                i,
            );
        }
        for h in 0..kvh {
            rope.apply(&mut keys[(i * kvh + h) * hd..(i * kvh + h + 1) * hd], i);
        }
    }
    let queries = maybe_round(queries, bf16);
    let keys = maybe_round(keys, bf16);
    let group = heads / kvh;
    let per: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            let kv = h / group;
            let pick = |src: &[f32], n: usize, idx: usize| -> Vec<f32> {
                (0..t)
                    .flat_map(|i| src[(i * n + idx) * hd..(i * n + idx + 1) * hd].to_vec())
                    .collect()
            };
            causal_attention(
                &pick(&queries, heads, h),
                &pick(&keys, kvh, kv),
                &pick(&vp, kvh, kv),
                t,
                hd,
            )
        })
        .collect();
    let mut merged = vec![0f32; t * heads * hd];
    for (h, head) in per.iter().enumerate() {
        for i in 0..t {
            merged[(i * heads + h) * hd..(i * heads + h + 1) * hd]
                .copy_from_slice(&head[i * hd..(i + 1) * hd]);
        }
    }
    let mut merged = maybe_round(merged, bf16);
    if c.attn_gate {
        for (m, g) in merged.iter_mut().zip(&gates) {
            *m *= 1.0 / (1.0 + (-g).exp());
        }
        merged = maybe_round(merged, bf16);
    }
    w.o.forward(&merged, t, bf16)
}
