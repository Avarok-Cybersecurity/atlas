// SPDX-License-Identifier: AGPL-3.0-only
// Portions follow the ARC-1.0 release documentation and the public Qwen3.5 gated-DeltaNet definition (causal conv,
// L2-normalized q and k, decay and beta gates, delta-rule state update, gated RMSNorm). The numerics were checked
// against the Atlas CPU references (crates/spark-model/src/layers/qwen3_ssm/debug.rs and the GDN parity tests, which
// were read, not modified). By Atlas contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the gated DeltaNet (linear attention) layer of the Qwen3.5 text stack on the CPU, for one sequence
//! with no cache, as the plain per-token recurrence (the chunked form used on the GPU is mathematically equal).

use rayon::prelude::*;

use super::TextConfig;
use super::ops::{Lin, l2_normalize, maybe_round, rms_norm_gated, silu, softplus};

/// Weights of one linear-attention layer.
pub struct GdnWeights {
    pub in_qkv: Lin,
    pub in_z: Lin,
    pub in_b: Lin,
    pub in_a: Lin,
    /// Depthwise conv taps, `[conv_dim, kernel]`.
    pub conv: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// Gated-norm weight `[value_head_dim]` (plain, not `1 +`).
    pub norm: Vec<f32>,
    pub out: Lin,
}

/// The per-token delta-rule recurrence for one head. `q`, `k` are `[t, dk]` (already normalized, `q` scaled), `v` is
/// `[t, dv]`, `g` the log-decay and `beta` the write strength per token. Returns `[t, dv]`.
pub fn delta_rule_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let t = g.len();
    let mut state = vec![0f32; dk * dv];
    let mut out = vec![0f32; t * dv];
    let mut kv_mem = vec![0f32; dv];
    for i in 0..t {
        let decay = g[i].exp();
        let (qi, ki, vi) = (
            &q[i * dk..(i + 1) * dk],
            &k[i * dk..(i + 1) * dk],
            &v[i * dv..(i + 1) * dv],
        );
        for s in state.iter_mut() {
            *s *= decay;
        }
        kv_mem.iter_mut().for_each(|m| *m = 0.0);
        for (kk, &kval) in ki.iter().enumerate() {
            for (m, s) in kv_mem.iter_mut().zip(&state[kk * dv..(kk + 1) * dv]) {
                *m += s * kval;
            }
        }
        let delta: Vec<f32> = (0..dv).map(|j| (vi[j] - kv_mem[j]) * beta[i]).collect();
        let o = &mut out[i * dv..(i + 1) * dv];
        for kk in 0..dk {
            let kval = ki[kk];
            let row = &mut state[kk * dv..(kk + 1) * dv];
            for j in 0..dv {
                row[j] += kval * delta[j];
                o[j] += row[j] * qi[kk];
            }
        }
    }
    out
}

/// Forward of one linear-attention layer: `x` is `[t, hidden]`, the result `[t, hidden]`.
pub fn gdn_forward(c: &TextConfig, w: &GdnWeights, x: &[f32], t: usize, bf16: bool) -> Vec<f32> {
    let (kh, vh, dk, dv, ck) = (
        c.lin_k_heads,
        c.lin_v_heads,
        c.lin_k_dim,
        c.lin_v_dim,
        c.conv_kernel,
    );
    let (key_dim, value_dim) = (kh * dk, vh * dv);
    let conv_dim = 2 * key_dim + value_dim;
    let qkv = w.in_qkv.forward(x, t, bf16);
    let z = w.in_z.forward(x, t, bf16);
    let b = w.in_b.forward(x, t, bf16);
    let a = w.in_a.forward(x, t, bf16);

    // causal depthwise conv over time, then SiLU
    let mut mixed = vec![0f32; t * conv_dim];
    mixed
        .par_chunks_mut(conv_dim)
        .enumerate()
        .for_each(|(i, row)| {
            for (ch, out) in row.iter_mut().enumerate() {
                let mut s = 0f32;
                for j in 0..ck {
                    let src = i + j;
                    if src + 1 >= ck {
                        s += w.conv[ch * ck + j] * qkv[(src + 1 - ck) * conv_dim + ch];
                    }
                }
                *out = s;
            }
        });
    let mixed = maybe_round(mixed, bf16);
    let mixed = maybe_round(mixed.into_iter().map(silu).collect(), bf16);

    let ratio = vh / kh;
    let beta = maybe_round(b.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect(), bf16);
    let mut g = vec![0f32; t * vh];
    for i in 0..t {
        for h in 0..vh {
            g[i * vh + h] = -w.a_log[h].exp() * softplus(a[i * vh + h] + w.dt_bias[h]);
        }
    }
    let heads: Vec<Vec<f32>> = (0..vh)
        .into_par_iter()
        .map(|h| {
            let kvh = h / ratio;
            let take = |base: usize, width: usize, head: usize| -> Vec<f32> {
                (0..t)
                    .flat_map(|i| {
                        mixed[i * conv_dim + base + head * width
                            ..i * conv_dim + base + (head + 1) * width]
                            .to_vec()
                    })
                    .collect()
            };
            let mut q = take(0, dk, kvh);
            let mut k = take(key_dim, dk, kvh);
            let v = take(2 * key_dim, dv, h);
            let scale = (dk as f32).powf(-0.5);
            for i in 0..t {
                l2_normalize(&mut q[i * dk..(i + 1) * dk], 1e-6);
                l2_normalize(&mut k[i * dk..(i + 1) * dk], 1e-6);
            }
            q.iter_mut().for_each(|v| *v *= scale);
            let gh: Vec<f32> = (0..t).map(|i| g[i * vh + h]).collect();
            let bh: Vec<f32> = (0..t).map(|i| beta[i * vh + h]).collect();
            delta_rule_head(&q, &k, &v, &gh, &bh, dk, dv)
        })
        .collect();
    let mut core = vec![0f32; t * value_dim];
    for (h, head) in heads.iter().enumerate() {
        for i in 0..t {
            core[i * value_dim + h * dv..i * value_dim + (h + 1) * dv]
                .copy_from_slice(&head[i * dv..(i + 1) * dv]);
        }
    }
    let core = maybe_round(core, bf16);
    let normed = rms_norm_gated(&core, &z, &w.norm, c.eps, bf16);
    w.out.forward(&normed, t, bf16)
}
