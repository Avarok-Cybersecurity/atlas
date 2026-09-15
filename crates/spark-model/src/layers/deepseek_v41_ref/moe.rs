// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 **MoE** (`Gate`, `Expert`, `MoE` in the reference), CPU reference against the
//! golden: sqrt-softplus routing with a correction bias that picks experts but does not scale
//! them, top-k renormalisation, `route_scale`, the SwiGLU clamp, and one shared expert every
//! token goes through.
//!
//! Precision follows the reference line by line. The gate runs in f32 from a bf16 `x`. Each
//! expert `Linear` is a bf16 GEMM (f32 accumulate, bf16 result), the SwiGLU is f32, the routing
//! weight multiplies in f32, the product is cast back to bf16 before `w2`, and the per-expert
//! outputs accumulate into an f32 `y` that is cast to bf16 once at the end. Accumulation order
//! inside a GEMM is the only thing that differs from torch, so outputs are compared at bf16
//! tolerance while the routing weights and indices are compared exactly.

use super::to_bf16_rne;

/// `F.softplus(x)` with torch's defaults (`beta = 1`, `threshold = 20`).
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `F.linear` on bf16 operands: `y[i][o] = bf16(sum_d x[i][d] * w[o][d])`, f32 accumulate.
/// `w` is `[out, in]` row-major, as `nn.Linear` stores it.
pub fn linear_bf16(x: &[f32], w: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * out_dim];
    for i in 0..rows {
        let xi = &x[i * in_dim..(i + 1) * in_dim];
        for o in 0..out_dim {
            let wo = &w[o * in_dim..(o + 1) * in_dim];
            let acc: f32 = xi.iter().zip(wo).map(|(a, b)| a * b).sum();
            y[i * out_dim + o] = to_bf16_rne(acc);
        }
    }
    y
}

/// `Gate.forward` for `score_func = "sqrtsoftplus"`. `x`: `[tokens, dim]` bf16 values;
/// `w`: `[n_routed, dim]` bf16 values; `bias`: `[n_routed]` f32. Returns
/// (`weights[tokens, topk]` f32, `indices[tokens, topk]`), indices in torch's `topk` order
/// (descending by `score + bias`).
pub fn gate(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    tokens: usize,
    dim: usize,
    n_routed: usize,
    topk: usize,
    gate_temp: f32,
    norm_topk_prob: bool,
    route_scale: f32,
) -> (Vec<f32>, Vec<usize>) {
    let mut weights = Vec::with_capacity(tokens * topk);
    let mut indices = Vec::with_capacity(tokens * topk);
    for t in 0..tokens {
        let xt = &x[t * dim..(t + 1) * dim];
        let scores: Vec<f32> = (0..n_routed)
            .map(|e| {
                let s: f32 = xt.iter().zip(&w[e * dim..(e + 1) * dim]).map(|(a, b)| a * b).sum();
                softplus(s / gate_temp).sqrt()
            })
            .collect();
        // the bias picks experts but does not scale them
        let mut order: Vec<usize> = (0..n_routed).collect();
        order.sort_by(|&a, &b| (scores[b] + bias[b]).partial_cmp(&(scores[a] + bias[a])).expect("finite scores"));
        let picked = &order[..topk];
        let mut wt: Vec<f32> = picked.iter().map(|&e| scores[e]).collect();
        if norm_topk_prob && topk > 1 {
            let sum: f32 = wt.iter().sum::<f32>() + 1e-20;
            for v in &mut wt {
                *v /= sum;
            }
        }
        for v in &mut wt {
            *v *= route_scale;
        }
        weights.extend(wt);
        indices.extend_from_slice(picked);
    }
    (weights, indices)
}

/// One SwiGLU expert. `w1`, `w3`: `[inter, dim]`; `w2`: `[dim, inter]`; all bf16 values.
/// `x`: `[rows, dim]` bf16 values; `weights`: per-row routing weight (f32) or `None` for the
/// shared expert. Returns `[rows, dim]` bf16 values.
pub fn expert(
    x: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    rows: usize,
    dim: usize,
    inter: usize,
    swiglu_limit: f32,
    weights: Option<&[f32]>,
) -> Vec<f32> {
    let gate_b = linear_bf16(x, w1, rows, dim, inter);
    let up_b = linear_bf16(x, w3, rows, dim, inter);
    let mut h = vec![0f32; rows * inter];
    for r in 0..rows {
        for j in 0..inter {
            let mut g = gate_b[r * inter + j];
            let mut u = up_b[r * inter + j];
            if swiglu_limit > 0.0 {
                u = u.clamp(-swiglu_limit, swiglu_limit);
                g = g.min(swiglu_limit);
            }
            let mut v = silu(g) * u;
            if let Some(wr) = weights {
                v *= wr[r];
            }
            h[r * inter + j] = to_bf16_rne(v);
        }
    }
    linear_bf16(&h, w2, rows, inter, dim)
}

/// The routed experts' parameters, `experts[i] = (w1, w2, w3)`.
pub struct MoeWeights<'a> {
    pub gate_w: &'a [f32],
    pub gate_bias: &'a [f32],
    pub experts: Vec<(&'a [f32], &'a [f32], &'a [f32])>,
    pub shared: (&'a [f32], &'a [f32], &'a [f32]),
}

pub struct MoeCfg {
    pub dim: usize,
    pub inter: usize,
    pub n_routed: usize,
    pub topk: usize,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,
}

/// `MoE.forward` (single rank). Returns (`y[tokens, dim]` bf16 values, routing `weights`,
/// routing `indices`).
pub fn moe(x: &[f32], tokens: usize, w: &MoeWeights, c: &MoeCfg) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let (weights, indices) = gate(
        x, w.gate_w, w.gate_bias, tokens, c.dim, c.n_routed, c.topk, c.gate_temp, c.norm_topk_prob, c.route_scale,
    );
    let mut y = vec![0f32; tokens * c.dim];
    for (e, (w1, w2, w3)) in w.experts.iter().enumerate() {
        // rows routed to expert e, in token order, with the matching routing weight
        let mut rows = Vec::new();
        let mut rw = Vec::new();
        for t in 0..tokens {
            for k in 0..c.topk {
                if indices[t * c.topk + k] == e {
                    rows.push(t);
                    rw.push(weights[t * c.topk + k]);
                }
            }
        }
        if rows.is_empty() {
            continue;
        }
        let xs: Vec<f32> = rows.iter().flat_map(|&t| x[t * c.dim..(t + 1) * c.dim].iter().copied()).collect();
        let out = expert(&xs, w1, w2, w3, rows.len(), c.dim, c.inter, c.swiglu_limit, Some(&rw));
        for (i, &t) in rows.iter().enumerate() {
            for d in 0..c.dim {
                y[t * c.dim + d] += out[i * c.dim + d];
            }
        }
    }
    let (s1, s2, s3) = w.shared;
    let shared = expert(x, s1, s2, s3, tokens, c.dim, c.inter, c.swiglu_limit, None);
    for (yv, sv) in y.iter_mut().zip(&shared) {
        *yv += sv;
    }
    (y.into_iter().map(to_bf16_rne).collect(), weights, indices)
}

#[cfg(test)]
#[path = "moe_tests.rs"]
mod tests;
