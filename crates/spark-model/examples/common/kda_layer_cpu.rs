// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `kda_layer_microtest.rs` to keep it under the 500-LoC cap.
//! Test-only harness code; no serving path runs any of it.

#![allow(unused_imports)]

use crate::*;
use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_model::layers::glm5next_kda::binding::{
    self, AttnBlockKind, KdaDtype, KdaTensorSource, RawTensor,
};
use spark_model::layers::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace, KdaSeqState,
};
use spark_model::layers::glm5next_kda_ref as kref;
use spark_model::weight_map::DenseWeight;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::collections::BTreeMap;

/// Host weights, already bf16-rounded exactly as they sit in the checkpoint.
pub(crate) struct Wts {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) conv: Vec<f32>,
    pub(crate) f_a: Vec<f32>,
    pub(crate) f_b: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) a_log: Vec<f32>,
    pub(crate) b: Vec<f32>,
    pub(crate) g_a: Vec<f32>,
    pub(crate) g_b: Vec<f32>,
    pub(crate) o_norm: Vec<f32>,
    pub(crate) o: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(crate) struct Dims {
    pub(crate) hid: usize,
    pub(crate) h: usize,
    pub(crate) d: usize,
    pub(crate) ks: usize,
}
impl Dims {
    pub(crate) fn qkv(&self) -> usize {
        self.h * self.d
    }
    pub(crate) fn conv_dim(&self) -> usize {
        3 * self.qkv()
    }
    pub(crate) fn qk(&self) -> usize {
        2 * self.qkv()
    }
}

/// `C = A @ B^T`, mirroring `dense_gemm_bf16`: bf16 operands, strictly ascending fp32
/// accumulation, bf16 result.
pub(crate) fn gemm_p(a: &[f32], w: &[f32], m: usize, n: usize, k: usize, pure: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = rq(acc, pure);
        }
    }
    out
}

pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// L2 over `head_dim` rows of the first `qk` channels, `1/sqrt(sum + eps)`, result bf16 —
/// what BOTH Atlas conv paths produce. V is left alone.
pub(crate) fn l2_qk_bf16(x: &mut [f32], dm: Dims, eps: f32, row_stride: usize, pure: bool) {
    for tok in x.chunks_exact_mut(row_stride) {
        for grp in tok[..dm.qk()].chunks_exact_mut(dm.d) {
            let inv = 1.0 / (grp.iter().map(|a| a * a).sum::<f32>() + eps).sqrt();
            for a in grp.iter_mut() {
                *a = rq(*a * inv, pure);
            }
        }
    }
}

/// Decode conv: shift-left 4-slot state, fp32 accumulate, SiLU, fused L2 on q|k, bf16 out.
pub(crate) fn cpu_conv_decode(
    state4: &mut [f32],
    tok: &[f32],
    w: &[f32],
    dm: Dims,
    eps: f32,
    pure: bool,
) -> Vec<f32> {
    let (dim, ks) = (dm.conv_dim(), dm.ks);
    let mut out = vec![0.0f32; dim];
    for ch in 0..dim {
        let s = &mut state4[ch * ks..(ch + 1) * ks];
        for i in 0..ks - 1 {
            s[i] = s[i + 1];
        }
        s[ks - 1] = tok[ch];
        let mut acc = 0.0f32;
        for k in 0..ks {
            acc += s[k] * w[ch * ks + k];
        }
        out[ch] = silu(acc);
    }
    l2_qk_bf16(&mut out, dm, eps, dim, pure);
    for x in out[dm.qk()..].iter_mut() {
        *x = rq(*x, pure);
    }
    out
}

/// Prefill conv: same window, SiLU only, bf16 out; L2 is a separate pass.
pub(crate) fn cpu_conv_prefill(
    state4: &mut [f32],
    toks: &[f32],
    w: &[f32],
    dm: Dims,
    t: usize,
    pure: bool,
) -> Vec<f32> {
    let (dim, ks) = (dm.conv_dim(), dm.ks);
    let mut out = vec![0.0f32; t * dim];
    for ch in 0..dim {
        let mut s = [0.0f32; 4];
        s[..ks].copy_from_slice(&state4[ch * ks..(ch + 1) * ks]);
        for (tt, o) in out.chunks_exact_mut(dim).enumerate() {
            let nv = toks[tt * dim + ch];
            for i in 0..ks - 1 {
                s[i] = s[i + 1];
            }
            s[ks - 1] = nv;
            let mut acc = 0.0f32;
            for k in 0..ks {
                acc += s[k] * w[ch * ks + k];
            }
            o[ch] = rq(silu(acc), pure);
        }
        state4[ch * ks..(ch + 1) * ks].copy_from_slice(&s[..ks]);
    }
    out
}

/// Every observable stage, on the host, in fp32 values that carry Atlas's exact bf16 ladder.
pub(crate) struct Stages {
    pub(crate) qkv_proj: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) beta: Vec<f32>,
    pub(crate) core: Vec<f32>,
    pub(crate) state: Vec<f32>,
    pub(crate) out_gate: Vec<f32>,
    pub(crate) o_norm: Vec<f32>,
    pub(crate) final_out: Vec<f32>,
    pub(crate) conv_state: Vec<f32>,
    /// Independent cross-check, populated on the pure-fp32 prefill pass only:
    /// `max_abs(final_out, glm5next_kda_ref::kda_reference_layer(...))`. That function is the
    /// handoff's named full-layer oracle and it drives the core through the **recurrent**
    /// formulation, so agreeing with it also re-proves chunk == recurrent at production
    /// geometry on the real weights.
    pub(crate) ref_layer_delta: Option<f64>,
}

/// CPU reference for one layer. `decode` selects the fused-conv + recurrent path; otherwise the
/// prefill-conv + separate-L2 + chunked path. `pure_f32 = true` drops every intermediate bf16
/// rounding, which is what floor A is measured with.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_layer(
    w: &Wts,
    dm: Dims,
    cfg: Glm5NextKdaConfig,
    hidden: &[f32],
    t: usize,
    conv_state4: &mut Vec<f32>,
    state: &mut Vec<f32>,
    decode: bool,
    chunk: usize,
    pure: bool,
) -> Stages {
    let (hid, qkv, hd) = (dm.hid, dm.qkv(), dm.d);
    let cd = dm.conv_dim();

    let mut qkv_proj = vec![0.0f32; t * cd];
    for (i, ww) in [&w.q, &w.k, &w.v].into_iter().enumerate() {
        let p = gemm_p(hidden, ww, t, qkv, hid, pure);
        for tt in 0..t {
            qkv_proj[tt * cd + i * qkv..tt * cd + (i + 1) * qkv]
                .copy_from_slice(&p[tt * qkv..(tt + 1) * qkv]);
        }
    }

    let mut pre_l2 = Vec::new();
    let conv_out = if decode {
        assert_eq!(t, 1);
        cpu_conv_decode(conv_state4, &qkv_proj, &w.conv, dm, cfg.l2_eps, pure)
    } else {
        let c = cpu_conv_prefill(conv_state4, &qkv_proj, &w.conv, dm, t, pure);
        pre_l2 = c.clone();
        let mut c = c;
        l2_qk_bf16(&mut c, dm, cfg.l2_eps, cd, pure);
        c
    };
    let pick_from = |src: &[f32], off: usize| -> Vec<f32> {
        (0..t)
            .flat_map(|tt| src[tt * cd + off..tt * cd + off + qkv].to_vec())
            .collect()
    };
    let (q, k, v) = (
        pick_from(&conv_out, 0),
        pick_from(&conv_out, qkv),
        pick_from(&conv_out, 2 * qkv),
    );

    let f_a = gemm_p(hidden, &w.f_a, t, hd, hid, pure);
    let g_raw = gemm_p(&f_a, &w.f_b, t, qkv, hd, pure);
    let kd = kref::KdaDims {
        hidden: hid,
        heads: dm.h,
        head_dim: hd,
        tokens: t,
    };
    let gate = kref::bounded_gate(&g_raw, &w.dt_bias, &w.a_log, kd, cfg.gate_lower_bound);
    let beta: Vec<f32> = gemm_p(hidden, &w.b, t, dm.h, hid, pure)
        .iter()
        .map(|x| sigmoid(*x))
        .collect();

    let core = if decode {
        kref::kda_recurrent_prenorm(&q, &k, &v, &gate, &beta, kd, state)
    } else {
        kref::kda_chunked_prenorm(&q, &k, &v, &gate, &beta, kd, chunk, state)
    };

    let g_a = gemm_p(hidden, &w.g_a, t, hd, hid, pure);
    let out_gate = gemm_p(&g_a, &w.g_b, t, qkv, hd, pure);
    let raw_norm = kref::rms_norm_gated(&core, &w.o_norm, &out_gate, hd, cfg.rms_norm_eps);
    let o_norm: Vec<f32> = if pure {
        raw_norm
    } else {
        round_bf16(&raw_norm)
    };
    let final_out = gemm_p(&o_norm, &w.o, t, hid, qkv, pure);

    // Cross-check against the handoff's named oracle. Only on the pure-fp32 prefill pass:
    // `kda_reference_layer` normalises q/k itself, so it needs the PRE-L2 conv output, and on
    // bf16 values a second normalisation would restore rounding (the Slice-4 hazard).
    let ref_layer_delta = if pure && !decode {
        let mut st2 = vec![0.0f32; dm.h * hd * hd];
        let rw = kref::KdaWeights {
            w_f_a: &w.f_a,
            w_f_b: &w.f_b,
            dt_bias: &w.dt_bias,
            a_log: &w.a_log,
            w_b: &w.b,
            w_g_a: &w.g_a,
            w_g_b: &w.g_b,
            o_norm_w: &w.o_norm,
            w_o: &w.o,
        };
        let out = kref::kda_reference_layer(
            hidden,
            &pick_from(&pre_l2, 0),
            &pick_from(&pre_l2, qkv),
            &pick_from(&pre_l2, 2 * qkv),
            &rw,
            kd,
            cfg.gate_lower_bound,
            cfg.rms_norm_eps,
            &mut st2,
        );
        Some(maxabs(&final_out, &out))
    } else {
        None
    };

    Stages {
        qkv_proj,
        q,
        k,
        v,
        gate,
        beta,
        core,
        state: state.clone(),
        out_gate,
        o_norm,
        final_out,
        conv_state: conv_state4.clone(),
        ref_layer_delta,
    }
}
