// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-parity + RST known-bad for `kda_decode.cu`. Compiles under ATLAS_SKIP_BUILD.
//!
//! CUDA launch is spark-model `kda_cuda` (spark2 nvcc). This file pins the
//! oracle and the planted mutants the RST sheet names.

use std::path::Path;

use super::kda::{
    KDA_L2_EPS, KdaConfig, KdaState, bounded_gate, kda_decode_token, kda_recurrent_step,
};
use super::situ::sigmoid;

fn kda_cu() -> String {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10/kimi-k3/bf16/kda_decode.cu");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn twin_step(conv_w: &[f32], beta: &[f32]) -> (Vec<f32>, KdaState) {
    let cfg = KdaConfig::twin_0_40b();
    let mut state = KdaState::new(&cfg);
    let x = vec![0.15f32; cfg.conv_dim()];
    let z = vec![0.1f32; cfg.qkv_dim()];
    let dt = vec![0.0f32; cfg.qkv_dim()];
    let a_log = vec![-0.5f32; cfg.heads];
    let gate = bounded_gate(
        &z,
        &dt,
        &a_log,
        cfg.heads,
        cfg.head_dim,
        cfg.gate_lower_bound,
    );
    let y = kda_decode_token(&x, conv_w, &gate, beta, &cfg, &mut state);
    (y, state)
}

/// CUDA-loop nest (V outer, K inner) of `k3_kda_recurrent_step_f32`.
fn recurrent_cuda_order(
    qkv: &[f32],
    gate: &[f32],
    beta: &[f32],
    cfg: &KdaConfig,
    recurrent: &mut [f32],
    apply_sigmoid: bool,
) -> Vec<f32> {
    let (h_n, d) = (cfg.heads, cfg.head_dim);
    let qkv_dim = h_n * d;
    let mut q = vec![0.0f32; qkv_dim];
    let mut k = vec![0.0f32; qkv_dim];
    for row in 0..h_n {
        let off = row * d;
        let mut qss = KDA_L2_EPS;
        let mut kss = KDA_L2_EPS;
        for i in 0..d {
            qss += qkv[off + i] * qkv[off + i];
            kss += qkv[qkv_dim + off + i] * qkv[qkv_dim + off + i];
        }
        let qinv = 1.0 / qss.sqrt();
        let kinv = 1.0 / kss.sqrt();
        for i in 0..d {
            q[off + i] = qkv[off + i] * qinv;
            k[off + i] = qkv[qkv_dim + off + i] * kinv;
        }
    }
    let v = &qkv[2 * qkv_dim..3 * qkv_dim];
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0f32; qkv_dim];
    for h in 0..h_n {
        let base = h * d;
        let s = &mut recurrent[h * d * d..(h + 1) * d * d];
        let b = if apply_sigmoid {
            sigmoid(beta[h])
        } else {
            beta[h]
        };
        for vi in 0..d {
            let mut kv = 0.0f32;
            for kk in 0..d {
                let idx = kk * d + vi;
                s[idx] *= gate[base + kk].exp();
                kv += s[idx] * k[base + kk];
            }
            let delta = (v[base + vi] - kv) * b;
            let mut acc = 0.0f32;
            for kk in 0..d {
                let idx = kk * d + vi;
                s[idx] += k[base + kk] * delta;
                acc += s[idx] * q[base + kk] * scale;
            }
            out[base + vi] = acc;
        }
    }
    out
}

#[test]
fn twin_0_40b_geometry() {
    let t = KdaConfig::twin_0_40b();
    assert_eq!(t.heads, 8);
    assert_eq!(t.head_dim, 32);
    assert_eq!(t.conv_kernel, 4);
    assert!(t.gate_lower_bound.is_none());
    assert_eq!(t.qkv_dim(), 256);
    assert_eq!(t.conv_dim(), 768);
}

#[test]
fn cuda_loop_order_matches_cpu_oracle() {
    let cfg = KdaConfig::twin_0_40b();
    let qkv: Vec<f32> = (0..cfg.conv_dim())
        .map(|i| ((i % 17) as f32) * 0.05 - 0.4)
        .collect();
    let gate = vec![-0.7f32; cfg.qkv_dim()];
    let beta = vec![0.3f32; cfg.heads];
    let mut a = vec![0.01f32; cfg.recurrent_elems()];
    let mut b = a.clone();
    let cpu = kda_recurrent_step(&qkv, &gate, &beta, &cfg, &mut a);
    let cuda = recurrent_cuda_order(&qkv, &gate, &beta, &cfg, &mut b, true);
    let err = cpu
        .iter()
        .zip(&cuda)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1e-5,
        "CUDA V-outer nest must match CPU oracle (max abs {err})"
    );
    assert_eq!(a, b);
}

#[test]
fn zero_conv_diverges() {
    let cfg = KdaConfig::twin_0_40b();
    let w = vec![0.2f32; cfg.conv_elems()];
    let zero = vec![0.0f32; cfg.conv_elems()];
    let beta = vec![0.4f32; cfg.heads];
    let (y, _) = twin_step(&w, &beta);
    let (z, _) = twin_step(&zero, &beta);
    let err = y
        .iter()
        .zip(&z)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err > 1e-4,
        "RST known-bad: zero conv must diverge (max abs {err})"
    );
}

#[test]
fn skip_sigmoid_diverges() {
    let cfg = KdaConfig::twin_0_40b();
    let qkv: Vec<f32> = (0..cfg.conv_dim())
        .map(|i| ((i % 11) as f32) * 0.08 - 0.3)
        .collect();
    let gate = vec![-0.4f32; cfg.qkv_dim()];
    let beta = vec![0.0f32; cfg.heads];
    let mut rec_sig = vec![0.0f32; cfg.recurrent_elems()];
    let mut rec_raw = rec_sig.clone();
    let with = kda_recurrent_step(&qkv, &gate, &beta, &cfg, &mut rec_sig);
    let skip = recurrent_cuda_order(&qkv, &gate, &beta, &cfg, &mut rec_raw, false);
    let err = with
        .iter()
        .zip(&skip)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err > 1e-4,
        "RST known-bad: skip sigmoid(beta=0) must diverge (max abs {err})"
    );
}

#[test]
fn kda_decode_cu_is_k3_not_gdn_shadow() {
    let src = kda_cu();
    assert!(src.contains("k3_kda_conv_update_f32"));
    assert!(src.contains("k3_kda_recurrent_step_f32"));
    assert!(
        src.contains("k3_sigmoid(beta"),
        "beta must be a logit; sigmoid lives in the kernel"
    );
    assert!(src.contains("k3_l2_row"));
    assert!(
        !src.contains("KDA_REC_BODY"),
        "must not copy common/kda_recurrent.cu"
    );
    assert!(
        !src.contains("gated_delta_rule_decode"),
        "must not paste GDN decode"
    );
    assert!(!src.contains("mamba2_ssm"));
}
