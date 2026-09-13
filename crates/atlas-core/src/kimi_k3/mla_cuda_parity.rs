// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-parity + RST known-bad for `mla_decode.cu`. Compiles under ATLAS_SKIP_BUILD.

use std::path::Path;

use super::cache::MlaKv;
use super::mla::{MlaConfig, apply_output_gate, maybe_rope, mla_decode_token, sdpa_one};
use super::situ::sigmoid;

fn mla_cu() -> String {
    let p =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10/kimi-k3/bf16/mla_decode.cu");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

/// CUDA-loop nest of `k3_mla_sdpa_gate_f32` (thread-0, recompute dots).
#[allow(clippy::too_many_arguments)]
fn sdpa_gate_cuda_order(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    t: usize,
    heads: usize,
    dq: usize,
    dv: usize,
    use_gate: bool,
) -> Vec<f32> {
    let scale = 1.0 / (dq as f32).sqrt();
    let mut out = vec![0.0f32; heads * dv];
    for h in 0..heads {
        let qrow = &q[h * dq..(h + 1) * dq];
        let mut m = f32::NEG_INFINITY;
        for kj in 0..t {
            let krow = &k[(kj * heads + h) * dq..(kj * heads + h) * dq + dq];
            let s: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>() * scale;
            if s > m {
                m = s;
            }
        }
        let mut z = 0.0f32;
        for kj in 0..t {
            let krow = &k[(kj * heads + h) * dq..(kj * heads + h) * dq + dq];
            let s: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>();
            z += (s * scale - m).exp();
        }
        for d in 0..dv {
            let mut o = 0.0f32;
            for kj in 0..t {
                let krow = &k[(kj * heads + h) * dq..(kj * heads + h) * dq + dq];
                let s: f32 = qrow.iter().zip(krow).map(|(a, b)| a * b).sum::<f32>();
                let a = (s * scale - m).exp() / z;
                o += a * v[(kj * heads + h) * dv + d];
            }
            if use_gate {
                o *= sigmoid(g[h * dv + d]);
            }
            out[h * dv + d] = o;
        }
    }
    out
}

fn packed(heads: usize, dim: usize, seed: f32) -> Vec<f32> {
    (0..heads * dim)
        .map(|i| ((i % 13) as f32) * 0.07 - seed)
        .collect()
}

#[test]
fn cuda_loop_order_matches_cpu_oracle() {
    let cfg = MlaConfig::twin_0_40b();
    let (h, dq, dv) = (cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    let t = 4usize;
    let q = packed(h, dq, 0.3);
    let k = packed(t * h, dq, 0.2);
    let v = packed(t * h, dv, 0.1);
    let g = packed(h, dv, 0.4);
    let cpu = apply_output_gate(&sdpa_one(&q, &k, &v, t, h, dq, dv), &g, true);
    let cuda = sdpa_gate_cuda_order(&q, &k, &v, &g, t, h, dq, dv, true);
    let err = cpu
        .iter()
        .zip(&cuda)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err < 1e-5,
        "CUDA nest must match sdpa_one+gate (max abs {err})"
    );
}

#[test]
fn skip_output_gate_diverges() {
    let cfg = MlaConfig::twin_0_40b();
    let (h, dq, dv) = (cfg.heads, cfg.qk_head_dim(), cfg.v_head_dim);
    let q = packed(h, dq, 0.2);
    let k = packed(h, dq, 0.1);
    let v = packed(h, dv, 0.3);
    let g = vec![0.0f32; h * dv];
    let with = sdpa_gate_cuda_order(&q, &k, &v, &g, 1, h, dq, dv, true);
    let skip = sdpa_gate_cuda_order(&q, &k, &v, &g, 1, h, dq, dv, false);
    let err = with
        .iter()
        .zip(&skip)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err > 1e-4,
        "RST known-bad: skip output-gate (g=0) must diverge (max abs {err})"
    );
}

#[test]
fn apply_rope_when_nope_diverges() {
    let cfg = MlaConfig::twin_0_40b();
    let mut nope = packed(cfg.heads, cfg.qk_head_dim(), 0.5);
    let mut rope = nope.clone();
    maybe_rope(
        &mut nope,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        3,
        10000.0,
        true,
    );
    maybe_rope(
        &mut rope,
        cfg.qk_nope_head_dim,
        cfg.qk_rope_head_dim,
        3,
        10000.0,
        false,
    );
    let err = nope
        .iter()
        .zip(&rope)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err > 1e-4,
        "RST known-bad: apply RoPE when NoPE must diverge (max abs {err})"
    );
}

#[test]
fn mla_decode_token_appends() {
    let cfg = MlaConfig::twin_0_40b();
    let mut q = packed(cfg.heads, cfg.qk_head_dim(), 0.2);
    let mut k = packed(cfg.heads, cfg.qk_head_dim(), 0.1);
    let v = packed(cfg.heads, cfg.v_head_dim, 0.3);
    let g = packed(cfg.heads, cfg.v_head_dim, 0.0);
    let mut kv = MlaKv::default();
    let y = mla_decode_token(&mut q, &mut k, &v, &g, &mut kv, &cfg, 1, 10000.0);
    assert_eq!(y.len(), cfg.heads * cfg.v_head_dim);
    assert_eq!(kv.seq_len, 1);
}

#[test]
fn mla_decode_cu_is_k3_not_qwen_shadow() {
    let src = mla_cu();
    assert!(src.contains("k3_mla_maybe_rope_f32"));
    assert!(src.contains("k3_mla_sdpa_gate_f32"));
    assert!(src.contains("k3_sigmoid"));
    assert!(
        src.contains("use_nope"),
        "NoPE must be a kernel flag, not dropped rope slots"
    );
    assert!(src.contains("use_gate"), "output gate must live in-kernel");
    assert!(!src.contains("ms_mla_decode"));
    assert!(!src.contains("glm5next_dsa"));
    assert!(!src.contains("mla_paged_decode"));
    assert!(!src.contains("gated_delta_rule"));
}
