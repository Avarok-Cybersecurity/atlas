// SPDX-License-Identifier: AGPL-3.0-only
//! Turbo3 + Turbo2 kernel parity: byte-exact quantizer checks against
//! CPU references (3-bit pack / 2-bit pack + matched-norm E4M3 scales)
//! and decode attention vs a CPU reference over the CPU-dequantized
//! cache. Shares E4M3 + test-value helpers with the other
//! `parity_turbo*.rs` files.

#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{GpuBackend, KernelArg};

const TURBO3_CODEBOOK: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];
const TURBO3_BOUNDS: [f32; 7] = [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748];
const TURBO3_MAX: f32 = 2.1520;

const TURBO2_CODEBOOK: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
const TURBO2_BOUNDS: [f32; 3] = [-0.9816, 0.0, 0.9816];
const TURBO2_MAX: f32 = 1.5104;

fn quantize(x: f32, bounds: &[f32]) -> u8 {
    let mut idx = 0u8;
    while (idx as usize) < bounds.len() && x >= bounds[idx as usize] {
        idx += 1;
    }
    idx
}

/// CPU mirror of one group of 16 through the turbo2/turbo3 append
/// kernels: (indices, E4M3 scale byte).
fn cpu_quant_group(vals: &[f32; 16], codebook: &[f32], bounds: &[f32], max: f32) -> ([u8; 16], u8) {
    let norm_sq: f32 = vals.iter().map(|v| v * v).sum();
    let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let inv = if amax > 1e-12 { max / amax } else { 1.0 };
    let mut idx = [0u8; 16];
    let mut recon_sq = 0.0f32;
    for i in 0..16 {
        idx[i] = quantize(vals[i] * inv, bounds);
        let c = codebook[idx[i] as usize];
        recon_sq += c * c;
    }
    let recon_norm = recon_sq.sqrt();
    let scale = if recon_norm > 1e-10 {
        norm_sq.sqrt() / recon_norm
    } else {
        amax / max
    };
    (idx, cpu_f32_to_e4m3(scale.min(448.0)))
}

fn pack8x3(idx: &[u8]) -> [u8; 3] {
    [
        idx[0] | (idx[1] << 3) | (idx[2] << 6),
        (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7),
        (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5),
    ]
}

/// Run one token through an append kernel and return host copies of
/// (k_data, k_scales) with the given per-token byte sizes.
#[allow(clippy::too_many_arguments)]
fn run_append(
    backend: &crate::metal_backend::MetalGpuBackend,
    kernel_mod: &str,
    new_k: &[half::bf16],
    new_v: &[half::bf16],
    num_kv_heads: u32,
    head_dim: u32,
    data_row_bytes: usize,
    scale_row_bytes: usize,
) -> (Vec<u8>, Vec<u8>) {
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let kernel = backend
        .kernel(kernel_mod, kernel_mod)
        .expect("kernel lookup");
    let k_src = backend.alloc(n_elems * 2).expect("alloc");
    let v_src = backend.alloc(n_elems * 2).expect("alloc");
    backend
        .copy_h2d(&bf16_slice_to_bytes(new_k), k_src)
        .expect("h2d");
    backend
        .copy_h2d(&bf16_slice_to_bytes(new_v), v_src)
        .expect("h2d");
    let k_data = backend.alloc(data_row_bytes).expect("alloc");
    let v_data = backend.alloc(data_row_bytes).expect("alloc");
    let k_scales = backend.alloc(scale_row_bytes).expect("alloc");
    let v_scales = backend.alloc(scale_row_bytes).expect("alloc");
    let cache_pos = 0u32;
    backend
        .launch_typed(
            kernel,
            [(num_groups as u32).div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&cache_pos.to_le_bytes()),
                KernelArg::Buffer(k_src),
                KernelArg::Buffer(v_src),
                KernelArg::Buffer(k_data),
                KernelArg::Buffer(v_data),
                KernelArg::Buffer(k_scales),
                KernelArg::Buffer(v_scales),
            ],
        )
        .expect("launch");
    backend.synchronize(backend.default_stream()).expect("sync");
    let mut data_h = vec![0u8; data_row_bytes];
    let mut scales_h = vec![0u8; scale_row_bytes];
    backend.copy_d2h(k_data, &mut data_h).expect("d2h");
    backend.copy_d2h(k_scales, &mut scales_h).expect("d2h");
    (data_h, scales_h)
}

/// `kv_cache_append_turbo3` produces byte-identical 3-bit packs +
/// E4M3 scales to the CPU reference.
#[test]
fn metal_kv_cache_append_turbo3_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let num_kv_heads = 2u32;
    let head_dim = 256u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let new_k: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i)))
        .collect();
    let new_v: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i + 100_000)))
        .collect();
    let (data_h, scales_h) = run_append(
        &backend,
        "kv_cache_append_turbo3",
        &new_k,
        &new_v,
        num_kv_heads,
        head_dim,
        n_elems * 3 / 8,
        num_groups,
    );
    for g in 0..num_groups {
        let mut vals = [0.0f32; 16];
        for i in 0..16 {
            vals[i] = f32::from(new_k[g * 16 + i]);
        }
        let (idx, want_scale) =
            cpu_quant_group(&vals, &TURBO3_CODEBOOK, &TURBO3_BOUNDS, TURBO3_MAX);
        assert_eq!(scales_h[g], want_scale, "k scale mismatch at group {g}");
        let lo = pack8x3(&idx[0..8]);
        let hi = pack8x3(&idx[8..16]);
        for b in 0..3 {
            assert_eq!(data_h[g * 6 + b], lo[b], "k pack lo mismatch g={g} b={b}");
            assert_eq!(
                data_h[g * 6 + 3 + b],
                hi[b],
                "k pack hi mismatch g={g} b={b}"
            );
        }
    }
}

/// `kv_cache_append_turbo2` produces byte-identical 2-bit packs +
/// E4M3 scales to the CPU reference.
#[test]
fn metal_kv_cache_append_turbo2_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let num_kv_heads = 2u32;
    let head_dim = 256u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let new_k: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i + 17)))
        .collect();
    let new_v: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i + 200_000)))
        .collect();
    let (data_h, scales_h) = run_append(
        &backend,
        "kv_cache_append_turbo2",
        &new_k,
        &new_v,
        num_kv_heads,
        head_dim,
        n_elems / 4,
        num_groups,
    );
    for g in 0..num_groups {
        let mut vals = [0.0f32; 16];
        for i in 0..16 {
            vals[i] = f32::from(new_k[g * 16 + i]);
        }
        let (idx, want_scale) =
            cpu_quant_group(&vals, &TURBO2_CODEBOOK, &TURBO2_BOUNDS, TURBO2_MAX);
        assert_eq!(scales_h[g], want_scale, "k scale mismatch at group {g}");
        for b in 0..4 {
            let want =
                idx[b * 4] | (idx[b * 4 + 1] << 2) | (idx[b * 4 + 2] << 4) | (idx[b * 4 + 3] << 6);
            assert_eq!(data_h[g * 4 + b], want, "k pack mismatch g={g} b={b}");
        }
    }
}

/// `attention_decode_turbo3` and `attention_decode_turbo2` match a CPU
/// reference attending over the CPU-dequantized cache (sparse-V gate
/// mirrored).
#[test]
fn metal_attention_decode_turbo23_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    for (append_mod, attn_mod, codebook, bounds, cmax, row_div) in [
        (
            "kv_cache_append_turbo3",
            "attention_decode_turbo3",
            &TURBO3_CODEBOOK[..],
            &TURBO3_BOUNDS[..],
            TURBO3_MAX,
            (3usize, 8usize),
        ),
        (
            "kv_cache_append_turbo2",
            "attention_decode_turbo2",
            &TURBO2_CODEBOOK[..],
            &TURBO2_BOUNDS[..],
            TURBO2_MAX,
            (1usize, 4usize),
        ),
    ] {
        let append = backend.kernel(append_mod, append_mod).expect("append");
        let attn = backend.kernel(attn_mod, attn_mod).expect("attn");

        let num_heads = 4u32;
        let num_kv_heads = 2u32;
        let head_dim = 128u32;
        let seq_len = 6u32;
        let n_elems = (num_kv_heads * head_dim) as usize;
        let num_groups = n_elems / 16;
        let row_bytes = n_elems * row_div.0 / row_div.1;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let k_data = backend.alloc(seq_len as usize * row_bytes).expect("alloc");
        let v_data = backend.alloc(seq_len as usize * row_bytes).expect("alloc");
        let k_scales = backend.alloc(seq_len as usize * num_groups).expect("alloc");
        let v_scales = backend.alloc(seq_len as usize * num_groups).expect("alloc");
        let k_src = backend.alloc(n_elems * 2).expect("alloc");
        let v_src = backend.alloc(n_elems * 2).expect("alloc");

        let mut k_deq = vec![0.0f32; seq_len as usize * n_elems];
        let mut v_deq = vec![0.0f32; seq_len as usize * n_elems];
        for s in 0..seq_len {
            let tok_k: Vec<half::bf16> = (0..n_elems)
                .map(|i| half::bf16::from_f32(test_val(i + 1000 * s as usize)))
                .collect();
            let tok_v: Vec<half::bf16> = (0..n_elems)
                .map(|i| half::bf16::from_f32(test_val(i + 1000 * s as usize + 500_000)))
                .collect();
            backend
                .copy_h2d(&bf16_slice_to_bytes(&tok_k), k_src)
                .expect("h2d");
            backend
                .copy_h2d(&bf16_slice_to_bytes(&tok_v), v_src)
                .expect("h2d");
            backend
                .launch_typed(
                    append,
                    [(num_groups as u32).div_ceil(64), 1, 1],
                    [64, 1, 1],
                    0,
                    backend.default_stream(),
                    &[
                        KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                        KernelArg::Bytes(&head_dim.to_le_bytes()),
                        KernelArg::Bytes(&s.to_le_bytes()),
                        KernelArg::Buffer(k_src),
                        KernelArg::Buffer(v_src),
                        KernelArg::Buffer(k_data),
                        KernelArg::Buffer(v_data),
                        KernelArg::Buffer(k_scales),
                        KernelArg::Buffer(v_scales),
                    ],
                )
                .expect("append launch");
            backend.synchronize(backend.default_stream()).expect("sync");

            for g in 0..num_groups {
                for (src, deq) in [(&tok_k, &mut k_deq), (&tok_v, &mut v_deq)] {
                    let mut vals = [0.0f32; 16];
                    for i in 0..16 {
                        vals[i] = f32::from(src[g * 16 + i]);
                    }
                    let (idx, scale_byte) = cpu_quant_group(&vals, codebook, bounds, cmax);
                    let gs = cpu_e4m3_to_f32(scale_byte);
                    for i in 0..16 {
                        deq[s as usize * n_elems + g * 16 + i] = codebook[idx[i] as usize] * gs;
                    }
                }
            }
        }

        let q: Vec<half::bf16> = (0..(num_heads * head_dim) as usize)
            .map(|i| half::bf16::from_f32(test_val(i + 333)))
            .collect();
        let q_buf = backend.alloc(q.len() * 2).expect("alloc");
        backend
            .copy_h2d(&bf16_slice_to_bytes(&q), q_buf)
            .expect("h2d");
        let out_buf = backend.alloc(q.len() * 2).expect("alloc");

        backend
            .launch_typed(
                attn,
                [num_heads, 1, 1],
                [32, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&seq_len.to_le_bytes()),
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                    KernelArg::Bytes(&head_dim.to_le_bytes()),
                    KernelArg::Bytes(&scale.to_le_bytes()),
                    KernelArg::Bytes(&1e-3f32.to_le_bytes()),
                    KernelArg::Buffer(q_buf),
                    KernelArg::Buffer(k_data),
                    KernelArg::Buffer(v_data),
                    KernelArg::Buffer(k_scales),
                    KernelArg::Buffer(v_scales),
                    KernelArg::Buffer(out_buf),
                ],
            )
            .expect("attn launch");
        backend.synchronize(backend.default_stream()).expect("sync");

        let mut out_bytes = vec![0u8; q.len() * 2];
        backend.copy_d2h(out_buf, &mut out_bytes).expect("d2h");
        let gpu_out = bytes_to_bf16_vec(&out_bytes);

        let group = (num_heads / num_kv_heads) as usize;
        for h in 0..num_heads as usize {
            let kv_h = h / group;
            let mut scores = vec![0.0f32; seq_len as usize];
            for s in 0..seq_len as usize {
                let mut dot = 0.0f32;
                for d in 0..head_dim as usize {
                    dot += f32::from(q[h * head_dim as usize + d])
                        * k_deq[s * n_elems + kv_h * head_dim as usize + d];
                }
                scores[s] = dot * scale;
            }
            let max = scores.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
            let exps: Vec<f32> = scores.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for d in 0..head_dim as usize {
                let mut acc = 0.0f32;
                for s in 0..seq_len as usize {
                    if exps[s] <= 1e-3 {
                        continue;
                    }
                    acc += exps[s] / sum * v_deq[s * n_elems + kv_h * head_dim as usize + d];
                }
                let got = f32::from(gpu_out[h * head_dim as usize + d]);
                let want = f32::from(half::bf16::from_f32(acc));
                assert!(
                    (got - want).abs() <= 0.08,
                    "{attn_mod} h={h} d={d}: got {got}, want {want}"
                );
            }
        }
    }
}
