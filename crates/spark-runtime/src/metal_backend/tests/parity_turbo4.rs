// SPDX-License-Identifier: AGPL-3.0-only
//! Turbo4 kernel parity: 4-bit Lloyd-Max cache append
//! (`kv_cache_append_turbo4`) and decode attention
//! (`attention_decode_turbo4`) against FP32 CPU references.
//! Shares the E4M3 + test-value helpers with `parity_turbo.rs`.

#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{GpuBackend, KernelArg};

const TURBO4_CODEBOOK: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];
const TURBO4_BOUNDS: [f32; 15] = [
    -2.4008, -1.8435, -1.4371, -1.0993, -0.7996, -0.5224, -0.2582, 0.0, 0.2582, 0.5224, 0.7996,
    1.0993, 1.4371, 1.8435, 2.4008,
];
const TURBO4_MAX: f32 = 2.7326;

fn turbo4_quantize(x: f32) -> u8 {
    let mut idx = 0u8;
    while (idx as usize) < 15 && x >= TURBO4_BOUNDS[idx as usize] {
        idx += 1;
    }
    idx
}

/// CPU mirror of one group through `kv_cache_append_turbo4.metal`:
/// returns (8 packed nibble bytes, E4M3 scale byte).
fn cpu_quant_group_turbo4(vals: &[f32; 16]) -> ([u8; 8], u8) {
    let norm_sq: f32 = vals.iter().map(|v| v * v).sum();
    let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let inv = if amax > 1e-12 { TURBO4_MAX / amax } else { 1.0 };
    let mut idx = [0u8; 16];
    let mut recon_sq = 0.0f32;
    for i in 0..16 {
        idx[i] = turbo4_quantize(vals[i] * inv);
        let c = TURBO4_CODEBOOK[idx[i] as usize];
        recon_sq += c * c;
    }
    let recon_norm = recon_sq.sqrt();
    let scale = if recon_norm > 1e-10 {
        norm_sq.sqrt() / recon_norm
    } else {
        amax / TURBO4_MAX
    };
    let scale_byte = cpu_f32_to_e4m3(scale.min(448.0));
    let mut packed = [0u8; 8];
    for i in (0..16).step_by(2) {
        packed[i / 2] = idx[i] | (idx[i + 1] << 4);
    }
    (packed, scale_byte)
}

/// `kv_cache_append_turbo4` produces byte-identical packed indices +
/// E4M3 scales to the CPU reference quantizer.
#[test]
fn metal_kv_cache_append_turbo4_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let kernel = backend
        .kernel("kv_cache_append_turbo4", "kv_cache_append_turbo4")
        .expect("kernel lookup");

    let num_kv_heads = 2u32;
    let head_dim = 256u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let max_seq = 4usize;
    let cache_pos = 1u32;

    let new_k: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i)))
        .collect();
    let new_v: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i + 100_000)))
        .collect();

    let k_src = backend.alloc(n_elems * 2).expect("alloc");
    let v_src = backend.alloc(n_elems * 2).expect("alloc");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&new_k), k_src)
        .expect("h2d");
    backend
        .copy_h2d(&bf16_slice_to_bytes(&new_v), v_src)
        .expect("h2d");

    let k_data = backend.alloc(max_seq * n_elems / 2).expect("alloc");
    let v_data = backend.alloc(max_seq * n_elems / 2).expect("alloc");
    let k_scales = backend.alloc(max_seq * num_groups).expect("alloc");
    let v_scales = backend.alloc(max_seq * num_groups).expect("alloc");

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

    let mut k_data_h = vec![0u8; max_seq * n_elems / 2];
    let mut k_scales_h = vec![0u8; max_seq * num_groups];
    backend.copy_d2h(k_data, &mut k_data_h).expect("d2h");
    backend.copy_d2h(k_scales, &mut k_scales_h).expect("d2h");

    let row = cache_pos as usize;
    for g in 0..num_groups {
        let mut vals = [0.0f32; 16];
        for i in 0..16 {
            vals[i] = f32::from(new_k[g * 16 + i]);
        }
        let (want_packed, want_scale) = cpu_quant_group_turbo4(&vals);
        assert_eq!(
            k_scales_h[row * num_groups + g],
            want_scale,
            "k scale byte mismatch at group {g}"
        );
        for b in 0..8 {
            assert_eq!(
                k_data_h[row * n_elems / 2 + g * 8 + b],
                want_packed[b],
                "k packed byte mismatch at group {g} byte {b}"
            );
        }
    }
}

/// `attention_decode_turbo4` matches a CPU reference attending over the
/// CPU-dequantized cache.
#[test]
fn metal_attention_decode_turbo4_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let append = backend
        .kernel("kv_cache_append_turbo4", "kv_cache_append_turbo4")
        .expect("append lookup");
    let attn = backend
        .kernel("attention_decode_turbo4", "attention_decode_turbo4")
        .expect("attn lookup");

    let num_heads = 4u32;
    let num_kv_heads = 2u32;
    let head_dim = 128u32;
    let seq_len = 7u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_data = backend
        .alloc(seq_len as usize * n_elems / 2)
        .expect("alloc");
    let v_data = backend
        .alloc(seq_len as usize * n_elems / 2)
        .expect("alloc");
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
                let (packed, scale_byte) = cpu_quant_group_turbo4(&vals);
                let gs = cpu_e4m3_to_f32(scale_byte);
                for i in 0..16 {
                    let idx = if i % 2 == 0 {
                        packed[i / 2] & 0xF
                    } else {
                        packed[i / 2] >> 4
                    };
                    deq[s as usize * n_elems + g * 16 + i] = TURBO4_CODEBOOK[idx as usize] * gs;
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
                // Mirror the kernel's sparse-V gate: rows with
                // exp(score - max) <= 1e-3 contribute nothing.
                if exps[s] <= 1e-3 {
                    continue;
                }
                acc += exps[s] / sum * v_deq[s * n_elems + kv_h * head_dim as usize + d];
            }
            let got = f32::from(gpu_out[h * head_dim as usize + d]);
            let want = f32::from(half::bf16::from_f32(acc));
            assert!(
                (got - want).abs() <= 0.05,
                "attn h={h} d={d}: got {got}, want {want}"
            );
        }
    }
}
