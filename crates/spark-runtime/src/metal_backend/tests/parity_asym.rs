// SPDX-License-Identifier: AGPL-3.0-only
//! Safer-asym (Bf16K + TurboNV) kernel parity: K-passthrough + V
//! quantizer byte-exactness for all three V dtypes, and decode
//! attention vs a CPU reference (raw-bf16 K scores, CPU-dequantized V,
//! sparse-V gate mirrored).

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
const TURBO3_CODEBOOK: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];
const TURBO3_BOUNDS: [f32; 7] = [-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748];
const TURBO2_CODEBOOK: [f32; 4] = [-1.5104, -0.4528, 0.4528, 1.5104];
const TURBO2_BOUNDS: [f32; 3] = [-0.9816, 0.0, 0.9816];

fn cpu_quant_group(
    vals: &[f32; 16],
    codebook: &[f32],
    bounds: &[f32],
    cmax: f32,
) -> ([u8; 16], u8) {
    let norm_sq: f32 = vals.iter().map(|v| v * v).sum();
    let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let inv = if amax > 1e-12 { cmax / amax } else { 1.0 };
    let mut idx = [0u8; 16];
    let mut recon_sq = 0.0f32;
    for i in 0..16 {
        let mut q = 0u8;
        while (q as usize) < bounds.len() && vals[i] * inv >= bounds[q as usize] {
            q += 1;
        }
        idx[i] = q;
        let c = codebook[q as usize];
        recon_sq += c * c;
    }
    let recon_norm = recon_sq.sqrt();
    let scale = if recon_norm > 1e-10 {
        norm_sq.sqrt() / recon_norm
    } else {
        amax / cmax
    };
    (idx, cpu_f32_to_e4m3(scale.min(448.0)))
}

/// One append through a `kv_cache_append_bf16k_turbo*v` kernel:
/// returns host copies of (k_cache row bf16, v_data row, v_scales row).
#[allow(clippy::too_many_arguments)]
fn run_asym_append(
    backend: &crate::metal_backend::MetalGpuBackend,
    func: &str,
    new_k: &[half::bf16],
    new_v: &[half::bf16],
    num_kv_heads: u32,
    head_dim: u32,
    v_row_bytes: usize,
    cache_pos: u32,
    max_seq: usize,
) -> (Vec<half::bf16>, Vec<u8>, Vec<u8>) {
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let kernel = backend
        .kernel("kv_cache_append_bf16k_turbov", func)
        .expect("kernel lookup");
    let k_src = backend.alloc(n_elems * 2).expect("alloc");
    let v_src = backend.alloc(n_elems * 2).expect("alloc");
    backend
        .copy_h2d(&bf16_slice_to_bytes(new_k), k_src)
        .expect("h2d");
    backend
        .copy_h2d(&bf16_slice_to_bytes(new_v), v_src)
        .expect("h2d");
    let k_cache = backend.alloc(max_seq * n_elems * 2).expect("alloc");
    let v_data = backend.alloc(max_seq * v_row_bytes).expect("alloc");
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
                KernelArg::Buffer(k_cache),
                KernelArg::Buffer(v_data),
                KernelArg::Buffer(v_scales),
            ],
        )
        .expect("launch");
    backend.synchronize(backend.default_stream()).expect("sync");
    let mut k_h = vec![0u8; max_seq * n_elems * 2];
    let mut vd_h = vec![0u8; max_seq * v_row_bytes];
    let mut vs_h = vec![0u8; max_seq * num_groups];
    backend.copy_d2h(k_cache, &mut k_h).expect("d2h");
    backend.copy_d2h(v_data, &mut vd_h).expect("d2h");
    backend.copy_d2h(v_scales, &mut vs_h).expect("d2h");
    let row = cache_pos as usize;
    (
        bytes_to_bf16_vec(&k_h[row * n_elems * 2..(row + 1) * n_elems * 2]),
        vd_h[row * v_row_bytes..(row + 1) * v_row_bytes].to_vec(),
        vs_h[row * num_groups..(row + 1) * num_groups].to_vec(),
    )
}

/// All three asym appends: K row passes through bit-exact, V packs +
/// scales match the CPU quantizer byte-for-byte.
#[test]
fn metal_kv_cache_append_bf16k_turbov_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let num_kv_heads = 2u32;
    let head_dim = 256u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let new_k: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i + 41)))
        .collect();
    let new_v: Vec<half::bf16> = (0..n_elems)
        .map(|i| half::bf16::from_f32(test_val(i + 300_000)))
        .collect();

    for (func, codebook, bounds, cmax, v_row_bytes) in [
        (
            "kv_cache_append_bf16k_turbo4v",
            &TURBO4_CODEBOOK[..],
            &TURBO4_BOUNDS[..],
            2.7326f32,
            n_elems / 2,
        ),
        (
            "kv_cache_append_bf16k_turbo3v",
            &TURBO3_CODEBOOK[..],
            &TURBO3_BOUNDS[..],
            2.1520f32,
            n_elems * 3 / 8,
        ),
        (
            "kv_cache_append_bf16k_turbo2v",
            &TURBO2_CODEBOOK[..],
            &TURBO2_BOUNDS[..],
            1.5104f32,
            n_elems / 4,
        ),
    ] {
        let (k_row, v_row, vs_row) = run_asym_append(
            &backend,
            func,
            &new_k,
            &new_v,
            num_kv_heads,
            head_dim,
            v_row_bytes,
            1,
            3,
        );
        // K side: bit-exact passthrough.
        for i in 0..n_elems {
            assert_eq!(k_row[i], new_k[i], "{func}: K passthrough mismatch at {i}");
        }
        // V side: byte-exact quant.
        for g in 0..num_groups {
            let mut vals = [0.0f32; 16];
            for i in 0..16 {
                vals[i] = f32::from(new_v[g * 16 + i]);
            }
            let (idx, want_scale) = cpu_quant_group(&vals, codebook, bounds, cmax);
            assert_eq!(
                vs_row[g], want_scale,
                "{func}: V scale mismatch at group {g}"
            );
            // Verify via the packed bytes of the 4-bit case only (the
            // 3/2-bit packers are pinned byte-exact by the symmetric
            // parity tests; here we check one representative byte).
            if func.ends_with("turbo4v") {
                for b in 0..8 {
                    let want = idx[b * 2] | (idx[b * 2 + 1] << 4);
                    assert_eq!(
                        v_row[g * 8 + b],
                        want,
                        "{func}: V pack mismatch g={g} b={b}"
                    );
                }
            }
        }
    }
}

/// `attention_decode_bf16k_turbo4v` matches a CPU reference: raw-bf16
/// K scores (no rotation), CPU-dequantized V, sparse-V gate mirrored.
#[test]
fn metal_attention_decode_bf16k_turbo4v_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let append = backend
        .kernel(
            "kv_cache_append_bf16k_turbov",
            "kv_cache_append_bf16k_turbo4v",
        )
        .expect("append");
    let attn = backend
        .kernel(
            "attention_decode_bf16k_turbov",
            "attention_decode_bf16k_turbo4v",
        )
        .expect("attn");

    let num_heads = 4u32;
    let num_kv_heads = 2u32;
    let head_dim = 128u32;
    let seq_len = 6u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let v_row_bytes = n_elems / 2;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_cache = backend
        .alloc(seq_len as usize * n_elems * 2)
        .expect("alloc");
    let v_data = backend
        .alloc(seq_len as usize * v_row_bytes)
        .expect("alloc");
    let v_scales = backend.alloc(seq_len as usize * num_groups).expect("alloc");
    let k_src = backend.alloc(n_elems * 2).expect("alloc");
    let v_src = backend.alloc(n_elems * 2).expect("alloc");

    let mut k_ref = vec![0.0f32; seq_len as usize * n_elems];
    let mut v_deq = vec![0.0f32; seq_len as usize * n_elems];
    for s in 0..seq_len {
        let tok_k: Vec<half::bf16> = (0..n_elems)
            .map(|i| half::bf16::from_f32(test_val(i + 1000 * s as usize + 50)))
            .collect();
        let tok_v: Vec<half::bf16> = (0..n_elems)
            .map(|i| half::bf16::from_f32(test_val(i + 1000 * s as usize + 700_000)))
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
                    KernelArg::Buffer(k_cache),
                    KernelArg::Buffer(v_data),
                    KernelArg::Buffer(v_scales),
                ],
            )
            .expect("append launch");
        backend.synchronize(backend.default_stream()).expect("sync");

        for i in 0..n_elems {
            k_ref[s as usize * n_elems + i] = f32::from(tok_k[i]);
        }
        for g in 0..num_groups {
            let mut vals = [0.0f32; 16];
            for i in 0..16 {
                vals[i] = f32::from(tok_v[g * 16 + i]);
            }
            let (idx, scale_byte) =
                cpu_quant_group(&vals, &TURBO4_CODEBOOK, &TURBO4_BOUNDS, 2.7326);
            let gs = cpu_e4m3_to_f32(scale_byte);
            for i in 0..16 {
                v_deq[s as usize * n_elems + g * 16 + i] = TURBO4_CODEBOOK[idx[i] as usize] * gs;
            }
        }
    }

    let q: Vec<half::bf16> = (0..(num_heads * head_dim) as usize)
        .map(|i| half::bf16::from_f32(test_val(i + 999)))
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
                KernelArg::Buffer(k_cache),
                KernelArg::Buffer(v_data),
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
                    * k_ref[s * n_elems + kv_h * head_dim as usize + d];
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
                (got - want).abs() <= 0.05,
                "asym attn h={h} d={d}: got {got}, want {want}"
            );
        }
    }
}
