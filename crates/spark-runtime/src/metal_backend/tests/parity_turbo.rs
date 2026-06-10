// SPDX-License-Identifier: AGPL-3.0-only
//! TurboQuant kernel parity: WHT rotation (`wht_bf16`), Turbo8 cache
//! append (`kv_cache_append_turbo8`), and Turbo8 decode attention
//! (`attention_decode_turbo8`) against FP32 CPU references.
//!
//! The metal common build does not define `TQ_PLUS_SIGNS`, so the CPU
//! WHT reference here is the plain (sign-free, self-inverse) transform
//! — mirroring the production GB10 model targets, whose KERNEL.tomls
//! also omit the define.

#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{GpuBackend, KernelArg};

// ── CPU references ───────────────────────────────────────────

/// Plain in-place WHT over one head of `n` f32 values + 1/sqrt(n)
/// normalization. Self-inverse.
fn cpu_wht(x: &mut [f32]) {
    let n = x.len();
    let mut stride = 1;
    while stride < n {
        let mut i = 0;
        while i < n {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let norm = 1.0f32 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= norm;
    }
}

/// float → FP8 E4M3 byte. Mirrors `f32_to_e4m3` in
/// `kv_cache_append_turbo8.metal` exactly (saturating, round-half-away
/// on the mantissa).
fn cpu_f32_to_e4m3(f: f32) -> u8 {
    let sign: u8 = if f < 0.0 { 0x80 } else { 0x00 };
    let a = f.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    if a < 0.001953125 {
        let m = (a * 512.0).round() as u32;
        return sign | m as u8;
    }
    let mut e = a.log2().floor() as i32;
    if e < -6 {
        e = -6;
    }
    let man = a / (e as f32).exp2();
    let mut m3 = ((man - 1.0) * 8.0).round() as u32;
    if m3 == 8 {
        e += 1;
        m3 = 0;
    }
    sign | (((e + 7) as u8) << 3) | m3 as u8
}

/// FP8 E4M3 byte → float. Mirrors `e4m3_to_f32` in
/// `attention_decode_turbo8.metal`.
fn cpu_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xF;
    let m = b & 7;
    if e == 0 {
        return sign * m as f32 * 0.001953125;
    }
    sign * (1.0 + m as f32 * 0.125) * ((e as i32 - 7) as f32).exp2()
}

/// Deterministic pseudo-random bf16-representable test value in ~[-2, 2].
fn test_val(i: usize) -> f32 {
    let raw = ((i * 2654435761) >> 7) % 4001;
    f32::from(half::bf16::from_f32(raw as f32 / 1000.0 - 2.0))
}

// ── Tests ────────────────────────────────────────────────────

/// `wht_bf16_inplace` matches the plain CPU WHT at head_dim 128 and
/// 256 across multiple heads.
#[test]
fn metal_wht_bf16_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let kernel = backend
        .kernel("wht_bf16", "wht_bf16_inplace")
        .expect("kernel lookup");

    for head_dim in [128usize, 256] {
        let num_heads = 3usize;
        let n = num_heads * head_dim;
        let input: Vec<half::bf16> = (0..n).map(|i| half::bf16::from_f32(test_val(i))).collect();

        let ptr = backend.alloc(n * 2).expect("alloc");
        backend
            .copy_h2d(&bf16_slice_to_bytes(&input), ptr)
            .expect("h2d");

        let hd = head_dim as u32;
        backend
            .launch_typed(
                kernel,
                [num_heads as u32, 1, 1],
                [32, 1, 1],
                0,
                backend.default_stream(),
                &[KernelArg::Bytes(&hd.to_le_bytes()), KernelArg::Buffer(ptr)],
            )
            .expect("launch");
        backend.synchronize(backend.default_stream()).expect("sync");

        let mut out_bytes = vec![0u8; n * 2];
        backend.copy_d2h(ptr, &mut out_bytes).expect("d2h");
        let gpu = bytes_to_bf16_vec(&out_bytes);

        for h in 0..num_heads {
            let mut reference: Vec<f32> = input[h * head_dim..(h + 1) * head_dim]
                .iter()
                .map(|v| f32::from(*v))
                .collect();
            cpu_wht(&mut reference);
            for d in 0..head_dim {
                let got = f32::from(gpu[h * head_dim + d]);
                let want = f32::from(half::bf16::from_f32(reference[d]));
                assert!(
                    (got - want).abs() <= 0.04,
                    "wht hd={head_dim} h={h} d={d}: got {got}, want {want}"
                );
            }
        }
    }
}

/// Forward WHT followed by the inverse kernel restores the input
/// (within bf16 round-trip error).
#[test]
fn metal_wht_bf16_inverse_roundtrip() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let fwd = backend
        .kernel("wht_bf16", "wht_bf16_inplace")
        .expect("fwd lookup");
    let inv = backend
        .kernel("wht_bf16", "wht_bf16_inplace_inv")
        .expect("inv lookup");

    for head_dim in [128usize, 256] {
        let num_heads = 2usize;
        let n = num_heads * head_dim;
        let input: Vec<half::bf16> = (0..n)
            .map(|i| half::bf16::from_f32(test_val(i + 7)))
            .collect();

        let ptr = backend.alloc(n * 2).expect("alloc");
        backend
            .copy_h2d(&bf16_slice_to_bytes(&input), ptr)
            .expect("h2d");

        let hd = head_dim as u32;
        for k in [fwd, inv] {
            backend
                .launch_typed(
                    k,
                    [num_heads as u32, 1, 1],
                    [32, 1, 1],
                    0,
                    backend.default_stream(),
                    &[KernelArg::Bytes(&hd.to_le_bytes()), KernelArg::Buffer(ptr)],
                )
                .expect("launch");
        }
        backend.synchronize(backend.default_stream()).expect("sync");

        let mut out_bytes = vec![0u8; n * 2];
        backend.copy_d2h(ptr, &mut out_bytes).expect("d2h");
        let gpu = bytes_to_bf16_vec(&out_bytes);

        for i in 0..n {
            let got = f32::from(gpu[i]);
            let want = f32::from(input[i]);
            assert!(
                (got - want).abs() <= 0.04,
                "roundtrip hd={head_dim} i={i}: got {got}, want {want}"
            );
        }
    }
}

/// `kv_cache_append_turbo8` produces byte-identical FP8 data + BF16
/// scales to the CPU reference quantizer.
#[test]
fn metal_kv_cache_append_turbo8_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let kernel = backend
        .kernel("kv_cache_append_turbo8", "kv_cache_append_turbo8")
        .expect("kernel lookup");

    let num_kv_heads = 2u32;
    let head_dim = 256u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let max_seq = 4usize;
    let cache_pos = 2u32;

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

    let k_data = backend.alloc(max_seq * n_elems).expect("alloc");
    let v_data = backend.alloc(max_seq * n_elems).expect("alloc");
    let k_scales = backend.alloc(max_seq * num_groups * 2).expect("alloc");
    let v_scales = backend.alloc(max_seq * num_groups * 2).expect("alloc");

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

    let mut k_data_h = vec![0u8; max_seq * n_elems];
    let mut k_scales_h = vec![0u8; max_seq * num_groups * 2];
    backend.copy_d2h(k_data, &mut k_data_h).expect("d2h");
    backend.copy_d2h(k_scales, &mut k_scales_h).expect("d2h");
    let k_scales_bf = bytes_to_bf16_vec(&k_scales_h);

    let row = cache_pos as usize;
    for g in 0..num_groups {
        let vals: Vec<f32> = (0..16).map(|i| f32::from(new_k[g * 16 + i])).collect();
        let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = (amax / 448.0).max(1e-12);
        let scale_bf = half::bf16::from_f32(scale);
        assert_eq!(
            k_scales_bf[row * num_groups + g],
            scale_bf,
            "k scale mismatch at group {g}"
        );
        let inv = 1.0 / f32::from(scale_bf);
        for i in 0..16 {
            let want = cpu_f32_to_e4m3(vals[i] * inv);
            let got = k_data_h[row * n_elems + g * 16 + i];
            assert_eq!(got, want, "k data mismatch at group {g} elem {i}");
        }
    }
}

/// `attention_decode_turbo8` matches a CPU reference that attends over
/// the CPU-dequantized cache (so only the kernel's softmax/dot math is
/// under test, not the quantization error).
#[test]
fn metal_attention_decode_turbo8_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };
    let append = backend
        .kernel("kv_cache_append_turbo8", "kv_cache_append_turbo8")
        .expect("append lookup");
    let attn = backend
        .kernel("attention_decode_turbo8", "attention_decode_turbo8")
        .expect("attn lookup");

    let num_heads = 4u32;
    let num_kv_heads = 2u32;
    let head_dim = 128u32;
    let seq_len = 9u32;
    let n_elems = (num_kv_heads * head_dim) as usize;
    let num_groups = n_elems / 16;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let k_data = backend.alloc(seq_len as usize * n_elems).expect("alloc");
    let v_data = backend.alloc(seq_len as usize * n_elems).expect("alloc");
    let k_scales = backend
        .alloc(seq_len as usize * num_groups * 2)
        .expect("alloc");
    let v_scales = backend
        .alloc(seq_len as usize * num_groups * 2)
        .expect("alloc");
    let k_src = backend.alloc(n_elems * 2).expect("alloc");
    let v_src = backend.alloc(n_elems * 2).expect("alloc");

    // Append seq_len tokens through the quantizer, mirroring on CPU.
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
        // copy_h2d writes straight into UMA shared memory, so the next
        // token's upload would race the in-flight append kernel —
        // drain the queue before reusing the staging buffers.
        backend.synchronize(backend.default_stream()).expect("sync");

        // CPU mirror of quant + dequant for the reference cache.
        for g in 0..num_groups {
            for (src, deq) in [(&tok_k, &mut k_deq), (&tok_v, &mut v_deq)] {
                let vals: Vec<f32> = (0..16).map(|i| f32::from(src[g * 16 + i])).collect();
                let amax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let s_bf = half::bf16::from_f32((amax / 448.0).max(1e-12));
                let s_f = f32::from(s_bf);
                for i in 0..16 {
                    let q = cpu_f32_to_e4m3(vals[i] / s_f);
                    deq[s as usize * n_elems + g * 16 + i] = cpu_e4m3_to_f32(q) * s_f;
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
                acc += exps[s] / sum * v_deq[s * n_elems + kv_h * head_dim as usize + d];
            }
            let got = f32::from(gpu_out[h * head_dim as usize + d]);
            let want = f32::from(half::bf16::from_f32(acc));
            assert!(
                (got - want).abs() <= 0.03,
                "attn h={h} d={d}: got {got}, want {want}"
            );
        }
    }
}
