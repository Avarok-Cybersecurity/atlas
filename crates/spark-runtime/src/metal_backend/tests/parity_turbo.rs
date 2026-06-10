// SPDX-License-Identifier: AGPL-3.0-only
//! TurboQuant kernel parity: WHT rotation (`wht_bf16`), Turbo8 cache
//! append (`kv_cache_append_turbo8`), and Turbo8 decode attention
//! (`attention_decode_turbo8`) against FP32 CPU references.
//!
//! The metal build defines `TQ_PLUS_SIGNS`, so the CPU WHT reference
//! is the two-sided Rademacher rotation S2·H·S1 (sign tables vendored
//! in helpers.rs) and the inverse-roundtrip test pins that the inverse
//! kernel reverses the sign order.

#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{GpuBackend, KernelArg};

// ── CPU references ───────────────────────────────────────────

/// In-place butterfly WHT over `n` f32 values + 1/sqrt(n) normalization.
fn cpu_fwht(x: &mut [f32]) {
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

/// Forward rotation matching `wht_bf16_inplace` with TQ_PLUS_SIGNS:
/// signs1 → FWHT → signs2.
fn cpu_wht(x: &mut [f32]) {
    let (s1, s2): (&[f32], &[f32]) = if x.len() == 256 {
        (&TQP_SIGNS1_256, &TQP_SIGNS2_256)
    } else {
        (&TQP_SIGNS1_128, &TQP_SIGNS2_128)
    };
    for (v, s) in x.iter_mut().zip(s1) {
        *v *= s;
    }
    cpu_fwht(x);
    for (v, s) in x.iter_mut().zip(s2) {
        *v *= s;
    }
}

// (E4M3 + test_val helpers shared with parity_turbo4.rs live in helpers.rs)

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
                (got - want).abs() <= 0.03,
                "attn h={h} d={d}: got {got}, want {want}"
            );
        }
    }
}
