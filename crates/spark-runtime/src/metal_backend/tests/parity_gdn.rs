// SPDX-License-Identifier: AGPL-3.0-only
//! Gated-Delta-Network and SSM-state kernel parity.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// `gated_delta_rule_decode` parity. Mirrors the CUDA reference
/// kernel `gated_delta_rule.cu::gated_delta_rule_decode` — same
/// math, same arg layout. CPU reference walks the kernel's
/// algorithm step-for-step (hk_dot → v_new → state update +
/// q_dot → 1/sqrt(k_dim) scaling) so any drift surfaces only
/// as BF16 / FP32 round error.
///
/// Skips the SSM state-norm clamp branch (the kernel includes
/// it but at our test magnitudes ||H||_F stays well below the
/// 1000 threshold, so the clamp doesn't fire).
#[test]
fn metal_gated_delta_rule_decode_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    // Qwen3.5-style dims (smaller for fast test).
    let batch_size: u32 = 1;
    let num_k_heads: u32 = 2;
    let num_v_heads: u32 = 4; // head_repeat = 2
    let k_dim: u32 = 128;
    let v_dim: u32 = 128;
    let head_repeat = num_v_heads / num_k_heads;

    // Initial H state, smaller magnitude so the norm clamp doesn't fire.
    let h_state: Vec<f32> = (0..(batch_size * num_v_heads * k_dim * v_dim) as usize)
        .map(|i| 0.001 * ((i as f32) * 0.0123).sin())
        .collect();

    let query: Vec<half::bf16> = (0..(batch_size * num_k_heads * k_dim))
        .map(|i| half::bf16::from_f32(0.05 + 0.001 * (i as f32 * 0.07).sin()))
        .collect();
    let key: Vec<half::bf16> = (0..(batch_size * num_k_heads * k_dim))
        .map(|i| half::bf16::from_f32(0.04 + 0.001 * (i as f32 * 0.05).cos()))
        .collect();
    let value: Vec<half::bf16> = (0..(batch_size * num_v_heads * v_dim))
        .map(|i| half::bf16::from_f32(0.06 + 0.001 * (i as f32 * 0.03).sin()))
        .collect();
    let gate: Vec<f32> = (0..(batch_size * num_v_heads))
        .map(|i| 0.95 - 0.01 * i as f32)
        .collect();
    let beta: Vec<f32> = (0..(batch_size * num_v_heads))
        .map(|i| 0.5 + 0.05 * i as f32)
        .collect();

    // ── CPU reference (matches kernel math step-for-step) ──
    let mut h_cpu = h_state.clone();
    let mut expected = vec![half::bf16::ZERO; (batch_size * num_v_heads * v_dim) as usize];
    for b in 0..batch_size as usize {
        for vh in 0..num_v_heads as usize {
            let kh = vh / head_repeat as usize;
            let g_raw = gate[b * num_v_heads as usize + vh];
            let g = g_raw.clamp(1e-6, 1.0 - 1e-6);
            let bt = beta[b * num_v_heads as usize + vh];
            let h_off = (b * num_v_heads as usize + vh) * k_dim as usize * v_dim as usize;
            let q_off = (b * num_k_heads as usize + kh) * k_dim as usize;
            let k_off = (b * num_k_heads as usize + kh) * k_dim as usize;
            let v_off = (b * num_v_heads as usize + vh) * v_dim as usize;

            for tid in 0..v_dim as usize {
                let v_i = value[v_off + tid].to_f32();

                // Step 1: hk_dot
                let mut hk_dot = 0.0f32;
                for j in 0..k_dim as usize {
                    let h_v = h_cpu[h_off + j * v_dim as usize + tid];
                    let k_v = key[k_off + j].to_f32();
                    hk_dot += h_v * k_v;
                }
                // Step 2: v_new
                let v_new_i = (v_i - g * hk_dot) * bt;
                // Steps 3+4: state update + output
                let mut q_dot = 0.0f32;
                for j in 0..k_dim as usize {
                    let h_old = h_cpu[h_off + j * v_dim as usize + tid];
                    let k_v = key[k_off + j].to_f32();
                    let q_v = query[q_off + j].to_f32();
                    let h_new = g * h_old + k_v * v_new_i;
                    h_cpu[h_off + j * v_dim as usize + tid] = h_new;
                    q_dot += h_new * q_v;
                }
                let inv_sqrt_d = (k_dim as f32).powf(-0.5);
                expected[(b * num_v_heads as usize + vh) * v_dim as usize + tid] =
                    half::bf16::from_f32(q_dot * inv_sqrt_d);
            }
        }
    }

    // ── Run kernel ──
    let h_state_bytes: Vec<u8> = h_state.iter().flat_map(|f| f.to_le_bytes()).collect();
    let q_bytes = bf16_slice_to_bytes(&query);
    let k_bytes = bf16_slice_to_bytes(&key);
    let v_bytes = bf16_slice_to_bytes(&value);
    let g_bytes: Vec<u8> = gate.iter().flat_map(|f| f.to_le_bytes()).collect();
    let bt_bytes: Vec<u8> = beta.iter().flat_map(|f| f.to_le_bytes()).collect();

    let h_ptr = backend.alloc(h_state_bytes.len()).unwrap();
    let q_ptr = backend.alloc(q_bytes.len()).unwrap();
    let k_ptr = backend.alloc(k_bytes.len()).unwrap();
    let v_ptr = backend.alloc(v_bytes.len()).unwrap();
    let g_ptr = backend.alloc(g_bytes.len()).unwrap();
    let bt_ptr = backend.alloc(bt_bytes.len()).unwrap();
    let out_ptr = backend
        .alloc((batch_size * num_v_heads * v_dim) as usize * 2)
        .unwrap();

    backend.copy_h2d(&h_state_bytes, h_ptr).unwrap();
    backend.copy_h2d(&q_bytes, q_ptr).unwrap();
    backend.copy_h2d(&k_bytes, k_ptr).unwrap();
    backend.copy_h2d(&v_bytes, v_ptr).unwrap();
    backend.copy_h2d(&g_bytes, g_ptr).unwrap();
    backend.copy_h2d(&bt_bytes, bt_ptr).unwrap();

    let kernel = backend
        .kernel("gated_delta_rule_decode", "gated_delta_rule_decode")
        .unwrap();
    let total_groups = num_v_heads * batch_size;
    backend
        .launch_typed(
            kernel,
            [total_groups, 1, 1],
            [128, 1, 1],
            0,
            backend.default_stream(),
            &[
                KernelArg::Buffer(h_ptr),
                KernelArg::Buffer(q_ptr),
                KernelArg::Buffer(k_ptr),
                KernelArg::Buffer(v_ptr),
                KernelArg::Buffer(g_ptr),
                KernelArg::Buffer(bt_ptr),
                KernelArg::Buffer(out_ptr),
                KernelArg::Bytes(&batch_size.to_le_bytes()),
                KernelArg::Bytes(&num_k_heads.to_le_bytes()),
                KernelArg::Bytes(&num_v_heads.to_le_bytes()),
                KernelArg::Bytes(&k_dim.to_le_bytes()),
                KernelArg::Bytes(&v_dim.to_le_bytes()),
            ],
        )
        .expect("launch gated_delta_rule_decode");
    backend.synchronize(backend.default_stream()).unwrap();

    let mut out_raw = vec![0u8; (batch_size * num_v_heads * v_dim) as usize * 2];
    backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
    let actual = bytes_to_bf16_vec(&out_raw);

    let mut max_abs_diff: f32 = 0.0;
    for i in 0..expected.len() {
        let e = expected[i].to_f32();
        let a = actual[i].to_f32();
        let d = (e - a).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
        assert!(a.is_finite(), "non-finite at idx {i}: {a}");
    }
    // ★ COSINE, NOT AN ABSOLUTE BOUND. The comment this replaced claimed an
    // output magnitude of ~0.05 and allowed 0.02 of drift; the inputs built
    // above (h_state = 0.001*sin(..)) actually produce outputs near 1e-3, so
    // the bound sat twenty times above the signal and a kernel that wrote
    // nothing passed. `COSINE_GATE` is the same 0.9999 the GDN microtest gates
    // on, and it is scale-free, so it cannot drift out of the units again.
    let cos_out = cosine_bf16(&expected, &actual);
    let mag_out = norm_ratio_bf16(&expected, &actual);
    assert!(
        mag_out >= COSINE_GATE,
        "gated_delta_rule_decode output: L2-norm ratio = {mag_out:.7} < {COSINE_GATE} \
         (cosine {cos_out:.7} cannot see a uniform gain, which is why this is here)"
    );
    assert!(
        cos_out >= COSINE_GATE,
        "gated_delta_rule_decode output: cosine = {cos_out:.7} < {COSINE_GATE} \
         (max |expected - actual| = {max_abs_diff})"
    );

    // Also verify the in-place state was updated correctly (read it
    // back and compare to h_cpu).
    let mut h_after_raw = vec![0u8; h_state_bytes.len()];
    backend.copy_d2h(h_ptr, &mut h_after_raw).unwrap();
    let h_after: Vec<f32> = h_after_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut h_max_diff: f32 = 0.0;
    for i in 0..h_cpu.len() {
        let d = (h_cpu[i] - h_after[i]).abs();
        if d > h_max_diff {
            h_max_diff = d;
        }
    }
    // Same reasoning for the in-place state: h_state's own values are ~1e-3,
    // so a 1e-3 absolute bound was the whole signal, not a tolerance.
    let cos_h = cosine_f32(&h_cpu, &h_after);
    let mag_h = norm_ratio_f32(&h_cpu, &h_after);
    assert!(
        mag_h >= COSINE_GATE,
        "h_state in-place update: L2-norm ratio = {mag_h:.7} < {COSINE_GATE}"
    );
    assert!(
        cos_h >= COSINE_GATE,
        "h_state in-place update: cosine = {cos_h:.7} < {COSINE_GATE} \
         (max |h_cpu - h_actual| = {h_max_diff})"
    );

    // Suppress unused warning for h_state (used as starting bytes only).
    let _ = h_state.len();
    for ptr in [h_ptr, q_ptr, k_ptr, v_ptr, g_ptr, bt_ptr, out_ptr] {
        backend.free(ptr).unwrap();
    }
}

/// `selective_scan_decode` parity. Drives 3 decode steps so the
/// in-place state buffer is genuinely exercised, not coincidentally
/// always reading the same values. CPU reference uses the exact
/// same arithmetic — softplus(dt + dt_bias), -exp(A_log) decay,
/// FP32 state update, FP32 head-reduction for y.
#[test]
fn metal_selective_scan_decode_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_heads: u32 = 4;
    let num_channels: u32 = 16;

    let a_log: Vec<f32> = (0..num_heads).map(|h| -1.0 - 0.1 * h as f32).collect();
    let dt_bias: Vec<half::bf16> = (0..num_heads)
        .map(|h| half::bf16::from_f32(-2.0 + 0.5 * h as f32))
        .collect();

    // Initial state: small non-zero values so the decay path is
    // visibly contributing.
    let mut state_cpu: Vec<half::bf16> = (0..(num_heads * num_channels))
        .map(|i| half::bf16::from_f32(0.05 * (i as f32 * 0.13).sin()))
        .collect();

    let a_log_bytes: Vec<u8> = a_log.iter().flat_map(|f| f.to_le_bytes()).collect();
    let dt_bias_bytes = bf16_slice_to_bytes(&dt_bias);
    let a_ptr = backend.alloc(a_log_bytes.len()).unwrap();
    let dtb_ptr = backend.alloc(dt_bias_bytes.len()).unwrap();
    let dt_ptr = backend.alloc(num_heads as usize * 2).unwrap();
    let b_ptr = backend.alloc(num_heads as usize * 2).unwrap();
    let c_ptr = backend.alloc(num_heads as usize * 2).unwrap();
    let x_ptr = backend.alloc(num_channels as usize * 2).unwrap();
    let state_ptr = backend
        .alloc((num_heads * num_channels) as usize * 2)
        .unwrap();
    let y_ptr = backend.alloc(num_channels as usize * 2).unwrap();
    backend.copy_h2d(&a_log_bytes, a_ptr).unwrap();
    backend.copy_h2d(&dt_bias_bytes, dtb_ptr).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&state_cpu), state_ptr)
        .unwrap();

    let kernel = backend
        .kernel("selective_scan_decode", "selective_scan_decode")
        .unwrap();

    for step in 0..3u32 {
        // Per-step inputs — varied so the state genuinely rotates.
        let dt_raw: Vec<half::bf16> = (0..num_heads)
            .map(|h| half::bf16::from_f32(0.5 + 0.1 * step as f32 + 0.05 * h as f32))
            .collect();
        let b_in: Vec<half::bf16> = (0..num_heads)
            .map(|h| half::bf16::from_f32(0.3 + 0.05 * h as f32 - 0.01 * step as f32))
            .collect();
        let c_in: Vec<half::bf16> = (0..num_heads)
            .map(|h| half::bf16::from_f32(0.2 - 0.03 * h as f32 + 0.02 * step as f32))
            .collect();
        let x_in: Vec<half::bf16> = (0..num_channels)
            .map(|c| half::bf16::from_f32(0.4 + 0.01 * c as f32 + 0.05 * step as f32))
            .collect();

        backend
            .copy_h2d(&bf16_slice_to_bytes(&dt_raw), dt_ptr)
            .unwrap();
        backend
            .copy_h2d(&bf16_slice_to_bytes(&b_in), b_ptr)
            .unwrap();
        backend
            .copy_h2d(&bf16_slice_to_bytes(&c_in), c_ptr)
            .unwrap();
        backend
            .copy_h2d(&bf16_slice_to_bytes(&x_in), x_ptr)
            .unwrap();

        // CPU reference: identical arithmetic.
        let mut expected = vec![half::bf16::ZERO; num_channels as usize];
        for c in 0..num_channels as usize {
            let xc = x_in[c].to_f32();
            let mut acc = 0.0f32;
            for h in 0..num_heads as usize {
                let a_eff = -(a_log[h]).exp();
                let dt_pre = dt_raw[h].to_f32() + dt_bias[h].to_f32();
                // Softplus with the same numeric guard as the kernel.
                let dt = if dt_pre > 20.0 {
                    dt_pre
                } else {
                    (1.0 + dt_pre.exp()).ln()
                };
                let decay = (dt * a_eff).exp();
                let bv = b_in[h].to_f32();
                let cv = c_in[h].to_f32();
                let old_s = state_cpu[h * num_channels as usize + c].to_f32();
                let new_s = old_s * decay + dt * bv * xc;
                state_cpu[h * num_channels as usize + c] = half::bf16::from_f32(new_s);
                acc += new_s * cv;
            }
            expected[c] = half::bf16::from_f32(acc);
        }

        backend
            .launch_typed(
                kernel,
                [num_channels, 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_heads.to_le_bytes()),
                    KernelArg::Bytes(&num_channels.to_le_bytes()),
                    KernelArg::Buffer(a_ptr),
                    KernelArg::Buffer(dtb_ptr),
                    KernelArg::Buffer(dt_ptr),
                    KernelArg::Buffer(b_ptr),
                    KernelArg::Buffer(c_ptr),
                    KernelArg::Buffer(x_ptr),
                    KernelArg::Buffer(state_ptr),
                    KernelArg::Buffer(y_ptr),
                ],
            )
            .expect("launch selective_scan_decode");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut y_raw = vec![0u8; num_channels as usize * 2];
        backend.copy_d2h(y_ptr, &mut y_raw).unwrap();
        let actual = bytes_to_bf16_vec(&y_raw);

        for c in 0..num_channels as usize {
            let e = expected[c].to_f32();
            let a = actual[c].to_f32();
            assert!(
                (e - a).abs() < 0.02,
                "step {step} ch {c}: expected {e}, got {a}"
            );
        }
    }

    for ptr in [
        a_ptr, dtb_ptr, dt_ptr, b_ptr, c_ptr, x_ptr, state_ptr, y_ptr,
    ] {
        backend.free(ptr).unwrap();
    }
}

/// `causal_conv1d_decode` parity. Drives a few decode steps
/// against the CPU reference so the in-place state shift is
/// pinned (a read-after-write bug there would corrupt the next
/// step's output silently).
#[test]
fn metal_causal_conv1d_decode_matches_reference() {
    let Some(backend) = maybe_backend() else {
        return;
    };

    let num_channels: u32 = 8;
    let kernel_size: u32 = 4;
    let state_len = (kernel_size - 1) as usize;

    // Per-channel weight vectors and an initial conv_state.
    let weights: Vec<half::bf16> = (0..(num_channels * kernel_size))
        .map(|i| {
            let c = i / kernel_size;
            let k = i % kernel_size;
            half::bf16::from_f32(0.1 * (c as f32 + 1.0) + 0.05 * k as f32)
        })
        .collect();
    let mut conv_state_cpu: Vec<half::bf16> = (0..(num_channels as usize * state_len))
        .map(|i| half::bf16::from_f32(0.01 * i as f32))
        .collect();

    let weights_bytes = bf16_slice_to_bytes(&weights);
    let weights_ptr = backend.alloc(weights_bytes.len()).unwrap();
    let state_ptr = backend
        .alloc(num_channels as usize * state_len * 2)
        .unwrap();
    let new_in_ptr = backend.alloc(num_channels as usize * 2).unwrap();
    let out_ptr = backend.alloc(num_channels as usize * 2).unwrap();

    backend.copy_h2d(&weights_bytes, weights_ptr).unwrap();
    backend
        .copy_h2d(&bf16_slice_to_bytes(&conv_state_cpu), state_ptr)
        .unwrap();

    let kernel = backend
        .kernel("causal_conv1d_decode", "causal_conv1d_decode")
        .unwrap();

    // Drive 3 decode steps with deterministic-but-changing inputs
    // so the state ring buffer is genuinely exercised, not
    // accidentally always reading the same value.
    for step in 0..3u32 {
        let new_input: Vec<half::bf16> = (0..num_channels)
            .map(|c| half::bf16::from_f32(0.5 + 0.1 * (step as f32 + c as f32)))
            .collect();
        backend
            .copy_h2d(&bf16_slice_to_bytes(&new_input), new_in_ptr)
            .unwrap();

        // CPU reference: snapshot past, compute output, then
        // shift state — exactly matches the kernel's algorithm.
        let mut expected = vec![half::bf16::ZERO; num_channels as usize];
        for c in 0..num_channels as usize {
            let mut past = vec![0.0f32; kernel_size as usize];
            for i in 0..state_len {
                past[i] = conv_state_cpu[c * state_len + i].to_f32();
            }
            past[state_len] = new_input[c].to_f32();
            let mut acc = 0.0f32;
            for i in 0..kernel_size as usize {
                let w = weights[c * kernel_size as usize + i].to_f32();
                acc += w * past[i];
            }
            expected[c] = half::bf16::from_f32(acc);
            // Update the CPU-side state for the next iteration.
            for i in 0..state_len {
                conv_state_cpu[c * state_len + i] = half::bf16::from_f32(past[i + 1]);
            }
        }

        backend
            .launch_typed(
                kernel,
                [num_channels.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                backend.default_stream(),
                &[
                    KernelArg::Bytes(&num_channels.to_le_bytes()),
                    KernelArg::Bytes(&kernel_size.to_le_bytes()),
                    KernelArg::Buffer(weights_ptr),
                    KernelArg::Buffer(new_in_ptr),
                    KernelArg::Buffer(state_ptr),
                    KernelArg::Buffer(out_ptr),
                ],
            )
            .expect("launch causal_conv1d_decode");
        backend.synchronize(backend.default_stream()).unwrap();

        let mut out_raw = vec![0u8; num_channels as usize * 2];
        backend.copy_d2h(out_ptr, &mut out_raw).unwrap();
        let actual = bytes_to_bf16_vec(&out_raw);

        for i in 0..num_channels as usize {
            let e = expected[i].to_f32();
            let a = actual[i].to_f32();
            assert!(
                (e - a).abs() < 0.02,
                "step {step} ch {i}: expected {e}, got {a}"
            );
        }
    }

    backend.free(weights_ptr).unwrap();
    backend.free(state_ptr).unwrap();
    backend.free(new_in_ptr).unwrap();
    backend.free(out_ptr).unwrap();
}
