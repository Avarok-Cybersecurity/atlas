// SPDX-License-Identifier: AGPL-3.0-only
//! Real-model parity: ViT block on `vision_tower.blocks.0` weights.

#[allow(unused_imports)]
use super::super::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use crate::gpu::{DevicePtr, GpuBackend, KernelArg};

/// Vision-block forward on real `vision_tower.blocks.0`
/// weights. Mirrors the layer-3 LLM attention test but for the
/// ViT-style vision tower:
///
///   x → layer_norm(norm1)
///     → dense_gemv_bf16(qkv) + bf16_add(qkv_bias)
///     → split into Q | K | V
///     → attention_full (non-causal self-attention)
///     → dense_gemv_bf16(proj) + bf16_add(proj_bias)
///     → bf16_add(x, attn_out)         (residual)
///     → layer_norm(norm2)
///     → dense_gemv_bf16(fc1) + bf16_add(fc1_bias)
///     → gelu
///     → dense_gemv_bf16(fc2) + bf16_add(fc2_bias)
///     → bf16_add(x_resid, ffn_out)    (residual)
///     → x_final
///
/// One token of synthetic ViT input (so num_tokens = seq_len = 1
/// — the kernel grid still exercises every reduction stage,
/// just with a degenerate single-key softmax). Validates that
/// every BF16-only kernel composes correctly on real vision
/// weights from `mlx-community/Qwen3.5-4B-MLX-8bit`.
#[test]
#[ignore = "requires local copy of mlx-community/Qwen3.5-4B-MLX-8bit"]
fn metal_real_model_vision_block_forward() {
    use safetensors::SafeTensors;

    let model_dir = std::env::var("AVAROK_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    if !st_path.exists() {
        eprintln!("skipping: {} not found", st_path.display());
        return;
    }
    let file = std::fs::File::open(&st_path).expect("open safetensors");
    let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap") };
    let st = SafeTensors::deserialize(&mmap).expect("parse safetensors");

    let block = "vision_tower.blocks.0";
    let Some(backend) = maybe_backend() else {
        return;
    };

    // Vision tower dims (Qwen3.5-VL): hidden=1024, qkv_total=3072,
    // 16 attention heads of 64 dim each, MLP intermediate=4096.
    let hidden: u32 = 1024;
    let qkv_total: u32 = 3072;
    let intermediate: u32 = 4096;
    let num_heads: u32 = 16;
    let head_dim: u32 = 64;
    let eps: f32 = 1e-6;

    let load_bf16 = |name: &str| -> (DevicePtr, &[u8]) {
        let t = st.tensor(name).unwrap_or_else(|_| panic!("missing {name}"));
        let p = backend.alloc(t.data().len()).unwrap();
        backend.copy_h2d(t.data(), p).unwrap();
        (p, t.data())
    };

    // Load every weight + bias for block 0.
    let (norm1_w, _) = load_bf16(&format!("{block}.norm1.weight"));
    let (norm1_b, _) = load_bf16(&format!("{block}.norm1.bias"));
    let (qkv_w, _) = load_bf16(&format!("{block}.attn.qkv.weight"));
    let (qkv_b, _) = load_bf16(&format!("{block}.attn.qkv.bias"));
    let (proj_w, _) = load_bf16(&format!("{block}.attn.proj.weight"));
    let (proj_b, _) = load_bf16(&format!("{block}.attn.proj.bias"));
    let (norm2_w, _) = load_bf16(&format!("{block}.norm2.weight"));
    let (norm2_b, _) = load_bf16(&format!("{block}.norm2.bias"));
    let (fc1_w, _) = load_bf16(&format!("{block}.mlp.linear_fc1.weight"));
    let (fc1_b, _) = load_bf16(&format!("{block}.mlp.linear_fc1.bias"));
    let (fc2_w, _) = load_bf16(&format!("{block}.mlp.linear_fc2.weight"));
    let (fc2_b, _) = load_bf16(&format!("{block}.mlp.linear_fc2.bias"));

    // Synthetic input: one patch token at hidden=1024.
    let x_init: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(0.3 * (i as f32 * 0.011).sin()))
        .collect();
    let x_bytes = bf16_slice_to_bytes(&x_init);

    let alloc_bf16 = |n: u32| -> DevicePtr { backend.alloc(n as usize * 2).unwrap() };
    let x = alloc_bf16(hidden);
    let x_norm = alloc_bf16(hidden);
    let qkv = alloc_bf16(qkv_total);
    let qkv_with_bias = alloc_bf16(qkv_total);
    let attn_out = alloc_bf16(hidden);
    let proj_out = alloc_bf16(hidden);
    let proj_with_bias = alloc_bf16(hidden);
    let x_resid = alloc_bf16(hidden);
    let x_norm2 = alloc_bf16(hidden);
    let fc1_out = alloc_bf16(intermediate);
    let fc1_with_bias = alloc_bf16(intermediate);
    let fc1_act = alloc_bf16(intermediate);
    let fc2_out = alloc_bf16(hidden);
    let fc2_with_bias = alloc_bf16(hidden);
    let x_final = alloc_bf16(hidden);
    backend.copy_h2d(&x_bytes, x).unwrap();

    let stream = backend.default_stream();
    let n_tokens: u32 = 1;

    // ── Helpers ─────────────────────────────────────────────
    let ln_kernel = backend.kernel("layer_norm", "layer_norm").unwrap();
    let launch_ln =
        |x_in: DevicePtr, w: DevicePtr, b: DevicePtr, x_out: DevicePtr, hid: u32, n_tok: u32| {
            backend
                .launch_typed(
                    ln_kernel,
                    [n_tok, 1, 1],
                    [128, 1, 1],
                    0,
                    stream,
                    &[
                        KernelArg::Bytes(&hid.to_le_bytes()),
                        KernelArg::Bytes(&eps.to_le_bytes()),
                        KernelArg::Buffer(x_in),
                        KernelArg::Buffer(w),
                        KernelArg::Buffer(b),
                        KernelArg::Buffer(x_out),
                    ],
                )
                .expect("layer_norm launch");
        };

    let gemv_kernel = backend
        .kernel("dense_gemv_bf16", "dense_gemv_bf16")
        .unwrap();
    let launch_gemv = |w: DevicePtr, x_in: DevicePtr, y: DevicePtr, n: u32, k: u32| {
        backend
            .launch_typed(
                gemv_kernel,
                [n, 1, 1],
                [64, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Bytes(&k.to_le_bytes()),
                    KernelArg::Buffer(w),
                    KernelArg::Buffer(x_in),
                    KernelArg::Buffer(y),
                ],
            )
            .expect("gemv launch");
    };

    let add_kernel = backend.kernel("bf16_add", "bf16_add").unwrap();
    let launch_add = |a: DevicePtr, b: DevicePtr, out: DevicePtr, n: u32| {
        backend
            .launch_typed(
                add_kernel,
                [n.div_ceil(64), 1, 1],
                [64, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&n.to_le_bytes()),
                    KernelArg::Buffer(a),
                    KernelArg::Buffer(b),
                    KernelArg::Buffer(out),
                ],
            )
            .expect("bf16_add launch");
    };

    // ── Stage 1: norm1 → qkv → +bias ────────────────────────
    launch_ln(x, norm1_w, norm1_b, x_norm, hidden, n_tokens);
    launch_gemv(qkv_w, x_norm, qkv, qkv_total, hidden);
    launch_add(qkv, qkv_b, qkv_with_bias, qkv_total);

    // Q/K/V are contiguous BF16 slices of qkv_with_bias at offsets
    // 0, hidden*2, 2*hidden*2 (in bytes). Each is [num_heads,
    // head_dim] = [16, 64] = hidden BF16 elements.
    let q_view = qkv_with_bias;
    let k_view = qkv_with_bias.offset(hidden as usize * 2);
    let v_view = qkv_with_bias.offset(2 * hidden as usize * 2);

    // ── Stage 2: attention_full (seq_len = num_tokens = 1) ──
    let seq_len: u32 = 1;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let attn_kernel = backend.kernel("attention_full", "attention_full").unwrap();
    backend
        .launch_typed(
            attn_kernel,
            [num_heads * n_tokens, 1, 1],
            [32, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_tokens.to_le_bytes()),
                KernelArg::Bytes(&seq_len.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()), // num_kv_heads = num_heads (no GQA in ViT)
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&scale.to_le_bytes()),
                KernelArg::Buffer(q_view),
                KernelArg::Buffer(k_view),
                KernelArg::Buffer(v_view),
                KernelArg::Buffer(attn_out),
            ],
        )
        .expect("attention_full launch");

    // ── Stage 3: o_proj → +bias → residual ──────────────────
    launch_gemv(proj_w, attn_out, proj_out, hidden, hidden);
    launch_add(proj_out, proj_b, proj_with_bias, hidden);
    launch_add(x, proj_with_bias, x_resid, hidden);

    // ── Stage 4: norm2 → fc1 → +bias → gelu → fc2 → +bias ──
    launch_ln(x_resid, norm2_w, norm2_b, x_norm2, hidden, n_tokens);
    launch_gemv(fc1_w, x_norm2, fc1_out, intermediate, hidden);
    launch_add(fc1_out, fc1_b, fc1_with_bias, intermediate);
    let gelu_kernel = backend.kernel("gelu", "gelu").unwrap();
    backend
        .launch_typed(
            gelu_kernel,
            [intermediate.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&intermediate.to_le_bytes()),
                KernelArg::Buffer(fc1_with_bias),
                KernelArg::Buffer(fc1_act),
            ],
        )
        .expect("gelu launch");
    launch_gemv(fc2_w, fc1_act, fc2_out, hidden, intermediate);
    launch_add(fc2_out, fc2_b, fc2_with_bias, hidden);

    // ── Stage 5: residual → x_final ─────────────────────────
    launch_add(x_resid, fc2_with_bias, x_final, hidden);

    backend.synchronize(stream).unwrap();

    let mut x_final_raw = vec![0u8; hidden as usize * 2];
    backend.copy_d2h(x_final, &mut x_final_raw).unwrap();
    let final_vals = bytes_to_bf16_vec(&x_final_raw);

    let mut nan_or_inf = 0;
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    for v in &final_vals {
        let f = v.to_f32();
        if !f.is_finite() {
            nan_or_inf += 1;
        }
        let a = f.abs();
        sum_abs += a;
        if a > max_abs {
            max_abs = a;
        }
    }
    let mean_abs = sum_abs / final_vals.len() as f32;

    assert_eq!(
        nan_or_inf,
        0,
        "vision block: {nan_or_inf} non-finite outputs out of {}",
        final_vals.len()
    );
    assert!(
        mean_abs > 1e-4 && mean_abs < 50.0,
        "vision block: x_final mean-abs {mean_abs} outside sanity band; max_abs={max_abs}"
    );
    assert!(
        max_abs < 1e3,
        "vision block: x_final max_abs {max_abs} suggests activation explosion"
    );

    // Free everything in batch — order doesn't matter on UMA.
    for ptr in [
        norm1_w,
        norm1_b,
        qkv_w,
        qkv_b,
        proj_w,
        proj_b,
        norm2_w,
        norm2_b,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        x,
        x_norm,
        qkv,
        qkv_with_bias,
        attn_out,
        proj_out,
        proj_with_bias,
        x_resid,
        x_norm2,
        fc1_out,
        fc1_with_bias,
        fc1_act,
        fc2_out,
        fc2_with_bias,
        x_final,
    ] {
        backend.free(ptr).unwrap();
    }
}
