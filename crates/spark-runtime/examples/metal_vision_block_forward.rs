// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone metal-backend smoke for the ViT vision tower: load
//! `vision_tower.blocks.0` of `mlx-community/Qwen3.5-4B-MLX-8bit`
//! and run a single non-causal attention + GeLU-MLP block forward.
//!
//! Run with:
//!
//!     AVAROK_TARGET_HW=metal \
//!     AVAROK_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
//!     AVAROK_TARGET_QUANT=mlx_int8 \
//!     cargo run --example metal_vision_block_forward \
//!         --features metal --no-default-features --release
//!
//! The vision tower stores plain BF16 weights (no MLX
//! quantization), uses LayerNorm with bias (not RMSNorm), and a
//! GeLU MLP — distinct from the LLM trunk's MLX-int8 + RMSNorm +
//! SwiGLU kernel chain. This binary is the parallel of
//! `metal_layer3_attention` for the vision-tower side.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::metal_backend::MetalGpuBackend;
use std::time::Instant;

fn bf16_slice_to_bytes(values: &[half::bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn bytes_to_bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        out.push(half::bf16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

fn main() -> Result<()> {
    let model_dir = std::env::var("AVAROK_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("$HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    if !st_path.exists() {
        anyhow::bail!(
            "model.safetensors not found at {} — set $AVAROK_MLX_MODEL_DIR or \
             run `git lfs clone https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-8bit \
             ~/models/Qwen3.5-4B-MLX-8bit`",
            st_path.display()
        );
    }
    let file = std::fs::File::open(&st_path).context("open safetensors")?;
    let mmap = unsafe { memmap2::Mmap::map(&file).context("mmap")? };
    let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;

    let modules = avarok_kernels::metallib_modules();
    if modules.is_empty() {
        anyhow::bail!(
            "metal kernel registry is empty — re-build with \
             AVAROK_TARGET_HW=metal AVAROK_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
             AVAROK_TARGET_QUANT=mlx_int8"
        );
    }
    println!("metal kernel registry: {} modules loaded", modules.len());
    let backend = MetalGpuBackend::new(0, &modules)?;

    // Real ViT block-0 dims: hidden=1024, qkv=3072, intermediate=4096,
    // 16 heads × 64 head_dim, no GQA. Plain BF16 throughout.
    let hidden: u32 = 1024;
    let qkv_total: u32 = 3072;
    let intermediate: u32 = 4096;
    let num_heads: u32 = 16;
    let head_dim: u32 = 64;
    let eps: f32 = 1e-6;
    let block = "vision_tower.blocks.0";

    let load_bf16 = |name: &str| -> Result<DevicePtr> {
        let t = st
            .tensor(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let p = backend.alloc(t.data().len())?;
        backend.copy_h2d(t.data(), p)?;
        Ok(p)
    };

    println!("loading vision_tower.blocks.0 weights (12 tensors)...");
    let t0 = Instant::now();
    let norm1_w = load_bf16(&format!("{block}.norm1.weight"))?;
    let norm1_b = load_bf16(&format!("{block}.norm1.bias"))?;
    let qkv_w = load_bf16(&format!("{block}.attn.qkv.weight"))?;
    let qkv_b = load_bf16(&format!("{block}.attn.qkv.bias"))?;
    let proj_w = load_bf16(&format!("{block}.attn.proj.weight"))?;
    let proj_b = load_bf16(&format!("{block}.attn.proj.bias"))?;
    let norm2_w = load_bf16(&format!("{block}.norm2.weight"))?;
    let norm2_b = load_bf16(&format!("{block}.norm2.bias"))?;
    let fc1_w = load_bf16(&format!("{block}.mlp.linear_fc1.weight"))?;
    let fc1_b = load_bf16(&format!("{block}.mlp.linear_fc1.bias"))?;
    let fc2_w = load_bf16(&format!("{block}.mlp.linear_fc2.weight"))?;
    let fc2_b = load_bf16(&format!("{block}.mlp.linear_fc2.bias"))?;
    println!("  → loaded in {:.2?}", t0.elapsed());

    // Synthetic patch token at hidden=1024.
    let x_init: Vec<half::bf16> = (0..hidden)
        .map(|i| half::bf16::from_f32(0.3 * (i as f32 * 0.011).sin()))
        .collect();

    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    let x = alloc_bf16(hidden)?;
    let x_norm = alloc_bf16(hidden)?;
    let qkv = alloc_bf16(qkv_total)?;
    let qkv_with_bias = alloc_bf16(qkv_total)?;
    let attn_out = alloc_bf16(hidden)?;
    let proj_out = alloc_bf16(hidden)?;
    let proj_with_bias = alloc_bf16(hidden)?;
    let x_resid = alloc_bf16(hidden)?;
    let x_norm2 = alloc_bf16(hidden)?;
    let fc1_out = alloc_bf16(intermediate)?;
    let fc1_with_bias = alloc_bf16(intermediate)?;
    let fc1_act = alloc_bf16(intermediate)?;
    let fc2_out = alloc_bf16(hidden)?;
    let fc2_with_bias = alloc_bf16(hidden)?;
    let x_final = alloc_bf16(hidden)?;
    backend.copy_h2d(&bf16_slice_to_bytes(&x_init), x)?;

    let stream = backend.default_stream();
    let n_tokens: u32 = 1;

    // ── Helper closures ─────────────────────────────────────────
    let ln_kernel = backend.kernel("layer_norm", "layer_norm")?;
    let launch_ln = |x_in: DevicePtr,
                     w: DevicePtr,
                     b: DevicePtr,
                     x_out: DevicePtr,
                     hid: u32,
                     n_tok: u32|
     -> Result<()> {
        backend.launch_typed(
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
    };

    let gemv_kernel = backend.kernel("dense_gemv_bf16", "dense_gemv_bf16")?;
    let launch_gemv = |w: DevicePtr, x_in: DevicePtr, y: DevicePtr, n: u32, k: u32| -> Result<()> {
        backend.launch_typed(
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
    };

    let add_kernel = backend.kernel("bf16_add", "bf16_add")?;
    let launch_add = |a: DevicePtr, b: DevicePtr, out: DevicePtr, n: u32| -> Result<()> {
        backend.launch_typed(
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
    };

    let t_fwd = Instant::now();

    // ── Stage 1: norm1 → qkv → +bias ────────────────────────────
    launch_ln(x, norm1_w, norm1_b, x_norm, hidden, n_tokens)?;
    launch_gemv(qkv_w, x_norm, qkv, qkv_total, hidden)?;
    launch_add(qkv, qkv_b, qkv_with_bias, qkv_total)?;

    // Q/K/V slices via DevicePtr offsets (each is `hidden` BF16 elements).
    let q_view = qkv_with_bias;
    let k_view = qkv_with_bias.offset(hidden as usize * 2);
    let v_view = qkv_with_bias.offset(2 * hidden as usize * 2);

    // ── Stage 2: attention_full (seq_len = num_tokens = 1) ──────
    let seq_len: u32 = 1;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let attn_kernel = backend.kernel("attention_full", "attention_full")?;
    backend.launch_typed(
        attn_kernel,
        [num_heads * n_tokens, 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&n_tokens.to_le_bytes()),
            KernelArg::Bytes(&seq_len.to_le_bytes()),
            KernelArg::Bytes(&num_heads.to_le_bytes()),
            KernelArg::Bytes(&num_heads.to_le_bytes()), // num_kv_heads = num_heads (no GQA)
            KernelArg::Bytes(&head_dim.to_le_bytes()),
            KernelArg::Bytes(&scale.to_le_bytes()),
            KernelArg::Buffer(q_view),
            KernelArg::Buffer(k_view),
            KernelArg::Buffer(v_view),
            KernelArg::Buffer(attn_out),
        ],
    )?;

    // ── Stage 3: o_proj → +bias → residual ──────────────────────
    launch_gemv(proj_w, attn_out, proj_out, hidden, hidden)?;
    launch_add(proj_out, proj_b, proj_with_bias, hidden)?;
    launch_add(x, proj_with_bias, x_resid, hidden)?;

    // ── Stage 4: norm2 → fc1 → +bias → gelu → fc2 → +bias ──────
    launch_ln(x_resid, norm2_w, norm2_b, x_norm2, hidden, n_tokens)?;
    launch_gemv(fc1_w, x_norm2, fc1_out, intermediate, hidden)?;
    launch_add(fc1_out, fc1_b, fc1_with_bias, intermediate)?;
    let gelu_kernel = backend.kernel("gelu", "gelu")?;
    backend.launch_typed(
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
    )?;
    launch_gemv(fc2_w, fc1_act, fc2_out, hidden, intermediate)?;
    launch_add(fc2_out, fc2_b, fc2_with_bias, hidden)?;
    launch_add(x_resid, fc2_with_bias, x_final, hidden)?;

    backend.synchronize(stream)?;
    let fwd_us = t_fwd.elapsed().as_micros();

    // ── Read back + report ──────────────────────────────────────
    let mut x_final_raw = vec![0u8; hidden as usize * 2];
    backend.copy_d2h(x_final, &mut x_final_raw)?;
    let final_vals = bytes_to_bf16_vec(&x_final_raw);

    let mut nan_inf = 0;
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    for v in &final_vals {
        let f = v.to_f32();
        if !f.is_finite() {
            nan_inf += 1;
        }
        let a = f.abs();
        sum_abs += a;
        if a > max_abs {
            max_abs = a;
        }
    }
    let mean_abs = sum_abs / final_vals.len() as f32;

    println!();
    println!("vision_tower.blocks.0 forward complete in {fwd_us} µs");
    println!(
        "  output [hidden={}]: mean|x| = {:.4}, max|x| = {:.4}, non-finite = {}",
        hidden, mean_abs, max_abs, nan_inf
    );
    println!(
        "  first 8 outputs: {:?}",
        &final_vals[..8]
            .iter()
            .map(|v| v.to_f32())
            .collect::<Vec<_>>()
    );

    if nan_inf != 0 {
        anyhow::bail!("non-finite output detected");
    }
    if !(mean_abs > 1e-4 && mean_abs < 50.0) {
        anyhow::bail!("output magnitude outside sanity band [1e-4, 50]");
    }

    println!();
    println!("✓ metal vision-tower block forward succeeded");
    Ok(())
}
