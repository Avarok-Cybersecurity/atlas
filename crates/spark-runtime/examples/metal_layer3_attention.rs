// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone metal-backend smoke: load layer 3 of
//! `mlx-community/Qwen3.5-4B-MLX-8bit` and run a single
//! full-attention block forward pass.
//!
//! Run with:
//!
//!     AVAROK_TARGET_HW=metal \
//!     AVAROK_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
//!     AVAROK_TARGET_QUANT=mlx_int8 \
//!     cargo run --example metal_layer3_attention \
//!         --features metal --no-default-features --release
//!
//! Override the model directory with `$AVAROK_MLX_MODEL_DIR`. The
//! default is `~/models/Qwen3.5-4B-MLX-8bit/`.
//!
//! This is essentially the integration-test code from
//! `metal_real_model_full_attention_block_layer3` lifted into a
//! standalone binary so anyone with the model on disk can confirm
//! their Apple Silicon Atlas build runs real kernels on real
//! weights — no `cargo test --include-ignored` invocation needed.

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelArg};
use spark_runtime::metal_backend::MetalGpuBackend;
use spark_runtime::weights::mlx_int8::MlxInt8Weight;
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
    // ── Locate + mmap the model ─────────────────────────────────
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
    let st = SafeTensors::deserialize(&mmap).context("parse safetensors header")?;

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

    // ── Real layer-3 dims (from upstream config.json) ───────────
    let hidden_size: u32 = 2560;
    let num_heads: u32 = 16;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 256;
    let intermediate: u32 = 9216;
    let rms_eps: f32 = 1e-6;
    let group_size: u32 = 64;
    let q_total: u32 = num_heads * head_dim * 2; // attn_output_gate
    let q_only: u32 = num_heads * head_dim;
    let kv_dim: u32 = num_kv_heads * head_dim;
    let layer = "language_model.model.layers.3";

    let load_bf16 = |name: &str| -> Result<DevicePtr> {
        let t = st
            .tensor(name)
            .with_context(|| format!("missing tensor {name}"))?;
        let p = backend.alloc(t.data().len())?;
        backend.copy_h2d(t.data(), p)?;
        Ok(p)
    };

    println!("loading layer-3 weights (12 tensors)...");
    let t0 = Instant::now();
    let input_ln = load_bf16(&format!("{layer}.input_layernorm.weight"))?;
    let q_norm = load_bf16(&format!("{layer}.self_attn.q_norm.weight"))?;
    let k_norm = load_bf16(&format!("{layer}.self_attn.k_norm.weight"))?;
    let post_ln = load_bf16(&format!("{layer}.post_attention_layernorm.weight"))?;
    let q_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.q_proj"),
        group_size,
    )?;
    let k_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.k_proj"),
        group_size,
    )?;
    let v_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.v_proj"),
        group_size,
    )?;
    let o_proj = MlxInt8Weight::load(
        &backend,
        &st,
        &format!("{layer}.self_attn.o_proj"),
        group_size,
    )?;
    let gate_p = MlxInt8Weight::load(&backend, &st, &format!("{layer}.mlp.gate_proj"), group_size)?;
    let up_p = MlxInt8Weight::load(&backend, &st, &format!("{layer}.mlp.up_proj"), group_size)?;
    let down_p = MlxInt8Weight::load(&backend, &st, &format!("{layer}.mlp.down_proj"), group_size)?;
    println!("  → loaded in {:.2?}", t0.elapsed());

    // ── Synthetic input residual stream ─────────────────────────
    let x_init: Vec<half::bf16> = (0..hidden_size)
        .map(|i| half::bf16::from_f32(0.4 * (i as f32 * 0.013).sin()))
        .collect();

    // Allocate intermediates.
    let alloc_bf16 = |n: u32| -> Result<DevicePtr> { Ok(backend.alloc(n as usize * 2)?) };
    let x = alloc_bf16(hidden_size)?;
    let x_norm = alloc_bf16(hidden_size)?;
    let q_full = alloc_bf16(q_total)?;
    let k = alloc_bf16(kv_dim)?;
    let v = alloc_bf16(kv_dim)?;
    let attn_out = alloc_bf16(q_only)?;
    let gated_attn = alloc_bf16(q_only)?;
    let o = alloc_bf16(hidden_size)?;
    let x_resid = alloc_bf16(hidden_size)?;
    let x_norm2 = alloc_bf16(hidden_size)?;
    let gate_act = alloc_bf16(intermediate)?;
    let up_act = alloc_bf16(intermediate)?;
    let ffn_act = alloc_bf16(intermediate)?;
    let ffn_out = alloc_bf16(hidden_size)?;
    let x_final = alloc_bf16(hidden_size)?;
    let max_seq: u32 = 1;
    let k_cache = alloc_bf16(max_seq * kv_dim)?;
    let v_cache = alloc_bf16(max_seq * kv_dim)?;
    backend.copy_h2d(&bf16_slice_to_bytes(&x_init), x)?;

    // RoPE inv_freq table — Qwen3.5-VL applies partial RoPE
    // (partial_rotary_factor=0.25), so only the first head_dim/4
    // elements of each head are rotated.
    let rope_theta: f32 = 10_000_000.0;
    let rotary_dim: u32 = head_dim / 4;
    let half_dim = rotary_dim / 2;
    let inv_freq: Vec<u8> = (0..half_dim)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / rotary_dim as f32))
        .flat_map(|f: f32| f.to_le_bytes())
        .collect();
    let inv_freq_ptr = backend.alloc(inv_freq.len())?;
    backend.copy_h2d(&inv_freq, inv_freq_ptr)?;
    let positions = 0u32.to_le_bytes().to_vec();
    let positions_ptr = backend.alloc(positions.len())?;
    backend.copy_h2d(&positions, positions_ptr)?;

    let stream = backend.default_stream();
    let n_tokens: u32 = 1;

    let t_fwd = Instant::now();

    // ── Stage 1: input_layernorm ────────────────────────────────
    let rms = backend.kernel("rms_norm", "rms_norm")?;
    let launch_rms =
        |x_in: DevicePtr, w: DevicePtr, x_out: DevicePtr, n_tok: u32, hid: u32| -> Result<()> {
            backend.launch_typed(
                rms,
                [n_tok, 1, 1],
                [128, 1, 1],
                0,
                stream,
                &[
                    KernelArg::Bytes(&hid.to_le_bytes()),
                    KernelArg::Bytes(&rms_eps.to_le_bytes()),
                    KernelArg::Buffer(x_in),
                    KernelArg::Buffer(w),
                    KernelArg::Buffer(x_out),
                ],
            )
        };
    launch_rms(x, input_ln, x_norm, n_tokens, hidden_size)?;

    // ── Stage 2: Q, K, V projections ────────────────────────────
    q_proj.gemv(&backend, x_norm, q_full, stream)?;
    k_proj.gemv(&backend, x_norm, k, stream)?;
    v_proj.gemv(&backend, x_norm, v, stream)?;

    // Split q_full → (Q | gate).
    let q_view = q_full;
    let gate_view = q_full.offset(q_only as usize * 2);

    // ── Stage 3: per-head q_norm / k_norm ───────────────────────
    let q_norm_out = alloc_bf16(q_only)?;
    let k_norm_out = alloc_bf16(kv_dim)?;
    launch_rms(q_view, q_norm, q_norm_out, num_heads, head_dim)?;
    launch_rms(k, k_norm, k_norm_out, num_kv_heads, head_dim)?;
    backend.copy_d2d_async(q_norm_out, q_view, q_only as usize * 2, stream)?;
    backend.copy_d2d_async(k_norm_out, k, kv_dim as usize * 2, stream)?;

    // ── Stage 4: RoPE ───────────────────────────────────────────
    let rope = backend.kernel("rope_apply", "rope_apply")?;
    let launch_rope = |x_inout: DevicePtr, n_h: u32| -> Result<()> {
        backend.launch_typed(
            rope,
            [half_dim, n_h, 1],
            [1, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_tokens.to_le_bytes()),
                KernelArg::Bytes(&n_h.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rotary_dim.to_le_bytes()),
                KernelArg::Buffer(positions_ptr),
                KernelArg::Buffer(inv_freq_ptr),
                KernelArg::Buffer(x_inout),
            ],
        )
    };
    launch_rope(q_view, num_heads)?;
    launch_rope(k, num_kv_heads)?;

    // ── Stage 5: KV cache append at pos 0 ───────────────────────
    let cache_pos: u32 = 0;
    let kvap = backend.kernel("kv_cache_append", "kv_cache_append")?;
    backend.launch_typed(
        kvap,
        [head_dim, num_kv_heads, 1],
        [1, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&head_dim.to_le_bytes()),
            KernelArg::Bytes(&cache_pos.to_le_bytes()),
            KernelArg::Buffer(k),
            KernelArg::Buffer(v),
            KernelArg::Buffer(k_cache),
            KernelArg::Buffer(v_cache),
        ],
    )?;

    // ── Stage 6: attention_decode ───────────────────────────────
    let seq_len: u32 = 1;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();
    let attn = backend.kernel("attention_decode", "attention_decode")?;
    backend.launch_typed(
        attn,
        [num_heads, 1, 1],
        [32, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&seq_len.to_le_bytes()),
            KernelArg::Bytes(&num_heads.to_le_bytes()),
            KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
            KernelArg::Bytes(&head_dim.to_le_bytes()),
            KernelArg::Bytes(&scale.to_le_bytes()),
            KernelArg::Buffer(q_view),
            KernelArg::Buffer(k_cache),
            KernelArg::Buffer(v_cache),
            KernelArg::Buffer(attn_out),
        ],
    )?;

    // ── Stage 7: sigmoid_gate(attn_gate, attn_out) ──────────────
    let sg = backend.kernel("sigmoid_gate", "sigmoid_gate")?;
    backend.launch_typed(
        sg,
        [q_only.div_ceil(64), 1, 1],
        [64, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&q_only.to_le_bytes()),
            KernelArg::Buffer(gate_view),
            KernelArg::Buffer(attn_out),
            KernelArg::Buffer(gated_attn),
        ],
    )?;

    // ── Stage 8: o_proj ─────────────────────────────────────────
    o_proj.gemv(&backend, gated_attn, o, stream)?;

    // ── Stage 9: residual ───────────────────────────────────────
    let add = backend.kernel("bf16_add", "bf16_add")?;
    let launch_add = |a: DevicePtr, b: DevicePtr, out: DevicePtr, n: u32| -> Result<()> {
        backend.launch_typed(
            add,
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
    launch_add(x, o, x_resid, hidden_size)?;

    // ── Stage 10: post_attention_layernorm + FFN ────────────────
    launch_rms(x_resid, post_ln, x_norm2, n_tokens, hidden_size)?;
    gate_p.gemv(&backend, x_norm2, gate_act, stream)?;
    up_p.gemv(&backend, x_norm2, up_act, stream)?;
    let silu = backend.kernel("silu_gate", "silu_gate")?;
    backend.launch_typed(
        silu,
        [intermediate.div_ceil(64), 1, 1],
        [64, 1, 1],
        0,
        stream,
        &[
            KernelArg::Bytes(&intermediate.to_le_bytes()),
            KernelArg::Buffer(gate_act),
            KernelArg::Buffer(up_act),
            KernelArg::Buffer(ffn_act),
        ],
    )?;
    down_p.gemv(&backend, ffn_act, ffn_out, stream)?;
    launch_add(x_resid, ffn_out, x_final, hidden_size)?;

    backend.synchronize(stream)?;
    let fwd_us = t_fwd.elapsed().as_micros();

    // ── Read back + report ──────────────────────────────────────
    let mut x_final_raw = vec![0u8; hidden_size as usize * 2];
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
    println!("layer-3 full-attention block forward complete in {fwd_us} µs");
    println!(
        "  output [hidden={}]: mean|x| = {:.4}, max|x| = {:.4}, non-finite = {}",
        hidden_size, mean_abs, max_abs, nan_inf
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
    if !(mean_abs > 1e-3 && mean_abs < 50.0) {
        anyhow::bail!("output magnitude outside sanity band [1e-3, 50]");
    }

    println!();
    println!("✓ metal layer-3 attention block forward succeeded");
    Ok(())
}
