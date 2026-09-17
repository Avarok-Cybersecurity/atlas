// SPDX-License-Identifier: AGPL-3.0-only

//! Throughput benchmark for the metal backend.
//!
//! Loads layer 3 of `mlx-community/Qwen3.5-4B-MLX-8bit` once, then
//! runs the full-attention block forward N times in a tight loop.
//! Reports min / p50 / p99 / mean latency and the equivalent decode
//! tokens/sec rate (one full-attention layer per token; the model
//! has 8 full-attention layers + 24 linear-attention layers, so the
//! per-token total is 8 × this measurement plus the SSM-layer cost).
//!
//! Run with:
//!
//!     AVAROK_TARGET_HW=metal \
//!     AVAROK_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
//!     AVAROK_TARGET_QUANT=mlx_int8 \
//!     AVAROK_METAL_BENCH_ITERS=200 \
//!     cargo run --release --example metal_layer3_bench \
//!         --features metal --no-default-features
//!
//! Default: 100 iterations + 10 warmup.

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

fn main() -> Result<()> {
    let n_iters: usize = std::env::var("AVAROK_METAL_BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let n_warmup: usize = (n_iters / 10).max(5);

    let model_dir = std::env::var("AVAROK_MLX_MODEL_DIR").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("$HOME unset");
        format!("{home}/models/Qwen3.5-4B-MLX-8bit")
    });
    let st_path = std::path::Path::new(&model_dir).join("model.safetensors");
    if !st_path.exists() {
        anyhow::bail!("model.safetensors not found at {}", st_path.display());
    }
    let file = std::fs::File::open(&st_path).context("open safetensors")?;
    let mmap = unsafe { memmap2::Mmap::map(&file).context("mmap")? };
    let st = SafeTensors::deserialize(&mmap).context("parse safetensors")?;

    let modules = avarok_kernels::metallib_modules();
    if modules.is_empty() {
        anyhow::bail!(
            "metal kernel registry empty — re-build with \
             AVAROK_TARGET_HW=metal AVAROK_TARGET_MODEL=qwen3-5-4b-vlm-mlx-int8 \
             AVAROK_TARGET_QUANT=mlx_int8"
        );
    }
    let backend = MetalGpuBackend::new(0, &modules)?;

    // Real layer-3 dims.
    let hidden_size: u32 = 2560;
    let num_heads: u32 = 16;
    let num_kv_heads: u32 = 4;
    let head_dim: u32 = 256;
    let intermediate: u32 = 9216;
    let rms_eps: f32 = 1e-6;
    let group_size: u32 = 64;
    let q_total: u32 = num_heads * head_dim * 2;
    let q_only: u32 = num_heads * head_dim;
    let kv_dim: u32 = num_kv_heads * head_dim;
    let layer = "language_model.model.layers.3";

    // Load weights once.
    let load_bf16 = |name: &str| -> Result<DevicePtr> {
        let t = st.tensor(name).with_context(|| format!("missing {name}"))?;
        let p = backend.alloc(t.data().len())?;
        backend.copy_h2d(t.data(), p)?;
        Ok(p)
    };

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

    let x_init: Vec<half::bf16> = (0..hidden_size)
        .map(|i| half::bf16::from_f32(0.4 * (i as f32 * 0.013).sin()))
        .collect();

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
    let q_norm_out = alloc_bf16(q_only)?;
    let k_norm_out = alloc_bf16(kv_dim)?;
    let max_seq: u32 = 1;
    let k_cache = alloc_bf16(max_seq * kv_dim)?;
    let v_cache = alloc_bf16(max_seq * kv_dim)?;
    backend.copy_h2d(&bf16_slice_to_bytes(&x_init), x)?;

    // Qwen3.5-VL partial RoPE — only first head_dim/4 (=64) elements
    // of each head are rotated.
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

    // Pre-resolve every kernel to keep the timing loop tight.
    let rms = backend.kernel("rms_norm", "rms_norm")?;
    let rope = backend.kernel("rope_apply", "rope_apply")?;
    let kvap = backend.kernel("kv_cache_append", "kv_cache_append")?;
    let attn = backend.kernel("attention_decode", "attention_decode")?;
    let sg = backend.kernel("sigmoid_gate", "sigmoid_gate")?;
    let add = backend.kernel("bf16_add", "bf16_add")?;
    let silu = backend.kernel("silu_gate", "silu_gate")?;

    // Single-iteration forward closure.
    let do_forward = || -> Result<()> {
        // norm1
        backend.launch_typed(
            rms,
            [n_tokens, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&hidden_size.to_le_bytes()),
                KernelArg::Bytes(&rms_eps.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(input_ln),
                KernelArg::Buffer(x_norm),
            ],
        )?;
        q_proj.gemv(&backend, x_norm, q_full, stream)?;
        k_proj.gemv(&backend, x_norm, k, stream)?;
        v_proj.gemv(&backend, x_norm, v, stream)?;

        let q_view = q_full;
        let gate_view = q_full.offset(q_only as usize * 2);

        // per-head q/k norm
        backend.launch_typed(
            rms,
            [num_heads, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rms_eps.to_le_bytes()),
                KernelArg::Buffer(q_view),
                KernelArg::Buffer(q_norm),
                KernelArg::Buffer(q_norm_out),
            ],
        )?;
        backend.launch_typed(
            rms,
            [num_kv_heads, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rms_eps.to_le_bytes()),
                KernelArg::Buffer(k),
                KernelArg::Buffer(k_norm),
                KernelArg::Buffer(k_norm_out),
            ],
        )?;
        backend.copy_d2d_async(q_norm_out, q_view, q_only as usize * 2, stream)?;
        backend.copy_d2d_async(k_norm_out, k, kv_dim as usize * 2, stream)?;

        // RoPE (partial — first rotary_dim of each head)
        backend.launch_typed(
            rope,
            [half_dim, num_heads, 1],
            [1, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_tokens.to_le_bytes()),
                KernelArg::Bytes(&num_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rotary_dim.to_le_bytes()),
                KernelArg::Buffer(positions_ptr),
                KernelArg::Buffer(inv_freq_ptr),
                KernelArg::Buffer(q_view),
            ],
        )?;
        backend.launch_typed(
            rope,
            [half_dim, num_kv_heads, 1],
            [1, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&n_tokens.to_le_bytes()),
                KernelArg::Bytes(&num_kv_heads.to_le_bytes()),
                KernelArg::Bytes(&head_dim.to_le_bytes()),
                KernelArg::Bytes(&rotary_dim.to_le_bytes()),
                KernelArg::Buffer(positions_ptr),
                KernelArg::Buffer(inv_freq_ptr),
                KernelArg::Buffer(k),
            ],
        )?;

        // KV cache + attention
        let cache_pos: u32 = 0;
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
        let seq_len: u32 = 1;
        let scale: f32 = 1.0 / (head_dim as f32).sqrt();
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

        // gate + o_proj + residual
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
        o_proj.gemv(&backend, gated_attn, o, stream)?;
        backend.launch_typed(
            add,
            [hidden_size.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&hidden_size.to_le_bytes()),
                KernelArg::Buffer(x),
                KernelArg::Buffer(o),
                KernelArg::Buffer(x_resid),
            ],
        )?;

        // norm2 + FFN + residual
        backend.launch_typed(
            rms,
            [n_tokens, 1, 1],
            [128, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&hidden_size.to_le_bytes()),
                KernelArg::Bytes(&rms_eps.to_le_bytes()),
                KernelArg::Buffer(x_resid),
                KernelArg::Buffer(post_ln),
                KernelArg::Buffer(x_norm2),
            ],
        )?;
        gate_p.gemv(&backend, x_norm2, gate_act, stream)?;
        up_p.gemv(&backend, x_norm2, up_act, stream)?;
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
        backend.launch_typed(
            add,
            [hidden_size.div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
            stream,
            &[
                KernelArg::Bytes(&hidden_size.to_le_bytes()),
                KernelArg::Buffer(x_resid),
                KernelArg::Buffer(ffn_out),
                KernelArg::Buffer(x_final),
            ],
        )?;
        backend.synchronize(stream)?;
        Ok(())
    };

    // Warmup.
    println!("warmup: {n_warmup} iterations");
    for _ in 0..n_warmup {
        do_forward()?;
    }

    // Timed loop.
    println!("running {n_iters} iterations...");
    let mut samples: Vec<u128> = Vec::with_capacity(n_iters);
    let t_total = Instant::now();
    for _ in 0..n_iters {
        let t = Instant::now();
        do_forward()?;
        samples.push(t.elapsed().as_micros());
    }
    let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

    samples.sort_unstable();
    let min = samples[0];
    let p50 = samples[n_iters / 2];
    let p99 = samples[(n_iters * 99 / 100).min(n_iters - 1)];
    let max = samples[n_iters - 1];
    let mean: f64 = samples.iter().map(|&v| v as f64).sum::<f64>() / n_iters as f64;

    println!();
    println!("=== layer-3 full-attention block timings (µs) ===");
    println!("  iterations: {n_iters}");
    println!("  total wall: {:.1} ms", total_ms);
    println!("  min:        {:>6} µs", min);
    println!("  p50:        {:>6} µs", p50);
    println!("  mean:       {:>6.0} µs", mean);
    println!("  p99:        {:>6} µs", p99);
    println!("  max:        {:>6} µs", max);
    println!();
    // Decode-equivalent throughput is per-block; the model has 32
    // layers (8 full_attention + 24 linear_attention). Reporting
    // both the per-block rate and an upper-bound full-model rate
    // assuming every layer cost the same as a full_attention layer.
    let blocks_per_sec = 1_000_000.0 / mean;
    println!(
        "  per-block:           {:>7.1} forwards/s  ({:>5.1} ms each)",
        blocks_per_sec,
        mean / 1000.0
    );
    let full_model_per_sec = blocks_per_sec / 32.0;
    println!(
        "  full-model upper bound (32 layers × this latency): {:>5.1} tok/s",
        full_model_per_sec
    );

    Ok(())
}
