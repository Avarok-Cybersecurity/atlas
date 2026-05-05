// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

/// Load an FP8 E4M3 block-scaled checkpoint weight as a native [`Fp8Weight`].
///
/// The FP8 checkpoint stores:
///   - `{prefix}.weight`: FP8E4M3 tensor [N, K]
///   - `{prefix}.weight_scale_inv`: BF16 tensor [N/block, K/block]
///
/// The `w8a16_gemv` kernel uses 2D block scales directly:
///   `dequant[i,j] = E4M3_LUT[fp8[i,j]] * block_scale[i/BS, j/BS]`
/// No per-row max reduction needed — the kernel loads the correct block
/// scale for each 128-element K chunk.
pub fn load_fp8_block_scaled_as_fp8weight(
    store: &WeightStore,
    prefix: &str,
    _gpu: &dyn GpuBackend,
) -> Result<Fp8Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    ensure!(
        w.dtype == WeightDtype::FP8E4M3,
        "Expected FP8E4M3 for {prefix}.weight, got {:?}",
        w.dtype,
    );
    ensure!(
        w.shape.len() == 2,
        "Expected 2D weight for {prefix}, got {:?}",
        w.shape
    );
    let n = w.shape[0];
    let k = w.shape[1];
    let weight_ptr = w.ptr;

    // Load block scale_inv: BF16 [N/BS, K/BS] — already on GPU from safetensors
    let scale_key = format!("{prefix}.weight_scale_inv");
    let s = store.get(&scale_key)?;
    ensure!(
        s.shape.len() == 2,
        "Expected 2D shape for {scale_key}, got {:?}",
        s.shape,
    );

    tracing::debug!(
        "FP8 block scales: {prefix} [{n},{k}] scale=[{},{}]",
        s.shape[0],
        s.shape[1],
    );

    // Pass block scales directly — the w8a16_gemv kernel handles 2D indexing
    Ok(Fp8Weight {
        weight: weight_ptr,
        row_scale: s.ptr, // BF16 [N/BS, K/BS] block scales on GPU
        n: n as u32,
        k: k as u32,
    })
}

/// Quantize a BF16 dense weight to NVFP4 on GPU.
///
/// Two-phase: (1) find global max, (2) per-group E2M1 quantization.
/// Halves weight bandwidth vs FP8 (0.5 bytes/weight + group scales vs 1 byte/weight).
/// Called once at model load time (not on the hot path).
pub(crate) fn quantize_to_nvfp4(
    bf16_weight: &DenseWeight,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_kernel: spark_runtime::gpu::KernelHandle,
    quantize_kernel: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<QuantizedWeight> {
    use spark_runtime::kernel_args::KernelLaunch;

    let total = n * k;

    // Phase 1: Find global absolute max
    let max_buf = gpu.alloc(4)?;
    gpu.memset(max_buf, 0, 4)?;

    let grid1 = (total / 256).clamp(1, 1024) as u32;
    KernelLaunch::new(gpu, absmax_kernel)
        .grid([grid1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(max_buf)
        .arg_u32(total as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;
    let mut max_bytes = [0u8; 4];
    gpu.copy_d2h(max_buf, &mut max_bytes)?;
    let global_max = f32::from_le_bytes(max_bytes);

    // scale2 = global_max / (6.0 * 448.0)  [FP8 E4M3 max = 448]
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };

    // Diagnostic: log absmax result for first few quantizations
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static QUANT_DIAG: AtomicUsize = AtomicUsize::new(0);
        if QUANT_DIAG.fetch_add(1, Ordering::Relaxed) < 5 {
            tracing::info!(
                "quantize_to_nvfp4: n={n} k={k} total={total} global_max={global_max:.6} scale2={scale2:.8} grid1={grid1}",
            );
        }
    }

    // Phase 2: Quantize
    let packed_buf = gpu.alloc(n * k / 2)?;
    let scale_buf = gpu.alloc(n * k / 16)?;

    KernelLaunch::new(gpu, quantize_kernel)
        .grid([n as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf16_weight.weight)
        .arg_ptr(packed_buf)
        .arg_ptr(scale_buf)
        .arg_f32(scale2)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;

    Ok(QuantizedWeight {
        weight: packed_buf,
        weight_scale: scale_buf,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
    })
}

/// Load attention weights for a full_attention layer.
pub(crate) fn load_attention(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    config: &atlas_core::config::ModelConfig,
) -> Result<AttentionWeights> {
    let p = format!("{layer_prefix}.self_attn");
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let h = config.hidden_size;
    let qkv_out = config.num_attention_heads * config.head_dim;
    let _kv_out = config.num_key_value_heads * config.head_dim;
    Ok(AttentionWeights {
        q_proj: dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?,
        k_proj: dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?,
        v_proj: dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?,
        o_proj: quantized_any(
            store,
            &format!("{p}.o_proj"),
            h,
            qkv_out,
            gpu,
            variant,
            qctx,
        )?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    })
}

/// Load SSM weights for a linear_attention layer.
pub(crate) fn load_ssm(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
    config: &atlas_core::config::ModelConfig,
) -> Result<SsmWeights> {
    let p = format!("{layer_prefix}.linear_attn");
    let h = config.hidden_size;
    // out_proj is [hidden_size, d_inner] where d_inner = linear_value_head_dim * linear_num_value_heads
    let d_inner = config.linear_value_head_dim * config.linear_num_value_heads;
    Ok(SsmWeights {
        in_proj_qkvz: dense_auto(store, &format!("{p}.in_proj_qkvz.weight"), gpu)?,
        in_proj_ba: dense_auto(store, &format!("{p}.in_proj_ba.weight"), gpu)?,
        conv1d: dense(store, &format!("{p}.conv1d.weight"))?,
        a_log: dense_keep_f32(store, &format!("{p}.A_log"), gpu)?,
        dt_bias: dense_keep_f32(store, &format!("{p}.dt_bias"), gpu)?,
        norm: dense(store, &format!("{p}.norm.weight"))?,
        out_proj: quantized_any(
            store,
            &format!("{p}.out_proj"),
            h,
            d_inner,
            gpu,
            variant,
            qctx,
        )?,
    })
}

/// Load MoE weights for a layer.
///
/// Under EP (ep_world_size > 1), only local experts are loaded from the store.
/// Remote experts get NULL pointers — kernels detect NULL and write zero output.
pub(crate) fn load_moe(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    load_moe_inner(
        store,
        layer_prefix,
        num_experts,
        gpu,
        config,
        variant,
        qctx,
        false,
    )
}

/// Load MoE with option to skip routed experts (native FP8 loads them separately).
pub(crate) fn load_moe_skip_experts(
    store: &WeightStore,
    layer_prefix: &str,
    num_experts: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<MoeWeights> {
    load_moe_inner(
        store,
        layer_prefix,
        num_experts,
        gpu,
        config,
        variant,
        qctx,
        true,
    )
}
