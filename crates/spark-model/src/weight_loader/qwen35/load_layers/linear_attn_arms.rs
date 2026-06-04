// SPDX-License-Identifier: AGPL-3.0-only
//
// Helper functions for the LinearAttention arms of `load_layers`. Two
// flavours: the native-FP8 path (currently dead-coded with `&& false`)
// and the standard NVFP4-quantized path.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3SsmLayer};
use crate::weight_map::{
    DenseWeight, Fp8Weight, Nvfp4Variant, QuantizedWeight, SsmWeights, gpu_concat_rows,
    interleave_ba, load_fp8_block_scaled_as_fp8weight, load_ssm_qwen35, quantize_to_nvfp4,
};

// Currently unused while the FP8 LinearAttention dispatch arm in the
// caller (`load_layers.rs`) is short-circuited; preserved for the
// in-progress FP8 GDN kernel work, see the comment in `load_layers.rs`
// above the `LayerType::LinearAttention` arm.
#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn build_linear_attention_fp8(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let p = format!("{lp}.linear_attn");
    tracing::info!("Layer {layer_idx}: loading SSM FP8 native");

    let qkv_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj_qkv"), gpu)?;
    let z_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.in_proj_z"), gpu)?;
    let out_fp8 = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.out_proj"), gpu)?;

    let qkv_rows = qkv_fp8.n as usize;
    let z_rows = z_fp8.n as usize;
    let qkvz_n = qkv_rows + z_rows;

    let qkvz_weight_ptr = gpu.alloc(qkvz_n * h)?;
    gpu.copy_d2d(qkv_fp8.weight, qkvz_weight_ptr, qkv_rows * h)?;
    gpu.copy_d2d(
        z_fp8.weight,
        qkvz_weight_ptr.offset(qkv_rows * h),
        z_rows * h,
    )?;

    let qkvz_scale_ptr = gpu.alloc(qkvz_n * 4)?;
    gpu.copy_d2d(qkv_fp8.row_scale, qkvz_scale_ptr, qkv_rows * 4)?;
    gpu.copy_d2d(
        z_fp8.row_scale,
        qkvz_scale_ptr.offset(qkv_rows * 4),
        z_rows * 4,
    )?;

    let qkvz_fp8 = Fp8Weight {
        weight: qkvz_weight_ptr,
        row_scale: qkvz_scale_ptr,
        n: qkvz_n as u32,
        k: h as u32,
    };
    tracing::info!(
        "Layer {layer_idx}: SSM QKVZ FP8 [{qkvz_n},{h}], out_proj FP8 [{},{}]",
        out_fp8.n,
        out_fp8.k
    );

    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant)?;

    let qkv_size = config.ssm_qkv_size();
    let z_size = config.ssm_z_size();
    let qkvz_dense = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        qkv_size,
        &ssm35.in_proj_z,
        z_size,
        h,
        gpu,
    )?;

    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let ba_dense = interleave_ba(
        &DenseWeight {
            weight: ssm35.in_proj_a.weight,
        },
        &DenseWeight {
            weight: ssm35.in_proj_b.weight,
        },
        nv,
        nk,
        h,
        gpu,
    )?;

    let _value_dim = nv * config.linear_value_head_dim;

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: ssm35.conv1d,
        a_log: ssm35.a_log,
        dt_bias: ssm35.dt_bias,
        norm: ssm35.norm,
        out_proj: QuantizedWeight::null(),
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        None,
        None,
        None,
        config,
        gpu,
    )?;
    layer.out_proj_dense = Some(ssm35.out_proj);
    layer.set_fp8_weights(Some(qkvz_fp8), Some(out_fp8));
    tracing::info!("Layer {layer_idx}: SSM using BF16 dense for prefill, FP8 for decode");
    Ok(Box::new(layer))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_linear_attention_nvfp4(
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let ssm35 = load_ssm_qwen35(store, lp, gpu, variant)?;

    let qkv_rows = config.ssm_qkv_size();
    let z_rows = config.ssm_z_size();
    let qkvz_dense = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        qkv_rows,
        &ssm35.in_proj_z,
        z_rows,
        h,
        gpu,
    )?;

    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let ba_dense = interleave_ba(
        &DenseWeight {
            weight: ssm35.in_proj_a.weight,
        },
        &DenseWeight {
            weight: ssm35.in_proj_b.weight,
        },
        nv,
        nk,
        h,
        gpu,
    )?;

    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 =
        quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;

    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 = quantize_to_nvfp4(
        &ssm35.out_proj,
        h,
        value_dim,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

    // Native FP8 SSM prefill GEMM (cross-port from qwen35_dense.rs,
    // 2026-05-20). Same conv-k SNR-collapse vulnerability as the dense
    // 27B: the MoE A3B's GDN config has identical asymmetric conv
    // weights (k-segment ~18× smaller than v-segment), so the triple-
    // quant FP8→BF16→NVFP4→BF16 chain attenuates direction in the
    // k-channel just as it did on dense. Bypass the NVFP4 intermediate
    // by installing a single-scale FP8 copy of `qkvz_dense` and
    // `ssm35.out_proj` and dispatching prefill through `fp8_gemm_n128`.
    // Unconditional for FP8-on-disk variants (mirrors dense).
    let (qkvz_fp8_prefill, out_proj_fp8_prefill) = if matches!(variant, Nvfp4Variant::Fp8Dequanted)
    {
        // Diagnostic: fires once per LinearAttention layer (~30
        // lines for 35B-A3B). Confirms the MoE Bug #1 cross-port
        // (commit 7d5e8fc) is active and the SSM prefill path
        // dispatches through fp8_gemm_n128, not w4a16_gemm.
        tracing::info!(
            "SSM[{lp}] in_proj_qkv + out_proj via native FP8 prefill GEMM \
                 (BF16 act × FP8 weight via fp8_gemm_n128)"
        );
        let b2f_k = gpu.kernel("w4a16", "bf16_to_fp8")?;
        let qkvz_total = (qkvz_size * h) as u32;
        let qkvz_fp8 = gpu.alloc(qkvz_size * h)?;
        crate::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            qkvz_dense.weight,
            qkvz_fp8,
            qkvz_total,
            stream,
        )?;
        let out_total = (h * value_dim) as u32;
        let out_fp8 = gpu.alloc(h * value_dim)?;
        crate::layers::ops::bf16_to_fp8(
            gpu,
            b2f_k,
            ssm35.out_proj.weight,
            out_fp8,
            out_total,
            stream,
        )?;
        gpu.synchronize(stream)?;
        (Some(qkvz_fp8), Some(out_fp8))
    } else {
        (None, None)
    };

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: ssm35.conv1d,
        a_log: ssm35.a_log,
        dt_bias: ssm35.dt_bias,
        norm: ssm35.norm,
        out_proj: out_proj_nvfp4,
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        Some(qkvz_nvfp4),
        Some(qkvz_nvfp4_t),
        Some(out_proj_nvfp4_t),
        config,
        gpu,
    )?;
    layer.predequant_for_prefill(gpu, config, stream)?;
    // Install native FP8 prefill weights AFTER `predequant_for_prefill`
    // (which sets `out_proj_fp8` from NVFP4 + scale2). The FP8 path
    // overrides both pointers when active, routing prefill through
    // `fp8_gemm_n128` instead of `w4a16_gemm_t`. Decode batch paths
    // retain their NVFP4 fallback via the `qkvz_nvfp4*` fields above.
    if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
        layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
    }
    // ATLAS_GDN_BF16_WEIGHTS=1 extension: also install BF16 out_proj so
    // the prefill dispatcher takes the dense_gemm BF16 path (highest
    // dispatch priority). Eliminates FP8/NVFP4 quant noise on out_proj
    // — the noise was previously amplified by post_attn_norm's RMSNorm
    // into wildly different gate inputs at the MoE block (cos=0.42 vs
    // HF). Test fix for long-context drift root cause (commit 1db7572
    // and onward investigation). ssm35.out_proj is the BF16 weight
    // (loaded via dense_auto with FP8→BF16 dequant).
    if matches!(
        std::env::var("ATLAS_GDN_BF16_WEIGHTS").ok().as_deref(),
        Some("1")
    ) {
        // ssm35.out_proj weight is BF16 on GPU (from load_ssm_qwen35 →
        // dense_auto on Fp8Dequanted variant). It's a separate buffer
        // from out_proj_nvfp4 / out_proj_fp8_prefill. Set as dense path.
        layer.out_proj_dense = Some(ssm35.out_proj);
        tracing::info!(
            "SSM[{lp}] ATLAS_GDN_BF16_WEIGHTS: out_proj routed through BF16 dense_gemm (overrides FP8/NVFP4)"
        );
    }
    Ok(Box::new(layer))
}
