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
    Ok(Box::new(layer))
}
