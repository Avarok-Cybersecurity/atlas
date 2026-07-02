// SPDX-License-Identifier: AGPL-3.0-only

//! `LinearAttention` (GDN SSM) layer loading for `Qwen35DenseWeightLoader`.
//! Extracted from `qwen35_dense.rs` so the parent file fits the 500-LoC
//! budget.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3SsmLayer};
use crate::weight_map::{
    DenseWeight, Nvfp4Variant, SsmWeights, dense, dense_auto, dense_f32_safe, dense_keep_f32,
    dequant_nvfp4_to_bf16, gpu_concat_rows, interleave_ba, quantize_to_nvfp4, quantized_auto,
};

/// Kernel handles shared across layers, threaded through so the caller loads
/// them once instead of re-resolving per layer.
pub(super) struct LinearAttnKernels {
    pub absmax_k: KernelHandle,
    pub quantize_k: KernelHandle,
    pub bf16_to_fp8_k: Option<KernelHandle>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_linear_attention_layer(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    stream: u64,
    lp: &str,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    kernels: &LinearAttnKernels,
    layer_idx: usize,
) -> Result<Box<dyn TransformerLayer>> {
    let h = config.hidden_size;
    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let qkv_rows = config.ssm_qkv_size();
    let z_rows = config.ssm_z_size();
    let value_dim = nv * config.linear_value_head_dim;
    let la = format!("{lp}.linear_attn");

    // Detect if QKV/Z/out_proj are native Standard NVFP4 (U8 packed).
    // When all three are U8, load directly as QuantizedWeight and
    // concat on GPU — skipping the NVFP4→BF16→NVFP4 double conversion.
    let qkv_is_u8 = matches!(
        store
            .get(&format!("{la}.in_proj_qkv.weight"))
            .map(|w| w.dtype),
        Ok(WeightDtype::UInt8)
    );
    let z_is_u8 = matches!(
        store
            .get(&format!("{la}.in_proj_z.weight"))
            .map(|w| w.dtype),
        Ok(WeightDtype::UInt8)
    );
    let out_is_u8 = matches!(
        store.get(&format!("{la}.out_proj.weight")).map(|w| w.dtype),
        Ok(WeightDtype::UInt8)
    );
    let native_nvfp4 = qkv_is_u8 && z_is_u8 && out_is_u8;

    // SSM projections: load per-projection by on-disk dtype.
    let load_ssm_proj = |name: &str, rows: usize, cols: usize| -> Result<DenseWeight> {
        if store.contains(&format!("{name}.weight_packed")) {
            dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
        } else if matches!(
            store.get(&format!("{name}.weight")).map(|w| w.dtype),
            Ok(WeightDtype::UInt8)
        ) {
            dequant_nvfp4_to_bf16(store, name, rows, cols, gpu)
        } else {
            dense_auto(store, &format!("{name}.weight"), gpu)
        }
    };

    // A, B: BF16 in most checkpoints, but may be NVFP4 (U8 packed)
    // in fully-quantized checkpoints (e.g. sakamakismile NVFP4-MTP).
    let in_proj_a = load_ssm_proj(&format!("{la}.in_proj_a"), nv, h)?;
    let in_proj_b = load_ssm_proj(&format!("{la}.in_proj_b"), nv, h)?;
    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
    let a_log = dense_keep_f32(store, &format!("{la}.A_log"), gpu)?;
    let dt_bias = dense_keep_f32(store, &format!("{la}.dt_bias"), gpu)?;
    let norm = dense_f32_safe(store, &format!("{la}.norm.weight"), gpu)?;
    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

    let qkvz_size = config.ssm_qkvz_size();
    let absmax_k = kernels.absmax_k;
    let quantize_k = kernels.quantize_k;

    if native_nvfp4 {
        // ── Native NVFP4 path: load pre-quantized weights directly ──
        if layer_idx == 0 {
            tracing::info!(
                "SSM native NVFP4: loading pre-quantized QKV/Z/out_proj \
                 (skipping BF16 roundtrip)"
            );
        }
        let qkv_qw = quantized_auto(
            store,
            &format!("{la}.in_proj_qkv"),
            gpu,
            Nvfp4Variant::Standard,
        )?;
        let z_qw = quantized_auto(
            store,
            &format!("{la}.in_proj_z"),
            gpu,
            Nvfp4Variant::Standard,
        )?;
        let qkvz_nvfp4 = qkv_qw
            .concat_rows(&z_qw, qkv_rows, z_rows, h, gpu)
            .with_context(|| format!("concat_rows({la}.in_proj_qkv, {la}.in_proj_z)"))?;
        let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

        let out_proj_nvfp4 = quantized_auto(
            store,
            &format!("{la}.out_proj"),
            gpu,
            Nvfp4Variant::Standard,
        )?;
        let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

        let ssm = SsmWeights {
            in_proj_qkvz: DenseWeight {
                weight: spark_runtime::gpu::DevicePtr::NULL,
            },
            in_proj_ba: ba_dense,
            conv1d,
            a_log,
            dt_bias,
            norm,
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
    } else {
        // ── Legacy path: dequant to BF16 then re-quantize ──
        let qkv_dense = load_ssm_proj(&format!("{la}.in_proj_qkv"), qkv_rows, h)?;
        let z_dense = load_ssm_proj(&format!("{la}.in_proj_z"), z_rows, h)?;
        let out_proj_dense = load_ssm_proj(&format!("{la}.out_proj"), h, value_dim)?;

        let qkvz_dense = gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;
        gpu.free(qkv_dense.weight)?;
        gpu.free(z_dense.weight)?;

        let qkvz_nvfp4 =
            quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;
        let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

        let out_proj_nvfp4 = quantize_to_nvfp4(
            &out_proj_dense,
            h,
            value_dim,
            gpu,
            absmax_k,
            quantize_k,
            stream,
        )?;
        let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

        let (qkvz_fp8_prefill, out_proj_fp8_prefill) = if let Some(b2f_k) = kernels.bf16_to_fp8_k {
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
                out_proj_dense.weight,
                out_fp8,
                out_total,
                stream,
            )?;
            gpu.synchronize(stream)?;
            (Some(qkvz_fp8), Some(out_fp8))
        } else {
            (None, None)
        };

        gpu.free(qkvz_dense.weight)?;
        gpu.free(out_proj_dense.weight)?;

        let ssm = SsmWeights {
            in_proj_qkvz: DenseWeight {
                weight: spark_runtime::gpu::DevicePtr::NULL,
            },
            in_proj_ba: ba_dense,
            conv1d,
            a_log,
            dt_bias,
            norm,
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
        if qkvz_fp8_prefill.is_some() || out_proj_fp8_prefill.is_some() {
            layer.set_fp8_prefill_only_weights(qkvz_fp8_prefill, out_proj_fp8_prefill);
        }
        Ok(Box::new(layer))
    }
}
