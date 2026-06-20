// SPDX-License-Identifier: AGPL-3.0-only
//
// Helper for the non-FP8 FullAttention arm of `load_layers`. Handles the
// CompressedTensors NVFP4 / Standard / Fp8Dequanted / Bf16Raw variants
// — anything but the native-FP8 (block-scaled-on-disk) path which stays
// inline because it owns enough closures to fight extraction.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Nvfp4Variant, dense, dense_auto, load_kv_scales,
    quantize_to_nvfp4, quantized_auto,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn build_full_attention_nvfp4(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    layer_kv_dtype: KvCacheDtype,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let p = format!("{lp}.self_attn");
    let tp_rank = config.tp_rank;
    let tp_size = config.tp_world_size.max(1);
    let i = layer_idx;

    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
        Nvfp4Variant::CompressedTensors => {
            let group_size = 16usize;
            let load_nvfp4 = |name: &str,
                              full_n: usize,
                              full_k: usize,
                              kind: TpShardKind|
             -> Result<crate::weight_map::QuantizedWeight> {
                let src = quantized_auto(store, &format!("{p}.{name}"), gpu, variant)?;
                if tp_size == 1 {
                    return Ok(src);
                }
                let sharded = shard_quantized_nvfp4(
                    &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
                )?;
                gpu.free(src.weight)?;
                gpu.free(src.weight_scale)?;
                Ok(sharded)
            };
            let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
            let dummy = DenseWeight {
                weight: spark_runtime::gpu::DevicePtr::NULL,
            };
            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
            let attn = AttentionWeights {
                q_proj: dummy,
                k_proj: dummy,
                v_proj: dummy,
                o_proj: o,
                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };
            (attn, Some(q), Some(k), Some(v))
        }
        Nvfp4Variant::Standard | Nvfp4Variant::Fp8Dequanted | Nvfp4Variant::Bf16Raw => {
            tracing::info!("Layer {i}: loading attention projections ({variant:?})");
            let load_bf16_then_nvfp4 =
                |name: &str,
                 full_n: usize,
                 full_k: usize,
                 kind: TpShardKind|
                 -> Result<(DenseWeight, crate::weight_map::QuantizedWeight)> {
                    let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                    let (sharded_ptr, local_n, local_k) =
                        shard_dense_bf16(src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu)?;
                    let sharded = DenseWeight {
                        weight: sharded_ptr,
                    };
                    // TQ+ weight pre-rotation: apply canonical Rademacher rotation
                    // S2·H·S1 to Q/K/V columns per-head BEFORE quantization. When
                    // active (TQ_PLUS_WEIGHT_ROTATION=1) the runtime wht_bf16_inplace
                    // launches on Q/K/V become redundant. O projection skipped (the
                    // input-side rotation needs a transpose). hd=128 only — 256/512
                    // sign arrays not yet vendored.
                    if super::tq_plus_weight_rotation::weight_rotation_enabled()
                        && (name == "q_proj" || name == "k_proj" || name == "v_proj")
                        && config.head_dim == 128
                    {
                        let n_heads = local_n / config.head_dim;
                        if n_heads * config.head_dim == local_n && n_heads > 0 {
                            let _ =
                                super::tq_plus_weight_rotation::apply_canonical_rotation_inplace(
                                    gpu,
                                    sharded_ptr,
                                    local_k,
                                    n_heads,
                                    config.head_dim,
                                    stream,
                                );
                            tracing::info!(
                                "TQ+ weight rotation applied: {name} ({n_heads}h × {}hd)",
                                config.head_dim
                            );
                        }
                    }
                    let q = quantize_to_nvfp4(
                        &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                    )?;
                    if sharded_ptr != src.weight {
                        gpu.free(sharded_ptr)?;
                    }
                    Ok((src, q))
                };
            tracing::info!("Layer {i}: BF16 → NVFP4 (TP-aware)");
            let [
                (q_dense, q_nvfp4),
                (k_dense, k_nvfp4),
                (v_dense, v_nvfp4),
                (_o_dense, o_nvfp4),
            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;
            tracing::info!(
                "Layer {i}: Q/K/V/O quantized, {:.1} GB free",
                gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

            let attn = AttentionWeights {
                q_proj: q_dense,
                k_proj: k_dense,
                v_proj: v_dense,
                o_proj: o_nvfp4,
                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };
            (attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4))
        }
    };

    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        q_nvfp4,
        k_nvfp4,
        v_nvfp4,
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;

    let num_heads = config.num_attention_heads;
    let num_kv_heads = config.num_key_value_heads;
    let head_dim = config.head_dim;
    let gated = config.attn_gated;
    let q_proj_n = if gated {
        num_heads * head_dim * 2
    } else {
        num_heads * head_dim
    };
    if let Some(ref qw) = q_nvfp4 {
        let qt = qw.transpose_for_gemm(gpu, q_proj_n, h)?;
        let kt = k_nvfp4
            .as_ref()
            .unwrap()
            .transpose_for_gemm(gpu, num_kv_heads * head_dim, h)?;
        let vt = v_nvfp4
            .as_ref()
            .unwrap()
            .transpose_for_gemm(gpu, num_kv_heads * head_dim, h)?;
        let ot = layer
            .attn
            .o_proj
            .transpose_for_gemm(gpu, h, num_heads * head_dim)?;
        layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
    }
    layer.predequant_for_prefill(gpu, config, stream)?;

    Ok(Box::new(layer))
}
