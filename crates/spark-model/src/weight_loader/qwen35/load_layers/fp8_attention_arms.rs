// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the native-FP8 FullAttention arms of `load_layers`. Two
// flavours: the BF16-dequant diagnostic path (TP=1, dense GEMM) and the
// native block-scaled FP8 path (w8a16_gemv decode + w8a16_gemm prefill).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, Qwen3AttentionLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_fp8_block_scaled};
use crate::weight_map::{
    AttentionWeights, DenseWeight, QuantizedWeight, dense, load_fp8_block_scaled_as_fp8weight,
    load_kv_scales,
};

/// BF16-dequant attention (diagnostic, TP=1). Dequant FP8 Q/K/V/O → BF16 on
/// GPU, store as dense weights, and leave q/k/v/o quant-weights None so both
/// prefill and decode fall through to the dense GEMM/GEMV paths.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_full_attention_fp8_bf16_dequant(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    layer_kv_dtype: KvCacheDtype,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let i = layer_idx;
    use crate::weight_map::dequant_fp8_blockscaled_to_bf16;
    if config.tp_world_size.max(1) != 1 {
        anyhow::bail!(
            "ATLAS_FP8_DEQUANT_ATTN_TO_BF16 supports TP=1 only (got tp={})",
            config.tp_world_size,
        );
    }
    let p = format!("{lp}.self_attn");
    tracing::info!("Layer {i}: dequanting attention Q/K/V/O FP8→BF16 (dense)");
    let q_bf16 = dequant_fp8_blockscaled_to_bf16(store, &format!("{p}.q_proj"), gpu)?;
    let k_bf16 = dequant_fp8_blockscaled_to_bf16(store, &format!("{p}.k_proj"), gpu)?;
    let v_bf16 = dequant_fp8_blockscaled_to_bf16(store, &format!("{p}.v_proj"), gpu)?;
    let o_bf16 = dequant_fp8_blockscaled_to_bf16(store, &format!("{p}.o_proj"), gpu)?;

    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let dummy_qw = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: q_bf16,
        k_proj: k_bf16,
        v_proj: v_bf16,
        o_proj: dummy_qw,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    // O-proj BF16 dense (decode + prefill both check this first).
    layer.set_o_dense_bf16(o_bf16);
    // Leave q/k/v/o quant-weights unset → dense fallback fires.
    Ok(Box::new(layer))
}

/// Native FP8 path: FP8 for both decode AND prefill. NO NVFP4 dequant —
/// saves ~30 GB peak memory on 122B EP=2. Decode uses w8a16_gemv, prefill
/// uses w8a16_gemm (both with E4M3 LUT + BF16 2D block scales from checkpoint).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_full_attention_fp8_native(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    stream: u64,
    layer_kv_dtype: KvCacheDtype,
    attn_idx: usize,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let i = layer_idx;
    let p = format!("{lp}.self_attn");
    tracing::info!("Layer {i}: loading attention FP8 native (zero-copy)");

    // FP8 block-scaled QKVO: column-parallel Q/K/V, row-parallel O.
    // Block size is 128 for Qwen3.5 native FP8 checkpoints.
    let tp_rank = config.tp_rank;
    let tp_size = config.tp_world_size.max(1);
    let block_size = 128usize;
    let load_fp8_proj = |name: &str,
                         _full_n: usize,
                         _full_k: usize,
                         kind: TpShardKind|
     -> Result<crate::weight_map::Fp8Weight> {
        let src = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)?;
        if tp_size == 1 {
            return Ok(src);
        }
        let sharded = shard_fp8_block_scaled(&src, kind, tp_rank, tp_size, block_size, gpu)?;
        gpu.free(src.weight)?;
        gpu.free(src.row_scale)?;
        Ok(sharded)
    };
    let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
    tracing::info!(
        "Layer {i}: FP8 Q/K/V/O loaded, {:.1} GB free",
        gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // O proj needs a QuantizedWeight placeholder for the AttentionWeights struct.
    // Use a dummy — the actual O proj uses o_fp8w via w8a16_gemv/gemm.
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let dummy = DenseWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
    };
    let dummy_qw = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: dummy,
        k_proj: dummy,
        v_proj: dummy,
        o_proj: dummy_qw,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };

    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None, // No NVFP4 — w8a16_gemm handles prefill
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;

    // Set checkpoint FP8 weights for decode (w8a16_gemv) and prefill fallback (w8a16_gemm).
    layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));

    // Transpose FP8 weights for fast prefill (w8a16_gemm_t: coalesced reads).
    // This allocates N*K bytes per projection but gives ~14x prefill speedup.
    if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
        tracing::warn!("Layer {i}: FP8 transpose failed, using non-transposed prefill: {e}");
    } else {
        tracing::info!("Layer {i}: FP8 weights transposed for fast prefill");
    }

    Ok(Box::new(layer))
}
