// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

mod ssm_layer;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, NemotronMoeLayer, Qwen3AttentionLayer};
use crate::tp_shard::{TpAttentionDims, TpShardKind, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    DenseWeight, MtpWeights, dense, load_nemotron_attention, load_nemotron_moe, quantize_to_nvfp4,
};

pub struct NemotronHWeightLoader;

impl ModelWeightLoader for NemotronHWeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers TP-sharded across both quant paths
        // (NVFP4-from-disk and BF16/FP8 → NVFP4). LinearAttention
        // (Mamba-2 SSM) and MoE layers run full-replica per rank —
        // SSM stays correct because hidden in/out is the same on
        // every rank; MoE under EP+TP composition only uses EP.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = &config.layer_types;
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;
        let h = config.hidden_size;

        // Runtime quantization kernels for BF16→NVFP4 conversion of unquantized layers.
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        // Pre-allocate a reusable scratch buffer for FP8→BF16 dequant intermediates.
        // On GB10 UVM, gpu.free() posts in-band TLB invalidations that corrupt
        // nearby allocations (BUG #29). Using a scratch buffer avoids all frees
        // during loading. Size = max(in_proj, out_proj, shared_up, shared_down) in BF16 bytes.
        let moe_input = config.moe_input_size();
        // Puzzle: intermediate size varies per MoE layer — size scratch to max.
        let max_moe_inter = config.max_moe_intermediate_size();
        let scratch_elems = (config.mamba2_in_proj_size() * h)
            .max(h * config.mamba2_d_inner())
            .max(config.shared_expert_intermediate_size * h)
            .max(h * config.shared_expert_intermediate_size)
            .max(max_moe_inter * moe_input)
            .max(moe_input * max_moe_inter);
        let scratch_bytes = scratch_elems * 2; // BF16 = 2 bytes
        let scratch = gpu.alloc(scratch_bytes)?;

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let norm = dense(store, &format!("{lp}.norm.weight"))?;

            match lt {
                atlas_core::config::LayerType::LinearAttention => {
                    let layer = Self::build_ssm_layer(
                        gpu, store, config, i, h, &lp, norm, quantize_k, absmax_k, scratch, stream,
                    )?;
                    layers.push(Box::new(layer));
                }
                atlas_core::config::LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                atlas_core::config::LayerType::Moe => {
                    // Standalone MoE FFN layer (uniform Super/Nano or Puzzle per-block)
                    let moe_inter = config.moe_intermediate_size_for(i);
                    let top_k = config.num_experts_per_tok_for(i);
                    let moe = load_nemotron_moe(
                        store,
                        i,
                        config.num_experts,
                        gpu,
                        config,
                        Some(absmax_k),
                        Some(quantize_k),
                        stream,
                        Some(scratch),
                        &lp,
                    )?;
                    if i < 4 || moe_inter != config.moe_intermediate_size {
                        tracing::info!(
                            "L{i} MoE: inter={moe_inter} top_k={top_k} latent={} has_fc1={} has_fc2={} shared_up_s2={:.6e} experts[0].up_s2={:.6e}",
                            config.moe_latent_size,
                            moe.fc1_latent_proj.is_some(),
                            moe.fc2_latent_proj.is_some(),
                            moe.shared_up.weight_scale_2,
                            moe.experts
                                .first()
                                .map(|e| e.up_proj.weight_scale_2)
                                .unwrap_or(0.0),
                        );
                    }
                    let mut moe_layer =
                        NemotronMoeLayer::new(moe, norm, config, gpu, moe_inter, top_k)?;
                    // Builds the transposed shared-expert weights (2 small matrices
                    // per layer) so prefill can use `w4a16_gemm_t` instead of the base
                    // `w4a16_gemm`. Routed-expert transposition stays disabled inside
                    // for LatentMoE — 512 experts x 40 layers would not fit.
                    moe_layer.prepare_prefill_weights(gpu, config);
                    layers.push(Box::new(moe_layer));
                }
                atlas_core::config::LayerType::FullAttention => {
                    // Attention layer — quantize BF16 Q/K/V/O directly from
                    // WeightStore pointers (no intermediate alloc/free needed).
                    let (mut attn, mut q_nvfp4, mut k_nvfp4, mut v_nvfp4, mut o_dense, is_nvfp4) =
                        load_nemotron_attention(store, i, gpu, &lp)?;
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let dims = TpAttentionDims::from_config(config);
                    if is_nvfp4 && tp_size > 1 {
                        // NVFP4-from-disk: shard packed weight + FP8 scales.
                        let group_size = 16usize;
                        if let Some(q) = q_nvfp4.as_ref() {
                            let s = shard_quantized_nvfp4(
                                q,
                                dims.full_q_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                group_size,
                                gpu,
                            )?;
                            gpu.free(q.weight)?;
                            gpu.free(q.weight_scale)?;
                            q_nvfp4 = Some(s);
                        }
                        if let Some(k) = k_nvfp4.as_ref() {
                            let s = shard_quantized_nvfp4(
                                k,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                group_size,
                                gpu,
                            )?;
                            gpu.free(k.weight)?;
                            gpu.free(k.weight_scale)?;
                            k_nvfp4 = Some(s);
                        }
                        if let Some(v) = v_nvfp4.as_ref() {
                            let s = shard_quantized_nvfp4(
                                v,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                group_size,
                                gpu,
                            )?;
                            gpu.free(v.weight)?;
                            gpu.free(v.weight_scale)?;
                            v_nvfp4 = Some(s);
                        }
                        // O proj is stored on attn.o_proj as QuantizedWeight in NVFP4-disk path.
                        let o_old = attn.o_proj;
                        let o_sharded = shard_quantized_nvfp4(
                            &o_old,
                            dims.h,
                            dims.full_o_in,
                            TpShardKind::RowParallel,
                            tp_rank,
                            tp_size,
                            group_size,
                            gpu,
                        )?;
                        gpu.free(o_old.weight)?;
                        gpu.free(o_old.weight_scale)?;
                        attn.o_proj = o_sharded;
                    }
                    let (q_nv, k_nv, v_nv) = if is_nvfp4 {
                        (q_nvfp4, k_nvfp4, v_nvfp4)
                    } else {
                        let num_heads = config.num_attention_heads;
                        let kv_heads = config.num_key_value_heads;
                        let hd = config.head_dim;
                        // BF16 / FP8-dequant fallback: shard the dense BF16
                        // before quantization. Dims here are TP-LOCAL after
                        // sharding (config head counts already TP-divided).
                        if tp_size > 1 {
                            let (qp, _, _) = shard_dense_bf16(
                                attn.q_proj.weight,
                                dims.full_q_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if qp != attn.q_proj.weight {
                                gpu.free(attn.q_proj.weight)?;
                            }
                            attn.q_proj.weight = qp;
                            let (kp, _, _) = shard_dense_bf16(
                                attn.k_proj.weight,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if kp != attn.k_proj.weight {
                                gpu.free(attn.k_proj.weight)?;
                            }
                            attn.k_proj.weight = kp;
                            let (vp, _, _) = shard_dense_bf16(
                                attn.v_proj.weight,
                                dims.full_kv_n,
                                dims.h,
                                TpShardKind::ColumnParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if vp != attn.v_proj.weight {
                                gpu.free(attn.v_proj.weight)?;
                            }
                            attn.v_proj.weight = vp;
                            let (op, _, _) = shard_dense_bf16(
                                o_dense.weight,
                                dims.h,
                                dims.full_o_in,
                                TpShardKind::RowParallel,
                                tp_rank,
                                tp_size,
                                gpu,
                            )?;
                            if op != o_dense.weight {
                                gpu.free(o_dense.weight)?;
                            }
                            o_dense.weight = op;
                        }
                        let q = quantize_to_nvfp4(
                            &attn.q_proj,
                            num_heads * hd,
                            h,
                            gpu,
                            absmax_k,
                            quantize_k,
                            stream,
                        )?;
                        let k = quantize_to_nvfp4(
                            &attn.k_proj,
                            kv_heads * hd,
                            h,
                            gpu,
                            absmax_k,
                            quantize_k,
                            stream,
                        )?;
                        let v = quantize_to_nvfp4(
                            &attn.v_proj,
                            kv_heads * hd,
                            h,
                            gpu,
                            absmax_k,
                            quantize_k,
                            stream,
                        )?;
                        let o = quantize_to_nvfp4(
                            &o_dense,
                            h,
                            num_heads * hd,
                            gpu,
                            absmax_k,
                            quantize_k,
                            stream,
                        )?;
                        attn.o_proj = o;
                        (Some(q), Some(k), Some(v))
                    };
                    // Transposed Q/K/V/O so prefill uses `w4a16_gemm_t` (FP8 MMA,
                    // N128/K32, cp.async) instead of the base `w4a16_gemm`. Same
                    // gap as the SSM/MoE layers: the setter existed but was never
                    // called for Nemotron. Q/K/V/O are small (h=4096, kv=2 heads),
                    // so the extra copies cost ~0.3 GB.
                    let q_dim = config.num_attention_heads * config.head_dim;
                    let kv_dim = config.num_key_value_heads * config.head_dim;
                    let qt = q_nv
                        .as_ref()
                        .and_then(|w| w.transpose_for_gemm(gpu, q_dim, h).ok());
                    let kt = k_nv
                        .as_ref()
                        .and_then(|w| w.transpose_for_gemm(gpu, kv_dim, h).ok());
                    let vt = v_nv
                        .as_ref()
                        .and_then(|w| w.transpose_for_gemm(gpu, kv_dim, h).ok());
                    let ot = attn.o_proj.transpose_for_gemm(gpu, h, q_dim).ok();

                    let mut attn_layer = Qwen3AttentionLayer::new_ungated(
                        norm,
                        attn,
                        DenseWeight {
                            weight: spark_runtime::gpu::DevicePtr::NULL,
                        },
                        FfnComponent::None,
                        attn_idx,
                        q_nv,
                        k_nv,
                        v_nv,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    attn_layer.set_prefill_weights(qt, kt, vt, ot);
                    layers.push(Box::new(attn_layer));
                    attn_idx += 1;
                }
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Nemotron-H weight loader: {} layers ({} SSM, {} MoE, {} attention)",
            layers.len(),
            config.num_ssm_layers(),
            config.num_moe_layers(),
            attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(
            store,
            &format!("{}.embeddings.weight", config.weight_prefix),
        )
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, &format!("{}.norm_f.weight", config.weight_prefix))
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            dense(
                store,
                &format!("{}.embeddings.weight", config.weight_prefix),
            )
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None) // Nemotron-H has no MTP
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nemotron_h_loader_exists() {
        let _loader = NemotronHWeightLoader;
    }
}
