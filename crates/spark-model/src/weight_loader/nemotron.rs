// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, NemotronMamba2Layer, NemotronMoeLayer, Qwen3AttentionLayer};
use crate::tp_shard::{TpAttentionDims, TpShardKind, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    DenseWeight, MtpWeights, NemotronSsmQuant, dense, dequant_fp8_to_bf16_into,
    load_nemotron_attention, load_nemotron_moe, load_nemotron_ssm, quantize_to_nvfp4,
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
        let scratch_elems = (config.mamba2_in_proj_size() * h)
            .max(h * config.mamba2_d_inner())
            .max(config.shared_expert_intermediate_size * h)
            .max(h * config.shared_expert_intermediate_size)
            .max(config.moe_intermediate_size * moe_input)
            .max(moe_input * config.moe_intermediate_size);
        let scratch_bytes = scratch_elems * 2; // BF16 = 2 bytes
        let scratch = gpu.alloc(scratch_bytes)?;

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let norm = dense(store, &format!("{lp}.norm.weight"))?;

            match lt {
                atlas_core::config::LayerType::LinearAttention => {
                    // Mamba-2 SSM layer (mixed quant: NVFP4, FP8, or BF16)
                    let (mut ssm, quant_kind) = load_nemotron_ssm(store, i, gpu, &lp)?;
                    tracing::info!(
                        "L{i} SSM quant={quant_kind:?} in_proj_size={} d_inner={} h={h}",
                        config.mamba2_in_proj_size(),
                        config.mamba2_d_inner(),
                    );
                    // TODO: Fix 3 — FP8 direct load causes CUDA 700 (illegal address).
                    // The WeightStore mmap pointers may be invalidated after loading.
                    // For now, keep the double-quant path (FP8→BF16→NVFP4).
                    if quant_kind != NemotronSsmQuant::Nvfp4 {
                        let p = format!("{lp}.mixer");
                        let in_proj_dense = if quant_kind == NemotronSsmQuant::Fp8 {
                            dequant_fp8_to_bf16_into(store, &format!("{p}.in_proj"), gpu, scratch)?
                        } else {
                            dense(store, &format!("{p}.in_proj.weight"))?
                        };
                        ssm.in_proj = quantize_to_nvfp4(
                            &in_proj_dense,
                            config.mamba2_in_proj_size(),
                            h,
                            gpu,
                            absmax_k,
                            quantize_k,
                            stream,
                        )?;
                        let out_fp8 = store.contains(&format!("{p}.out_proj.weight_scale"));
                        let out_proj_dense = if out_fp8 {
                            dequant_fp8_to_bf16_into(store, &format!("{p}.out_proj"), gpu, scratch)?
                        } else {
                            dense(store, &format!("{p}.out_proj.weight"))?
                        };
                        ssm.out_proj = quantize_to_nvfp4(
                            &out_proj_dense,
                            h,
                            config.mamba2_d_inner(),
                            gpu,
                            absmax_k,
                            quantize_k,
                            stream,
                        )?;
                    }
                    layers.push(Box::new(NemotronMamba2Layer::new(
                        norm, ssm, config, gpu, i,
                    )?));
                }
                atlas_core::config::LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                atlas_core::config::LayerType::Moe => {
                    // Standalone MoE FFN layer
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
                    if i < 4 {
                        tracing::info!(
                            "L{i} MoE: latent={} has_fc1={} has_fc2={} shared_up_s2={:.6e} shared_down_s2={:.6e} experts[0].up_s2={:.6e}",
                            config.moe_latent_size,
                            moe.fc1_latent_proj.is_some(),
                            moe.fc2_latent_proj.is_some(),
                            moe.shared_up.weight_scale_2,
                            moe.shared_down.weight_scale_2,
                            moe.experts
                                .first()
                                .map(|e| e.up_proj.weight_scale_2)
                                .unwrap_or(0.0),
                        );
                    }
                    layers.push(Box::new(NemotronMoeLayer::new(moe, norm, config, gpu)?));
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
                    layers.push(Box::new(Qwen3AttentionLayer::new_ungated(
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
                    )?));
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

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
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

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            self.load_embedding(store, config)
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
