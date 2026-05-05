// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, Nvfp4Variant, SsmWeights, dense, dense_auto,
    dequant_nvfp4_to_bf16, detect_nvfp4_variant, gpu_concat_rows, interleave_ba, load_dense_ffn,
    load_kv_scales, load_ssm_qwen35, quantize_to_nvfp4, quantized_auto,
};

pub struct Qwen35DenseWeightLoader;

impl ModelWeightLoader for Qwen35DenseWeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers are TP-sharded (NVFP4-from-disk and BF16
        // → NVFP4 paths). LinearAttention (GDN SSM) layers run
        // full-replica per rank — see qwen35.rs for the rationale.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect::<Vec<_>>()
        } else {
            config.layer_types.clone()
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // Dense FFN instead of MoE
            let ffn_weights = load_dense_ffn(
                store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
            )?;
            let ffn = FfnComponent::Dense(DenseFfnLayer::new(ffn_weights, gpu)?);

            match lt {
                LayerType::FullAttention => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let (attn, q_nvfp4, k_nvfp4, v_nvfp4) = match variant {
                        Nvfp4Variant::CompressedTensors => {
                            // NVFP4-from-disk path: column-parallel Q/K/V, row-parallel O.
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
                        Nvfp4Variant::Standard
                        | Nvfp4Variant::Fp8Dequanted
                        | Nvfp4Variant::Bf16Raw => {
                            // BF16 → NVFP4 path: shard BF16 then quantize per-rank.
                            let load_bf16_then_nvfp4 = |name: &str,
                                                        full_n: usize,
                                                        full_k: usize,
                                                        kind: TpShardKind|
                             -> Result<(
                                DenseWeight,
                                crate::weight_map::QuantizedWeight,
                            )> {
                                let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
                                let (sharded_ptr, local_n, local_k) = shard_dense_bf16(
                                    src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu,
                                )?;
                                let sharded = DenseWeight {
                                    weight: sharded_ptr,
                                };
                                let q = quantize_to_nvfp4(
                                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                                )?;
                                if sharded_ptr != src.weight {
                                    gpu.free(sharded_ptr)?;
                                }
                                Ok((src, q))
                            };
                            let [
                                (q_dense, q_nvfp4),
                                (k_dense, k_nvfp4),
                                (v_dense, v_nvfp4),
                                (_o_dense, o_nvfp4),
                            ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;

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

                    layers.push(Box::new(Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        q_nvfp4,
                        k_nvfp4,
                        v_nvfp4,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?));
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let nv = config.linear_num_value_heads;
                    let nk = config.linear_num_key_heads;
                    let qkv_rows = config.ssm_qkv_size();
                    let z_rows = config.ssm_z_size();
                    let value_dim = nv * config.linear_value_head_dim;
                    let la = format!("{lp}.linear_attn");

                    // SSM projections may be BF16 or NVFP4 depending on quantizer.
                    // If NVFP4 (weight_packed exists), dequant to BF16 for concat pipeline.
                    let ssm_quantized = store.contains(&format!("{la}.in_proj_qkv.weight_packed"));

                    let (qkv_dense, z_dense, out_proj_dense) = if ssm_quantized {
                        let qkv = dequant_nvfp4_to_bf16(
                            store,
                            &format!("{la}.in_proj_qkv"),
                            qkv_rows,
                            h,
                            gpu,
                        )?;
                        let z = dequant_nvfp4_to_bf16(
                            store,
                            &format!("{la}.in_proj_z"),
                            z_rows,
                            h,
                            gpu,
                        )?;
                        let out = dequant_nvfp4_to_bf16(
                            store,
                            &format!("{la}.out_proj"),
                            h,
                            value_dim,
                            gpu,
                        )?;
                        (qkv, z, out)
                    } else {
                        let ssm35 = load_ssm_qwen35(store, &lp, gpu, variant)?;
                        (ssm35.in_proj_qkv, ssm35.in_proj_z, ssm35.out_proj)
                    };

                    // A, B are always BF16
                    let in_proj_a = dense(store, &format!("{la}.in_proj_a.weight"))?;
                    let in_proj_b = dense(store, &format!("{la}.in_proj_b.weight"))?;
                    let conv1d = dense(store, &format!("{la}.conv1d.weight"))?;
                    let a_log = dense(store, &format!("{la}.A_log"))?;
                    let dt_bias = dense(store, &format!("{la}.dt_bias"))?;
                    let norm = dense(store, &format!("{la}.norm.weight"))?;

                    let qkvz_dense =
                        gpu_concat_rows(&qkv_dense, qkv_rows, &z_dense, z_rows, h, gpu)?;

                    let ba_dense = interleave_ba(&in_proj_a, &in_proj_b, nv, nk, h, gpu)?;

                    let qkvz_size = config.ssm_qkvz_size();
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &qkvz_dense,
                        qkvz_size,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;

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

                    let ssm = SsmWeights {
                        in_proj_qkvz: qkvz_dense,
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
                    layers.push(Box::new(layer));
                }
                LayerType::Moe => unreachable!("Qwen3.5 dense has no standalone MoE layers"),
            }

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Qwen3.5 dense weight loader: {} layers ({} attention, {} SSM, dense FFN)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        for pattern in &[
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ] {
            if store.contains(pattern) {
                return dense(store, pattern);
            }
        }
        self.load_embedding(store, config)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
