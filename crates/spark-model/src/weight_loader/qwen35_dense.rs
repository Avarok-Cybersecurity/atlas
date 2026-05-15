// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, Qwen3AttentionLayer, Qwen3SsmLayer};
use crate::tp_shard::{
    TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_fp8_block_scaled, shard_quantized_nvfp4,
};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, MtpWeights, Nvfp4Variant, QuantizedWeight,
    SsmWeights, dense, dense_auto, dequant_nvfp4_to_bf16, detect_nvfp4_variant, gpu_concat_rows,
    interleave_ba, load_dense_ffn, load_dense_ffn_fp8, load_fp8_block_scaled_as_fp8weight,
    load_kv_scales, load_mtp, load_ssm_qwen35, quantize_to_nvfp4, quantized_auto,
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
        // Route native-FP8 checkpoints (e.g. Qwen3.6-27B-FP8) through the
        // FP8 attention + dense-FFN kernels instead of the default
        // FP8→BF16→NVFP4 re-quantization. The 4-bit downgrade was the
        // root cause of the documented "prose-attractor" failure mode
        // on dense Qwen 3.6 27B (see
        // `kernels/gb10/qwen3.6-27b/MODEL.toml:103-108`). Detection
        // mirrors `qwen35/load_layers.rs:69-80`.
        //
        // `ATLAS_DENSE_NATIVE_FP8=0` forces the legacy
        // FP8→BF16→NVFP4 path — for A/B-testing this loader against
        // the prior behaviour without rebuilding.
        let env_native_fp8 = std::env::var("ATLAS_DENSE_NATIVE_FP8")
            .ok()
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);
        let native_fp8 = env_native_fp8 && variant == Nvfp4Variant::Fp8Dequanted;
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}, native_fp8: {}",
            weight_format,
            variant,
            native_fp8,
        );

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // Dense FFN. Native FP8 path keeps the checkpoint's block-scaled
            // FP8 weights (1 byte/weight + BF16 block scales) and installs
            // them via `set_fp8_weights`. Default path runs the existing
            // FP8 → BF16 → NVFP4 quantization through `load_dense_ffn`.
            let ffn = {
                let mut dense_layer = if native_fp8 {
                    // Dummy NVFP4 slots — never read because `forward` /
                    // `forward_prefill` short-circuit when `fp8_weights`
                    // is Some. Mirrors the attention dummy pattern at
                    // `qwen35/load_layers.rs:251-266`.
                    let dummy = crate::layers::dense_ffn::DenseFfnWeights {
                        gate_proj: QuantizedWeight::null(),
                        up_proj: QuantizedWeight::null(),
                        down_proj: QuantizedWeight::null(),
                    };
                    DenseFfnLayer::new(dummy, gpu)?
                } else {
                    let ffn_weights = load_dense_ffn(
                        store, &lp, gpu, variant, absmax_k, quantize_k, stream, config,
                    )?;
                    DenseFfnLayer::new(ffn_weights, gpu)?
                };
                if native_fp8 {
                    let fp8w = load_dense_ffn_fp8(store, &lp, gpu)?;
                    dense_layer
                        .set_fp8_weights(fp8w.gate_proj, fp8w.up_proj, fp8w.down_proj)?;
                    tracing::info!(
                        "Layer {i}: dense FFN loaded as native FP8 (gate/up/down block-scaled)"
                    );
                }
                FfnComponent::Dense(dense_layer)
            };

            match lt {
                LayerType::FullAttention if native_fp8 => {
                    // Native FP8 attention: zero-copy load of block-scaled
                    // FP8 Q/K/V/O from the checkpoint, no dequant to BF16
                    // and no re-quant to NVFP4. Mirrors the MoE FP8 path
                    // at `qwen35/load_layers.rs:213-298`.
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let block_size = 128usize;
                    let load_fp8_proj = |name: &str,
                                         _full_n: usize,
                                         _full_k: usize,
                                         kind: TpShardKind|
                     -> Result<Fp8Weight> {
                        let src = load_fp8_block_scaled_as_fp8weight(
                            store,
                            &format!("{p}.{name}"),
                            gpu,
                        )?;
                        if tp_size == 1 {
                            return Ok(src);
                        }
                        let sharded =
                            shard_fp8_block_scaled(&src, kind, tp_rank, tp_size, block_size, gpu)?;
                        gpu.free(src.weight)?;
                        gpu.free(src.row_scale)?;
                        Ok(sharded)
                    };
                    let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;

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
                        None,
                        gpu,
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));
                    if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
                        tracing::warn!(
                            "Layer {i}: FP8 transpose failed, using non-transposed prefill: {e}"
                        );
                    } else {
                        tracing::info!("Layer {i}: FP8 attention transposed for fast prefill");
                    }
                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
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
                        layer_kv_dtypes[attn_idx],
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;

                    // Wire prefill weights: transpose Q/K/V/O for the fast
                    // `w4a16_gemm_t` kernel and pre-dequant NVFP4→FP8 for
                    // the FP8 prefill path. Same setup the MoE loader
                    // performs at `qwen35/load_layers/attention_arms.rs:153-175`.
                    // Without these, prefill falls through to the
                    // non-transposed `w4a16_gemm` fallback (numerically
                    // correct but slower). The `q_proj_n` accounting
                    // doubles for `attn_output_gate=true` checkpoints
                    // (Qwen3.6 family) — see MODEL.toml:25.
                    let num_heads = config.num_attention_heads;
                    let num_kv_heads = config.num_key_value_heads;
                    let head_dim = config.head_dim;
                    let gated = config.attn_gated;
                    let q_proj_n = if gated {
                        num_heads * head_dim * 2
                    } else {
                        num_heads * head_dim
                    };
                    if let (Some(qw), Some(kw), Some(vw)) =
                        (q_nvfp4.as_ref(), k_nvfp4.as_ref(), v_nvfp4.as_ref())
                    {
                        let qt = qw.transpose_for_gemm(gpu, q_proj_n, h)?;
                        let kt = kw.transpose_for_gemm(gpu, num_kv_heads * head_dim, h)?;
                        let vt = vw.transpose_for_gemm(gpu, num_kv_heads * head_dim, h)?;
                        let ot = layer.attn.o_proj.transpose_for_gemm(
                            gpu,
                            h,
                            num_heads * head_dim,
                        )?;
                        layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                    }
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    layers.push(Box::new(layer));
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

            crate::loader_progress::inc();
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
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading dense MTP weights (variant={:?}, hidden={}, inter={})",
            variant,
            config.hidden_size,
            config.intermediate_size,
        );
        // `load_mtp` auto-detects MoE vs dense FFN by inspecting the weight
        // names. For dense Qwen3.6-27B-FP8 it returns a MtpWeights with
        // `dense_ffn = Some(...)` and NULL placeholders for the MoE fields.
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        if mtp.dense_ffn.is_some() {
            tracing::info!("Dense MTP head ready (FP8 e4m3 projections + dense gate/up/down MLP)");
        } else {
            tracing::info!(
                "MoE MTP head ready ({} experts) — dense loader sees MoE bundle",
                mtp.experts.len(),
            );
        }
        Ok(Some(mtp))
    }
}
