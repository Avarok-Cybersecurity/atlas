// SPDX-License-Identifier: AGPL-3.0-only

//! Step 3.7 Flash weight loader.
//!
//! Hybrid of MiniMax M2 and Qwen 3.5 patterns:
//!   * Sigmoid MoE routing + correction bias (MiniMax M2 pattern)
//!   * Shared expert per MoE layer (Qwen 3.5 pattern)
//!   * Attention gate g_proj (Qwen 3.5 pattern)
//!   * Partial RoPE 0.5 (MiniMax M2 pattern)
//!   * Per-head q_norm / k_norm
//!   * Mixed dense FFN (layers 0-2) + MoE (layers 3-44)
//!   * 3 MTP modules at layers 45-47 (different prefix: `model.layers.`)
//!
//! Weight prefix: `model.language_model.layers.{i}` for main layers.
//! MTP prefix: `model.layers.{45|46|47}` (different namespace!).
//!
//! KEY ARCHITECTURAL DIFFERENCE: Step 3.7 stores expert weights as FUSED
//! tensors — one tensor per projection type containing ALL 288 experts
//! concatenated. Atlas needs per-expert QuantizedWeight entries, so we
//! slice by computing byte offsets into the fused GPU allocations.
//!
//! NVFP4 format: ModelOpt style with `weight`, `weight_scale`, `weight_scale_2`,
//! `input_scale` per projection. Shared expert is BF16.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, MtpWeights, QuantizedWeight,
    dense, dense_auto, detect_nvfp4_variant, load_kv_scales, quantize_to_nvfp4,
};

pub struct Step3p7WeightLoader;

/// Slice a fused NVFP4 tensor into per-expert QuantizedWeight entries.
///
/// Step 3.7 stores all experts in one contiguous tensor per projection:
///   weight: [num_experts * n, k] packed NVFP4 (0.5 bytes/element)
///   weight_scale: [num_experts * n, k/group_size] FP8 per-group scales
///   input_scale: [num_experts * n] (optional, activation quantization)
///
/// This function creates `num_experts` QuantizedWeight entries, each
/// pointing to a different offset within the fused allocations.
fn slice_fused_experts(
    fused_weight: DevicePtr,
    fused_scale: DevicePtr,
    fused_input_scale: DevicePtr,
    global_scale_2: f32,
    num_experts: usize,
    n: usize,  // rows per expert (e.g. moe_intermediate for gate/up, hidden for down)
    k: usize,  // cols per expert (e.g. hidden for gate/up, moe_intermediate for down)
) -> Vec<QuantizedWeight> {
    let group_size = 16usize;
    // NVFP4 packed: 2 elements per byte → n*k/2 bytes per expert
    let packed_bytes_per_expert = n * k / 2;
    // Per-group FP8 scales: 1 byte per group → n * ceil(k/group_size) bytes per expert
    let scale_bytes_per_expert = n * k.div_ceil(group_size);
    // Input scale: 1 float per row → n * 4 bytes per expert (if present)
    let input_scale_bytes_per_expert = n * 4;

    (0..num_experts)
        .map(|e| QuantizedWeight {
            weight: fused_weight.offset(e * packed_bytes_per_expert),
            weight_scale: fused_scale.offset(e * scale_bytes_per_expert),
            weight_scale_2: global_scale_2,
            input_scale: if fused_input_scale == DevicePtr::NULL {
                DevicePtr::NULL
            } else {
                fused_input_scale.offset(e * input_scale_bytes_per_expert)
            },
        })
        .collect()
}

/// Load a fused NVFP4 tensor from the store (Standard ModelOpt format).
fn load_fused_nvfp4(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, f32)> {
    let weight = store.get(&format!("{prefix}.weight"))?.ptr;
    let weight_scale = store.get(&format!("{prefix}.weight_scale"))?.ptr;

    // weight_scale_2 is a scalar F32 — copy from GPU to CPU
    let ws2_key = format!("{prefix}.weight_scale_2");
    let ws2_ptr = store.get(&ws2_key)?.ptr;
    let mut ws2_buf = [0u8; 4];
    gpu.copy_d2h(ws2_ptr, &mut ws2_buf)?;
    let weight_scale_2 = f32::from_le_bytes(ws2_buf);

    // input_scale is optional
    let is_key = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&is_key) {
        store.get(&is_key)?.ptr
    } else {
        DevicePtr::NULL
    };

    Ok((weight, weight_scale, input_scale, weight_scale_2))
}

impl ModelWeightLoader for Step3p7WeightLoader {
    fn supports_tp(&self) -> bool {
        false // Single-GPU initial bring-up
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;
        let variant = detect_nvfp4_variant(store, config);
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        tracing::info!(
            "step3p7: loading {} layers, variant={:?}, hidden_size={h}, \
             experts={}, moe_inter={inter}, shared_inter={shared_inter}",
            config.num_hidden_layers,
            variant,
            config.num_experts,
        );

        let prefix = if config.weight_prefix.is_empty() {
            "model.language_model"
        } else {
            &config.weight_prefix
        };

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_layer_idx = 0usize;

        for i in 0..config.num_hidden_layers {
            let lp = format!("{prefix}.layers.{i}");
            tracing::debug!("step3p7: layer {i}");

            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // ── FFN: detect MoE vs dense by probing for gate weight ─────
            let moe_gate_key = format!("{lp}.moe.gate.weight");
            let is_moe = store.contains(&moe_gate_key);

            let ffn = if is_moe {
                // ── MoE layer ───────────────────────────────────────────
                let gate = dense(store, &moe_gate_key)?;

                // Router bias (sigmoid routing)
                let bias_key = format!("{lp}.moe.router_bias");
                let correction_bias = if store.contains(&bias_key) {
                    Some(dense(store, &bias_key)?)
                } else {
                    None
                };

                // Load fused expert tensors and slice per-expert
                let moe_p = format!("{lp}.moe");
                let (gp_w, gp_s, gp_is, gp_s2) = load_fused_nvfp4(store, &format!("{moe_p}.gate_proj"), gpu)?;
                let (up_w, up_s, up_is, up_s2) = load_fused_nvfp4(store, &format!("{moe_p}.up_proj"), gpu)?;
                let (dp_w, dp_s, dp_is, dp_s2) = load_fused_nvfp4(store, &format!("{moe_p}.down_proj"), gpu)?;

                let gate_projs = slice_fused_experts(gp_w, gp_s, gp_is, gp_s2, config.num_experts, inter, h);
                let up_projs = slice_fused_experts(up_w, up_s, up_is, up_s2, config.num_experts, inter, h);
                let down_projs = slice_fused_experts(dp_w, dp_s, dp_is, dp_s2, config.num_experts, h, inter);

                let experts: Vec<ExpertWeight> = (0..config.num_experts)
                    .map(|e| ExpertWeight {
                        gate_proj: gate_projs[e],
                        up_proj: up_projs[e],
                        down_proj: down_projs[e],
                    })
                    .collect();

                // Shared expert (BF16, needs runtime quantization to NVFP4)
                let se_p = format!("{lp}.share_expert");
                let se_gate = dense_auto(store, &format!("{se_p}.gate_proj.weight"), gpu)?;
                let se_up = dense_auto(store, &format!("{se_p}.up_proj.weight"), gpu)?;
                let se_down = dense_auto(store, &format!("{se_p}.down_proj.weight"), gpu)?;
                let shared_expert = ExpertWeight {
                    gate_proj: quantize_to_nvfp4(&se_gate, shared_inter, h, gpu, absmax_k, quantize_k, stream)?,
                    up_proj: quantize_to_nvfp4(&se_up, shared_inter, h, gpu, absmax_k, quantize_k, stream)?,
                    down_proj: quantize_to_nvfp4(&se_down, h, shared_inter, gpu, absmax_k, quantize_k, stream)?,
                };

                // Step 3.7 doesn't have an explicit shared_expert_gate sigmoid
                // weight — the shared expert is always active (unconditional addition).
                // Create a dummy shared_expert_gate with a single 1.0 BF16 value.
                let dummy_seg_buf = [0x80u8, 0x3F]; // 1.0 in BF16
                let seg_ptr = gpu.alloc(2)?;
                gpu.copy_h2d(&dummy_seg_buf, seg_ptr)?;
                let shared_expert_gate = DenseWeight { weight: seg_ptr };

                // Quantize router gate for prefill
                let gate_nvfp4 = quantize_to_nvfp4(
                    &gate, config.num_experts, h, gpu, absmax_k, quantize_k, stream,
                )?;

                let moe_weights = MoeWeights {
                    gate,
                    shared_expert,
                    shared_expert_gate,
                    experts,
                    router_pre_norm: None,
                    correction_bias,
                };

                let mut moe_layer = MoeLayer::new(
                    moe_weights,
                    config.num_experts,
                    Some(gate_nvfp4),
                    gpu,
                    config,
                )?;
                moe_layer.predequant_for_prefill(gpu, config, stream)?;
                // Note: transpose_for_prefill is deferred to post-load in factory::build
                // for memory management (same as MiniMax M2)
                FfnComponent::Moe(moe_layer)
            } else {
                // ── Dense FFN layer (layers 0-2) ────────────────────────
                tracing::info!("step3p7: layer {i} is dense FFN");
                let gate_w = dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?;
                let up_w = dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?;
                let down_w = dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?;

                let gate_q = quantize_to_nvfp4(&gate_w, config.intermediate_size, h, gpu, absmax_k, quantize_k, stream)?;
                let up_q = quantize_to_nvfp4(&up_w, config.intermediate_size, h, gpu, absmax_k, quantize_k, stream)?;
                let down_q = quantize_to_nvfp4(&down_w, h, config.intermediate_size, gpu, absmax_k, quantize_k, stream)?;

                let dense_weights = DenseFfnWeights {
                    gate_proj: gate_q,
                    up_proj: up_q,
                    down_proj: down_q,
                };
                FfnComponent::Dense(DenseFfnLayer::new(dense_weights, gpu)?)
            };

            // ── Attention ───────────────────────────────────────────────
            let p = format!("{lp}.self_attn");
            let q_proj = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
            let k_proj = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
            let v_proj = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
            let o_proj_w = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;

            let q_proj_n = config.num_attention_heads * config.head_dim;
            let kv_proj_n = config.num_key_value_heads * config.head_dim;
            let q_nvfp4 = quantize_to_nvfp4(&q_proj, q_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
            let k_nvfp4 = quantize_to_nvfp4(&k_proj, kv_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
            let v_nvfp4 = quantize_to_nvfp4(&v_proj, kv_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
            let o_nvfp4 = quantize_to_nvfp4(&o_proj_w, h, q_proj_n, gpu, absmax_k, quantize_k, stream)?;

            // Per-head q_norm / k_norm
            let q_norm = dense(store, &format!("{p}.q_norm.weight"))?;
            let k_norm = dense(store, &format!("{p}.k_norm.weight"))?;

            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

            let attn = AttentionWeights {
                q_proj,
                k_proj,
                v_proj,
                o_proj: o_nvfp4,
                q_norm,
                k_norm,
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
                attn_layer_idx,
                Some(q_nvfp4),
                Some(k_nvfp4),
                Some(v_nvfp4),
                gpu,
                layer_kv_dtypes[attn_layer_idx],
                config.fp8_kv_calibration_tokens,
                config,
            )?;

            // Transpose attention NVFP4 for prefill
            let qt = q_nvfp4.transpose_for_gemm(gpu, q_proj_n, h)?;
            let kt = k_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
            let vt = v_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
            let ot = layer.attn.o_proj.transpose_for_gemm(gpu, h, q_proj_n)?;
            layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));

            layers.push(Box::new(layer));
            attn_layer_idx += 1;
        }

        tracing::info!("step3p7: built {} layers ({} attention)", layers.len(), attn_layer_idx);
        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        let prefix = if config.weight_prefix.is_empty() {
            "model.language_model"
        } else {
            &config.weight_prefix
        };
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = if config.weight_prefix.is_empty() {
            "model.language_model"
        } else {
            &config.weight_prefix
        };
        dense(store, &format!("{prefix}.norm.weight"))
    }

    fn load_lm_head(&self, store: &WeightStore, _config: &ModelConfig) -> Result<DenseWeight> {
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None) // Multi-module MTP — use load_mtp_weights_multi
    }

    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        // Step 3.7 MTP layers at `model.layers.{45|46|47}` — note different
        // prefix from main layers.
        let first_mtp_idx = config.num_hidden_layers;
        let probe = format!("model.layers.{first_mtp_idx}.input_layernorm.weight");
        if !store.contains(&probe) {
            tracing::info!(
                "step3p7: no MTP module weights found (expected at layer {first_mtp_idx}); MTP disabled"
            );
            return Ok(Vec::new());
        }

        // Initial bring-up: skip MTP. Run with --speculative 0.
        tracing::info!(
            "step3p7: MTP module weights detected at layers {}-{} but MTP loader \
             not yet implemented. Run with --speculative 0 for non-MTP decode.",
            first_mtp_idx,
            first_mtp_idx + config.mtp_num_hidden_layers - 1,
        );
        Ok(Vec::new())
    }
}
