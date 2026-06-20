// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::{DenseFfnLayer, FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_auto,
    detect_nvfp4_variant, load_kv_scales, quantize_to_nvfp4,
};

use super::{
    has_per_expert_tensors, load_fused_nvfp4, offset_norm_weights_plus_one, slice_fused_experts,
};

pub(super) fn load_layers(
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

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_layer_idx = 0usize;

    for i in 0..config.num_hidden_layers {
        let lp = format!("{prefix}.layers.{i}");
        tracing::debug!("step3p7: layer {i}");

        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

        // Step 3.7 shifted RMSNorm: add 1.0 to norm weights so standard
        // kernel computes (x/rms) * (weight+1) correctly.
        offset_norm_weights_plus_one(&input_norm, h, gpu)?;
        offset_norm_weights_plus_one(&post_attn_norm, h, gpu)?;

        // ── FFN: detect MoE vs dense by probing for gate weight ─────
        let moe_gate_key = format!("{lp}.moe.gate.weight");
        let is_moe = store.contains(&moe_gate_key);

        let ffn = if is_moe {
            load_moe_ffn(
                store,
                config,
                gpu,
                &lp,
                &moe_gate_key,
                h,
                inter,
                shared_inter,
                absmax_k,
                quantize_k,
                stream,
            )?
        } else {
            load_dense_ffn(
                store,
                gpu,
                &lp,
                config.intermediate_size,
                h,
                absmax_k,
                quantize_k,
                stream,
                i,
            )?
        };

        // ── Attention ───────────────────────────────────────────────
        let layer = load_attention_layer(
            store,
            config,
            gpu,
            &lp,
            input_norm,
            post_attn_norm,
            ffn,
            attn_layer_idx,
            h,
            layer_kv_dtypes,
            absmax_k,
            quantize_k,
            stream,
            i,
        )?;

        layers.push(Box::new(layer));
        attn_layer_idx += 1;
    }

    tracing::info!(
        "step3p7: built {} layers ({} attention)",
        layers.len(),
        attn_layer_idx
    );
    Ok(layers)
}

#[allow(clippy::too_many_arguments)]
fn load_moe_ffn(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    moe_gate_key: &str,
    h: usize,
    inter: usize,
    shared_inter: usize,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
) -> Result<FfnComponent> {
    let gate = dense(store, moe_gate_key)?;

    // Router bias (sigmoid routing)
    let bias_key = format!("{lp}.moe.router_bias");
    let correction_bias = if store.contains(&bias_key) {
        Some(dense(store, &bias_key)?)
    } else {
        None
    };

    let moe_p = format!("{lp}.moe");
    let use_per_expert = has_per_expert_tensors(store, lp);

    let experts: Vec<ExpertWeight> = if use_per_expert {
        tracing::debug!("step3p7: layer {lp} using per-expert tensor format");
        (0..config.num_experts)
            .map(|e| {
                let ep = format!("{moe_p}.experts.{e}");
                let gp_key = format!("{ep}.gate_proj.weight");
                if !store.contains(&gp_key) {
                    return Ok(ExpertWeight::null());
                }
                let load_expert_proj = |proj: &str| -> Result<QuantizedWeight> {
                    let pp = format!("{ep}.{proj}");
                    let weight = store.get(&format!("{pp}.weight"))?.ptr;
                    let weight_scale = store.get(&format!("{pp}.weight_scale"))?.ptr;
                    let ws2_key = format!("{pp}.weight_scale_2");
                    let global_ws2_key = format!("{moe_p}.{proj}.weight_scale_2");
                    let ws2_ptr = if store.contains(&ws2_key) {
                        store.get(&ws2_key)?.ptr
                    } else if store.contains(&global_ws2_key) {
                        store.get(&global_ws2_key)?.ptr
                    } else {
                        anyhow::bail!(
                            "weight_scale_2 not found for {pp} \
                             (tried per-expert and global)"
                        );
                    };
                    let mut ws2_buf = [0u8; 4];
                    gpu.copy_d2h(ws2_ptr, &mut ws2_buf).ok();
                    let weight_scale_2 = f32::from_le_bytes(ws2_buf);
                    let is_key = format!("{pp}.input_scale");
                    let input_scale = if store.contains(&is_key) {
                        store
                            .get(&is_key)
                            .ok()
                            .map(|t| t.ptr)
                            .unwrap_or(DevicePtr::NULL)
                    } else {
                        DevicePtr::NULL
                    };
                    Ok(QuantizedWeight {
                        weight,
                        weight_scale,
                        weight_scale_2,
                        input_scale,
                    })
                };
                let gate_proj = load_expert_proj("gate_proj")?;
                let up_proj = load_expert_proj("up_proj")?;
                let down_proj = load_expert_proj("down_proj")?;
                Ok(ExpertWeight {
                    gate_proj,
                    up_proj,
                    down_proj,
                })
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        tracing::debug!("step3p7: layer {lp} using fused tensor format");
        let (gp_w, gp_s, gp_is, gp_s2) =
            load_fused_nvfp4(store, &format!("{moe_p}.gate_proj"), gpu)?;
        let (up_w, up_s, up_is, up_s2) = load_fused_nvfp4(store, &format!("{moe_p}.up_proj"), gpu)?;
        let (dp_w, dp_s, dp_is, dp_s2) =
            load_fused_nvfp4(store, &format!("{moe_p}.down_proj"), gpu)?;

        let gate_projs =
            slice_fused_experts(gp_w, gp_s, gp_is, gp_s2, config.num_experts, inter, h);
        let up_projs = slice_fused_experts(up_w, up_s, up_is, up_s2, config.num_experts, inter, h);
        let down_projs =
            slice_fused_experts(dp_w, dp_s, dp_is, dp_s2, config.num_experts, h, inter);

        (0..config.num_experts)
            .map(|e| ExpertWeight {
                gate_proj: gate_projs[e],
                up_proj: up_projs[e],
                down_proj: down_projs[e],
            })
            .collect()
    };

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

    let shared_expert_gate = DenseWeight {
        weight: DevicePtr::NULL,
    };

    let gate_nvfp4 = quantize_to_nvfp4(
        &gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
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
    Ok(FfnComponent::Moe(moe_layer))
}

#[allow(clippy::too_many_arguments)]
fn load_dense_ffn(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    lp: &str,
    intermediate_size: usize,
    h: usize,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    i: usize,
) -> Result<FfnComponent> {
    tracing::info!("step3p7: layer {i} is dense FFN");
    let gate_w = dense_auto(store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?;
    let up_w = dense_auto(store, &format!("{lp}.mlp.up_proj.weight"), gpu)?;
    let down_w = dense_auto(store, &format!("{lp}.mlp.down_proj.weight"), gpu)?;

    let gate_q = quantize_to_nvfp4(
        &gate_w,
        intermediate_size,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let up_q = quantize_to_nvfp4(
        &up_w,
        intermediate_size,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let down_q = quantize_to_nvfp4(
        &down_w,
        h,
        intermediate_size,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;

    let dense_weights = DenseFfnWeights {
        gate_proj: gate_q,
        up_proj: up_q,
        down_proj: down_q,
        // Transposed copies for the fast w4a16_gemm_t_m128 prefill kernel.
        gate_proj_t: Some(gate_q.transpose_for_gemm(gpu, intermediate_size, h)?),
        up_proj_t: Some(up_q.transpose_for_gemm(gpu, intermediate_size, h)?),
        down_proj_t: Some(down_q.transpose_for_gemm(gpu, h, intermediate_size)?),
    };
    Ok(FfnComponent::Dense(DenseFfnLayer::new(dense_weights, gpu)?))
}

#[allow(clippy::too_many_arguments)]
fn load_attention_layer(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    lp: &str,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
    attn_layer_idx: usize,
    h: usize,
    layer_kv_dtypes: &[KvCacheDtype],
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
    i: usize,
) -> Result<Qwen3AttentionLayer> {
    let p = format!("{lp}.self_attn");
    let q_proj = dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?;
    let k_proj = dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?;
    let v_proj = dense_auto(store, &format!("{p}.v_proj.weight"), gpu)?;
    let o_proj_w = dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?;

    let q_proj_shape = store.get(&format!("{p}.q_proj.weight"))?.shape.clone();
    let q_proj_n = q_proj_shape[0];
    let kv_proj_n = config.num_key_value_heads * config.head_dim;
    let actual_q_heads = q_proj_n / config.head_dim;
    tracing::info!(
        "step3p7: layer {i} attention: q_proj_n={q_proj_n} \
         ({actual_q_heads} Q heads), kv_proj_n={kv_proj_n}"
    );
    let q_nvfp4 = quantize_to_nvfp4(&q_proj, q_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
    let k_nvfp4 = quantize_to_nvfp4(&k_proj, kv_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
    let v_nvfp4 = quantize_to_nvfp4(&v_proj, kv_proj_n, h, gpu, absmax_k, quantize_k, stream)?;
    let o_nvfp4 = quantize_to_nvfp4(&o_proj_w, h, q_proj_n, gpu, absmax_k, quantize_k, stream)?;

    // Per-head attention gate (g_proj)
    let g_proj_key = format!("{p}.g_proj.weight");
    let g_proj_weight = if store.contains(&g_proj_key) {
        let w = dense_auto(store, &g_proj_key, gpu)?;
        tracing::info!("step3p7: layer {i} loaded g_proj gate [{actual_q_heads}, {h}]");
        Some(w)
    } else {
        None
    };

    // Per-head q_norm / k_norm (also shifted RMSNorm)
    let q_norm = dense(store, &format!("{p}.q_norm.weight"))?;
    let k_norm = dense(store, &format!("{p}.k_norm.weight"))?;
    offset_norm_weights_plus_one(&q_norm, config.head_dim, gpu)?;
    offset_norm_weights_plus_one(&k_norm, config.head_dim, gpu)?;

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

    let mut layer = Qwen3AttentionLayer::new_ungated(
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

    layer.set_dimension_overrides(config.head_dim, actual_q_heads, config.num_key_value_heads);

    let is_sliding = if !config.layer_types.is_empty() {
        config.layer_types.get(i).copied() == Some(atlas_core::config::LayerType::SlidingAttention)
    } else {
        !i.is_multiple_of(4)
    };

    if is_sliding && config.sliding_window > 0 {
        layer.set_sliding_window(Some(config.sliding_window));
    } else {
        layer.set_sliding_window(None);
    }

    if is_sliding {
        layer.set_rope_overrides(10000.0, config.head_dim as u32);
    }

    if let Some(gw) = g_proj_weight {
        layer.set_head_gate_weight(gw);
    }

    let qt = q_nvfp4.transpose_for_gemm(gpu, q_proj_n, h)?;
    let kt = k_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
    let vt = v_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
    let ot = layer.attn.o_proj.transpose_for_gemm(gpu, h, q_proj_n)?;
    layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));

    Ok(layer)
}
