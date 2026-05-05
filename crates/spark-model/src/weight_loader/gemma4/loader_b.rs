// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 weight loader: per-layer helpers + auxiliary methods (embedding,
//! norm, lm_head, mtp, kv_layer_dims).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::FfnComponent;
use crate::weight_map::{DenseWeight, MtpWeights, QuantizeCtx, dense};

/// Build the Gemma-4 26B MoE FFN component for one layer (None for dense).
pub(super) fn build_moe_ffn(
    store: &WeightStore,
    lp: &str,
    i: usize,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    variant: crate::weight_map::Nvfp4Variant,
    qctx: QuantizeCtx,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<Option<(FfnComponent, DenseWeight, DenseWeight, DenseWeight)>> {
    if config.num_experts == 0 {
        return Ok(None);
    }
    use crate::weight_map::load_moe_gemma4;
    tracing::info!(
        "L{i}: loading MoE ({} experts, top-{})...",
        config.num_experts,
        config.num_experts_per_tok
    );
    let moe_weights = load_moe_gemma4(store, lp, config.num_experts, gpu, config, variant, qctx)?;
    gpu.synchronize(stream)?;
    let gate_nvfp4 = crate::weight_map::quantize_to_nvfp4(
        &moe_weights.gate,
        config.num_experts,
        h,
        gpu,
        absmax_k,
        quantize_k,
        stream,
    )?;
    let mut moe_layer = crate::layers::MoeLayer::new(
        moe_weights,
        config.num_experts,
        Some(gate_nvfp4),
        gpu,
        config,
    )?;
    moe_layer.set_gelu_activation(gpu)?;
    // Set pre-expert norm: router sees raw input, experts see normed input.
    let pre_expert_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
    moe_layer.set_pre_expert_norm(pre_expert_norm);
    gpu.synchronize(stream)?;
    tracing::info!("L{i}: MoE layer built (GeGLU activation, pre-expert norm)");

    // Load extra norms for dual FFN
    let pre_moe_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm_2.weight"))?;
    let post_moe_out_norm = dense(store, &format!("{lp}.post_feedforward_layernorm_2.weight"))?;
    let post_dense_ffn_norm = dense(store, &format!("{lp}.post_feedforward_layernorm_1.weight"))?;
    Ok(Some((
        FfnComponent::Moe(moe_layer),
        pre_moe_norm,
        post_moe_out_norm,
        post_dense_ffn_norm,
    )))
}

/// Allocate a BF16 ones-filled buffer of length `head_dim` for v_norm
/// (Gemma-4 v_norm is identity-shaped per HF reference).
pub(super) fn make_v_norm_ones_bf16(gpu: &dyn GpuBackend, head_dim: usize) -> Result<DenseWeight> {
    let bytes = head_dim * 2; // BF16
    let ptr = gpu.alloc(bytes)?;
    // BF16 1.0 = 0x3F80 little-endian → bytes 0x80, 0x3F
    let ones_host: Vec<u8> = std::iter::repeat_with(|| [0x80u8, 0x3Fu8])
        .take(head_dim)
        .flatten()
        .collect();
    gpu.copy_h2d(&ones_host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// Optional BF16 dequant of MLP gate/up/down (Gemma-4 dense, NVFP4 on disk).
pub(super) fn build_bf16_mlp(
    store: &WeightStore,
    lp: &str,
    bf16_mlp: bool,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    h: usize,
) -> Result<Option<(DenseWeight, DenseWeight, DenseWeight)>> {
    let mlp_is_nvfp4 = store.contains(&format!("{lp}.mlp.gate_proj.weight_scale"));
    if !(bf16_mlp && mlp_is_nvfp4) {
        return Ok(None);
    }
    use crate::weight_map::dequant_nvfp4_to_bf16;
    let gate_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.gate_proj"),
        config.intermediate_size,
        h,
        gpu,
    )?;
    let up_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.up_proj"),
        config.intermediate_size,
        h,
        gpu,
    )?;
    let down_bf16 = dequant_nvfp4_to_bf16(
        store,
        &format!("{lp}.mlp.down_proj"),
        h,
        config.intermediate_size,
        gpu,
    )?;
    Ok(Some((gate_bf16, up_bf16, down_bf16)))
}

pub(super) fn load_embedding_impl(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<DenseWeight> {
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.embed_tokens.weight"))
}

pub(super) fn load_final_norm_impl(
    store: &WeightStore,
    config: &ModelConfig,
) -> Result<DenseWeight> {
    // Gemma-4's model-specific rms_norm kernel uses the absolute
    // convention (`out = x * rms * weight`), so norms load as-is.
    let prefix = &config.weight_prefix;
    dense(store, &format!("{prefix}.norm.weight"))
}

pub(super) fn load_lm_head_impl(store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
    // Gemma-4 uses tied embeddings (no separate lm_head tensor).
    // Check for explicit lm_head first, fall back to embed_tokens.
    for pattern in &[
        "lm_head.weight",
        "language_model.lm_head.weight",
        "model.lm_head.weight",
    ] {
        if store.contains(pattern) {
            return dense(store, pattern);
        }
    }
    load_embedding_impl(store, config)
}

pub(super) fn load_mtp_weights_impl(
    _store: &WeightStore,
    _config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<Option<MtpWeights>> {
    // Gemma-4 has no MTP head.
    Ok(None)
}

pub(super) fn kv_layer_dims_impl(config: &ModelConfig) -> Vec<(usize, usize)> {
    // Gemma-4 has a 5:1 sliding→full pattern:
    //   Sliding:  nkv=num_kv_heads (16), hd=256
    //   Full:     nkv=num_global_kv_heads (4), hd=global_head_dim (512)
    // Layer i is full when (i+1) % 6 == 0, else sliding.
    let sliding_nkv = config.num_key_value_heads;
    let sliding_hd = 256;
    let full_nkv = 4;
    let full_hd = 512;
    let mut dims = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        if (i + 1) % 6 == 0 {
            dims.push((full_nkv, full_hd));
        } else {
            dims.push((sliding_nkv, sliding_hd));
        }
    }
    dims
}
