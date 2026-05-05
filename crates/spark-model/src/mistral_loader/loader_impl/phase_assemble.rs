// SPDX-License-Identifier: AGPL-3.0-only

//! Phase G: build MlaWeights, the dummy AttentionWeights stub, the MoE
//! FFN, and the final TransformerLayer.

use anyhow::{Result, anyhow};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;

use super::ctx::MistralLayerCtx;
use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::MlaWeights;
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight, dense, load_moe_mistral};

pub(super) fn assemble_layer(
    ctx: MistralLayerCtx<'_>,
    yarn_inv_freq: DevicePtr,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Box<dyn TransformerLayer>> {
    let i = ctx.layer_idx;
    let prefix = format!("layers.{i}");
    let gpu = ctx.gpu;
    let config = ctx.config;
    let q_lora = ctx.q_lora;
    let kv_lora = ctx.kv_lora;
    let nope = ctx.nope;
    let rope = ctx.rope;
    let v_dim = ctx.v_dim;

    // Dummy attention weights — never accessed when mla is Some, except
    // o_proj which is unused on the MLA decode path (reads come from
    // mla.wo / wo_nvfp4).
    let null = DenseWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
    };
    let o_dummy_quant = QuantizedWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
        weight_scale: spark_runtime::gpu::DevicePtr::NULL,
        weight_scale_2: 0.0,
        input_scale: spark_runtime::gpu::DevicePtr::NULL,
    };
    let attn = AttentionWeights {
        q_proj: null,
        k_proj: null,
        v_proj: null,
        o_proj: o_dummy_quant,
        q_norm: null,
        k_norm: null,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };

    // MLA projections default to NVFP4 for GB10 decode throughput.
    // Set ATLAS_NVFP4_MLA={0,false,no,off} (case-insensitive) to force BF16.
    let disable_nvfp4_mla = std::env::var("ATLAS_NVFP4_MLA")
        .map(|v| {
            let v = v.trim().to_ascii_lowercase();
            matches!(v.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false);

    // Every required MLA tensor must have been populated by an earlier
    // phase. Surface a typed error naming the missing field instead of
    // panicking, so a malformed checkpoint variant produces a useful
    // startup diagnostic in the server log.
    let require = |v: Option<DenseWeight>, name: &'static str| -> Result<DenseWeight> {
        v.ok_or_else(|| anyhow!("Mistral L{i} phase_assemble: missing required tensor `{name}`"))
    };

    let wq_a_dense = require(ctx.wq_a_dense, "wq_a_dense")?;
    let wkv_a_dense = require(ctx.wkv_a_dense, "wkv_a_dense")?;
    let mla_weights = MlaWeights {
        wq_a: wq_a_dense,
        wq_a_nvfp4: if disable_nvfp4_mla {
            None
        } else {
            ctx.wq_a_nvfp4
        },
        wq_b: require(ctx.wq_b, "wq_b")?,
        wq_b_nvfp4: if disable_nvfp4_mla {
            None
        } else {
            ctx.wq_b_nvfp4
        },
        q_a_norm: require(ctx.q_a_norm, "q_a_norm")?,
        wkv_a: wkv_a_dense,
        wkv_a_nvfp4: if disable_nvfp4_mla {
            None
        } else {
            ctx.wkv_a_nvfp4
        },
        wkv_b: require(ctx.wkv_b, "wkv_b")?,
        kv_a_norm: require(ctx.kv_a_norm, "kv_a_norm")?,
        wkv_a_rope: require(ctx.wkv_a_rope_dense, "wkv_a_rope_dense")?,
        wkv_a_merged: DenseWeight {
            weight: wkv_a_dense.weight,
        },
        wo: require(ctx.o_dense_bf16, "o_dense_bf16")?,
        wo_nvfp4: if disable_nvfp4_mla { None } else { ctx.o_nvfp4 },
        wq_b_rope: require(ctx.wq_b_rope, "wq_b_rope")?,
        w_uk_t: require(ctx.w_uk_t, "w_uk_t")?,
        w_uv: require(ctx.w_uv, "w_uv")?,
        w_qk_absorbed: require(ctx.w_qk_absorbed, "w_qk_absorbed")?,
        w_uk_block_diag: require(ctx.w_uk_block_diag, "w_uk_block_diag")?,
        w_uv_block_diag: require(ctx.w_uv_block_diag, "w_uv_block_diag")?,
        yarn_inv_freq,
        q_lora_rank: q_lora,
        kv_lora_rank: kv_lora,
        nope,
        rope,
        v_dim,
    };

    let input_norm = dense(ctx.store, &format!("{prefix}.attention_norm.weight"))?;
    let post_norm = dense(ctx.store, &format!("{prefix}.ffn_norm.weight"))?;
    let kv_dtype = layer_kv_dtypes.get(i).copied().unwrap_or(KvCacheDtype::Fp8);

    // ── MoE experts (w1=gate, w2=down, w3=up) ──
    let ffn = build_moe_ffn(ctx.store, i, gpu, config);

    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm, attn, post_norm, ffn, i, None, None, None, gpu, kv_dtype, 0, config,
    )?;
    layer.set_mla_weights(mla_weights);
    Ok(Box::new(layer))
}

fn build_moe_ffn(
    store: &spark_runtime::weights::WeightStore,
    i: usize,
    gpu: &dyn GpuBackend,
    config: &atlas_core::config::ModelConfig,
) -> FfnComponent {
    if config.num_experts == 0 {
        return FfnComponent::None;
    }
    match load_moe_mistral(store, i, config.num_experts, gpu, config) {
        Ok(moe_weights) => {
            match MoeLayer::new(moe_weights, config.num_experts, None, gpu, config) {
                Ok(mut moe) => {
                    // Skip MoE transpose for Mistral on single GPU: saves
                    // ~1.5 GB per layer (54 GB total). Prefill uses the
                    // untransposed path (slightly slower but avoids OOM).
                    // EP: enough memory per rank, transpose for faster prefill.
                    if config.ep_world_size > 1
                        && let Err(e) = moe.transpose_for_prefill(gpu, config)
                    {
                        tracing::warn!("L{i}: MoE transpose failed: {e}, using untransposed");
                    }
                    FfnComponent::Moe(moe)
                }
                Err(e) => {
                    tracing::warn!("L{i}: MoE construction failed: {e}, using None");
                    FfnComponent::None
                }
            }
        }
        Err(e) => {
            tracing::warn!("L{i}: MoE weight load failed: {e}, using None");
            FfnComponent::None
        }
    }
}
