// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat-Flash(-Lite) weight loader — the backbone behind the n-gram
//! embeddings (`longcat_flash_ngram`).
//!
//! Architecture (HF `modeling_longcat_flash.py`), and how it maps onto Atlas:
//!
//! - Each CHECKPOINT layer is a dual-sublayer "shortcut" block: two MLA
//!   attentions, two dense SwiGLU MLPs, and ONE shortcut MoE whose output is
//!   computed on sublayer 1's post-attention normed input but added at the END
//!   of sublayer 2. Atlas serves each SUBLAYER as one `Qwen3AttentionLayer`
//!   (`num_hidden_layers` is already 2x at parse), with the shortcut carried
//!   between the pair via `set_shortcut_moe` / `set_shortcut_carry_in`.
//! - MLA is the DeepSeek-lineage q-LoRA form Atlas already serves; the two
//!   LongCat deltas (interleaved rope, sqrt LoRA scaling) fold into the
//!   WEIGHTS at load (see `prep`), so the runtime is unchanged.
//! - The MoE router is softmax + `e_score_correction_bias` over
//!   `n_routed + zero_expert_num` logits, with the zero (identity) experts
//!   folded inside the router kernel (see `moe_topk_softmax_bias.cu`).
//!
//! Tensor names are HF-standard under `model.layers.{L}.`:
//!   `self_attn.{0,1}.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
//!                     kv_a_layernorm,kv_b_proj,o_proj}`
//!   `mlps.{0,1}.{gate,up,down}_proj`, `input_layernorm.{0,1}`,
//!   `post_attention_layernorm.{0,1}`,
//!   `mlp.router.{classifier.weight,e_score_correction_bias}`,
//!   `mlp.experts.{e}.{gate,up,down}_proj`.

mod ngram;
mod prep;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::MlaWeights;
use crate::layers::vision_encoder::VisionEncoder;
use crate::mistral_loader::loader_impl::{
    ctx as mctx, phase_block_diag, phase_per_head, phase_qk_absorbed,
};
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizedWeight, dense, quantize_to_nvfp4,
};

pub struct LongcatWeightLoader;

/// Tokens the shortcut-MoE carry buffer must hold: the largest prefill chunk
/// a sublayer can be handed. Sized from `max_prefill_tokens`' ceiling; the
/// producer/consumer both `ensure!` against it rather than overrunning.
const CARRY_TOKENS: usize = 8192;

impl ModelWeightLoader for LongcatWeightLoader {
    fn supports_tp(&self) -> bool {
        // MLA TP would need the same wq_b/wkv_b column sharding Mistral does,
        // plus a per-rank shortcut carry. Not validated — refuse rather than
        // serve a silently wrong shard split.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        anyhow::ensure!(
            config.num_hidden_layers.is_multiple_of(2),
            "longcat: num_hidden_layers ({}) must be even — each checkpoint \
             layer is TWO engine sublayers",
            config.num_hidden_layers
        );
        let ckpt_layers = config.num_hidden_layers / 2;
        let h = config.hidden_size;
        let nope = config.qk_nope_head_dim;
        let rope = config.qk_rope_head_dim;
        let q_lora = config.q_lora_rank;
        let kv_lora = config.kv_lora_rank;
        let n_heads = config.num_attention_heads;
        // The reference's mla_scale_{q,kv}_lora flags (both true on Lite).
        let scale_q = (h as f32 / q_lora as f32).sqrt();
        let scale_kv = (h as f32 / kv_lora as f32).sqrt();
        // Head-width padding (see prep.rs §3): LongCat's qk head is 192 and its
        // v head is 128 — the first MLA model where they differ, and 192 does
        // not compile. Both are padded to the stock 256 in the WEIGHTS.
        let true_qk_hd = nope + rope;
        let padded_hd = 256usize;
        let padded_nope = padded_hd - rope;
        // The softmax scale must stay 1/sqrt(TRUE qk head width).
        let attn_scale = 1.0f32 / (true_qk_hd as f32).sqrt();

        tracing::info!(
            "LongCat: {ckpt_layers} checkpoint layers → {} engine sublayers \
             (MLA q_lora={q_lora} kv_lora={kv_lora} nope={nope} rope={rope}; \
             rope de-interleave + q/kv LoRA scale folded at load: \
             q×{scale_q:.4}, kv-norm×{scale_kv:.4}); {} routed + {} zero experts",
            config.num_hidden_layers,
            config.num_experts,
            config.zero_expert_num,
        );

        // Same measurement lever the Mistral MLA loader has: ATLAS_NVFP4_MLA=0
        // keeps the MLA projections in BF16, which separates "the port's math
        // is wrong" from "4-bit quantization of these projections is lossy".
        let disable_nvfp4_mla = std::env::var("ATLAS_NVFP4_MLA")
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                matches!(v.as_str(), "0" | "false" | "no" | "off")
            })
            .unwrap_or(false);
        if disable_nvfp4_mla {
            tracing::info!("LongCat: ATLAS_NVFP4_MLA=0 — MLA projections stay BF16");
        }
        if super::longcat::ffn::bf16_dense_ffn() {
            tracing::info!("LongCat: ATLAS_LONGCAT_BF16_FFN=1 — per-sublayer dense FFN stays BF16");
        }
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let mut yarn_shared = DevicePtr::NULL;
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);

        for l in 0..ckpt_layers {
            let lp = format!("model.layers.{l}");
            // One carry buffer per checkpoint layer (producer sublayer 0 →
            // consumer sublayer 1). Allocated per block so two blocks in
            // flight (chunked prefill) cannot alias.
            let carry = gpu.alloc(CARRY_TOKENS * h * 2)?;

            for s in 0..2usize {
                let ap = format!("{lp}.self_attn.{s}");
                let global_idx = l * 2 + s;

                // ── MLA: name-bound loads + the two LongCat folds ──
                let wq_a = dense(store, &format!("{ap}.q_a_proj.weight"))?;
                let wq_b = prep::prep_q_b(
                    store,
                    &format!("{ap}.q_b_proj.weight"),
                    n_heads,
                    nope,
                    rope,
                    q_lora,
                    scale_q,
                    padded_hd,
                    gpu,
                )?;
                let q_a_norm = dense(store, &format!("{ap}.q_a_layernorm.weight"))?;
                let wkv_a = prep::prep_kv_a(
                    store,
                    &format!("{ap}.kv_a_proj_with_mqa.weight"),
                    kv_lora,
                    rope,
                    h,
                    gpu,
                )?;
                let kv_a_norm = prep::prep_kv_a_norm(
                    store,
                    &format!("{ap}.kv_a_layernorm.weight"),
                    kv_lora,
                    scale_kv,
                    gpu,
                )?;
                let wkv_b = prep::prep_kv_b(
                    store,
                    &format!("{ap}.kv_b_proj.weight"),
                    n_heads,
                    nope,
                    config.v_head_dim,
                    rope,
                    kv_lora,
                    padded_hd,
                    gpu,
                )?;
                let wo = prep::prep_o_proj(
                    store,
                    &format!("{ap}.o_proj.weight"),
                    h,
                    n_heads,
                    config.v_head_dim,
                    padded_hd,
                    gpu,
                )?;

                // ── shared MLA precompute (per-head transpose → absorbed QK
                //    → block-diagonals), reusing the Mistral phases ──
                let mut c = mctx::MistralLayerCtx::new(
                    store, config, gpu, absmax_k, quantize_k, stream, global_idx,
                );
                // Everything downstream indexes the PADDED weights.
                c.hd = padded_hd;
                c.nope = padded_nope;
                c.v_dim = padded_hd;
                c.wq_a_dense = Some(wq_a);
                c.wq_b = Some(wq_b);
                c.q_a_norm = Some(q_a_norm);
                c.wkv_a_dense = Some(wkv_a);
                c.wkv_a_rope_dense = Some(DenseWeight {
                    weight: wkv_a.weight.offset(kv_lora * h * 2),
                });
                c.wkv_b = Some(wkv_b);
                c.kv_a_norm = Some(kv_a_norm);
                c.wq_a_nvfp4 = Some(quantize_to_nvfp4(
                    &wq_a, q_lora, h, gpu, absmax_k, quantize_k, stream,
                )?);
                c.wq_b_nvfp4 = Some(quantize_to_nvfp4(
                    &wq_b,
                    n_heads * padded_hd,
                    q_lora,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?);
                c.wkv_a_nvfp4 = Some(quantize_to_nvfp4(
                    &wkv_a,
                    kv_lora + rope,
                    h,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?);
                phase_per_head::build_per_head_views(&mut c)?;
                phase_qk_absorbed::build_w_qk_absorbed(&mut c)?;
                phase_block_diag::build_block_diagonals(&mut c)?;
                let o_nvfp4 = quantize_to_nvfp4(
                    &wo,
                    h,
                    n_heads * padded_hd,
                    gpu,
                    absmax_k,
                    quantize_k,
                    stream,
                )?;
                let yarn = mctx::ensure_yarn_inv_freq(&mut yarn_shared, config, rope, gpu)?;

                let null = DenseWeight {
                    weight: DevicePtr::NULL,
                };
                let mla = MlaWeights {
                    wq_a,
                    wq_a_fp8: None,
                    wq_a_nvfp4: if disable_nvfp4_mla {
                        None
                    } else {
                        c.wq_a_nvfp4
                    },
                    wq_b,
                    wq_b_fp8: None,
                    wq_b_nvfp4: if disable_nvfp4_mla {
                        None
                    } else {
                        c.wq_b_nvfp4
                    },
                    q_a_norm,
                    wkv_a,
                    wkv_a_nvfp4: if disable_nvfp4_mla {
                        None
                    } else {
                        c.wkv_a_nvfp4
                    },
                    wkv_b,
                    kv_a_norm,
                    wkv_a_rope: c.wkv_a_rope_dense.expect("set above"),
                    wkv_a_merged: DenseWeight {
                        weight: wkv_a.weight,
                    },
                    wo,
                    wo_nvfp4: if disable_nvfp4_mla {
                        None
                    } else {
                        Some(o_nvfp4)
                    },
                    wo_a: null,
                    wo_a_nvfp4: None,
                    wo_b: null,
                    wo_b_nvfp4: None,
                    wo_b_fp8: None,
                    wo_a_fp8: None,
                    wkv_a_fp8: None,
                    wq_b_rope: c.wq_b_rope.context("longcat: wq_b_rope")?,
                    w_uk_t: c.w_uk_t.context("longcat: w_uk_t")?,
                    w_uv: c.w_uv.context("longcat: w_uv")?,
                    w_qk_absorbed: c.w_qk_absorbed.context("longcat: w_qk_absorbed")?,
                    w_uk_block_diag: c.w_uk_block_diag.context("longcat: w_uk_bd")?,
                    w_uv_block_diag: c.w_uv_block_diag.context("longcat: w_uv_bd")?,
                    yarn_inv_freq: yarn,
                    main_inv_freq: yarn,
                    q_lora_rank: q_lora,
                    kv_lora_rank: kv_lora,
                    o_lora_rank: 0,
                    nope: padded_nope,
                    rope,
                    v_dim: padded_hd,
                    compressor: None,
                    attn_sink: DevicePtr::NULL,
                };

                // Dummy attention weights (never read on the MLA path).
                let o_dummy = QuantizedWeight {
                    weight: DevicePtr::NULL,
                    weight_scale: DevicePtr::NULL,
                    weight_scale_2: 0.0,
                    input_scale: DevicePtr::NULL,
                    weight_scale_2_vec: DevicePtr::NULL,
                };
                let attn = AttentionWeights {
                    q_proj: null,
                    k_proj: null,
                    v_proj: null,
                    o_proj: o_dummy,
                    q_norm: null,
                    k_norm: null,
                    q_norm_full: None,
                    k_norm_full: None,
                    k_scale: 1.0,
                    v_scale: 1.0,
                };

                // ── dense SwiGLU FFN for this sublayer ──
                let ffn = build_dense_ffn(store, &format!("{lp}.mlps.{s}"), config, gpu)?;
                let input_norm = dense(store, &format!("{lp}.input_layernorm.{s}.weight"))?;
                let post_norm = dense(store, &format!("{lp}.post_attention_layernorm.{s}.weight"))?;
                let kv_dtype = layer_kv_dtypes
                    .get(global_idx)
                    .copied()
                    .unwrap_or(KvCacheDtype::Bf16);

                let mut layer = Qwen3AttentionLayer::new_ungated(
                    input_norm, attn, post_norm, ffn, global_idx, None, None, None, gpu, kv_dtype,
                    0, config,
                )?;
                layer.set_mla_weights(mla);
                // Padding widened the head to 256; the scale must remain
                // 1/sqrt(192), the TRUE qk head width.
                layer.set_attn_scale_override(attn_scale);
                // The attention chain strides Q/K/V by `head_dim_override`,
                // which must be the PADDED width the weights now emit — the
                // config's 192 would slice every head short.
                layer.set_dimension_overrides(padded_hd, n_heads, n_heads);

                if s == 0 {
                    // Sublayer 0 owns the block's shortcut MoE; its output is
                    // stashed and added at the end of sublayer 1.
                    let moe = build_shortcut_moe(store, &lp, config, gpu)?;
                    layer.set_shortcut_moe(moe, carry, CARRY_TOKENS);
                } else {
                    layer.set_shortcut_carry_in(carry, CARRY_TOKENS);
                }
                layers.push(Box::new(layer));
            }

            if (l + 1) % 4 == 0 || l == ckpt_layers - 1 {
                let free = gpu.free_memory().unwrap_or(0);
                tracing::info!(
                    "LongCat L{}/{ckpt_layers} — {:.1} GB free",
                    l + 1,
                    free as f64 / 1e9
                );
            }
        }
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight").context("longcat: embedding")
    }

    fn load_ngram_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        max_tokens: usize,
    ) -> Result<Option<crate::layers::ngram_embed::NgramEmbedding>> {
        ngram::build(store, config, gpu, max_tokens)
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight").context("longcat: final norm")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else if config.tie_word_embeddings {
            dense(store, "model.embed_tokens.weight")
        } else {
            anyhow::bail!("longcat: lm_head.weight not found")
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // The checkpoint ships `model.mtp.*`, but the MTP head shape is not
        // the Qwen-style one Atlas builds. Ignored (matches HF's own
        // `_keys_to_ignore_on_load_unexpected = [r"model\\.mtp.*"]`).
        Ok(None)
    }

    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionEncoder>> {
        Ok(None)
    }
}

/// One sublayer's dense SwiGLU FFN (`mlps.{s}`), NVFP4-quantized at load.
mod ffn;
use ffn::{build_dense_ffn, build_shortcut_moe};
