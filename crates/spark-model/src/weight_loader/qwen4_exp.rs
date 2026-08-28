// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3.8-Flash-Next` (`qwen4_exp`) weight loader. Port tracked in Avarok
//! #753.
//!
//! **The mHC highway runs; PLE does not.** The low-rank multi-hyperconnection
//! residual is wired on all 48 layers and validated against the reference
//! (`ops/hyper_connection_lowrank_tests.rs`, PLAN.md phases A-C). What is
//! still missing:
//!
//! * **PLE n-gram injection** — refused at LOAD unless
//!   `ATLAS_QWEN4EXP_NO_PLE=1`, because skipping it does not crash and does
//!   not look wrong. It produces fluent text from a model missing an input.
//! * **The QSA indexer** — provably inert at or below `indexer_budget`, which
//!   is the context this fits today; required above it, and refused there.
//!   See PLAN.md §1.5.
//! * **Batched / multi-sequence decode** — refused by name; v1 is C=1.
//!
//! WHY THIS IS MOSTLY qwen35's LOADER. Qwen3.8-Flash-Next and Qwen3.6-35B-A3B
//! share far more than the version numbers suggest: 3:1 GDN/full-attention
//! interleave, MoE with a shared expert, gated attention, mRoPE, a ViT tower,
//! vocab 248320, rope_theta 1e7, head_dim 256, partial rotary 0.25, and the
//! same GDN key geometry. Critically, `load_ssm_qwen35` already reads
//! `in_proj_qkv` and `in_proj_z` as SEPARATE tensors and concatenates them —
//! which is exactly this model's layout, not a coincidence to be re-derived.
//! So the GDN and full-attention arms are called directly, with
//! `config.weight_prefix = "model.language_model"` making
//! `config.layer_prefix(i)` yield the real keys.
//!
//! WHAT IS GENUINELY DIFFERENT, and why each needs care:
//!
//! 1. **There are no per-layer norms.** No `input_layernorm`, no
//!    `post_attention_layernorm`, no final `model.norm`. Normalization lives
//!    inside the hyper-connection blocks as `hc_norm [hc_mult*hidden]`, and
//!    the model-level `hyper_connection_mixer` — which collapses the streams
//!    back to one before `lm_head` — carries the final norm. A loader that
//!    "helpfully" defaults these would be inventing weights.
//! 2. **mHC is 4 residual streams**, mixed low-rank (rank 320). Atlas's mHC
//!    plumbing is DeepSeek-V4's, whose mixer is Sinkhorn-normalized — same
//!    stream layout, different math.
//! 3. **A QSA indexer** on the 12 full-attention layers.
//! 4. **PLE n-gram injection** at one layer, off a ~320M-row table served
//!    from NVMe rather than resident.

use anyhow::{Context, Result};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

// `aux_sites`, NOT `aux`: bare `aux` is a RESERVED filename on Windows
// (CON/PRN/AUX/NUL...) — git checkout of `aux.rs` fails with "invalid
// path" on every Windows runner, which killed the release-matrix builds.
#[path = "qwen4_exp/aux_sites.rs"]
mod aux;
/// Offline safetensors-header readers. Test-only: they exist so a checkpoint's
/// layout can be checked without uploading it.
#[cfg(test)]
mod ckpt_header;
mod ffn;
mod hc;
mod mtp;
mod ple;
mod probe;
mod probe_mtp;

#[cfg(test)]
pub use ckpt_header::ple_shard_layout;
pub use mtp::{Qwen4ExpMtpModule, load_qwen4_exp_mtp_module};
pub use probe::audit_namespace;
pub use probe_mtp::{MTP_LAYER_PREFIX, MtpExpertLayout, MtpNamespaceReport, audit_mtp_namespace};

pub struct Qwen4ExpWeightLoader;

impl ModelWeightLoader for Qwen4ExpWeightLoader {
    fn supports_tp(&self) -> bool {
        // Not attempted. mHC would need the stream buffer sharded alongside
        // every projection, and the PLE row cache is a single-device arena.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let report = audit_namespace(store, config);
        report.log();
        report.ensure_loadable()?;

        let h = config.hidden_size;
        let variant = crate::weight_map::detect_nvfp4_variant(store, config);
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        tracing::info!(
            "Qwen3.8-Flash-Next: {} layers ({} GDN + {} full attention), \
             {} experts top-{}, hc {} streams x rank {}, indexer budget {}, \
             PLE at {:?}; NVFP4 variant {:?}",
            config.num_hidden_layers,
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            config.num_experts,
            config.num_experts_per_tok,
            config.hc_mult,
            config.hc_lowrank,
            config.index_topk,
            config.ple_layer_ids,
            variant,
        );

        // The model-level mixer collapses the streams before `lm_head` and
        // carries the FINAL NORM (this checkpoint has no `model.norm.weight`).
        // Replicated onto every layer; only the last one consumes it.
        let hc_head = if config.hc_mult > 0 {
            Some(hc::load_head(store, config)?)
        } else {
            None
        };

        // PLE scratch is sized once, for the largest prefill CHUNK a pass can
        // present — not the model's context.
        //
        // Deliberately NOT `config.max_position_embeddings`: `--max-seq-len`
        // is never written back into it, so on this model that field is the
        // architectural 262144 and any clamp of it over-allocates. The six
        // buffers total `tokens * 10240 * 14` bytes, which at 8192 is 1.26 GB
        // — enough to push a 94.6 GB resident model past the util pledge on a
        // box with 2.7 GB of headroom, which is exactly what it did.
        //
        // 2048 covers the chunk sizes this model runs at; a larger chunk gets
        // the layer's refusal, which names this variable, rather than a
        // silent overrun.
        let max_ple_tokens: usize = std::env::var("ATLAS_PLE_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);
        // With PLE disabled for bisection, skip the 21 MB arena and the
        // 128-shard open entirely rather than building what we will not run.
        let ple_off = std::env::var("ATLAS_QWEN4EXP_NO_PLE").as_deref() == Ok("1");
        // GDN projections stay BF16 by DEFAULT on this checkpoint. Measured,
        // both arms, same prompt, util 0.85 / 16K / bf16 KV:
        //
        //                      requantized NVFP4      BF16 (default)
        //   layer construction   7.43 GB / 154.8 MB/L   1.39 GB / 28.9 MB/L
        //   attn/GDN arms        7.34 GB                1.32 GB
        //   pre-KV               95.8 GB                90.0 GB
        //   KV budget            3.9 GB / 172144 tok    9.7 GB / 424464 tok
        //   decode               2.207 tok/s            2.188 tok/s
        //
        // 6.04 GB back for ~1% of decode. GDN weight bandwidth is simply not
        // what bounds decode here at C=1 — something else dominates — so the
        // usual w4a16-is-faster argument does not apply yet. Revisit if that
        // changes.
        //
        // And it is not only a memory lever: ONLY the routed experts are
        // quantized in this checkpoint. The GDN projections ship BF16, so
        // requantizing them was a lossy round trip we chose, on 36 of 48
        // layers. `=0` opts back into it for A/B.
        let bf16_gdn = std::env::var("ATLAS_QWEN4EXP_BF16_GDN").as_deref() != Ok("0");
        tracing::info!(
            "GDN projections: {} on the {} linear-attention layers",
            if bf16_gdn {
                "BF16 as shipped (no runtime NVFP4 requantization)"
            } else {
                "requantized to NVFP4 (ATLAS_QWEN4EXP_BF16_GDN=0)"
            },
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
        );

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        // Per-arm memory attribution. Layer construction costs 7.41 GB on this
        // model (154.5 MB/layer, measured) on top of the 85.2 GB of uploaded
        // shards, and nothing said which arm spent it. Summed here and logged
        // once, so the answer is read rather than guessed.
        let (mut moe_bytes, mut arm_bytes, mut hc_bytes) = (0u64, 0u64, 0u64);
        let free_now = |g: &dyn GpuBackend| g.free_memory().unwrap_or(0) as u64;

        for i in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(i);
            let f0 = free_now(gpu);
            let ffn = ffn::build_moe(store, &lp, config, gpu, variant)?;
            let f1 = free_now(gpu);
            moe_bytes += f0.saturating_sub(f1);

            // Norm placeholders — see module docs. This model keeps its
            // normalization inside the hyper-connection blocks, so there is
            // no per-layer norm tensor to load. Ones-filled buffers keep the
            // shared arms' shape contract without inventing a scale, and they
            // are unreachable at runtime because the mHC forward refuses
            // before any layer executes.
            let input_norm = ones_norm(h, gpu)?;
            let post_attn_norm = ones_norm(h, gpu)?;

            let layer = match config.layer_types[i] {
                LayerType::LinearAttention if bf16_gdn => {
                    // Keep the GDN projections BF16 instead of requantizing
                    // them to NVFP4 at load.
                    //
                    // Two reasons, and the second is the interesting one.
                    // (1) MEMORY: the requantization is where this model's
                    // build spends its 7.34 GB (152.8 MB/layer, measured —
                    // the MoE costs zero because its experts ship NVFP4 and
                    // upload straight through).
                    // (2) PRECISION: these tensors ship as BF16 in this
                    // checkpoint. Only the routed experts are quantized. So
                    // BF16 -> NVFP4 here is a lossy round trip we chose, not
                    // one the checkpoint forced, and it lands on the GDN
                    // projections of 36 of 48 layers.
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_dense_bf16(
                        i, store, &lp, gpu, variant, config, h,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i} (BF16)"))?
                }
                LayerType::LinearAttention => {
                    crate::weight_loader::qwen35::load_layers::linear_attn_arms::build_linear_attention_nvfp4(
                        store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: GDN layer {i}"))?
                }
                LayerType::FullAttention => {
                    let kv_dtype = layer_kv_dtypes
                        .get(attn_idx)
                        .copied()
                        .unwrap_or(KvCacheDtype::Bf16);
                    let l = crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
                        i, store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        kv_dtype, attn_idx, input_norm, post_attn_norm, ffn,
                    )
                    .with_context(|| format!("qwen4_exp: full-attention layer {i}"))?;
                    attn_idx += 1;
                    l
                }
                other => anyhow::bail!(
                    "qwen4_exp layer {i} has type {other:?}; this architecture is \
                     only linear_attention / full_attention"
                ),
            };
            let f2 = free_now(gpu);
            arm_bytes += f1.saturating_sub(f2);

            // mHC: two sites per layer wrapping attention and the MoE. The
            // residual this model carries is `hc_mult * hidden` wide, so
            // without these the layer would run on a stream it never mixed.
            let mut layer = layer;
            if config.hc_mult > 0 {
                let (attn, ffn) = hc::load_layer_sites(store, &lp, config)?;
                aux::attach_hc(&mut layer, i, attn, ffn, hc_head.clone(), config)?;
            }
            aux::attach_qsa(&mut layer, i, &lp, store, config, gpu)?;
            // PLE lands on exactly one layer, which on this checkpoint is a
            // GDN one. `load` returns None for every other layer.
            let ple_layer = if ple_off {
                None
            } else {
                ple::load(store, config, i, max_ple_tokens, gpu)?
            };
            if let Some(p) = ple_layer {
                aux::attach_ple(&mut layer, i, p)?;
            }
            layers.push(layer);
            hc_bytes += f2.saturating_sub(free_now(gpu));
        }
        tracing::info!(
            "qwen4_exp layer construction: MoE {:.2} GB ({:.1} MB/layer), \
             attn/GDN arms {:.2} GB ({:.1} MB/layer), mHC+PLE {:.2} GB",
            moe_bytes as f64 / 1e9,
            moe_bytes as f64 / 1e6 / config.num_hidden_layers as f64,
            arm_bytes as f64 / 1e9,
            arm_bytes as f64 / 1e6 / config.num_hidden_layers as f64,
            hc_bytes as f64 / 1e9,
        );

        // PLE is wired (PLAN.md phase D) and validated against the reference
        // in `ops/ple_tests.rs`. The escape hatch stays, inverted: it now
        // DISABLES a mechanism that is present, for bisecting, and says so.
        if !config.ple_layer_ids.is_empty()
            && std::env::var("ATLAS_QWEN4EXP_NO_PLE").as_deref() == Ok("1")
        {
            tracing::warn!(
                "ATLAS_QWEN4EXP_NO_PLE=1: PLE n-gram injection at model layer {} \
                 is DISABLED. Output is wrong by construction — this arm exists \
                 to bisect the mHC spine, nothing else.",
                config.ple_layer_ids[0].saturating_sub(1),
            );
        }
        tracing::info!(
            "Qwen3.8-Flash-Next loaded {} layers with the mHC highway live on \
             all of them ({} GDN + {} full-attention).",
            layers.len(),
            layers.len()
                - config
                    .layer_types
                    .iter()
                    .filter(|t| **t == LayerType::FullAttention)
                    .count(),
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
        );
        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: embedding")
    }

    /// **This model has no final norm tensor.**
    ///
    /// There is no `model.norm.weight` anywhere in the checkpoint. The
    /// model-level `hyper_connection_mixer` — which collapses the `hc_mult`
    /// residual streams back to a single hidden state before `lm_head` —
    /// carries `hc_norm [hc_mult*hidden]`, and that IS the final
    /// normalization. It is the wrong width to stand in here (10240 against
    /// 2560), and applying it as though it were a plain final norm would be
    /// inventing math.
    ///
    /// A ones-filled buffer keeps the shape contract so the footprint can be
    /// measured at load. It is unreachable at inference because the mHC
    /// forward refuses first; if that ever stops being true, this is the
    /// first thing to fix.
    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        aux::final_norm_placeholder(store, config, gpu)
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        anyhow::ensure!(
            config.tie_word_embeddings,
            "qwen4_exp: no lm_head.weight and tie_word_embeddings is false"
        );
        let pfx = embed_prefix(config);
        dense(store, &format!("{pfx}.embed_tokens.weight")).context("qwen4_exp: tied lm_head")
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        // The ViT tower IS the Qwen3-VL family shape the qwen35 loader
        // already reads: 27 blocks under `model.visual.*`, patch 16,
        // spatial-merge 2, plain BF16 weights (no quant tensors under
        // `visual` in this checkpoint), empty deepstack list. The
        // qwen3.8-flash-next kernel target ships its own vision_encoder.cu
        // shadow, so kernels resolve per-target as usual.
        crate::weight_loader::qwen35::Qwen35WeightLoader.load_vision_encoder(store, config, gpu)
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // NOT a deferral any more — a routing decision, the same one
        // `DeepSeekV4WeightLoader::load_mtp_weights` makes.
        //
        // `MtpWeights` is STRUCTURALLY the wrong container for this family:
        //   * it demands a fused `fc [h, 2h]` over `concat(norm(embed),
        //     norm(hidden))`; this checkpoint ships two square projections,
        //     `mtp.fc_embedding` and `mtp.fc_hidden`;
        //   * it types `pre_fc_norm_hidden` as `[h]`; this ships
        //     `[hc_mult * h]` (10240 against 2560), because the incoming
        //     hidden state is the four-stream mHC highway, not a collapsed
        //     one;
        //   * it requires `input_layernorm`, `post_attn_layernorm` and a final
        //     `norm`, none of which exist in this architecture — normalization
        //     lives inside the hyper-connection blocks;
        //   * and `MtpHead`'s forward hard-codes a single pre-norm residual
        //     stream, which an mHC bracket does not have.
        //
        // So qwen4_exp is Track B, and the real loader is
        // `mtp::load_qwen4_exp_mtp_module` — a bespoke `Qwen4ExpMtpModule`
        // built out of this file's own per-layer helpers at
        // `lp = "mtp.layers.0"`. Do not re-derive this refusal; extend that.
        //
        // NO PREFLIGHT CHANGE IS NEEDED, checked: `check_layer_count` parses
        // `mtp.layers.0` as index 0 (it splits on the first `.layers.`), so
        // `max_layer_idx` stays 47, `47 + 1 > 48` is false, and neither
        // `check_mtp_consumability` nor its `MTP_SUPPORTED_MODEL_TYPES`
        // allowlist is ever reached. `check_expert_count` sees experts 0..511
        // against `num_experts = 512` and passes.
        Ok(None)
    }
}

/// A ones-filled `[n]` BF16 norm scale.
///
/// BF16 1.0 is `0x3F80`, so the buffer cannot be produced with `memset`.
fn ones_norm(n: usize, gpu: &dyn GpuBackend) -> Result<DenseWeight> {
    let host: Vec<u8> = std::iter::repeat_n([0x80u8, 0x3Fu8], n).flatten().collect();
    let ptr = gpu.alloc(host.len())?;
    gpu.copy_h2d(&host, ptr)?;
    Ok(DenseWeight { weight: ptr })
}

/// `model.language_model` for the multimodal layout, `model` otherwise.
fn embed_prefix(config: &ModelConfig) -> String {
    if config.weight_prefix.is_empty() {
        "model".to_string()
    } else {
        config.weight_prefix.clone()
    }
}

/// The model-level hyper-connection mixer that collapses the residual streams.
fn mixer_prefix(config: &ModelConfig) -> String {
    format!("{}.hyper_connection_mixer", embed_prefix(config))
}
