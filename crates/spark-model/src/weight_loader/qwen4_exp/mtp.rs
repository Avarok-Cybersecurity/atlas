// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3.8-Flash-Next` (`qwen4_exp`) MTP draft-module loader — Track B.
//!
//! Atlas has two MTP tracks. Track A fills in `MtpWeights` and lets
//! `MtpHead` build a hand-rolled GQA drafter; Track B builds a bespoke module
//! out of the model's OWN layer type and surfaces it to a proposer. DeepSeek-V4
//! is Track B, and so is this: `MtpWeights` demands a fused `fc [h, 2h]` where
//! this ships `fc_embedding` + `fc_hidden`, a `pre_fc_norm_hidden [h]` where
//! this ships `[hc_mult * h]`, and `input_layernorm` / `post_attn_layernorm` /
//! `norm` fields this architecture does not have at all.
//!
//! The body is the existing per-layer construction from `qwen4_exp.rs` replayed
//! ONCE with `lp = "mtp.layers.0"` and the full-attention arm only. Every
//! helper it calls already takes `lp: &str`; nothing shared is modified, and
//! the prefix comes down as a literal from here rather than by teaching
//! `ModelConfig::layer_prefix` a top-level `mtp.*` namespace it cannot express.
//!
//! **Nothing runs this yet.** There is no proposer, so `has_proposer()` stays
//! false, `use_speculative` stays false, and `step_mtp` is never reached.
//! `--speculative` on qwen4_exp loads and audits this module and says so.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{MTP_LAYER_PREFIX, MtpExpertLayout, MtpNamespaceReport, audit_mtp_namespace};
use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::HcHeadWeights;
use crate::weight_map::{DenseWeight, dense_auto};

/// A loaded qwen4_exp MTP draft module: the reused full-attention body (mHC +
/// QSA + its own 512-expert MoE) plus the MTP-specific combiner and the
/// module's own stream-collapsing mixer.
///
/// # Invariant: the body has NO KV cache of its own yet
///
/// `body` is built with `attn_idx = 0`. That index is used as a RAW slice
/// index into whatever `PagedKvCache` it is handed
/// (`paged_impl.rs`: `self.layers[layer_idx]`), so decoding this body against
/// the main model's pool would read and write full-attention layer 0's K/V.
/// The module must ONLY ever be decoded against its own single-layer cache,
/// which does not exist yet. That is why it is stored in a dedicated
/// `Option` field on the model, is never pushed into `model.layers`, and is
/// reachable from no main-model path.
//
// Consumed by the forthcoming `Qwen4ExpMtpHead` proposer — the same deferral
// `DeepseekV4MtpModule` carried between its loader and its head landing.
#[allow(dead_code)]
pub struct Qwen4ExpMtpModule {
    /// Reused `Qwen3AttentionLayer` body built from the `mtp.layers.0` prefix.
    pub body: Box<dyn TransformerLayer>,
    /// The module's OWN collapse-only mixer (`mtp.hyper_connection_mixer`).
    /// The body was built at `layer_idx = num_hidden_layers`, so both
    /// `is_first_model_layer` and `is_last_model_layer` are false and it runs
    /// the MIDDLE mHC mixing only — the proposer owns both ends.
    pub hc_head: Option<HcHeadWeights>,
    /// RMSNorm on the next-token embedding, `[hidden]`.
    pub pre_fc_norm_embedding: DenseWeight,
    /// RMSNorm on the incoming hidden state, `[hc_mult * hidden]` — the
    /// FOUR-STREAM mHC highway, not a collapsed hidden. See the audit.
    pub pre_fc_norm_hidden: DenseWeight,
    /// `[hidden, hidden]` projection of the normed embedding.
    pub fc_embedding: DenseWeight,
    /// `[hidden, hidden]` projection of the normed hidden state.
    pub fc_hidden: DenseWeight,
    /// Which routed-expert layout the audit resolved. Recorded so the
    /// proposer's memory accounting does not have to re-derive it.
    pub expert_layout: MtpExpertLayout,
}

/// Load the MTP draft module, or `Ok(None)` when none is declared.
///
/// # Ordering is a correctness contract
///
/// The audit runs BEFORE the first `build_moe`. `load_moe_qwen35` `gpu.free()`s
/// the fused source tensors once it has quantized them, leaving the store
/// holding freed pointers for `experts.gate_up_proj` / `experts.down_proj`; an
/// audit that ran afterwards would be reading dangling allocations.
///
/// # An absent-but-declared MTP is an ERROR, not a skip
///
/// Every `store.contains` in the helpers this reuses is a GATE, not a check:
/// `attach_qsa` skips silently, `load_moe_qwen35`'s fused detection just picks
/// the other branch. So if the tensors were declared and are not in the store —
/// the `skip_mtp` upload filter still on, or a GGUF/RDMA loader path — this
/// returns `Err` naming the cause rather than logging "no MTP in checkpoint"
/// and loading nothing.
pub fn load_qwen4_exp_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Qwen4ExpMtpModule>> {
    if config.num_mtp_modules == 0 {
        return Ok(None);
    }
    // The combiner's `fc_embedding` is the cheapest MTP-only marker (the role
    // `mtp.0.enorm.weight` plays for DeepSeek-V4): it exists in no other
    // namespace and is not something a shard split can move.
    if !store.contains("mtp.fc_embedding.weight") {
        anyhow::bail!(
            "qwen4_exp: config declares num_mtp_modules={} but the weight store \
             holds no `mtp.fc_embedding.weight`. The tensors were NOT loaded. \
             Most likely `skip_mtp` (spark-server serve_phases/weights.rs) is \
             still filtering `mtp.*` out at upload — it only lets them through \
             under `--speculative` or ATLAS_QWEN4EXP_MTP=1 — or the checkpoint \
             came in through the GGUF or RDMA loader path, neither of which \
             carries the mtp namespace. Refusing rather than reporting \"no MTP \
             in checkpoint\" and loading nothing.",
            config.num_mtp_modules,
        );
    }

    // BEFORE anything is built. See the ordering note above.
    let report: MtpNamespaceReport = audit_mtp_namespace(store, config);
    report.log();
    report.ensure_loadable(config)?;

    let h = config.hidden_size;
    let lp = MTP_LAYER_PREFIX;
    let variant = crate::weight_map::detect_nvfp4_variant(store, config);
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let free_now = |g: &dyn GpuBackend| g.free_memory().unwrap_or(0) as u64;
    let f0 = free_now(gpu);

    // ── (1) MoE. `build_moe` is called verbatim: its `is_fused` detection
    // handles BOTH shipped layouts with no new code. On the fused-BF16
    // snapshot this uploads ~5.0 GB of BF16 experts that collapse to ~1.3 GB
    // NVFP4 and frees the sources; on the per-expert NVFP4 snapshot the
    // packed tensors pass straight through.
    let ffn = super::ffn::build_moe(store, lp, config, gpu, variant)
        .context("qwen4_exp MTP: MoE block")?;
    let f1 = free_now(gpu);

    // ── (2) Norm placeholders. Same as every main layer: this architecture
    // keeps its normalization inside the hyper-connection blocks, so there is
    // no `input_layernorm` / `post_attention_layernorm` tensor to load and a
    // ones-filled buffer keeps the shared arm's shape contract.
    let input_norm = super::ones_norm(h, gpu)?;
    let post_attn_norm = super::ones_norm(h, gpu)?;

    // ── (3) The body, full-attention arm only (`mtp.layer_types ==
    // ["full_attention"]`, refused by the config parser otherwise).
    //
    // `layer_idx = config.num_hidden_layers` is log-only here, and it is also
    // what makes `attach_hc` set is_first == is_last == false below.
    //
    // KvCacheDtype::Bf16 is passed EXPLICITLY rather than looked up in
    // `layer_kv_dtypes`, for two reasons: an out-of-range index there falls
    // back to Bf16 SILENTLY (qwen4_exp.rs), and QSA selection refuses anything
    // but plain BF16 KV, so a draft body carrying an indexer has no other
    // legal choice.
    //
    // `attn_idx = 0`: the module's future private single-layer cache. See the
    // struct invariant — this body must never be decoded against the main pool.
    let mut body =
        crate::weight_loader::qwen35::load_layers::attention_arms::build_full_attention_nvfp4(
            config.num_hidden_layers,
            store,
            lp,
            gpu,
            variant,
            config,
            h,
            absmax_k,
            quantize_k,
            stream,
            KvCacheDtype::Bf16,
            0,
            input_norm,
            post_attn_norm,
            ffn,
        )
        .context("qwen4_exp MTP: full-attention body")?;
    let f2 = free_now(gpu);

    // ── (4) mHC sites + the module's own mixer.
    let hc_head = if config.hc_mult > 0 {
        let (attn_site, ffn_site) =
            super::hc::load_layer_sites(store, lp, config).context("qwen4_exp MTP: mHC sites")?;
        let head = super::hc::load_head_at(store, "mtp.hyper_connection_mixer", config.hc_lowrank)
            .context("qwen4_exp MTP: hyper_connection_mixer")?;
        // `head = None` on the BODY: at layer_idx == num_hidden_layers both
        // is_first_model_layer and is_last_model_layer are false, so the body
        // runs middle-only mixing and never calls hc_head. The head is
        // surfaced on the module instead, for the proposer to collapse with —
        // exactly the DeepSeek-V4 device.
        super::aux::attach_hc(
            &mut body,
            config.num_hidden_layers,
            attn_site,
            ffn_site,
            None,
            config,
        )?;
        Some(head)
    } else {
        None
    };

    // ── (5) QSA indexer. The tensors exist in `mtp.*`, and a dense body past
    // `indexer_budget` is not the reference model. It also exercises the three
    // audited indexer tensors end to end.
    super::aux::attach_qsa(&mut body, config.num_hidden_layers, lp, store, config, gpu)?;

    // ── (6) NO PLE. `mtp.*` carries zero PLE tensors, and PLE only ever
    // attaches to a `Qwen3SsmLayer` while this body is a `Qwen3AttentionLayer`.
    // `ple::load` would return None for this index anyway; not calling it is a
    // decision, not an accident.

    // ── (7) The four MTP-only combiner tensors. Plain BF16 in both shipped
    // checkpoints; `dense_auto` survives a future FP8 re-quant. Re-asserted
    // against the store's own shapes at load — the audit already checked them,
    // and this is the line that keeps that true if the audit is ever loosened.
    let dense_checked = |name: &str, want: &[usize]| -> Result<DenseWeight> {
        let got = &store.get(name)?.shape;
        anyhow::ensure!(
            got == want,
            "qwen4_exp MTP: {name} has shape {got:?}, expected {want:?}"
        );
        dense_auto(store, name, gpu).with_context(|| format!("qwen4_exp MTP: {name}"))
    };
    let hc_h = config.hc_mult * h;
    let pre_fc_norm_embedding = dense_checked("mtp.pre_fc_norm_embedding.weight", &[h])?;
    let pre_fc_norm_hidden = dense_checked("mtp.pre_fc_norm_hidden.weight", &[hc_h])?;
    let fc_embedding = dense_checked("mtp.fc_embedding.weight", &[h, h])?;
    let fc_hidden = dense_checked("mtp.fc_hidden.weight", &[h, h])?;

    let f3 = free_now(gpu);
    // ⚠ SIGNED deltas. `saturating_sub` on unsigned free-memory readings
    // clamps to 0.00 GB precisely on the arm that matters: with FUSED experts
    // `build_moe` frees ~5 GB of already-resident BF16 sources while
    // allocating ~1.3 GB of NVFP4, so free memory RISES across the call and an
    // unsigned delta reports "0.00 GB" — which reads as "the MTP block is
    // free" when it is nothing of the sort. Negative = net freed.
    let gb = |a: u64, b: u64| (a as i128 - b as i128) as f64 / 1e9;
    tracing::info!(
        "qwen4_exp MTP module loaded ({:?} experts): MoE {:+.2} GB, attention body \
         {:+.2} GB, mHC+QSA+combiner {:+.2} GB, {:+.2} GB net. Deltas are free-memory \
         differences, so a NEGATIVE figure means that stage freed more than it \
         allocated (the fused arm frees its BF16 sources) — it is NOT the resident \
         cost of the module. NOTE: no proposer is wired — this module is loaded and \
         audited, nothing decodes it.",
        report.expert_layout,
        gb(f0, f1),
        gb(f1, f2),
        gb(f2, f3),
        gb(f0, f3),
    );

    Ok(Some(Qwen4ExpMtpModule {
        body,
        hc_head,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc_embedding,
        fc_hidden,
        expert_layout: report.expert_layout,
    }))
}
