// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash Multi-Token-Prediction (MTP) draft-module loader.
//!
//! `nvidia/DeepSeek-V4-Flash-NVFP4` ships one MTP module
//! (`num_nextn_predict_layers = 1`, 1575 tensors). Its body is structurally a
//! main V4 layer — MLA attention, manifold-constrained hyper-connections (mHC),
//! and a 256-expert NVFP4 MoE, full attention (no compressor) — stored under the
//! `mtp.0.*` prefix. On top of the body sit the MTP-specific pieces:
//!
//!   h_in = e_proj(rmsnorm(embed(token), enorm)) + h_proj(rmsnorm(h_prev, hnorm))
//!   h_out = body(h_in)                          // reused V4 layer forward
//!   logits = lm_head(rmsnorm(h_out, norm))      // SHARED embed + lm_head
//!
//! The body is built by reusing [`super::assemble::assemble_layer`] with the
//! `mtp.0` prefix; only the combiner (`enorm`/`hnorm` + `e_proj`/`h_proj`) and the
//! final `norm` are MTP-specific. The token embedding and lm_head are shared with
//! the parent model and supplied to the proposer at build time (not duplicated in
//! `mtp.*`). See `docs/deepseek_v4_mtp_support.md`.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::qwen3_attention::HcHeadWeights;
use crate::weight_map::{DenseWeight, dense_auto};

/// A loaded DeepSeek-V4 MTP draft module: the reused V4 transformer body plus the
/// MTP-specific input combiner and final norm. Embedding + lm_head are shared with
/// the parent model and supplied at proposer-build time.
//
// Consumed by the (forthcoming, runtime-verified) `DeepseekV4MtpHead` proposer.
#[allow(dead_code)]
pub struct DeepseekV4MtpModule {
    /// Reused V4 layer body (MLA + mHC + MoE), built from the `mtp.0` prefix.
    pub body: Box<dyn TransformerLayer>,
    /// RMSNorm applied to the next-token embedding before `e_proj`.
    pub enorm: DenseWeight,
    /// RMSNorm applied to the previous hidden state before `h_proj`.
    pub hnorm: DenseWeight,
    /// Projects the normed embedding `[hidden, hidden]`; summed with `h_proj`.
    pub e_proj: DenseWeight,
    /// Projects the normed hidden state `[hidden, hidden]`; summed with `e_proj`.
    pub h_proj: DenseWeight,
    /// Final RMSNorm applied before the shared lm_head.
    pub norm: DenseWeight,
    /// The MTP module's OWN head hyper-connection (`mtp.0.hc_head_*`). The body
    /// was built with `layer_idx = num_hidden_layers`, so its `decode_inner_hc`
    /// runs the MIDDLE mHC mixing only — it does NOT call `hc_head`. The
    /// proposer must collapse `hc_streams → h_out` itself after `body.decode`,
    /// so we surface the head weights here (a clone of the pointers already
    /// handed to the body) rather than reaching into the private body. `None`
    /// when `hc_mult == 0` (no mHC).
    pub hc_head: Option<HcHeadWeights>,
}

/// Loads the DeepSeek-V4 MTP draft module if the checkpoint contains MTP weights.
///
/// Returns `Ok(None)` (no-op) when MTP is disabled (`num_mtp_modules == 0`) or the
/// checkpoint ships no MTP tensors (e.g. the RedHat NVFP4-FP8 re-quant) — so it is
/// safe to call unconditionally on the V4 load path.
pub fn load_v4_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Option<DeepseekV4MtpModule>> {
    if config.num_mtp_modules == 0 {
        return Ok(None);
    }
    // The combiner's `enorm` is the cheapest MTP-only marker: present iff the
    // checkpoint actually ships MTP weights.
    if !store.contains("mtp.0.enorm.weight") {
        tracing::info!(
            "DeepSeek-V4: num_mtp_modules={} but no mtp.0.* tensors in checkpoint — MTP disabled",
            config.num_mtp_modules
        );
        return Ok(None);
    }

    let prefix = "mtp.0";
    let ap = format!("{prefix}.attn");
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };

    // ── Body pre-loads ──
    // Mirrors the V4-Flash branch of `load_all_layers` (o_lora_rank > 0): direct
    // KV projection + grouped low-rank O projection, so the V3 absorption tensors
    // (w_uk_t / w_uv / w_qk_absorbed / block-diagonals) are NULL/unused.
    let input_norm = dense_auto(store, &format!("{prefix}.attn_norm.weight"), gpu)?;
    let post_attn_norm = dense_auto(store, &format!("{prefix}.ffn_norm.weight"), gpu)?;
    let wq_a = dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?;
    let wq_b = dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?;
    let q_a_norm = dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?;
    let wkv_a = dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?;
    let kv_a_norm = dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?;
    let wo_a = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
    let wo_b = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;

    // The MTP module carries its OWN head hyper-connection (`mtp.0.hc_head_*`),
    // distinct from the model-level `hc_head` shared by the main layers.
    let hc_head = if config.hc_mult > 0 {
        let hc = config.hc_mult;
        let hc_dim = hc * config.hidden_size;
        let head_fn = super::assemble::load_hc_f32(
            store,
            &[format!("{prefix}.hc_head_fn")],
            hc * hc_dim,
            gpu,
        )?;
        let head_base =
            super::assemble::load_hc_f32(store, &[format!("{prefix}.hc_head_base")], hc, gpu)?;
        let head_scale =
            super::assemble::load_hc_f32(store, &[format!("{prefix}.hc_head_scale")], 1, gpu)?;
        Some(HcHeadWeights {
            hc_fn: head_fn,
            hc_base: head_base,
            hc_scale: head_scale,
        })
    } else {
        None
    };

    let mut yarn_inv_freq = DevicePtr::NULL;
    yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

    // Build the body by reusing `assemble_layer` with the `mtp.0` prefix.
    // layer_idx = num_hidden_layers ⇒ compress_ratios.get()/hash-layer/kv-dtype
    // all fall to safe defaults (no compressor, no hash routing, bf16 KV).
    let body = super::assemble::assemble_layer(
        config.num_hidden_layers,
        prefix,
        true, // force_all_experts — MTP draft runs no-EP on rank 0, needs all experts
        input_norm,
        post_attn_norm,
        wq_a,
        None, // wq_a_nvfp4 — V4 attn is FP8/BF16, not NVFP4
        wq_b,
        None, // wq_b_nvfp4
        q_a_norm,
        wkv_a,
        None, // wkv_a_nvfp4
        null, // wkv_b — unused for V4-Flash
        kv_a_norm,
        wo_b, // o_dense
        None, // o_nvfp4
        null, // w_uk_t
        null, // w_uv
        null, // wq_b_rope
        null, // w_qk_absorbed
        null, // w_uk_block_diag
        null, // w_uv_block_diag
        yarn_inv_freq,
        wo_a,
        hc_head.clone(),
        store,
        config,
        gpu,
        layer_kv_dtypes,
    )?;

    // ── MTP-specific combiner + final norm ──
    // enorm/hnorm/norm are HF-vanilla RMSNorms — loaded exactly, normalized by
    // `rms_norm_vanilla` (see `DeepseekV4MtpHead::rms_norm_k`).
    // e_proj/h_proj are FP8 block-scaled linears in the checkpoint (dense_auto dequants).
    let enorm = dense_auto(store, &format!("{prefix}.enorm.weight"), gpu)?;
    let hnorm = dense_auto(store, &format!("{prefix}.hnorm.weight"), gpu)?;
    let norm = dense_auto(store, &format!("{prefix}.norm.weight"), gpu)?;
    let e_proj = dense_auto(store, &format!("{prefix}.e_proj.weight"), gpu)?;
    let h_proj = dense_auto(store, &format!("{prefix}.h_proj.weight"), gpu)?;

    tracing::info!(
        "DeepSeek-V4 MTP module loaded: reused V4 body (MLA + mHC + 256-expert MoE) \
         + combiner (enorm/hnorm + e_proj/h_proj) + final norm"
    );

    Ok(Some(DeepseekV4MtpModule {
        body,
        enorm,
        hnorm,
        e_proj,
        h_proj,
        norm,
        hc_head,
    }))
}
