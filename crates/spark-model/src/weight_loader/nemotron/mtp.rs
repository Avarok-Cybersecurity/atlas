// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H MTP draft-module loader (Nemotron-3.5 Lightning 30B).
//!
//! transformers-5.x Nemotron-H ships a DeepSeek-style 1-step MTP head under
//! the `mtp.layers.*` prefix (`num_nextn_predict_layers = 1`,
//! `mtp_layers_block_type = ["attention", "moe"]`):
//!
//! ```text
//!   mtp.layers.0.{enorm,hnorm,eh_proj}.weight       — combiner (embed ‖ hidden)
//!   mtp.layers.0.norm.weight                        — attention block input norm
//!   mtp.layers.0.mixer.{q,k,v,o}_proj.weight        — ungated GQA 32q/2kv hd128, NoPE, BF16
//!   mtp.layers.1.norm.weight                        — MoE block input norm
//!   mtp.layers.1.mixer.gate.{weight,e_score_correction_bias}
//!   mtp.layers.1.mixer.experts.{0..127}.{up,down}_proj.weight   — relu² MLP, BF16
//!   mtp.layers.1.mixer.shared_experts.{up,down}_proj.weight
//!   mtp.layers.1.final_layernorm.weight             — pre-LM-head norm
//! ```
//!
//! The two blocks are byte-identical in SHAPE to the backbone's attention and
//! MoE layers (same `mixer.` naming, same 128-expert top-6 sigmoid router with
//! correction bias, same relu² up/down experts), so this loader reuses
//! `load_nemotron_attention` / `load_nemotron_moe` with the `mtp.layers.N`
//! prefix and assembles real `Qwen3AttentionLayer` / `NemotronMoeLayer`
//! instances. The proposer (`layers::nemotron_mtp`) then delegates its block
//! forwards to `TransformerLayer::decode`, exactly like the DeepSeek-V4 MTP
//! proposer delegates to its reused V4 body — no new kernels.
//!
//! Precision follows the backbone conventions: attention stays checkpoint-BF16
//! (the same reasoning as `ATLAS_NEMOTRON_BF16_ATTN` — the head is ~47 MB and
//! draft quality is what MTP acceptance lives on), MoE experts are
//! runtime-quantized BF16 → NVFP4 (the relu²-fused decode kernels speak NVFP4).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use crate::layers::{FfnComponent, NemotronMoeLayer, Qwen3AttentionLayer};
use crate::weight_map::{DenseWeight, dense, load_nemotron_attention, load_nemotron_moe};

/// Loaded Nemotron MTP draft module: DeepSeek-style combiner + one attention
/// block + one MoE block + final norm. The LM head and embedding are shared
/// with the target model (wired in by the proposer, not loaded here).
pub struct NemotronMtpModule {
    /// Embedding-branch RMS norm `[H]` (applied to `embed(token)`).
    pub enorm: DenseWeight,
    /// Hidden-branch RMS norm `[H]` (applied to the target's saved hidden).
    pub hnorm: DenseWeight,
    /// Combiner projection `[H, 2H]` BF16: `cat(enorm(e), hnorm(h)) → H`.
    pub eh_proj: DenseWeight,
    /// `mtp.layers.0`: ungated GQA attention block (own input norm inside;
    /// NoPE via `config.rotary_dim() == 0`; writes the proposer's OWN
    /// single-layer KV cache at `attn_layer_idx = 0`).
    pub attn_layer: Qwen3AttentionLayer,
    /// `mtp.layers.1`: relu² MoE block (own input norm inside; sigmoid
    /// router + e_score_correction_bias + routed_scaling_factor, shared
    /// experts — same machinery as the backbone MoE layers).
    pub moe_layer: NemotronMoeLayer,
    /// Final RMS norm `[H]` before the shared LM head.
    pub final_norm: DenseWeight,
}

/// Load the Nemotron MTP module from the `mtp.layers.*` tensors.
///
/// Returns `Ok(None)` when the config declares no MTP head or the checkpoint
/// does not ship the tensors (e.g. Nano 30B, which routes to the same kernel
/// target but has no `mtp.*` shard).
pub fn load_nemotron_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<NemotronMtpModule>> {
    if config.mtp_num_hidden_layers == 0 {
        return Ok(None);
    }
    if !store.contains("mtp.layers.0.eh_proj.weight") {
        tracing::info!(
            "Nemotron MTP: config declares num_nextn_predict_layers={} but the \
             checkpoint has no mtp.layers.* tensors — MTP disabled",
            config.mtp_num_hidden_layers,
        );
        return Ok(None);
    }
    if config.ep_world_size > 1 {
        // The draft MoE loads ALL experts locally (the drafter runs only on
        // rank 0); under EP the `is_local_expert` partition inside
        // `load_nemotron_moe` would silently null out the remote experts.
        tracing::warn!("Nemotron MTP: EP world size > 1 not supported — MTP disabled");
        return Ok(None);
    }

    // Runtime BF16 → NVFP4 quantization kernels for the MoE experts.
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();

    // Combiner + norms (all BF16 vectors, loaded exactly; the nemotron kernel
    // target's `rms_norm` override is the ABSOLUTE formula `x·w/rms`, matching
    // the checkpoint's vanilla-stored norm weights — same as the backbone).
    let enorm = dense(store, "mtp.layers.0.enorm.weight")?;
    let hnorm = dense(store, "mtp.layers.0.hnorm.weight")?;
    let eh_proj = dense(store, "mtp.layers.0.eh_proj.weight")?;
    let attn_norm = dense(store, "mtp.layers.0.norm.weight")?;
    let moe_norm = dense(store, "mtp.layers.1.norm.weight")?;
    let final_norm = dense(store, "mtp.layers.1.final_layernorm.weight")?;

    // ── Attention block (mtp.layers.0) ──
    // Lightning ships the MTP head unquantized (BF16, no weight_scale), so
    // `load_nemotron_attention` takes its BF16 arm; keep it BF16 like the
    // backbone attention layers (draft acceptance is precision-sensitive and
    // the four projections total ~47 MB). The layer indexes the proposer's
    // own single-layer KV cache, hence `attn_layer_idx = 0`; BF16 KV avoids
    // the FP8 unit-scale collapse documented on the Qwen MTP path.
    let (attn, q_nv, k_nv, v_nv, o_dense, is_nvfp4) =
        load_nemotron_attention(store, 0, gpu, "mtp.layers.0")?;
    let mut attn_layer = Qwen3AttentionLayer::new_ungated(
        attn_norm,
        attn,
        DenseWeight {
            weight: DevicePtr::NULL,
        },
        FfnComponent::None,
        0, // attn_layer_idx: sole layer of the proposer's own KV cache
        q_nv,
        k_nv,
        v_nv,
        gpu,
        KvCacheDtype::Bf16,
        0, // no FP8-KV calibration on a BF16 drafter cache
        config,
    )?;
    if !is_nvfp4 {
        attn_layer.set_o_dense_bf16(o_dense);
    }

    // ── MoE block (mtp.layers.1) ──
    // BF16 experts on disk → `load_nemotron_moe` runtime-quantizes to NVFP4
    // (same fused relu²+down decode kernels as the backbone). The layer-index
    // argument only feeds `moe_intermediate_size_for` (uniform on Lightning —
    // out-of-range falls back to the scalar) and load-time logging.
    let moe = load_nemotron_moe(
        store,
        config.num_hidden_layers, // virtual "layer 52" — log label + uniform-size fallback
        config.num_experts,
        gpu,
        config,
        Some(absmax_k),
        Some(quantize_k),
        stream,
        None, // BF16-on-disk: no FP8-dequant scratch needed
        "mtp.layers.1",
    )?;
    let moe_layer = NemotronMoeLayer::new(
        moe,
        moe_norm,
        config,
        gpu,
        config.moe_intermediate_size,
        config.num_experts_per_tok,
    )?;

    tracing::info!(
        "Nemotron MTP module loaded: combiner [H,2H] + ungated GQA block (BF16) + \
         relu² MoE block ({} experts → NVFP4, top-{})",
        config.num_experts,
        config.num_experts_per_tok,
    );

    Ok(Some(NemotronMtpModule {
        enorm,
        hnorm,
        eh_proj,
        attn_layer,
        moe_layer,
        final_norm,
    }))
}
