// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA env/config leaves: the `$ATLAS_LORA_*` runtime hatches (eager / rotate /
//! peer), the full-attention layer enumerator, and the build-time
//! `validate_peft_config` gate. These sit on the model-integration side of the
//! eventual `lora-core` carve. Split out of the former monolithic `lora/mod.rs`
//! (SDD seam: ENV/CONFIG) — visibility unchanged.

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig, PeftAdapterConfig};

use super::LoraModule;

/// Permanent LoRA debugging hatch: `ATLAS_LORA_EAGER=1` (or `true`) forces
/// eager decode (no CUDA-graph capture) when an adapter is active, so
/// graph-vs-eager output parity can be compared in the field. Read ONCE —
/// the decode graph gate runs per token.
/// Resolved at the point of use rather than cached in a static: the model
/// carries this as `ModelLevers::lora_eager` for the per-token decode gate, and
/// the remaining callers are one-shot startup checks where a getenv is free.
pub fn lora_eager_env() -> bool {
    crate::layers::ops::ModelLevers::from_env().lora_eager
}

/// `ATLAS_LORA_ROTATE=1` (or `true`) ARMS runtime adapter rotation: it forces
/// eager decode (no CUDA-graph capture) so a `set_active_lora` re-point is
/// immediately live (eager-on-rotate — the graph would otherwise replay the
/// previously-captured slot pointers). A pool with >1 resident adapter arms
/// this automatically (see `TransformerModel::lora_rotatable`), so this env is
/// only needed to arm rotation on a SINGLE resident adapter (e.g. RDMA
/// slot-swap-in-place). Unset + a single startup adapter = today's behaviour
/// exactly (graphs ON, slot-0 pointers baked).
/// See [`lora_eager_env`] on why this is not cached.
pub fn lora_rotate_env() -> bool {
    crate::layers::ops::ModelLevers::from_env().lora_rotate
}

/// `$ATLAS_LORA_PEER` (host:port of an `atlas-weight-peer` staging a rotation
/// set) — when set, arms rotation (eager decode) even for a single resident
/// slot, because an RDMA swap re-points that slot in place. Unset = disk path
/// only, byte-identical to today.
pub fn lora_peer_env() -> Option<String> {
    std::env::var("ATLAS_LORA_PEER")
        .ok()
        .filter(|s| !s.is_empty())
}

/// Feature-1 (MoE expert + router LoRA) master switch. `ATLAS_LORA_EXPERTS=1`
/// (or `true`) opts INTO loading + applying routed-expert / router deltas.
/// DEFAULT OFF: an adapter that targets `mlp.experts.*` / `mlp.gate` is a NAMED
/// reject at load unless this is set, so the base path stays byte-identical and
/// the (correctness-first, host-synced, non-graphable) expert side-path is never
/// silently on. Read once.
pub fn lora_experts_env() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_LORA_EXPERTS")
            .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
    })
}

/// Feature-1 padded expert/router LoRA rank cap (`ATLAS_LORA_EXPERT_RANK`,
/// default 16). Separate from `--max-lora-rank` (the attention pool) because the
/// per-(layer,expert,proj) pool grows ~`num_experts × num_layers` faster, so a
/// low cap bounds the expert-pool VRAM blow-up. An adapter with `r` above this
/// is a named reject.
pub fn max_lora_expert_rank() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ATLAS_LORA_EXPERT_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&r: &usize| r > 0)
            .unwrap_or(16)
    })
}

/// `ATLAS_LORA_PREFILL_BGMV=1` — force prefill LoRA through the per-row BGMV
/// instead of the tensor-core GEMM.
///
/// Default OFF because the GEMM is ~4.8x faster on a 2K prompt (841 vs 176
/// tok/s measured on qwen3.8-27B) and the prefill call site is uniform-slot by
/// construction. The BGMV is the only form that can honour per-row slots
/// (including base rows), so this exists for the day a prefill batches rows
/// from different sequences — and as the bisect handle if the GEMM path is
/// ever suspected of a numerics difference, since the two are NOT bit-identical
/// (GEMV-per-row vs one GEMM).
pub fn prefill_bgmv_forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_LORA_PREFILL_BGMV").as_deref() == Ok("1"))
}

/// `ATLAS_LORA_NO_BATCH_VERIFY=1` — restore the old refusal of cross-sequence
/// batched speculative verify while a LoRA adapter is resident.
///
/// Default OFF: the batched path applies the deltas on every op it batches,
/// and all rows share one adapter (mixed batches are refused upstream). The
/// refusal used to be unconditional and undocumented, and it flattened DFlash
/// throughput to ~34 tok/s at every concurrency. This is the bisect handle if
/// a batched-verify numerics difference is ever suspected under an adapter.
pub fn no_batch_verify() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_LORA_NO_BATCH_VERIFY").as_deref() == Ok("1"))
}

pub fn full_attention_layers(cfg: &ModelConfig) -> Vec<usize> {
    (0..cfg.num_hidden_layers)
        .filter(|&i| cfg.layer_type(i) == LayerType::FullAttention)
        .collect()
}

/// Adapter-config gates that need build-time context (`--max-lora-rank`).
/// Parse-time gates (peft_type/DoRA/bias/regex target_modules/…) already
/// ran in `atlas_core::config::parse_peft_adapter_config`.
pub fn validate_peft_config(peft: &PeftAdapterConfig, max_lora_rank: usize) -> Result<()> {
    if peft.r > max_lora_rank {
        bail!(
            "REJECT[rank-exceeds-pool]: r={} > --max-lora-rank={}",
            peft.r,
            max_lora_rank
        );
    }
    let mut unsupported: Vec<&str> = Vec::new();
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        // `gate` is the MoE router (Feature-1), distinct from `gate_proj`. Expert
        // projections reuse the dense leaves (gate_proj/up_proj/down_proj), so
        // the LoraModule allow-list already covers them.
        let ok = last == "gate" || LoraModule::ALL.iter().any(|m| m.peft_name() == last);
        if !ok {
            unsupported.push(t.as_str());
        }
    }
    if !unsupported.is_empty() {
        if !allow_partial_targets() {
            bail!(
                "REJECT[unsupported-target]: target_modules {unsupported:?} \
                 (allowed: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj gate). \
                 Set ATLAS_LORA_ALLOW_PARTIAL=1 to load anyway, applying only the \
                 supported modules — the adapter will then be PARTIALLY applied and \
                 will not reproduce its training behaviour."
            );
        }
        // Opt-in partial load. Loud and once per adapter: a silently partial
        // adapter reads as "the model is behaving oddly", which is a far worse
        // debugging experience than a refused load. Real hybrid-model adapters
        // hit this constantly — Qwen3.8-27B community LoRAs target `out_proj`
        // (the SSM/GDN output projection, 48 of its 64 layers), which has no
        // LoraModule variant and no wiring in the SSM layers.
        tracing::warn!(
            "LoRA PARTIAL LOAD (ATLAS_LORA_ALLOW_PARTIAL=1): target_modules \
             {unsupported:?} are NOT supported and will be SKIPPED. Their \
             trained deltas will not be applied; output will differ from the \
             adapter's intent. Supported: q_proj k_proj v_proj o_proj \
             gate_proj up_proj down_proj gate."
        );
    }
    Ok(())
}

/// `ATLAS_LORA_ALLOW_PARTIAL=1` — load an adapter that names target modules
/// Atlas cannot apply, skipping those and applying the rest.
///
/// Delegates to the atlas-core definition rather than re-reading the env:
/// the parse-time allow-list down there is the FIRST gate an adapter meets,
/// so the flag has to be defined below this layer. Two OnceLocks reading one
/// variable is exactly the hand-synced drift this repo keeps getting bitten
/// by (cf. the 384-vs-3072 thinking-budget bug).
pub use atlas_core::config::allow_partial_targets;
