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
    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        // `gate` is the MoE router (Feature-1), distinct from `gate_proj`. Expert
        // projections reuse the dense leaves (gate_proj/up_proj/down_proj), so
        // the LoraModule allow-list already covers them.
        let ok = last == "gate" || LoraModule::ALL.iter().any(|m| m.peft_name() == last);
        if !ok {
            bail!(
                "REJECT[unsupported-target]: target_modules entry '{t}' \
                 (allowed: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj gate)"
            );
        }
    }
    Ok(())
}
