// SPDX-License-Identifier: AGPL-3.0-only

//! Capability-gate tests: a tier the model cannot populate is REJECTED at
//! startup with an actionable message — never a silent no-op — while capable
//! models and the no-env default path are untouched.

use atlas_core::config::{LayerType, ModelConfig};

use super::*;

/// Hybrid SSM+attention MoE (the Holo/Qwen3-next family shape).
fn hybrid() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

/// Pure-attention dense model: no recurrent state, no experts.
fn dense() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3".to_string();
    c.num_hidden_layers = 28;
    c.layer_types = vec![LayerType::FullAttention; 28];
    c.num_experts = 0;
    c.linear_num_key_heads = 0;
    c.linear_key_head_dim = 0;
    c.linear_num_value_heads = 0;
    c.linear_value_head_dim = 0;
    c
}

// ── The honest predicates (config-derived, SSOT) ───────────────────────

#[test]
fn capability_predicates_are_honest() {
    assert!(hybrid().has_recurrent_state());
    assert!(hybrid().has_experts());
    assert!(!dense().has_recurrent_state());
    assert!(!dense().has_experts());
    // Pure-attention MoE (DeepSeek/Mixtral shape): experts yes, SSM no.
    let mut moe = dense();
    moe.num_experts = 512;
    assert!(moe.has_experts());
    assert!(!moe.has_recurrent_state());
}

// ── Startup rejection: dense model + any SSM tier var ──────────────────

#[test]
fn dense_model_with_ssm_tier_var_is_rejected() {
    let err = ensure_ssm_tier_capability_from(&dense(), &["ATLAS_SSM_TIER"]).unwrap_err();
    let msg = format!("{err:#}");
    // Actionable: names the model and the exact var(s) to unset.
    assert!(msg.contains("qwen3"), "names the model: {msg}");
    assert!(msg.contains("ATLAS_SSM_TIER"), "names the var: {msg}");
    assert!(msg.contains("no recurrent state"), "says why: {msg}");
}

#[test]
fn dense_model_rejects_every_tier_selector_var() {
    for var in [
        "ATLAS_SSM_TIER",
        "ATLAS_SSM_RDMA_TIER",
        "ATLAS_SSM_SWAP",
        "ATLAS_SSM_DECODE_TIER",
        "ATLAS_SSM_DECODE_RING_ROLL",
    ] {
        let err = ensure_ssm_tier_capability_from(&dense(), &[var]).unwrap_err();
        assert!(
            format!("{err:#}").contains(var),
            "gate must reject and name {var}"
        );
    }
}

#[test]
fn dense_model_error_lists_all_set_vars() {
    let err =
        ensure_ssm_tier_capability_from(&dense(), &["ATLAS_SSM_TIER", "ATLAS_SSM_DECODE_TIER"])
            .unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("ATLAS_SSM_TIER") && msg.contains("ATLAS_SSM_DECODE_TIER"));
}

// ── The unchanged paths: no-env default and capable models ─────────────

#[test]
fn dense_model_without_tier_vars_is_ok() {
    // The byte-identical default path: env presence only, no error.
    ensure_ssm_tier_capability_from(&dense(), &[]).unwrap();
}

#[test]
fn hybrid_model_with_all_tier_vars_is_ok() {
    ensure_ssm_tier_capability_from(
        &hybrid(),
        &[
            "ATLAS_SSM_TIER",
            "ATLAS_SSM_RDMA_TIER",
            "ATLAS_SSM_SWAP",
            "ATLAS_SSM_DECODE_TIER",
            "ATLAS_SSM_DECODE_RING_ROLL",
        ],
    )
    .unwrap();
}
