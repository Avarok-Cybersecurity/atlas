// SPDX-License-Identifier: AGPL-3.0-only

//! Laguna-specific GGUF name-translation tests, split out of `tests.rs`
//! (<=500 LoC cap).

use super::*;

fn direct(name: &str) -> Option<GgufName> {
    Some(GgufName::Direct(name.to_string()))
}

/// Laguna (`general.architecture = laguna`) name translation: the three
/// overrides (shared expert, score-correction bias, attention output gate)
/// plus fall-through to the default translator for everything else. Fails
/// without `translate_laguna_layer` — the arch would take the default path
/// and drop `attn_gate` / mis-key the shared expert.
#[test]
fn laguna_layer_overrides() {
    assert_eq!(
        translate("blk.4.attn_gate.weight", "laguna"),
        direct("model.layers.4.self_attn.g_proj.weight")
    );
    assert_eq!(
        translate("blk.4.ffn_gate_shexp.weight", "laguna"),
        direct("model.layers.4.mlp.shared_expert.gate_proj.weight")
    );
    assert_eq!(
        translate("blk.4.ffn_up_shexp.weight", "laguna"),
        direct("model.layers.4.mlp.shared_expert.up_proj.weight")
    );
    assert_eq!(
        translate("blk.4.ffn_down_shexp.weight", "laguna"),
        direct("model.layers.4.mlp.shared_expert.down_proj.weight")
    );
    // Bare key: no trailing `.weight`/`.bias`.
    assert_eq!(
        translate("blk.4.exp_probs_b.bias", "laguna"),
        direct("model.layers.4.mlp.experts.e_score_correction_bias")
    );
    // Fall-through to the default translator: attention, router, expert stacks.
    assert_eq!(
        translate("blk.4.attn_q.weight", "laguna"),
        direct("model.layers.4.self_attn.q_proj.weight")
    );
    assert_eq!(
        translate("blk.4.ffn_gate_inp.weight", "laguna"),
        direct("model.layers.4.mlp.gate.weight")
    );
    assert_eq!(
        translate("blk.4.ffn_gate_exps.weight", "laguna"),
        Some(GgufName::ExpertStack {
            layer: 4,
            proj: "gate"
        })
    );
}
