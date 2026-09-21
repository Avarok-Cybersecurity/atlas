// SPDX-License-Identifier: AGPL-3.0-only

//! qwen35moe shared-expert + NextN name map.

use super::*;

fn direct(name: &str) -> Option<GgufName> {
    Some(GgufName::Direct(name.to_string()))
}

#[test]
fn qwen35moe_stacked_experts_map_to_expert_stack() {
    assert_eq!(
        translate("blk.0.ffn_gate_exps.weight", "qwen35moe"),
        Some(GgufName::ExpertStack {
            layer: 0,
            proj: "gate",
        })
    );
    assert_eq!(
        translate("blk.0.ffn_up_exps.weight", "qwen35moe"),
        Some(GgufName::ExpertStack {
            layer: 0,
            proj: "up",
        })
    );
    assert_eq!(
        translate("blk.0.ffn_down_exps.weight", "qwen35moe"),
        Some(GgufName::ExpertStack {
            layer: 0,
            proj: "down",
        })
    );
    assert_eq!(
        expert_name(0, "down", 7),
        "model.layers.0.mlp.experts.7.down_proj.weight"
    );
}

#[test]
fn qwen35moe_shared_expert_maps_to_hf_shared_expert() {
    assert_eq!(
        translate("blk.5.ffn_gate_shexp.weight", "qwen35moe"),
        direct("model.layers.5.mlp.shared_expert.gate_proj.weight")
    );
    assert_eq!(
        translate("blk.5.ffn_up_shexp.weight", "qwen35moe"),
        direct("model.layers.5.mlp.shared_expert.up_proj.weight")
    );
    assert_eq!(
        translate("blk.5.ffn_down_shexp.weight", "qwen35moe"),
        direct("model.layers.5.mlp.shared_expert.down_proj.weight")
    );
    assert_eq!(
        translate("blk.5.ffn_gate_inp_shexp.weight", "qwen35moe"),
        direct("model.layers.5.mlp.shared_expert_gate.weight")
    );
}

#[test]
fn qwen35moe_nextn_tensors_are_dropped() {
    assert_eq!(
        translate("blk.40.nextn.eh_proj.weight", "qwen35moe"),
        Some(GgufName::Drop)
    );
    assert_eq!(
        translate("blk.40.nextn.enorm.weight", "qwen35moe"),
        Some(GgufName::Drop)
    );
    assert_eq!(
        translate("blk.40.nextn.embed_tokens.weight", "qwen35moe"),
        Some(GgufName::Drop)
    );
}
