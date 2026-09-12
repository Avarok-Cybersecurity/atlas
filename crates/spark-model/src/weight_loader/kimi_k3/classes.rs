// SPDX-License-Identifier: AGPL-3.0-only

//! Production text + vision tensor-class names from
//! `docs/k3/official-weight-classes.tsv` (59 classes, 96 shards).
//! Expert MXFP4 keys are `*.weight_packed` + `*.weight_scale`.

/// Replace every all-numeric path segment with `*`.
pub fn canonicalize(name: &str) -> String {
    name.split('.')
        .map(|seg| {
            if !seg.is_empty() && seg.chars().all(|c| c.is_ascii_digit()) {
                "*"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Every `language_model.*` class in the official TSV. Required for C0 dry-run.
pub const TEXT_CLASSES: &[&str] = &[
    "language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight_packed",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight_scale",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight_packed",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight_scale",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight_packed",
    "language_model.model.layers.*.block_sparse_moe.experts.*.w3.weight_scale",
    "language_model.model.layers.*.input_layernorm.weight",
    "language_model.model.layers.*.mlp_res_norm.weight",
    "language_model.model.layers.*.mlp_res_proj.weight",
    "language_model.model.layers.*.post_attention_layernorm.weight",
    "language_model.model.layers.*.self_attention_res_norm.weight",
    "language_model.model.layers.*.self_attention_res_proj.weight",
    "language_model.model.layers.*.self_attn.g_proj.weight",
    "language_model.model.layers.*.self_attn.o_proj.weight",
    "language_model.model.layers.*.block_sparse_moe.gate.e_score_correction_bias",
    "language_model.model.layers.*.block_sparse_moe.gate.weight",
    "language_model.model.layers.*.block_sparse_moe.routed_expert_down_proj.weight",
    "language_model.model.layers.*.block_sparse_moe.routed_expert_norm.weight",
    "language_model.model.layers.*.block_sparse_moe.routed_expert_up_proj.weight",
    "language_model.model.layers.*.block_sparse_moe.shared_experts.down_proj.weight",
    "language_model.model.layers.*.block_sparse_moe.shared_experts.gate_proj.weight",
    "language_model.model.layers.*.block_sparse_moe.shared_experts.up_proj.weight",
    "language_model.model.layers.*.self_attn.A_log",
    "language_model.model.layers.*.self_attn.b_proj.weight",
    "language_model.model.layers.*.self_attn.dt_bias",
    "language_model.model.layers.*.self_attn.f_a_proj.weight",
    "language_model.model.layers.*.self_attn.f_b_proj.weight",
    "language_model.model.layers.*.self_attn.k_conv1d.weight",
    "language_model.model.layers.*.self_attn.k_proj.weight",
    "language_model.model.layers.*.self_attn.o_norm.weight",
    "language_model.model.layers.*.self_attn.q_conv1d.weight",
    "language_model.model.layers.*.self_attn.q_proj.weight",
    "language_model.model.layers.*.self_attn.v_conv1d.weight",
    "language_model.model.layers.*.self_attn.v_proj.weight",
    "language_model.model.layers.*.self_attn.kv_a_layernorm.weight",
    "language_model.model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
    "language_model.model.layers.*.self_attn.kv_b_proj.weight",
    "language_model.model.layers.*.self_attn.q_a_layernorm.weight",
    "language_model.model.layers.*.self_attn.q_a_proj.weight",
    "language_model.model.layers.*.self_attn.q_b_proj.weight",
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.model.layers.*.mlp.down_proj.weight",
    "language_model.model.layers.*.mlp.gate_proj.weight",
    "language_model.model.layers.*.mlp.up_proj.weight",
    "language_model.model.norm.weight",
    "language_model.model.output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight",
];

/// Vision / mm_projector classes. Optional under `language_model_only`.
pub const VISION_CLASSES: &[&str] = &[
    "vision_tower.encoder.blocks.*.mlp.fc0.weight",
    "vision_tower.encoder.blocks.*.mlp.fc1.weight",
    "vision_tower.encoder.blocks.*.norm0.weight",
    "vision_tower.encoder.blocks.*.norm1.weight",
    "vision_tower.encoder.blocks.*.wo.weight",
    "vision_tower.encoder.blocks.*.wqkv.weight",
    "mm_projector.proj.*.weight",
    "mm_projector.post_norm.weight",
    "vision_tower.encoder.final_layernorm.weight",
    "vision_tower.patch_embed.pos_emb.weight",
    "vision_tower.patch_embed.proj.weight",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassKind {
    Text,
    Vision,
}

pub fn classify(name: &str) -> Option<(&'static str, ClassKind)> {
    let canon = canonicalize(name);
    if let Some(c) = TEXT_CLASSES.iter().copied().find(|c| *c == canon) {
        return Some((c, ClassKind::Text));
    }
    if let Some(c) = VISION_CLASSES.iter().copied().find(|c| *c == canon) {
        return Some((c, ClassKind::Vision));
    }
    None
}

/// One concrete tensor name per class, for synthetic dry-run maps.
#[cfg(test)]
pub fn example_key(class: &str) -> String {
    class.replacen('*', "0", 1).replacen('*', "0", 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kimi_k3_text_classes_match_official_tsv() {
        let tsv = include_str!("../../../../../docs/k3/official-weight-classes.tsv");
        let mut from_tsv: Vec<&str> = tsv
            .lines()
            .filter_map(|line| {
                let cols: Vec<&str> = line.split('\t').collect();
                let class = *cols.last()?;
                class.starts_with("language_model.").then_some(class)
            })
            .collect();
        from_tsv.sort_unstable();
        let mut ours = TEXT_CLASSES.to_vec();
        ours.sort_unstable();
        assert_eq!(ours, from_tsv);
    }

    #[test]
    fn kimi_k3_vision_classes_match_official_tsv() {
        let tsv = include_str!("../../../../../docs/k3/official-weight-classes.tsv");
        let mut from_tsv: Vec<&str> = tsv
            .lines()
            .filter_map(|line| {
                let cols: Vec<&str> = line.split('\t').collect();
                let class = *cols.last()?;
                (class.starts_with("vision_tower.") || class.starts_with("mm_projector."))
                    .then_some(class)
            })
            .collect();
        from_tsv.sort_unstable();
        let mut ours = VISION_CLASSES.to_vec();
        ours.sort_unstable();
        assert_eq!(ours, from_tsv);
    }

    #[test]
    fn canonicalize_collapses_layer_and_expert_indices() {
        assert_eq!(
            canonicalize(
                "language_model.model.layers.12.block_sparse_moe.experts.7.w1.weight_packed"
            ),
            "language_model.model.layers.*.block_sparse_moe.experts.*.w1.weight_packed"
        );
    }
}
