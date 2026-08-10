// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for GGUF → HF tensor-name translation ([`super`]).

use super::*;

fn direct(name: &str) -> Option<GgufName> {
    Some(GgufName::Direct(name.to_string()))
}

#[test]
fn top_level_tensors() {
    assert_eq!(
        translate("token_embd.weight", "llama"),
        direct("model.embed_tokens.weight")
    );
    assert_eq!(
        translate("output_norm.weight", "llama"),
        direct("model.norm.weight")
    );
    assert_eq!(
        translate("output.weight", "llama"),
        direct("lm_head.weight")
    );
    assert_eq!(
        translate("rope_freqs.weight", "llama"),
        Some(GgufName::Drop)
    );
}

#[test]
fn attention_projections() {
    assert_eq!(
        translate("blk.0.attn_q.weight", "qwen3"),
        direct("model.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(
        translate("blk.7.attn_k.weight", "qwen3"),
        direct("model.layers.7.self_attn.k_proj.weight")
    );
    assert_eq!(
        translate("blk.7.attn_v.weight", "qwen3"),
        direct("model.layers.7.self_attn.v_proj.weight")
    );
    assert_eq!(
        translate("blk.7.attn_output.weight", "qwen3"),
        direct("model.layers.7.self_attn.o_proj.weight")
    );
}

#[test]
fn norms_and_mlp() {
    assert_eq!(
        translate("blk.3.attn_norm.weight", "llama"),
        direct("model.layers.3.input_layernorm.weight")
    );
    assert_eq!(
        translate("blk.3.ffn_norm.weight", "llama"),
        direct("model.layers.3.post_attention_layernorm.weight")
    );
    assert_eq!(
        translate("blk.3.ffn_gate.weight", "llama"),
        direct("model.layers.3.mlp.gate_proj.weight")
    );
    assert_eq!(
        translate("blk.3.ffn_up.weight", "llama"),
        direct("model.layers.3.mlp.up_proj.weight")
    );
    assert_eq!(
        translate("blk.3.ffn_down.weight", "llama"),
        direct("model.layers.3.mlp.down_proj.weight")
    );
}

#[test]
fn qwen3_qk_norms() {
    assert_eq!(
        translate("blk.11.attn_q_norm.weight", "qwen3"),
        direct("model.layers.11.self_attn.q_norm.weight")
    );
    assert_eq!(
        translate("blk.11.attn_k_norm.weight", "qwen3"),
        direct("model.layers.11.self_attn.k_norm.weight")
    );
}

#[test]
fn qwen2_qkv_biases() {
    assert_eq!(
        translate("blk.2.attn_q.bias", "qwen2"),
        direct("model.layers.2.self_attn.q_proj.bias")
    );
    assert_eq!(
        translate("blk.2.attn_k.bias", "qwen2"),
        direct("model.layers.2.self_attn.k_proj.bias")
    );
    assert_eq!(
        translate("blk.2.attn_v.bias", "qwen2"),
        direct("model.layers.2.self_attn.v_proj.bias")
    );
}

#[test]
fn moe_router_and_expert_stack() {
    assert_eq!(
        translate("blk.4.ffn_gate_inp.weight", "qwen3"),
        direct("model.layers.4.mlp.gate.weight")
    );
    assert_eq!(
        translate("blk.4.ffn_gate_exps.weight", "qwen3"),
        Some(GgufName::ExpertStack {
            layer: 4,
            proj: "gate"
        })
    );
    assert_eq!(
        translate("blk.4.ffn_up_exps.weight", "qwen3"),
        Some(GgufName::ExpertStack {
            layer: 4,
            proj: "up"
        })
    );
    assert_eq!(
        translate("blk.4.ffn_down_exps.weight", "qwen3"),
        Some(GgufName::ExpertStack {
            layer: 4,
            proj: "down"
        })
    );
}

#[test]
fn expert_name_expansion() {
    assert_eq!(
        expert_name(4, "gate", 17),
        "model.layers.4.mlp.experts.17.gate_proj.weight"
    );
    assert_eq!(
        expert_name(0, "down", 0),
        "model.layers.0.mlp.experts.0.down_proj.weight"
    );
}

#[test]
fn robust_layer_index_parsing() {
    assert_eq!(
        translate("blk.123.attn_q.weight", "llama"),
        direct("model.layers.123.self_attn.q_proj.weight")
    );
    assert_eq!(
        translate("blk.5.attn_q_norm.weight", "qwen3"),
        direct("model.layers.5.self_attn.q_norm.weight")
    );
}

#[test]
fn unrecognized_names_return_none() {
    assert_eq!(translate("blk.x.attn_q.weight", "llama"), None);
    assert_eq!(translate("blk.0.something_new.weight", "llama"), None);
    assert_eq!(translate("blk.0.attn_q.scale", "llama"), None);
    assert_eq!(translate("garbage", "llama"), None);
}

#[test]
fn qwen35_gdn_and_norm_mappings() {
    // Second RMSNorm renamed vs. the default `ffn_norm`.
    assert_eq!(
        translate("blk.0.post_attention_norm.weight", "qwen35"),
        direct("model.layers.0.post_attention_layernorm.weight")
    );
    // GDN / linear-attention projections → `linear_attn.*`.
    assert_eq!(
        translate("blk.0.attn_qkv.weight", "qwen35"),
        direct("model.layers.0.linear_attn.in_proj_qkv.weight")
    );
    assert_eq!(
        translate("blk.0.attn_gate.weight", "qwen35"),
        direct("model.layers.0.linear_attn.in_proj_z.weight")
    );
    assert_eq!(
        translate("blk.2.ssm_alpha.weight", "qwen35"),
        direct("model.layers.2.linear_attn.in_proj_a.weight")
    );
    assert_eq!(
        translate("blk.2.ssm_beta.weight", "qwen35"),
        direct("model.layers.2.linear_attn.in_proj_b.weight")
    );
    assert_eq!(
        translate("blk.4.ssm_conv1d.weight", "qwen35"),
        direct("model.layers.4.linear_attn.conv1d.weight")
    );
    assert_eq!(
        translate("blk.4.ssm_norm.weight", "qwen35"),
        direct("model.layers.4.linear_attn.norm.weight")
    );
    assert_eq!(
        translate("blk.4.ssm_out.weight", "qwen35"),
        direct("model.layers.4.linear_attn.out_proj.weight")
    );
    // Bare (extension-less) F32 gate params keyed without a suffix.
    assert_eq!(
        translate("blk.1.ssm_a", "qwen35"),
        direct("model.layers.1.linear_attn.A_log")
    );
    assert_eq!(
        translate("blk.1.ssm_dt.bias", "qwen35"),
        direct("model.layers.1.linear_attn.dt_bias")
    );
}

#[test]
fn qwen35_shared_names_fall_through_to_default() {
    // Full-attention + FFN + norms shared with the default translator.
    assert_eq!(
        translate("blk.3.attn_q.weight", "qwen35"),
        direct("model.layers.3.self_attn.q_proj.weight")
    );
    assert_eq!(
        translate("blk.3.attn_output.weight", "qwen35"),
        direct("model.layers.3.self_attn.o_proj.weight")
    );
    assert_eq!(
        translate("blk.3.attn_q_norm.weight", "qwen35"),
        direct("model.layers.3.self_attn.q_norm.weight")
    );
    assert_eq!(
        translate("blk.0.attn_norm.weight", "qwen35"),
        direct("model.layers.0.input_layernorm.weight")
    );
    assert_eq!(
        translate("blk.0.ffn_down.weight", "qwen35"),
        direct("model.layers.0.mlp.down_proj.weight")
    );
    assert_eq!(
        translate("token_embd.weight", "qwen35"),
        direct("model.embed_tokens.weight")
    );
    assert_eq!(
        translate("output.weight", "qwen35"),
        direct("lm_head.weight")
    );
}

#[test]
fn nllb_top_level_and_positions() {
    // Tied embedding → model.shared (NOT the default model.embed_tokens).
    assert_eq!(
        translate("token_embd.weight", "nllb"),
        direct("model.shared.weight")
    );
    assert_eq!(
        translate("token_embd.weight", "m2m_100"),
        direct("model.shared.weight")
    );
    // Learned positions dropped (sinusoidal regenerated at runtime).
    assert_eq!(
        translate("position_embd.weight", "nllb"),
        Some(GgufName::Drop)
    );
    // Per-stack final norm.
    assert_eq!(
        translate("enc.output_norm.weight", "nllb"),
        direct("model.encoder.layer_norm.weight")
    );
    assert_eq!(
        translate("enc.output_norm.bias", "nllb"),
        direct("model.encoder.layer_norm.bias")
    );
    assert_eq!(
        translate("dec.output_norm.weight", "nllb"),
        direct("model.decoder.layer_norm.weight")
    );
    assert_eq!(
        translate("dec.output_norm.bias", "m2m_100"),
        direct("model.decoder.layer_norm.bias")
    );
}

#[test]
fn nllb_encoder_self_attn_and_ffn() {
    assert_eq!(
        translate("enc.blk.0.attn_q.weight", "nllb"),
        direct("model.encoder.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(
        translate("enc.blk.5.attn_o.bias", "nllb"),
        direct("model.encoder.layers.5.self_attn.out_proj.bias")
    );
    assert_eq!(
        translate("enc.blk.7.attn_norm.weight", "nllb"),
        direct("model.encoder.layers.7.self_attn_layer_norm.weight")
    );
    assert_eq!(
        translate("enc.blk.3.ffn_up.weight", "nllb"),
        direct("model.encoder.layers.3.fc1.weight")
    );
    assert_eq!(
        translate("enc.blk.3.ffn_down.bias", "nllb"),
        direct("model.encoder.layers.3.fc2.bias")
    );
    assert_eq!(
        translate("enc.blk.3.ffn_norm.weight", "nllb"),
        direct("model.encoder.layers.3.final_layer_norm.weight")
    );
}

#[test]
fn nllb_decoder_self_and_cross_attn() {
    assert_eq!(
        translate("dec.blk.1.attn_k.weight", "nllb"),
        direct("model.decoder.layers.1.self_attn.k_proj.weight")
    );
    assert_eq!(
        translate("dec.blk.23.cross_attn_v.weight", "nllb"),
        direct("model.decoder.layers.23.encoder_attn.v_proj.weight")
    );
    assert_eq!(
        translate("dec.blk.0.cross_attn_o.bias", "m2m_100"),
        direct("model.decoder.layers.0.encoder_attn.out_proj.bias")
    );
    assert_eq!(
        translate("dec.blk.1.cross_attn_norm.bias", "nllb"),
        direct("model.decoder.layers.1.encoder_attn_layer_norm.bias")
    );
}

#[test]
fn nllb_unrecognized_and_no_default_fallthrough() {
    // token_embd must NOT take the decoder-only default mapping under nllb.
    assert_ne!(
        translate("token_embd.weight", "nllb"),
        direct("model.embed_tokens.weight")
    );
    // Unknown per-block sub-name → None (loader skips).
    assert_eq!(translate("enc.blk.0.something_new.weight", "nllb"), None);
    assert_eq!(translate("enc.blk.x.attn_q.weight", "nllb"), None);
    assert_eq!(translate("garbage", "nllb"), None);
}

#[test]
fn unknown_arch_falls_through_to_default() {
    assert_eq!(
        translate("blk.0.attn_q.weight", "some_future_arch"),
        direct("model.layers.0.self_attn.q_proj.weight")
    );
}

#[test]
fn clip_block_attn_and_ffn() {
    assert_eq!(
        translate("v.blk.0.attn_qkv.weight", "clip"),
        direct("model.visual.blocks.0.attn.qkv.weight")
    );
    assert_eq!(
        translate("v.blk.0.attn_qkv.bias", "clip"),
        direct("model.visual.blocks.0.attn.qkv.bias")
    );
    assert_eq!(
        translate("v.blk.13.attn_out.weight", "clip"),
        direct("model.visual.blocks.13.attn.proj.weight")
    );
    assert_eq!(
        translate("v.blk.13.attn_out.bias", "clip"),
        direct("model.visual.blocks.13.attn.proj.bias")
    );
    assert_eq!(
        translate("v.blk.26.ffn_up.weight", "clip"),
        direct("model.visual.blocks.26.mlp.linear_fc1.weight")
    );
    assert_eq!(
        translate("v.blk.26.ffn_down.weight", "clip"),
        direct("model.visual.blocks.26.mlp.linear_fc2.weight")
    );
    assert_eq!(
        translate("v.blk.5.ffn_up.bias", "clip"),
        direct("model.visual.blocks.5.mlp.linear_fc1.bias")
    );
}

#[test]
fn clip_block_norms() {
    assert_eq!(
        translate("v.blk.0.ln1.weight", "clip"),
        direct("model.visual.blocks.0.norm1.weight")
    );
    assert_eq!(
        translate("v.blk.0.ln1.bias", "clip"),
        direct("model.visual.blocks.0.norm1.bias")
    );
    assert_eq!(
        translate("v.blk.7.ln2.weight", "clip"),
        direct("model.visual.blocks.7.norm2.weight")
    );
    assert_eq!(
        translate("v.blk.7.ln2.bias", "clip"),
        direct("model.visual.blocks.7.norm2.bias")
    );
}

#[test]
fn clip_patch_bias_and_position_embed() {
    assert_eq!(
        translate("v.patch_embd.bias", "clip"),
        direct("model.visual.patch_embed.proj.bias")
    );
    // The temporal-split conv WEIGHT frames are intercepted by the loader,
    // so the name map deliberately leaves them unmatched.
    assert_eq!(translate("v.patch_embd.weight", "clip"), None);
    assert_eq!(translate("v.patch_embd.weight.1", "clip"), None);
    assert_eq!(
        translate("v.position_embd.weight", "clip"),
        direct("model.visual.pos_embed.weight")
    );
}

#[test]
fn clip_merger_and_projector() {
    assert_eq!(
        translate("v.post_ln.weight", "clip"),
        direct("model.visual.merger.norm.weight")
    );
    assert_eq!(
        translate("v.post_ln.bias", "clip"),
        direct("model.visual.merger.norm.bias")
    );
    assert_eq!(
        translate("mm.0.weight", "clip"),
        direct("model.visual.merger.linear_fc1.weight")
    );
    assert_eq!(
        translate("mm.0.bias", "clip"),
        direct("model.visual.merger.linear_fc1.bias")
    );
    assert_eq!(
        translate("mm.2.weight", "clip"),
        direct("model.visual.merger.linear_fc2.weight")
    );
    assert_eq!(
        translate("mm.2.bias", "clip"),
        direct("model.visual.merger.linear_fc2.bias")
    );
}

#[test]
fn clip_unrecognized_names_return_none() {
    assert_eq!(translate("v.blk.0.unknown.weight", "clip"), None);
    assert_eq!(translate("v.blk.0.attn_qkv.scale", "clip"), None);
    assert_eq!(translate("v.blk.x.ln1.weight", "clip"), None);
    assert_eq!(translate("mm.1.weight", "clip"), None); // GELU, no tensor
    // clip names never leak into the default (backbone) translation.
    assert_eq!(translate("v.blk.0.attn_qkv.weight", "llama"), None);
}

#[test]
fn keep_packed_proj_scoped_to_dense_ffn() {
    // The three FFN projections keep-pack (Tier-1 decode scope).
    assert!(is_keep_packed_proj("model.layers.0.mlp.gate_proj.weight"));
    assert!(is_keep_packed_proj("model.layers.31.mlp.up_proj.weight"));
    assert!(is_keep_packed_proj("model.layers.7.mlp.down_proj.weight"));

    // Tier-1c: FULL-ATTENTION q/k/v/o keep-pack DIRECTLY (transform-free).
    assert!(is_keep_packed_proj(
        "model.layers.0.self_attn.q_proj.weight"
    ));
    assert!(is_keep_packed_proj(
        "model.layers.3.self_attn.k_proj.weight"
    ));
    assert!(is_keep_packed_proj(
        "model.layers.3.self_attn.v_proj.weight"
    ));
    assert!(is_keep_packed_proj(
        "model.layers.0.self_attn.o_proj.weight"
    ));

    // Deliberately NOT matched by this filter: lm_head, embeddings, norms, and
    // the GDN/linear-attn projections. The GDN input projections keep-pack via
    // `value_transform::packed_reorder_rows` (a packed row-permute), NOT here;
    // `out_proj` (column reorder) stays on its NVFP4/BF16 path.
    assert!(!is_keep_packed_proj("lm_head.weight"));
    assert!(!is_keep_packed_proj("model.embed_tokens.weight"));
    assert!(!is_keep_packed_proj(
        "model.layers.0.input_layernorm.weight"
    ));
    assert!(!is_keep_packed_proj(
        "model.layers.0.linear_attn.in_proj_qkv.weight"
    ));
    assert!(!is_keep_packed_proj(
        "model.layers.0.linear_attn.out_proj.weight"
    ));
    // A bias must never match (only .weight projections are packed).
    assert!(!is_keep_packed_proj("model.layers.0.mlp.gate_proj.bias"));
}
