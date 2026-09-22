// SPDX-License-Identifier: AGPL-3.0-only

//! Kimi K3 GGUF (`general.architecture = kimi-k3`) name map.
//!
//! Maps llama.cpp / Unsloth tensor names onto the HuggingFace keys
//! `layer_keys` expects (empty `weight_prefix` -> `model.layers.N.*`).
//! Experts use K3 w1/w2/w3 naming, not Qwen mlp.experts.*.{gate,up,down}_proj.

use super::GgufName;

const HF: &str = "model";

/// Translate one GGUF tensor name for `kimi-k3` / `kimi_k3`.
pub(super) fn translate_kimi_k3(gguf_name: &str) -> Option<GgufName> {
    match gguf_name {
        "token_embd.weight" => {
            return Some(GgufName::Direct(format!("{HF}.embed_tokens.weight")));
        }
        "output_norm.weight" => {
            return Some(GgufName::Direct(format!("{HF}.norm.weight")));
        }
        "output.weight" => return Some(GgufName::Direct("lm_head.weight".into())),
        "output_res_score.weight" => {
            return Some(GgufName::Direct(format!(
                "{HF}.output_attn_res_proj.weight"
            )));
        }
        "rope_freqs.weight" => return Some(GgufName::Drop),
        _ => {}
    }

    let rest = gguf_name.strip_prefix("blk.")?;
    let (n_str, sub) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;

    match sub {
        "ffn_gate_exps.weight" => {
            return Some(GgufName::ExpertStack {
                layer,
                proj: "gate",
            });
        }
        "ffn_up_exps.weight" => return Some(GgufName::ExpertStack { layer, proj: "up" }),
        "ffn_down_exps.weight" => {
            return Some(GgufName::ExpertStack {
                layer,
                proj: "down",
            });
        }
        _ => {}
    }

    if sub == "ssm_a" {
        return Some(GgufName::Direct(format!(
            "{HF}.layers.{layer}.self_attn.A_log"
        )));
    }
    if sub == "ssm_dt.bias" {
        return Some(GgufName::Direct(format!(
            "{HF}.layers.{layer}.self_attn.dt_bias"
        )));
    }

    let (stem, ext) = match sub.rsplit_once('.') {
        Some((stem, ext @ ("weight" | "bias"))) => (stem, ext),
        _ => return None,
    };

    let hf_sub: Option<&str> = match stem {
        "attn_norm" => Some("input_layernorm"),
        "ffn_norm" => Some("post_attention_layernorm"),
        "attn_res_score" => Some("self_attention_res_proj"),
        "ffn_res_score" => Some("mlp_res_proj"),
        "attn_q" => Some("self_attn.q_proj"),
        "attn_k" => Some("self_attn.k_proj"),
        "attn_v" => Some("self_attn.v_proj"),
        "attn_output" => Some("self_attn.o_proj"),
        "attn_gate" => Some("self_attn.g_proj"),
        "attn_q_a" => Some("self_attn.q_a_proj"),
        "attn_q_b" => Some("self_attn.q_b_proj"),
        "attn_q_a_norm" => Some("self_attn.q_a_layernorm"),
        "attn_kv_a_mqa" => Some("self_attn.kv_a_proj_with_mqa"),
        "attn_kv_a_norm" => Some("self_attn.kv_a_layernorm"),
        "attn_k_b" => Some("self_attn.k_b_proj"),
        "attn_v_b" => Some("self_attn.v_b_proj"),
        "ssm_beta" => Some("self_attn.b_proj"),
        "ssm_g" => Some("self_attn.g_proj"),
        "ssm_f_a" => Some("self_attn.f_a_proj"),
        "ssm_f_b" => Some("self_attn.f_b_proj"),
        "ssm_norm" => Some("self_attn.o_norm"),
        "ssm_conv1d_q" => Some("self_attn.q_conv1d"),
        "ssm_conv1d_k" => Some("self_attn.k_conv1d"),
        "ssm_conv1d_v" => Some("self_attn.v_conv1d"),
        "ffn_gate" => Some("mlp.gate_proj"),
        "ffn_up" => Some("mlp.up_proj"),
        "ffn_down" => Some("mlp.down_proj"),
        "ffn_gate_inp" => Some("block_sparse_moe.gate"),
        "exp_probs_b" => Some("block_sparse_moe.gate.e_score_correction"),
        "ffn_routed_up" => Some("block_sparse_moe.routed_expert_up_proj"),
        "ffn_routed_down" => Some("block_sparse_moe.routed_expert_down_proj"),
        "ffn_routed_norm" => Some("block_sparse_moe.routed_expert_norm"),
        "ffn_gate_shexp" => Some("block_sparse_moe.shared_experts.gate_proj"),
        "ffn_up_shexp" => Some("block_sparse_moe.shared_experts.up_proj"),
        "ffn_down_shexp" => Some("block_sparse_moe.shared_experts.down_proj"),
        _ => None,
    };

    let hf_sub = hf_sub?;
    let name = if stem == "exp_probs_b" {
        format!("{HF}.layers.{layer}.{hf_sub}_bias")
    } else {
        format!("{HF}.layers.{layer}.{hf_sub}.{ext}")
    };
    Some(GgufName::Direct(name))
}

/// K3 expert HF name: gate->w1, up->w3, down->w2.
pub fn kimi_k3_expert_name(layer: usize, proj: &str, e: usize) -> String {
    let w = match proj {
        "gate" => "w1",
        "up" => "w3",
        "down" => "w2",
        other => other,
    };
    format!("{HF}.layers.{layer}.block_sparse_moe.experts.{e}.{w}.weight")
}

/// Stacked expert tensors stay on disk (DeferredTensor). Non-experts stay
/// keep-packed on GPU. 896 x 92 x 3 experts will not fit as BF16.
pub fn kimi_k3_deferred_name(gguf_name: &str) -> Option<String> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (n_str, sub) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;
    let proj = match sub {
        "ffn_gate_exps.weight" => "w1",
        "ffn_up_exps.weight" => "w3",
        "ffn_down_exps.weight" => "w2",
        _ => return None,
    };
    Some(format!(
        "model.layers.{layer}.block_sparse_moe.experts._stack.{proj}"
    ))
}

#[cfg(test)]
mod tests {
    use super::{GgufName, translate_kimi_k3};

    fn direct_suffix(name: &str) -> String {
        match translate_kimi_k3(name) {
            Some(GgufName::Direct(n)) => n,
            other => panic!("{name} -> {other:?}"),
        }
    }

    #[test]
    fn attn_k_b_and_v_b_stay_split() {
        // Oracle: GGUF stems map to the stored split tensors. Known-bad is a fused kv_b.
        let k = direct_suffix("blk.3.attn_k_b.weight");
        let v = direct_suffix("blk.3.attn_v_b.weight");
        assert!(k.ends_with("self_attn.k_b_proj.weight"), "{k}");
        assert!(v.ends_with("self_attn.v_b_proj.weight"), "{v}");
        assert_ne!(k, v);
        assert!(translate_kimi_k3("blk.3.attn_kv_b.weight").is_none());
        assert!(translate_kimi_k3("blk.3.attn_kv_b_proj.weight").is_none());
    }

    #[test]
    fn routed_down_gguf_name_is_not_expert_stack() {
        match translate_kimi_k3("blk.1.ffn_routed_down.weight") {
            Some(GgufName::Direct(n)) => {
                assert!(n.contains("routed_expert_down_proj"), "{n}");
                assert!(!n.contains("experts."));
            }
            other => panic!("{other:?}"),
        }
    }
}
