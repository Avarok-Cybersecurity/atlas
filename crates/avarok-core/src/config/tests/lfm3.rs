// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: the LFM3 config contract (`parsers/lfm3.rs`): the real
//! `LiquidAI/LFM3-1.7B` `config.json` parses to the 13/5/6 layer split, the
//! RoPE and stop-token facts, and the GDN-schema sentinel; every math-changing
//! field the loader does not implement is refused with a clear error.

#![allow(unused_imports)]

use super::super::*;

/// `LiquidAI/LFM3-1.7B` `config.json` (transformers 5.12.1), verbatim.
const CHECKPOINT: &str = r#"{
  "architectures": ["Lfm3ForCausalLM"],
  "auto_map": {
    "AutoConfig": "configuration_lfm3.Lfm3Config",
    "AutoModel": "modeling_lfm3.Lfm3Model",
    "AutoModelForCausalLM": "modeling_lfm3.Lfm3ForCausalLM"
  },
  "bos_token_id": 124894,
  "conv_L_cache": 7,
  "conv_bias": false,
  "dtype": "bfloat16",
  "eos_token_id": 124900,
  "head_dim": 128,
  "hidden_size": 2048,
  "intermediate_size": 7680,
  "layer_types": [
    "conv", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv",
    "sliding_attention", "conv", "full_attention", "conv", "sliding_attention", "conv",
    "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"
  ],
  "max_position_embeddings": 262144,
  "model_type": "lfm3",
  "norm_eps": 1e-05,
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "num_key_value_heads": 4,
  "pad_token_id": 124893,
  "rope_parameters": {
    "full_attention": {
      "partial_rotary_factor": 0.5,
      "rope_theta": 1000000.0,
      "rope_type": "default"
    },
    "sliding_attention": {
      "rope_theta": 10000,
      "rope_type": "default"
    }
  },
  "sliding_window": 256,
  "tie_word_embeddings": true,
  "transformers_version": "5.12.1",
  "use_cache": true,
  "use_sandwich_norm": true,
  "vocab_size": 128000
}"#;

fn parse() -> ModelConfig {
    parse_config(CHECKPOINT).expect("the real LFM3 config.json parses")
}

/// The checkpoint JSON with `edit` applied to its top-level object.
fn edited(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
    let mut v: serde_json::Value = serde_json::from_str(CHECKPOINT).unwrap();
    edit(v.as_object_mut().unwrap());
    v.to_string()
}

fn refusal(json: &str) -> String {
    parse_config(json)
        .expect_err("this config must be refused")
        .to_string()
}

#[test]
fn layer_split_is_13_conv_5_sliding_6_global() {
    let c = parse();
    assert_eq!(c.model_type, "lfm3");
    assert_eq!(c.num_hidden_layers, 24);
    assert_eq!(c.layer_types.len(), 24);
    let count = |t: LayerType| c.layer_types.iter().filter(|x| **x == t).count();
    assert_eq!(count(LayerType::LinearAttention), 13);
    assert_eq!(count(LayerType::SlidingAttention), 5);
    assert_eq!(count(LayerType::FullAttention), 6);
    // Indices 0, 1, 3, 5, ..., 23 are conv; 2, 6, ..., 22 full; 4, 8, ..., 20 sliding.
    for i in 0..24 {
        let want = match i {
            0 | 1 => LayerType::LinearAttention,
            _ if i % 4 == 2 => LayerType::FullAttention,
            _ if i % 4 == 0 => LayerType::SlidingAttention,
            _ => LayerType::LinearAttention,
        };
        assert_eq!(c.layer_type(i), want, "layer {i}");
    }
    assert_eq!(c.num_attention_layers(), 11);
    assert_eq!(c.num_ssm_layers(), 13);
}

#[test]
fn dimensions_match_the_checkpoint() {
    let c = parse();
    assert_eq!(c.hidden_size, 2048);
    assert_eq!(c.intermediate_size, 7680);
    assert_eq!(c.vocab_size, 128_000);
    assert_eq!(c.num_attention_heads, 16);
    assert_eq!(c.num_key_value_heads, 4);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.gqa_ratio(), 4);
    assert_eq!(c.max_position_embeddings, 262_144);
    assert_eq!(c.sliding_window, 256);
    assert_eq!(c.rms_norm_eps, 1e-5);
    assert!(c.tie_word_embeddings);
    assert_eq!(c.weight_prefix, "");
    assert_eq!(c.layer_prefix(3), "model.layers.3");
    assert!(!c.attn_gated, "the head gate is its own projection");
    assert_eq!(c.qk_norm_type, "per_head");
    assert_eq!((c.num_experts, c.mtp_num_hidden_layers), (0, 0));
}

#[test]
fn rope_is_full_head_with_a_half_rotated_global_layer() {
    let c = parse();
    assert_eq!(c.rope_theta, 1_000_000.0);
    assert_eq!(c.partial_rotary_factor, 0.5);
    // The zero-frequency tail keeps the rotated dimension at the whole head.
    assert_eq!(c.rotary_dim(), 128);
}

#[test]
fn stop_tokens_are_the_im_end_id() {
    let c = parse();
    assert_eq!(c.eos_token_id, 124_900);
    assert_eq!(c.eos_ids(), vec![124_900]);
    assert!(c.is_eos(124_900));
    assert!(!c.is_eos(124_894), "BOS is not a stop token");
    assert_eq!(c.bos_token_id, 124_894);
}

#[test]
fn a_stop_token_array_keeps_the_first_as_primary() {
    let json = edited(|o| {
        o.insert("eos_token_id".into(), serde_json::json!([124_900, 2]));
    });
    let c = parse_config(&json).unwrap();
    assert_eq!(c.eos_token_id, 124_900);
    assert_eq!(c.eos_ids(), vec![124_900, 2]);
}

/// The conv geometry on the GDN schema: `ssm_qkvz_size()` is the `3 * hidden`
/// BCx width so a scratch sized from it holds the in_proj output, the SSM
/// pool is a few hundred KB per slot, and the conv window is 7 wide.
#[test]
fn conv_geometry_rides_the_gdn_schema() {
    let c = parse();
    assert_eq!(c.linear_conv_kernel_dim, 7);
    assert_eq!(c.ssm_qkvz_size(), 3 * c.hidden_size);
    assert_eq!(c.ssm_z_size(), c.hidden_size);
    assert_eq!(c.ssm_h_state_bytes(), 16 * 128 * 64 * 4);
    assert_eq!(c.ssm_conv_state_bytes(), (2 * 16 * 64 + 16 * 128) * 7 * 4);
    assert!(c.has_recurrent_state());
}

#[test]
fn the_default_layer_pattern_is_the_checkpoints() {
    let json = edited(|o| {
        o.remove("layer_types");
    });
    assert_eq!(
        parse_config(&json).unwrap().layer_types,
        parse().layer_types
    );
}

#[test]
fn absent_optional_blocks_take_lfm3config_defaults() {
    let json = edited(|o| {
        for key in [
            "rope_parameters",
            "sliding_window",
            "norm_eps",
            "head_dim",
            "conv_L_cache",
            "conv_bias",
            "use_sandwich_norm",
            "tie_word_embeddings",
            "eos_token_id",
        ] {
            o.remove(key);
        }
    });
    let c = parse_config(&json).unwrap();
    assert_eq!(c.rope_theta, 1_000_000.0);
    assert_eq!(c.partial_rotary_factor, 0.5);
    assert_eq!(c.sliding_window, 256);
    assert_eq!(c.rms_norm_eps, 1e-5);
    assert_eq!(c.head_dim, 128);
    assert!(c.tie_word_embeddings);
    assert_eq!(c.eos_token_id, 124_900);
}

#[test]
fn a_native_fp8_checkpoint_keeps_its_quantization_block() {
    // Shape of a compressed-tensors `float-quantized` block (128x128 weight
    // blocks); the loader reads the tensor dtypes, not this block, so the test
    // only pins that the parser keeps it for the memory estimates.
    let json = edited(|o| {
        o.insert(
            "quantization_config".into(),
            serde_json::json!({
                "quant_method": "compressed-tensors",
                "format": "float-quantized",
                "ignore": ["lm_head", "re:.*conv.*", "re:.*gate_proj"],
                "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}}
            }),
        );
    });
    let c = parse_config(&json).unwrap();
    let q = c.quantization_config.expect("quantization block kept");
    assert_eq!(q.quant_method, "compressed-tensors");
    assert_eq!(q.format, "float-quantized");
    assert_eq!(q.quant_algo, "FP8");
    assert!(q.ignore_modules.iter().any(|m| m == "lm_head"));
}

#[test]
fn the_head_stays_bf16_unless_the_operator_asks() {
    let mut c = parse();
    assert!(
        c.skip_lm_head_quantization(),
        "tied BF16 head, no NVFP4 requant"
    );
    c.lm_head_bf16_override = Some(false);
    assert!(
        !c.skip_lm_head_quantization(),
        "--lm-head-dtype fp8/nvfp4 wins"
    );
}

// ── refusals ──

#[test]
fn conv_bias_is_refused() {
    let e = refusal(&edited(|o| {
        o.insert("conv_bias".into(), serde_json::json!(true));
    }));
    assert!(e.contains("conv_bias"), "{e}");
}

#[test]
fn a_model_without_sandwich_norms_is_refused() {
    let e = refusal(&edited(|o| {
        o.insert("use_sandwich_norm".into(), serde_json::json!(false));
    }));
    assert!(e.contains("use_sandwich_norm"), "{e}");
}

#[test]
fn a_conv_width_other_than_7_is_refused() {
    let e = refusal(&edited(|o| {
        o.insert("conv_L_cache".into(), serde_json::json!(3));
    }));
    assert!(e.contains("conv_L_cache 3") && e.contains("7-tap"), "{e}");
}

#[test]
fn rope_types_other_than_default_are_refused() {
    for kind in ["full_attention", "sliding_attention"] {
        let e = refusal(&edited(|o| {
            o["rope_parameters"][kind]["rope_type"] = "yarn".into();
        }));
        assert!(e.contains(kind) && e.contains("yarn"), "{e}");
    }
}

#[test]
fn a_sliding_theta_the_loader_cannot_build_is_refused() {
    let e = refusal(&edited(|o| {
        o["rope_parameters"]["sliding_attention"]["rope_theta"] = serde_json::json!(5000.0);
    }));
    assert!(e.contains("sliding_attention rope_theta"), "{e}");
}

#[test]
fn a_partially_rotated_sliding_layer_is_refused() {
    let e = refusal(&edited(|o| {
        o["rope_parameters"]["sliding_attention"]["partial_rotary_factor"] = serde_json::json!(0.5);
    }));
    assert!(e.contains("sliding_attention partial_rotary_factor"), "{e}");
}

#[test]
fn a_legacy_rope_scaling_block_is_refused() {
    let e = refusal(&edited(|o| {
        o.insert(
            "rope_scaling".into(),
            serde_json::json!({"rope_type": "yarn", "factor": 4.0}),
        );
    }));
    assert!(e.contains("rope_scaling"), "{e}");
}

#[test]
fn bad_layer_schedules_are_refused() {
    let e = refusal(&edited(|o| {
        o["layer_types"][0] = "mamba".into();
    }));
    assert!(e.contains("mamba"), "{e}");
    let e = refusal(&edited(|o| {
        o.insert("num_hidden_layers".into(), serde_json::json!(23));
    }));
    assert!(e.contains("24 entries") && e.contains("23"), "{e}");
}

#[test]
fn sliding_layers_need_a_window() {
    let e = refusal(&edited(|o| {
        o.insert("sliding_window".into(), serde_json::Value::Null);
    }));
    assert!(e.contains("sliding_window"), "{e}");
}

#[test]
fn inconsistent_head_counts_are_refused() {
    let e = refusal(&edited(|o| {
        o.insert("num_key_value_heads".into(), serde_json::json!(5));
    }));
    assert!(e.contains("divisible"), "{e}");
    let e = refusal(&edited(|o| {
        o.insert("head_dim".into(), serde_json::json!(127));
    }));
    assert!(e.contains("even"), "{e}");
}

#[test]
fn the_other_lfm3_model_types_do_not_route_here() {
    // `lfm3_vl` (the d1 and VL checkpoints) carries a vision tower and is not
    // this parser's family: it must not be silently parsed as text-only.
    let json = edited(|o| {
        o.insert("model_type".into(), "lfm3_vl".into());
    });
    let routed = parse_config(&json).map(|c| c.model_type);
    assert_ne!(routed.ok().as_deref(), Some("lfm3"));
}
