// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. Source of the field list and defaults: configuration_lfm3.py (Lfm3Config).

//! Responsibility: parse a Liquid AI LFM3 `config.json` into a [`ModelConfig`].
//!
//! LFM3 is a 24-layer hybrid: gated short convolutions (`conv`, 13 layers),
//! full attention (6) and sliding-window attention (5, window 256), all with
//! sandwich RMSNorms, a dense SwiGLU FFN and a tied embedding table. Layer
//! kinds map to the existing [`LayerType`] enum (no variant is added):
//!
//! | `layer_types` entry | [`LayerType`] |
//! |---|---|
//! | `conv` | `LinearAttention` (the layer opts out of the GDN pool, `uses_ssm_pool() == false`) |
//! | `full_attention` | `FullAttention` |
//! | `sliding_attention` | `SlidingAttention` |
//!
//! ## RoPE
//! Full-attention layers use theta `rope_parameters.full_attention.rope_theta`
//! with `partial_rotary_factor` 0.5: HF keeps the frequency progression of the
//! WHOLE head (`1 / theta^(2i/head_dim)`) for the first `factor * head_dim / 2`
//! pairs and zeroes the rest, so the rotated dimension is still `head_dim`.
//! `config.rotary_dim` is therefore `head_dim` (an exact identity on the
//! zero-frequency pairs) and `partial_rotary_factor` records the checkpoint's
//! 0.5; the loader builds the zero-tail table. Sliding layers rotate the full
//! head with theta 10000. `ModelConfig` has one `rope_theta`, which carries the
//! full-attention value; the sliding theta is fixed in the loader
//! (`weight_loader/lfm3`, `SLIDING_ROPE_THETA`) and this parser REFUSES any
//! other value rather than load a table the config cannot describe.
//!
//! ## What is refused (math-changing fields the loader does not implement)
//! `conv_bias = true`, `use_sandwich_norm = false`, a `conv_L_cache` other than
//! 7 (the conv kernels are 7-tap), any `rope_type` other than `default`, a
//! sliding-layer `partial_rotary_factor` below 1, a legacy `rope_scaling`
//! block, and unknown `layer_types` entries.
//!
//! ## GDN-schema sentinel
//! A `LinearAttention` layer makes `validate_config` require the GDN fields,
//! and `ssm_qkvz_size()` and the SSM state pool are sized from them. The conv
//! layer has no recurrent matrix state, so the fields are set to the conv
//! geometry mapped onto the GDN schema (see [`SSM_KEY_HEADS`] and friends):
//! `ssm_qkvz_size()` is exactly the `[3 * hidden]` BCx width, so a scratch
//! buffer sized from it fits the in_proj output.

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};

use super::super::{LayerType, ModelConfig, finalize_config};

/// The conv kernels are 7-tap (`weights [hidden, 7]`, state `[hidden, 6]`).
pub(crate) const CONV_WIDTH: usize = 7;
/// Theta of the sliding-attention layers, fixed in the loader.
pub(crate) const SLIDING_ROPE_THETA: f64 = 10_000.0;

/// GDN-schema sentinel: q and k are `2 * 16 * 64 = 2048` wide, v and z are
/// `16 * 128 = 2048` each, so `ssm_qkvz_size() == 6144 == 3 * hidden_size` and
/// `ssm_z_size() == hidden_size` for the 2048-wide model.
pub(crate) const SSM_KEY_HEADS: usize = 16;
pub(crate) const SSM_KEY_HEAD_DIM: usize = 64;
pub(crate) const SSM_VALUE_HEADS: usize = 16;
pub(crate) const SSM_VALUE_HEAD_DIM: usize = 128;

fn field<'a>(raw: &'a Value, key: &str) -> Option<&'a Value> {
    raw.get(key).filter(|v| !v.is_null())
}

fn uint(raw: &Value, key: &str, default: Option<usize>) -> Result<usize> {
    match field(raw, key) {
        Some(v) => v
            .as_u64()
            .map(|n| n as usize)
            .with_context(|| format!("lfm3 config field `{key}` must be an unsigned integer")),
        None => default.with_context(|| format!("lfm3 config missing required field `{key}`")),
    }
}

fn float(raw: &Value, key: &str, default: Option<f64>) -> Result<f64> {
    match field(raw, key) {
        Some(v) => v
            .as_f64()
            .with_context(|| format!("lfm3 config field `{key}` must be numeric")),
        None => default.with_context(|| format!("lfm3 config missing required field `{key}`")),
    }
}

fn flag(raw: &Value, key: &str, default: bool) -> Result<bool> {
    match field(raw, key) {
        Some(v) => v
            .as_bool()
            .with_context(|| format!("lfm3 config field `{key}` must be a boolean")),
        None => Ok(default),
    }
}

/// `layer_types` entries after validation, as the HF strings.
fn layer_type_names(raw: &Value, n_layers: usize) -> Result<Vec<String>> {
    let names: Vec<String> = match field(raw, "layer_types") {
        Some(v) => v
            .as_array()
            .context("lfm3 layer_types must be an array of strings")?
            .iter()
            .map(|e| {
                e.as_str()
                    .map(str::to_owned)
                    .context("lfm3 layer_types entries must be strings")
            })
            .collect::<Result<_>>()?,
        // The HF default pattern: a full-attention layer every fourth layer
        // from the third, a sliding layer in between, convolutions elsewhere.
        None => (0..n_layers)
            .map(|i| match i % 4 {
                2 => "full_attention",
                0 if i > 0 => "sliding_attention",
                _ => "conv",
            })
            .map(str::to_owned)
            .collect(),
    };
    ensure!(
        names.len() == n_layers,
        "lfm3 layer_types has {} entries but num_hidden_layers is {n_layers}",
        names.len()
    );
    for name in &names {
        ensure!(
            matches!(
                name.as_str(),
                "conv" | "full_attention" | "sliding_attention"
            ),
            "lfm3 layer_types entry `{name}` is not one of conv, full_attention, sliding_attention"
        );
    }
    Ok(names)
}

/// One RoPE block, validated: `(rope_theta, partial_rotary_factor)`.
fn rope_block(
    raw: &Value,
    kind: &str,
    default_theta: f64,
    default_factor: f64,
) -> Result<(f64, f64)> {
    let block = raw
        .get("rope_parameters")
        .and_then(|p| p.get(kind))
        .filter(|b| !b.is_null());
    let Some(block) = block else {
        return Ok((default_theta, default_factor));
    };
    let rope_type = block
        .get("rope_type")
        .and_then(Value::as_str)
        .unwrap_or("default");
    ensure!(
        rope_type == "default",
        "lfm3 rope_parameters.{kind}.rope_type `{rope_type}` is not supported (only `default`)"
    );
    let theta = float(block, "rope_theta", Some(default_theta))?;
    let factor = float(block, "partial_rotary_factor", Some(default_factor))?;
    ensure!(
        theta.is_finite() && theta > 1.0,
        "lfm3 rope_parameters.{kind}.rope_theta must be finite and above 1"
    );
    ensure!(
        factor > 0.0 && factor <= 1.0,
        "lfm3 rope_parameters.{kind}.partial_rotary_factor must be in (0, 1]"
    );
    Ok((theta, factor))
}

pub(crate) fn parse_lfm3(raw: &Value) -> Result<ModelConfig> {
    let obj = raw
        .as_object()
        .context("lfm3 config.json must be an object")?;

    // ── Refuse what changes the math and is not implemented ──
    ensure!(
        !flag(raw, "conv_bias", false)?,
        "lfm3 conv_bias = true is not supported (the conv kernels and projections are bias-free)"
    );
    ensure!(
        flag(raw, "use_sandwich_norm", true)?,
        "lfm3 use_sandwich_norm = false is not supported (the loader requires post_operator_norm and post_ffn_norm)"
    );
    if field(raw, "rope_scaling").is_some() {
        bail!("lfm3 rope_scaling is not supported; LFM3 declares RoPE through rope_parameters");
    }
    let conv_width = uint(raw, "conv_L_cache", Some(CONV_WIDTH))?;
    ensure!(
        conv_width == CONV_WIDTH,
        "lfm3 conv_L_cache {conv_width} is not supported (the conv kernels are {CONV_WIDTH}-tap)"
    );

    // ── Dimensions (defaults are Lfm3Config's) ──
    let hidden = uint(raw, "hidden_size", Some(2048))?;
    let n_layers = uint(raw, "num_hidden_layers", Some(24))?;
    let heads = uint(raw, "num_attention_heads", Some(16))?;
    let kv_heads = uint(raw, "num_key_value_heads", Some(4))?;
    let inter = uint(raw, "intermediate_size", Some(7680))?;
    let vocab = uint(raw, "vocab_size", Some(128_000))?;
    let head_dim = uint(raw, "head_dim", Some(hidden / heads.max(1)))?;
    let max_pos = uint(raw, "max_position_embeddings", Some(128_000))?;
    let norm_eps = float(raw, "norm_eps", Some(1e-5))?;
    for (name, value) in [
        ("hidden_size", hidden),
        ("num_hidden_layers", n_layers),
        ("num_attention_heads", heads),
        ("num_key_value_heads", kv_heads),
        ("intermediate_size", inter),
        ("vocab_size", vocab),
        ("head_dim", head_dim),
    ] {
        ensure!(value > 0, "lfm3 {name} must be non-zero");
    }
    ensure!(
        heads.is_multiple_of(kv_heads),
        "lfm3 num_attention_heads ({heads}) must be divisible by num_key_value_heads ({kv_heads})"
    );
    ensure!(
        head_dim.is_multiple_of(2),
        "lfm3 head_dim ({head_dim}) must be even (RoPE rotates pairs)"
    );
    ensure!(
        norm_eps.is_finite() && norm_eps > 0.0,
        "lfm3 norm_eps must be finite and positive"
    );

    // ── Layer schedule ──
    let names = layer_type_names(raw, n_layers)?;
    let layer_types: Vec<LayerType> = names
        .iter()
        .map(|n| match n.as_str() {
            "conv" => LayerType::LinearAttention,
            "full_attention" => LayerType::FullAttention,
            _ => LayerType::SlidingAttention,
        })
        .collect();
    let has_sliding = layer_types.contains(&LayerType::SlidingAttention);
    // HF: absent = 256; an explicit null is `None`, which is an error when a
    // sliding layer exists.
    let sliding_window = match raw.get("sliding_window") {
        None => 256,
        Some(Value::Null) => 0,
        Some(_) => uint(raw, "sliding_window", None)?,
    };
    ensure!(
        !has_sliding || sliding_window > 0,
        "lfm3 has sliding_attention layers but sliding_window is not positive"
    );

    // ── RoPE ──
    let has_full = layer_types.contains(&LayerType::FullAttention);
    let (full_theta, full_factor) = rope_block(raw, "full_attention", 1_000_000.0, 0.5)?;
    let (sliding_theta, sliding_factor) =
        rope_block(raw, "sliding_attention", SLIDING_ROPE_THETA, 1.0)?;
    ensure!(
        !has_sliding || sliding_factor == 1.0,
        "lfm3 sliding_attention partial_rotary_factor {sliding_factor} is not supported (sliding layers rotate the full head)"
    );
    ensure!(
        !has_sliding || sliding_theta == SLIDING_ROPE_THETA,
        "lfm3 sliding_attention rope_theta {sliding_theta} is not supported (the loader builds the table for {SLIDING_ROPE_THETA})"
    );

    // ── Stop tokens: scalar or array; the primary is element 0 ──
    let primary_eos = match field(raw, "eos_token_id") {
        None => 124_900,
        Some(Value::Number(n)) => n
            .as_u64()
            .context("lfm3 eos_token_id must be an unsigned integer")?,
        Some(Value::Array(ids)) => ids
            .first()
            .and_then(Value::as_u64)
            .context("lfm3 eos_token_id array must start with an unsigned integer")?,
        Some(_) => bail!("lfm3 eos_token_id must be an integer or an integer array"),
    };

    // Feed serde a normalized copy: the layer kinds spelled as LayerType
    // names, the EOS collapsed to its primary, the norm epsilon under both of
    // its ModelConfig names.
    let mut normalized = Value::Object(obj.clone());
    {
        let o = normalized
            .as_object_mut()
            .expect("normalized copy of an object is an object");
        o.insert(
            "layer_types".into(),
            Value::Array(
                layer_types
                    .iter()
                    .map(|t| Value::from(t.hf_name()))
                    .collect(),
            ),
        );
        o.insert("eos_token_id".into(), json!(primary_eos));
        o.insert("hidden_size".into(), json!(hidden));
        o.insert("num_hidden_layers".into(), json!(n_layers));
        o.insert("model_type".into(), json!("lfm3"));
    }
    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse lfm3 config.json")?;

    config.model_type = "lfm3".to_string();
    config.hidden_size = hidden;
    config.num_hidden_layers = n_layers;
    config.intermediate_size = inter;
    config.vocab_size = vocab;
    config.num_attention_heads = heads;
    config.num_attention_heads_per_layer.clear();
    config.num_key_value_heads = kv_heads;
    config.head_dim = head_dim;
    config.max_position_embeddings = max_pos;
    config.rms_norm_eps = norm_eps;
    config.layer_types = layer_types;
    config.sliding_window = if has_sliding {
        sliding_window as u32
    } else {
        0
    };
    config.tie_word_embeddings = flag(raw, "tie_word_embeddings", true)?;

    // Full-attention RoPE: theta is the config's `rope_theta`; the rotated
    // dimension is the whole head with a zero-frequency tail (module docs).
    config.rope_theta = if has_full { full_theta } else { sliding_theta };
    config.partial_rotary_factor = if has_full { full_factor } else { 1.0 };
    config.rotary_dim = head_dim;

    // Attention: ungated Q (the sigmoid head gate is a separate projection),
    // per-head QK norm.
    config.attn_gated = false;
    config.qk_norm_type = "per_head".to_string();
    config.weight_prefix.clear();
    config.nested_config = false;

    // Dense FFN, no MoE, no MTP head, no vision tower.
    config.num_experts = 0;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 0;
    config.shared_expert_intermediate_size = 0;
    config.mtp_num_hidden_layers = 0;
    config.num_mtp_modules = 0;
    config.mtp_transformer_layers = 0;
    config.vision = None;

    // The conv geometry on the GDN schema (module docs).
    config.linear_conv_kernel_dim = conv_width;
    config.linear_num_key_heads = SSM_KEY_HEADS;
    config.linear_key_head_dim = SSM_KEY_HEAD_DIM;
    config.linear_num_value_heads = SSM_VALUE_HEADS;
    config.linear_value_head_dim = SSM_VALUE_HEAD_DIM;

    finalize_config(&mut config, raw)?;
    Ok(config)
}
