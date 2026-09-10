// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{ModelConfig, VisionConfig};

pub(crate) fn parse_vision_config(raw: &serde_json::Value) -> Option<VisionConfig> {
    let vc = raw.get("vision_config")?;
    let get_usize = |key: &str| -> usize {
        vc.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
    };
    let deepstack_visual_indexes = vc
        .get("deepstack_visual_indexes")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    // Some checkpoints declare the image placeholder token at the TOP
    // level (Qwen3.6: `image_token_id`). Older VL configs embed it under
    // `vision_config`. Read both; fall back to 0 which downstream treats
    // as "use the Qwen3-VL default 151655".
    let image_pad_token_id = raw
        .get("image_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("image_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as u32;
    // Same two-location dance as the image token, and the same fallback.
    let video_pad_token_id = raw
        .get("video_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("video_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as u32;
    Some(VisionConfig {
        depth: get_usize("depth"),
        hidden_size: get_usize("hidden_size"),
        num_heads: get_usize("num_heads"),
        patch_size: get_usize("patch_size"),
        temporal_patch_size: get_usize("temporal_patch_size"),
        spatial_merge_size: get_usize("spatial_merge_size"),
        intermediate_size: get_usize("intermediate_size"),
        out_hidden_size: get_usize("out_hidden_size"),
        deepstack_visual_indexes,
        image_pad_token_id,
        video_pad_token_id,
        // Not in config.json — it comes from preprocessor_config.json (or the
        // operator's flag), which this parser does not see. Resolved and
        // installed by the server right after config load, before the encoder
        // is built. `None` here means "not yet resolved", never "unbounded".
        max_pixels: None,
        // `vision_config.model_type` is the ONLY family discriminant the
        // preprocessor is allowed to dispatch on. Absent on older VL configs,
        // where the empty string keeps the historical Qwen arm.
        model_type: vc
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        projection_intermediate_size: vc
            .get("projection_intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize),
        swiglu_limit: vc
            .get("swiglu_limit")
            .and_then(serde_json::Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v as f32),
        // Default TRUE: every tower Atlas binds today is biased, and a
        // checkpoint that says otherwise should be refused at bind time
        // rather than silently served with the biases it does not have.
        attention_bias: vc
            .get("attention_bias")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        // Same rationale as `max_pixels`: the token budget and the per-channel
        // normalisation live in the PROCESSOR config, which this parser never
        // sees. The server resolves them right after config load, in the same
        // block that installs `max_pixels`.
        min_image_tokens: None,
        max_image_tokens: None,
        image_mean: None,
        image_std: None,
    })
}
