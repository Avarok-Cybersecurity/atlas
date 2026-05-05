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
    })
}
