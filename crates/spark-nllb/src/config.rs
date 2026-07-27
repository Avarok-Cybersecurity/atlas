// SPDX-License-Identifier: AGPL-3.0-only

//! NLLB / M2M-100 model configuration, parsed from HuggingFace `config.json`.

use anyhow::{Context, Result};
use serde::Deserialize;

/// M2M-100 / NLLB encoder-decoder configuration.
///
/// Field names mirror the HuggingFace `M2M100Config` JSON keys so the raw
/// `config.json` deserializes directly.
#[derive(Debug, Clone, Deserialize)]
pub struct NllbConfig {
    #[serde(rename = "d_model")]
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_pad")]
    pub pad_token_id: u32,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    #[serde(default = "default_eos")]
    pub decoder_start_token_id: u32,
    #[serde(default = "default_true")]
    pub scale_embedding: bool,
    #[serde(default = "default_activation")]
    pub activation_function: String,
    // HF configs frequently ship `"max_length": null`; `serde(default)` only
    // fires on an ABSENT key, so tolerate an explicit null too.
    #[serde(
        default = "default_max_length",
        deserialize_with = "null_or_default_len"
    )]
    pub max_length: usize,
}

/// Deserialize `usize`, mapping an explicit JSON `null` to the default length.
fn null_or_default_len<'de, D>(de: D) -> Result<usize, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<usize>::deserialize(de)?.unwrap_or_else(default_max_length))
}

fn default_pad() -> u32 {
    1
}
fn default_eos() -> u32 {
    2
}
fn default_true() -> bool {
    true
}
fn default_activation() -> String {
    "relu".to_string()
}
#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

fn default_max_length() -> usize {
    200
}

impl NllbConfig {
    /// Parse a HuggingFace `config.json` string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("failed to parse NLLB config.json")
    }

    /// Head dimension (shared by encoder + decoder — NLLB uses one `d_model`).
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Embedding scale factor (`sqrt(d_model)` when `scale_embedding`).
    pub fn embed_scale(&self) -> f32 {
        if self.scale_embedding {
            (self.d_model as f32).sqrt()
        } else {
            1.0
        }
    }
}
