// SPDX-License-Identifier: AGPL-3.0-only

//! Config / model-dir / vocab-cap helpers.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn merge_sidecar_quant_config(model_dir: &Path, config: &mut ModelConfig) {
    if config.quantization_config.is_some() {
        return;
    }
    let hf_quant_path = model_dir.join("hf_quant_config.json");
    if !hf_quant_path.exists() {
        return;
    }
    match std::fs::read_to_string(&hf_quant_path) {
        Ok(raw_hq) => {
            let wrapped = format!(r#"{{"quantization_config":{raw_hq}}}"#);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&wrapped) {
                config.quantization_config = atlas_core::config::parse_quantization_config(&v);
            }
        }
        Err(e) => tracing::warn!("Failed to read sibling hf_quant_config.json: {e}"),
    }
}

pub(crate) fn load_model_config(model_dir: &Path) -> Result<(ModelConfig, String)> {
    let config_path = model_dir.join("config.json");
    let params_path = model_dir.join("params.json");
    let config_json = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else if params_path.exists() {
        std::fs::read_to_string(&params_path)
            .with_context(|| format!("Failed to read {}", params_path.display()))?
    } else {
        anyhow::bail!(
            "No config.json or params.json found in {}",
            model_dir.display()
        );
    };
    let config = if params_path.exists() && !config_path.exists() {
        atlas_core::config::parse_mistral_params(&config_json)
            .context("Failed to parse params.json (Mistral format)")?
    } else {
        atlas_core::config::parse_config(&config_json).context("Failed to parse config.json")?
    };
    Ok((config, config_json))
}

pub(crate) fn resolve_model_dir(args: &cli::ServeArgs) -> Result<std::path::PathBuf> {
    use crate::model_resolver;
    if let Some(ref path) = args.model_from_path {
        model_resolver::resolve_model_dir(
            path.to_str().context("Invalid model path")?,
            args.cache_dir.as_deref(),
        )
    } else {
        let model_spec = args
            .model
            .as_deref()
            .context("Either MODEL or --model-from-path is required")?;
        model_resolver::resolve_model_dir(model_spec, args.cache_dir.as_deref())
    }
}

pub(crate) fn cap_vocab_size_to_tokenizer(model_dir: &Path, config: &mut ModelConfig) {
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists()
        && let Ok(tok) = tokenizers::Tokenizer::from_file(&tok_path)
    {
        let tok_vocab = tok.get_vocab_size(true);
        if tok_vocab > 0 && tok_vocab < config.vocab_size {
            tracing::info!(
                "Capping vocab_size from {} to {} (tokenizer)",
                config.vocab_size,
                tok_vocab,
            );
            config.vocab_size = tok_vocab;
        }
    }
}

pub(crate) fn apply_model_default_num_drafts(
    args: &mut cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) {
    if ptx_set.behavior.default_num_drafts > 0 && args.num_drafts == 1 {
        let model_default = ptx_set.behavior.default_num_drafts as usize;
        if model_default != args.num_drafts {
            tracing::info!(
                "num_drafts: using MODEL.toml default_num_drafts={} (K={}) — pass --num-drafts to override",
                model_default,
                model_default + 1,
            );
            args.num_drafts = model_default;
        }
    }
}
