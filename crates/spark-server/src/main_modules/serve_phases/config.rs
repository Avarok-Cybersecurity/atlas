// SPDX-License-Identifier: AGPL-3.0-only

//! Config / model-dir / vocab-cap helpers.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn merge_sidecar_quant_config(model_dir: &Path, config: &mut ModelConfig) {
    let hf_quant_path = model_dir.join("hf_quant_config.json");
    if !hf_quant_path.exists() {
        return;
    }
    // The ModelOpt sidecar is the authoritative dump for its checkpoints:
    // transformers 5.x ADDITIONALLY embeds a compressed-tensors mirror in
    // config.json whose `config_groups.group_0` describes only ONE of a
    // mixed-precision checkpoint's groups (Nemotron-3.5 Lightning: group_0
    // is the FP8 mamba-projection group), so an embedded-block-wins rule
    // mislabels the whole model FP8 and NVFP4 detection breaks. When the
    // sidecar exists and parses, it wins; the embedded block only serves
    // checkpoints that ship no sidecar.
    match std::fs::read_to_string(&hf_quant_path) {
        Ok(raw_hq) => {
            let wrapped = format!(r#"{{"quantization_config":{raw_hq}}}"#);
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&wrapped)
                && let Some(sidecar) = atlas_core::config::parse_quantization_config(&v)
            {
                if let Some(ref embedded) = config.quantization_config
                    && embedded.quant_algo != sidecar.quant_algo
                {
                    tracing::info!(
                        "hf_quant_config.json overrides embedded quantization_config \
                         ({} -> {})",
                        embedded.quant_algo,
                        sidecar.quant_algo
                    );
                }
                config.quantization_config = Some(sidecar);
            }
        }
        Err(e) => tracing::warn!("Failed to read sibling hf_quant_config.json: {e}"),
    }
}

pub(crate) fn load_model_config(model_dir: &Path) -> Result<(ModelConfig, String)> {
    let config_path = model_dir.join("config.json");
    let params_path = model_dir.join("params.json");

    // Bare-GGUF directory (no config.json/params.json): synthesize the config
    // from the GGUF metadata block. Weight loading already routes to GgufLoader.
    if !config_path.exists()
        && !params_path.exists()
        && spark_runtime::weights::find_gguf(model_dir).is_some()
    {
        let config = spark_runtime::weights::config_from_gguf_dir(model_dir)
            .context("Failed to build ModelConfig from GGUF metadata")?;
        tracing::info!(
            "Built ModelConfig from GGUF metadata (model_type={}, layers={}, hidden={})",
            config.model_type,
            config.num_hidden_layers,
            config.hidden_size,
        );
        // No config.json string exists; the only downstream consumer
        // (resolve_model_name) falls back to the directory name.
        return Ok((config, String::new()));
    }

    let config_json = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else if params_path.exists() {
        std::fs::read_to_string(&params_path)
            .with_context(|| format!("Failed to read {}", params_path.display()))?
    } else {
        anyhow::bail!(
            "No config.json, params.json, or .gguf found in {}",
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

/// Where the effective `num_drafts` came from. The CLI → MODEL.toml → engine
/// precedence is explicit (PCND); `apply_model_default_num_drafts` logs per
/// source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NumDraftsSource {
    Cli,
    ModelDefault,
    EngineDefault,
}

/// Resolve the effective draft count. An explicitly passed `--num-drafts`
/// ALWAYS wins — including `--num-drafts 1`: the previous sentinel test
/// (`args.num_drafts == 1` against the clap default) could not tell an
/// explicit 1 from an omitted flag and silently served the MODEL.toml
/// default instead. An omitted flag falls back to MODEL.toml
/// `[behavior].default_num_drafts` when set (> 0), else the engine default.
pub(crate) fn resolve_num_drafts(
    cli_num_drafts: Option<usize>,
    model_default_num_drafts: u32,
) -> (usize, NumDraftsSource) {
    let model_default = (model_default_num_drafts > 0).then_some(model_default_num_drafts as usize);
    match (cli_num_drafts, model_default) {
        (Some(v), _) => (v, NumDraftsSource::Cli),
        (None, Some(md)) => (md, NumDraftsSource::ModelDefault),
        (None, None) => (cli::DEFAULT_NUM_DRAFTS, NumDraftsSource::EngineDefault),
    }
}

pub(crate) fn apply_model_default_num_drafts(
    args: &mut cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) {
    let (effective, source) =
        resolve_num_drafts(args.num_drafts, ptx_set.behavior.default_num_drafts);
    match source {
        NumDraftsSource::Cli => {
            let model_default = ptx_set.behavior.default_num_drafts as usize;
            if ptx_set.behavior.default_num_drafts > 0 && model_default != effective {
                tracing::info!(
                    "num_drafts: {} (K={}) from --num-drafts, overriding MODEL.toml default_num_drafts={}",
                    effective,
                    effective + 1,
                    model_default,
                );
            }
        }
        NumDraftsSource::ModelDefault => {
            tracing::info!(
                "num_drafts: using MODEL.toml default_num_drafts={} (K={}) — pass --num-drafts to override",
                effective,
                effective + 1,
            );
        }
        NumDraftsSource::EngineDefault => {}
    }
    args.num_drafts = Some(effective);
}

#[cfg(test)]
mod tests {
    use super::{NumDraftsSource, resolve_num_drafts};

    /// The observed dgx2 bug: `--num-drafts 1` on a model with
    /// `default_num_drafts = 3` must serve 1 (K=2), not 3 (K=4).
    #[test]
    fn explicit_cli_value_equal_to_engine_default_beats_model_default() {
        assert_eq!(resolve_num_drafts(Some(1), 3), (1, NumDraftsSource::Cli));
    }

    #[test]
    fn explicit_cli_value_beats_model_default() {
        assert_eq!(resolve_num_drafts(Some(2), 3), (2, NumDraftsSource::Cli));
        assert_eq!(resolve_num_drafts(Some(3), 1), (3, NumDraftsSource::Cli));
    }

    #[test]
    fn omitted_flag_falls_back_to_model_default() {
        assert_eq!(
            resolve_num_drafts(None, 3),
            (3, NumDraftsSource::ModelDefault)
        );
    }

    #[test]
    fn omitted_flag_without_model_default_uses_engine_default() {
        assert_eq!(
            resolve_num_drafts(None, 0),
            (
                crate::cli::DEFAULT_NUM_DRAFTS,
                NumDraftsSource::EngineDefault
            )
        );
    }
}

#[cfg(test)]
mod sidecar_tests {
    use super::merge_sidecar_quant_config;
    use atlas_core::config::ModelConfig;

    /// Nemotron-3.5 Lightning class: transformers 5.x embeds a
    /// compressed-tensors `quantization_config` whose group_0 is the FP8
    /// (mamba-projection) group, while the ModelOpt sidecar carries the
    /// authoritative MIXED_PRECISION layout. Fails without the
    /// sidecar-wins rule: the early return keeps the embedded FP8 label.
    #[test]
    fn modelopt_sidecar_overrides_embedded_group0_label() {
        let dir = std::env::temp_dir().join(format!("sidecar-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("hf_quant_config.json"),
            r#"{"producer":{"name":"modelopt","version":"0.44.0"},
                "quantization":{"quant_algo":"MIXED_PRECISION",
                                "kv_cache_quant_algo":"FP8",
                                "quantized_layers":{
                                  "backbone.layers.0.mixer.in_proj":{"quant_algo":"FP8"}}}}"#,
        )
        .unwrap();

        // Embedded compressed-tensors block as parse_config would have left
        // it: group_0 (8, "float") mined into a model-wide "FP8" label.
        let embedded = serde_json::json!({
            "quantization_config": {
                "config_groups": {
                    "group_0": { "weights": { "num_bits": 8, "type": "float" } }
                }
            }
        });
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.quantization_config = None;
        config.quantization_config = atlas_core::config::parse_quantization_config(&embedded);
        assert_eq!(
            config.quantization_config.as_ref().unwrap().quant_algo,
            "FP8",
            "precondition: embedded group_0 mining labels the model FP8"
        );

        merge_sidecar_quant_config(&dir, &mut config);
        let qc = config.quantization_config.expect("sidecar must parse");
        assert_eq!(qc.quant_algo, "MIXED_PRECISION");
        assert_eq!(qc.quant_method, "modelopt");

        std::fs::remove_dir_all(&dir).ok();
    }
}
