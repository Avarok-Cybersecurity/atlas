// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime helpers run after model build: EOS / sampling defaults from
//! generation_config.json, dump-writer open, response-store / behavior
//! audit logging, model-name resolution, tool-call parser dispatch.

use std::path::Path;

use anyhow::Result;

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) fn load_eos_tokens(model_dir: &Path, config: &ModelConfig) -> Vec<u32> {
    let gen_config_path = model_dir.join("generation_config.json");
    if let Ok(gen_json) = std::fs::read_to_string(&gen_config_path) {
        if let Ok(gen_cfg) = serde_json::from_str::<serde_json::Value>(&gen_json) {
            return match gen_cfg.get("eos_token_id") {
                Some(serde_json::Value::Array(arr)) => {
                    let ids: Vec<u32> = arr
                        .iter()
                        .filter_map(|v| v.as_u64().map(|n| n as u32))
                        .collect();
                    if !ids.is_empty() {
                        tracing::info!("EOS tokens (from generation_config.json): {:?}", ids);
                        ids
                    } else {
                        vec![config.eos_token_id]
                    }
                }
                Some(serde_json::Value::Number(n)) => {
                    let id = n.as_u64().unwrap_or(0) as u32;
                    tracing::info!("EOS token (from generation_config.json): {}", id);
                    vec![id]
                }
                _ => vec![config.eos_token_id],
            };
        }
        return vec![config.eos_token_id];
    }
    tracing::info!("EOS token (from config.json): {}", config.eos_token_id);
    vec![config.eos_token_id]
}

pub(crate) struct SamplingDefaults {
    pub(crate) temperature: f32,
    pub(crate) top_k: u32,
    pub(crate) top_p: f32,
    pub(crate) top_n_sigma: f32,
    pub(crate) min_p: f32,
}

pub(crate) fn load_sampling_defaults(model_dir: &Path, args: &cli::ServeArgs) -> SamplingDefaults {
    let gen_config_path = model_dir.join("generation_config.json");
    let gen_cfg = std::fs::read_to_string(&gen_config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    let temperature = gen_cfg
        .as_ref()
        .and_then(|v| v.get("temperature")?.as_f64())
        .map(|t| t as f32)
        .unwrap_or(0.6);
    let top_k = gen_cfg
        .as_ref()
        .and_then(|v| v.get("top_k")?.as_u64())
        .map(|k| k as u32)
        .unwrap_or(20);
    let top_p = gen_cfg
        .as_ref()
        .and_then(|v| v.get("top_p")?.as_f64())
        .map(|p| p as f32)
        .unwrap_or(0.95);
    let top_n_sigma = gen_cfg
        .as_ref()
        .and_then(|v| v.get("top_n_sigma")?.as_f64())
        .map(|s| s as f32)
        .unwrap_or(args.default_top_n_sigma);
    let min_p = gen_cfg
        .as_ref()
        .and_then(|v| v.get("min_p")?.as_f64())
        .map(|p| p as f32)
        .unwrap_or(args.default_min_p);
    tracing::info!(
        "Default sampling: temperature={temperature}, top_k={top_k}, top_p={top_p}, top_n_sigma={top_n_sigma}, min_p={min_p}"
    );
    SamplingDefaults {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
    }
}

pub(crate) fn open_dump_writer(args: &cli::ServeArgs) -> Option<crate::request_dumper::DumpHandle> {
    use crate::request_dumper;
    match args.dump.as_deref() {
        Some(arg) => {
            let path = request_dumper::resolve_path(arg);
            match request_dumper::DumpHandle::open(path) {
                Ok(h) => {
                    tracing::info!(
                        path = %h.path().display(),
                        "Request dump enabled (JSONL append)"
                    );
                    Some(h)
                }
                Err(e) => {
                    tracing::error!("Failed to open --dump target: {e}. Dumping is disabled.");
                    None
                }
            }
        }
        None => None,
    }
}

pub(crate) fn log_response_store_audit(
    response_store: &crate::response_store::ResponseStore,
    rate_limiter: &crate::rate_limiter::RateLimiter,
) {
    if rate_limiter.config().is_enabled() {
        let cfg = rate_limiter.config();
        tracing::info!(
            "Rate limiter active: {} req/min, {} tok/min (bursts {}/{})",
            cfg.rpm,
            cfg.tpm,
            cfg.burst_rpm,
            cfg.burst_tpm
        );
    }
    tracing::info!(
        "Response store: max_entries={}, ttl={:?}, persist={}",
        response_store.max_entries(),
        response_store.ttl(),
        match response_store.persist_dir() {
            Some(p) => format!("filesystem ({})", p.display()),
            None => "memory-only".to_string(),
        }
    );
    if response_store.is_persistent() && response_store.len() > 0 {
        tracing::info!(
            "Response store: replayed {} entries from disk",
            response_store.len()
        );
    }
}

pub(crate) fn log_behavior_audit(args: &cli::ServeArgs, ptx_set: &atlas_kernels::TargetPtxSet) {
    if !ptx_set.behavior.thinking_in_tools {
        tracing::info!("Model behavior: thinking disabled when tools active (MODEL.toml)");
    }
    let effective_thinking_budget = args
        .max_thinking_budget
        .unwrap_or(ptx_set.behavior.max_thinking_budget);
    tracing::info!(
        "Model behavior: max_thinking_budget={}{}, thinking_default={}",
        effective_thinking_budget,
        if args.max_thinking_budget.is_some() {
            " (CLI override)"
        } else {
            ""
        },
        ptx_set.behavior.thinking_default,
    );
    crate::scheduler::set_enable_loop_watchdog(ptx_set.behavior.enable_loop_watchdog);
    if ptx_set.behavior.enable_loop_watchdog {
        tracing::info!(
            "Model behavior: content-loop watchdog ENABLED (period-{}…{} repetition detector)",
            crate::scheduler::CONTENT_LOOP_PERIOD_MIN,
            crate::scheduler::CONTENT_LOOP_PERIOD_MAX,
        );
    }
    if args.disable_thinking {
        tracing::info!("--disable-thinking set: thinking is forced OFF for every request");
    }
    if let Some(threshold) = args.auto_compact {
        tracing::info!(
            "Auto-compact enabled: threshold={:.0}% of max_seq_len ({})",
            threshold * 100.0,
            args.max_seq_len
        );
    }
}

pub(crate) fn resolve_model_name(
    args: &cli::ServeArgs,
    config_json: &str,
    model_dir: &Path,
) -> String {
    args.model_name
        .clone()
        .or_else(|| args.model.clone())
        .or_else(|| {
            serde_json::from_str::<serde_json::Value>(config_json)
                .ok()
                .and_then(|v| v.get("_name_or_path")?.as_str().map(String::from))
        })
        .unwrap_or_else(|| {
            model_dir
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "atlas".to_string())
        })
}

pub(crate) fn resolve_tool_call_parser(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    config: &ModelConfig,
) -> Result<Option<std::sync::Arc<dyn crate::tool_parser::ToolCallParser>>> {
    use crate::tool_parser;
    let tool_call_format: Option<tool_parser::ToolCallFormat> =
        if let Some(ref parser) = args.tool_call_parser {
            let format: tool_parser::ToolCallFormat =
                parser.parse().map_err(|e: String| anyhow::anyhow!(e))?;
            tracing::info!("Tool call parser: {} (user-specified)", format.name());
            Some(format)
        } else if !ptx_set.behavior.tool_call_parser.is_empty() {
            let format: tool_parser::ToolCallFormat = ptx_set
                .behavior
                .tool_call_parser
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;
            tracing::info!(
                "Tool call parser: {} (MODEL.toml [behavior].tool_call_parser)",
                format.name()
            );
            Some(format)
        } else {
            let defaults: toml::Table = toml::from_str(include_str!("../../../tool_defaults.toml"))
                .expect("invalid tool_defaults.toml");
            let auto_format = defaults
                .get("model_type")
                .and_then(|t| t.as_table())
                .and_then(|t| t.get(config.model_type.as_str()))
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse::<tool_parser::ToolCallFormat>().ok());
            if let Some(format) = auto_format {
                tracing::info!(
                    "Tool call parser: {} (auto-detected from model_type '{}')",
                    format.name(),
                    config.model_type
                );
                Some(format)
            } else {
                tracing::info!(
                    "Tool call parser: disabled (no mapping for model_type '{}')",
                    config.model_type
                );
                None
            }
        };

    if let Some(format) = tool_call_format {
        if format.has_grammar() {
            tracing::info!(
                "Tool call parser: '{}' has registered XGrammar grammar — constrained decoding ENABLED for tool requests",
                format.name()
            );
        } else {
            tracing::warn!(
                "Tool call parser: '{}' has NO XGrammar grammar registered — constrained decoding DISABLED. \
                 Tool calls rely entirely on model-trained behavior; degraded quality possible.",
                format.name()
            );
        }
    }
    Ok(tool_call_format.map(|f| std::sync::Arc::from(f.into_parser())))
}
