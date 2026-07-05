// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime helpers run after model build: EOS / sampling defaults from
//! generation_config.json, dump-writer open, response-store / behavior
//! audit logging, model-name resolution, tool-call parser dispatch.

use std::path::Path;

use anyhow::Result;

use atlas_core::config::ModelConfig;
use atlas_kernels::SamplingCategory;

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

pub(crate) fn load_sampling_defaults(
    model_dir: &Path,
    args: &cli::ServeArgs,
    preset: &SamplingCategory,
) -> SamplingDefaults {
    let gen_config_path = model_dir.join("generation_config.json");
    let gen_cfg = std::fs::read_to_string(&gen_config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    if gen_cfg.is_none() {
        tracing::warn!(
            "No parseable generation_config.json at {} — falling back to MODEL.toml sampling preset (temperature={}, top_k={}, top_p={})",
            gen_config_path.display(),
            preset.temperature,
            preset.top_k,
            preset.top_p
        );
    }
    let defaults = resolve_sampling_defaults(
        gen_cfg.as_ref(),
        preset,
        args.default_top_n_sigma,
        args.default_min_p,
    );
    tracing::info!(
        "Default sampling: temperature={}, top_k={}, top_p={}, top_n_sigma={}, min_p={}",
        defaults.temperature,
        defaults.top_k,
        defaults.top_p,
        defaults.top_n_sigma,
        defaults.min_p
    );
    defaults
}

/// Resolve the request-level sampling defaults from `generation_config.json`,
/// field-by-field, falling back to the model's curated MODEL.toml `preset`
/// (not hard-coded constants) whenever the config is absent, unparseable, or
/// missing a given field.
///
/// The preset fallback is the guard: an absent/unparseable `generation_config`
/// used to leave `temperature`/`top_k` at hard-coded values that could drift
/// from the model's curated preset; sourcing the fallback from the same preset
/// that drives the penalties keeps a single source of truth and guarantees the
/// defaults are non-degenerate (never silently `0` → greedy). A config that is
/// *present* and legitimately requests `temperature=0` is still honored.
fn resolve_sampling_defaults(
    gen_cfg: Option<&serde_json::Value>,
    preset: &SamplingCategory,
    default_top_n_sigma: f32,
    default_min_p: f32,
) -> SamplingDefaults {
    let temperature = gen_cfg
        .and_then(|v| v.get("temperature")?.as_f64())
        .map(|t| t as f32)
        .unwrap_or(preset.temperature);
    let top_k = gen_cfg
        .and_then(|v| v.get("top_k")?.as_u64())
        .map(|k| k as u32)
        .unwrap_or(preset.top_k);
    let top_p = gen_cfg
        .and_then(|v| v.get("top_p")?.as_f64())
        .map(|p| p as f32)
        .unwrap_or(preset.top_p);
    let top_n_sigma = gen_cfg
        .and_then(|v| v.get("top_n_sigma")?.as_f64())
        .map(|s| s as f32)
        .unwrap_or(default_top_n_sigma);
    let min_p = gen_cfg
        .and_then(|v| v.get("min_p")?.as_f64())
        .map(|p| p as f32)
        .unwrap_or(default_min_p);
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
    // 2026-05-24: ATLAS_DISABLE_WATCHDOGS env var disables ALL
    // auto-watchdogs (content-loop, inter-tool prose, F2 confidence,
    // mid-word </think>, thinking-loop). Empirical test toggle —
    // surface its state prominently at boot.
    if crate::scheduler::disable_watchdogs() {
        tracing::warn!(
            "Model behavior: ALL auto-watchdogs DISABLED via ATLAS_DISABLE_WATCHDOGS=1 \
             (content-loop, inter-tool prose, F2 confidence early-stop, mid-word </think> \
             defer, thinking-loop). User-set max_thinking_budget and safety masks unaffected. \
             Use only for empirical-test runs — re-enable for production."
        );
    }
    // Phase-A: per-model watchdog tunables from MODEL.toml [behavior].
    let b = &ptx_set.behavior;
    crate::scheduler::set_watchdog_params(crate::scheduler::WatchdogParams {
        think_loop_min_repeats: b.think_loop_min_repeats as usize,
        think_loop_scan_window: b.think_loop_scan_window as usize,
        confidence_early_stop: b.confidence_early_stop,
        confidence_run_length: b.confidence_run_length,
        fuzzy_repeat_tolerance_div: b.fuzzy_repeat_tolerance_div as usize,
        max_inter_tool_prose: b.max_inter_tool_prose,
        max_post_think_content_tokens: b.max_post_think_content_tokens,
        rollback_resteer: b.rollback_resteer,
    });
    if !b.confidence_early_stop {
        tracing::info!("Model behavior: F2 confidence early-stop DISABLED");
    }
    // Phase-C: watchdog rollback + re-steer (arXiv:2603.27905).
    if b.rollback_resteer {
        tracing::info!(
            "Model behavior: watchdog rollback+re-steer ENABLED (cap {} per sequence)",
            atlas_kernels::ROLLBACK_RESTEER_CAP,
        );
    } else {
        tracing::info!("Model behavior: watchdog rollback+re-steer DISABLED (legacy hard-stop)");
    }
    // Phase-C ROM (arXiv:2603.22016) scaffold. A trained repetition-onset
    // detection head can be dropped in via MODEL.toml [behavior].rom_head;
    // the runtime would load the artifact and call `set_rom_head`. No
    // trained head ships with Atlas, so when `rom_head` is empty (the
    // default) the F2 confidence heuristic stays as the fallback —
    // unchanged. Loading the artifact is intentionally a future step:
    // only the optional hook (the `RomHead` trait seam) is wired now.
    if !b.rom_head.is_empty() {
        tracing::warn!(
            rom_head = b.rom_head,
            "Model behavior: [behavior].rom_head is set but ROM artifact \
             loading is not yet implemented — F2 confidence heuristic \
             remains the active detector (Phase-C scaffold only)"
        );
    }
    // Phase-B: TSCG tool-schema compilation (MODEL.toml [behavior].tscg).
    crate::tscg::set_tscg_enabled(b.tscg);
    if b.tscg {
        tracing::info!("Model behavior: TSCG tool-schema compilation ENABLED (compact signatures)");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn preset() -> SamplingCategory {
        // A curated, deliberately non-greedy preset (mirrors a MODEL.toml
        // `[sampling.non_thinking]`).
        atlas_kernels::SamplingPresets::default().non_thinking
    }

    #[test]
    fn missing_generation_config_falls_back_to_preset_not_greedy() {
        let p = preset();
        // gen_cfg = None models an absent or unparseable generation_config.json.
        let d = resolve_sampling_defaults(None, &p, 0.0, 0.0);
        assert_eq!(d.temperature, p.temperature);
        assert_eq!(d.top_k, p.top_k);
        assert_eq!(d.top_p, p.top_p);
        // The guard's whole point: the fallback must not be a silent greedy 0.
        assert!(
            d.temperature > 0.0,
            "fallback temperature must be non-greedy"
        );
        assert!(d.top_k > 0, "fallback top_k must not collapse to 0");
        assert!(d.top_p > 0.0, "fallback top_p must be usable");
        // top_n_sigma / min_p fall back to the CLI-arg defaults.
        assert_eq!(d.top_n_sigma, 0.0);
        assert_eq!(d.min_p, 0.0);
    }

    #[test]
    fn present_generation_config_overrides_preset() {
        let p = preset();
        let cfg = serde_json::json!({
            "temperature": 0.15,
            "top_k": 7,
            "top_p": 0.5,
            "top_n_sigma": 1.5,
            "min_p": 0.02
        });
        let d = resolve_sampling_defaults(Some(&cfg), &p, 0.0, 0.0);
        assert_eq!(d.temperature, 0.15);
        assert_eq!(d.top_k, 7);
        assert_eq!(d.top_p, 0.5);
        assert_eq!(d.top_n_sigma, 1.5);
        assert_eq!(d.min_p, 0.02);
    }

    #[test]
    fn partial_generation_config_falls_back_per_field() {
        let p = preset();
        // Only temperature is present; every other field must fall back.
        let cfg = serde_json::json!({ "temperature": 0.9 });
        let d = resolve_sampling_defaults(Some(&cfg), &p, 3.0, 0.05);
        assert_eq!(d.temperature, 0.9); // from config
        assert_eq!(d.top_k, p.top_k); // preset fallback
        assert_eq!(d.top_p, p.top_p); // preset fallback
        assert_eq!(d.top_n_sigma, 3.0); // arg fallback
        assert_eq!(d.min_p, 0.05); // arg fallback
    }

    #[test]
    fn present_config_may_legitimately_request_greedy() {
        // A config that is present and explicitly asks for temperature=0 is
        // honored — the guard only backfills *absent* fields, it does not
        // clamp an intentional greedy request.
        let p = preset();
        let cfg = serde_json::json!({ "temperature": 0.0 });
        let d = resolve_sampling_defaults(Some(&cfg), &p, 0.0, 0.0);
        assert_eq!(d.temperature, 0.0);
    }
}
