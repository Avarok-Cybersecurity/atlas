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

pub(crate) fn load_sampling_defaults(
    model_dir: &Path,
    args: &cli::ServeArgs,
    preset: &atlas_kernels::SamplingCategory,
) -> SamplingDefaults {
    let gen_config_path = model_dir.join("generation_config.json");
    let gen_cfg = std::fs::read_to_string(&gen_config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
    if gen_cfg.is_none() {
        tracing::warn!(
            "generation_config.json absent or unparseable at {}; sampling defaults fall back to \
             the MODEL.toml [sampling.non_thinking] preset",
            gen_config_path.display()
        );
    }
    let d = resolve_sampling_defaults(gen_cfg.as_ref(), args, preset);
    tracing::info!(
        "Default sampling: temperature={}, top_k={}, top_p={}, top_n_sigma={}, min_p={}",
        d.temperature,
        d.top_k,
        d.top_p,
        d.top_n_sigma,
        d.min_p,
    );
    d
}

/// Pure per-field resolution: a field present in `generation_config.json` is
/// honored verbatim (a config that legitimately asks for `temperature=0` gets
/// it); an ABSENT field backfills from the model's curated MODEL.toml
/// `[sampling.non_thinking]` preset (temperature/top_k/top_p/min_p) or the CLI
/// defaults. No hard-coded constants: the old `0.6 / 20 / 0.95` literals
/// duplicated — and could drift from — the preset that already drives the
/// penalties, and a missing config must not silently mean "someone's idea of
/// typical" when the model ships its own numbers.
///
/// `min_p` is three-tier because the preset carries it only when MODEL.toml
/// says so: config → preset (`Some`) → `--default-min-p`. A model card that
/// specifies `min_p = 0` previously had no way to express it — the server's
/// 0.08 default reached every request regardless, and `[behavior].min_p_floor`
/// can only raise min_p, never pin it. A preset that stays silent (`None`)
/// leaves the CLI default owning the field exactly as before.
///
/// `top_n_sigma` remains CLI-only: it is not carried by any model card we
/// serve, and it is separately suspected of being a no-op on several targets,
/// so giving it a MODEL.toml voice would advertise a knob that does nothing.
pub(crate) fn resolve_sampling_defaults(
    gen_cfg: Option<&serde_json::Value>,
    args: &cli::ServeArgs,
    preset: &atlas_kernels::SamplingCategory,
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
        .unwrap_or(args.default_top_n_sigma);
    let min_p = gen_cfg
        .and_then(|v| v.get("min_p")?.as_f64())
        .map(|p| p as f32)
        .or(preset.min_p)
        .unwrap_or(args.default_min_p);
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
    if ptx_set.behavior.use_sampling_presets_for_core {
        let non_thinking = &ptx_set.sampling.non_thinking;
        let tools = &ptx_set.sampling.tools;
        tracing::info!(
            "Model behavior: MODEL sampling defaults enabled (non-thinking: temp={}, top_k={}, top_p={}; tools: temp={}, top_k={}, top_p={})",
            non_thinking.temperature,
            non_thinking.top_k,
            non_thinking.top_p,
            tools.temperature,
            tools.top_k,
            tools.top_p,
        );
    }
    // The content-loop watchdog value reaches the scheduler as
    // `SchedLevers::loop_watchdog`, armed in `serve`; this phase only audits it.
    if !ptx_set.behavior.enable_think_loop_watchdog {
        tracing::info!(
            "Model behavior: THINKING-loop watchdog DISABLED (per MODEL.toml \
             [behavior].enable_think_loop_watchdog = false)"
        );
    }
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
    if crate::scheduler::parse_disable_watchdogs(
        std::env::var("ATLAS_DISABLE_WATCHDOGS").ok().as_deref(),
    ) {
        tracing::warn!(
            "Model behavior: ALL auto-watchdogs DISABLED via ATLAS_DISABLE_WATCHDOGS=1 \
             (content-loop, inter-tool prose, F2 confidence early-stop, mid-word </think> \
             defer, thinking-loop). User-set max_thinking_budget and safety masks unaffected. \
             Use only for empirical-test runs — re-enable for production."
        );
    }
    // Phase-A: per-model watchdog tunables from MODEL.toml [behavior]. The
    // values themselves reach the scheduler as `SchedCtx::watchdog`, built in
    // `serve` before the scheduler thread spawns; this phase only audits them.
    let b = &ptx_set.behavior;
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
    // The value itself reaches the renderers via `AppState::prompt_levers`;
    // this phase only audits it.
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
mod sampling_defaults_tests {
    use clap::Parser;

    use super::resolve_sampling_defaults;
    use crate::cli;

    /// A preset with values distinct from both the old hard-coded constants
    /// (0.6 / 20 / 0.95) and the CLI defaults, so a wrong fallback source is
    /// unmistakable in every assertion below.
    fn preset() -> atlas_kernels::SamplingCategory {
        atlas_kernels::SamplingCategory {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 40,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
            min_p: None,
            top_n_sigma: None,
        }
    }

    fn args() -> cli::ServeArgs {
        cli::ServeArgs::parse_from(["spark", "org/model"])
    }

    #[test]
    fn missing_config_falls_back_to_preset_not_constants() {
        let d = resolve_sampling_defaults(None, &args(), &preset());
        assert_eq!(d.temperature, 0.7, "preset, not the old 0.6 literal");
        assert_eq!(d.top_k, 40, "preset, not the old 20 literal");
        assert_eq!(d.top_p, 0.8, "preset, not the old 0.95 literal");
        assert!(d.temperature > 0.0, "absent config must never mean greedy");
    }

    #[test]
    fn present_config_overrides_preset() {
        let cfg = serde_json::json!({"temperature": 1.1, "top_k": 64, "top_p": 0.9});
        let d = resolve_sampling_defaults(Some(&cfg), &args(), &preset());
        assert_eq!(d.temperature, 1.1);
        assert_eq!(d.top_k, 64);
        assert_eq!(d.top_p, 0.9);
    }

    #[test]
    fn partial_config_backfills_per_field() {
        let cfg = serde_json::json!({"top_k": 64});
        let d = resolve_sampling_defaults(Some(&cfg), &args(), &preset());
        assert_eq!(d.top_k, 64, "present field honored");
        assert_eq!(d.temperature, 0.7, "absent field from preset");
        assert_eq!(d.top_p, 0.8, "absent field from preset");
    }

    #[test]
    fn explicit_temperature_zero_is_honored() {
        // A config that is PRESENT and asks for greedy gets greedy — the
        // preset guard only backfills absent fields, it never overrides.
        let cfg = serde_json::json!({"temperature": 0.0});
        let d = resolve_sampling_defaults(Some(&cfg), &args(), &preset());
        assert_eq!(d.temperature, 0.0);
    }

    #[test]
    fn sigma_and_min_p_fall_back_to_cli_args() {
        // top_n_sigma is CLI-owned outright; min_p is CLI-owned only while
        // the preset stays silent (`min_p: None`), which `preset()` is.
        let a = args();
        let d = resolve_sampling_defaults(None, &a, &preset());
        assert_eq!(d.top_n_sigma, a.default_top_n_sigma);
        assert_eq!(d.min_p, a.default_min_p);
    }

    #[test]
    fn model_declared_min_p_zero_beats_the_cli_default() {
        // The reason this field exists. A card that specifies min_p = 0 must
        // get 0, not the server's 0.08 — and 0.0 is exactly the value a
        // non-Option field could not distinguish from "unset".
        let a = args();
        assert!(a.default_min_p > 0.0, "guard: CLI default must be nonzero");
        let mut p = preset();
        p.min_p = Some(0.0);
        let d = resolve_sampling_defaults(None, &a, &p);
        assert_eq!(d.min_p, 0.0, "model preset must win over --default-min-p");
    }

    #[test]
    fn generation_config_still_outranks_a_declared_min_p() {
        // Same precedence as temperature/top_k/top_p: the checkpoint's own
        // generation_config.json is the most specific source and wins.
        let cfg = serde_json::json!({"min_p": 0.31});
        let mut p = preset();
        p.min_p = Some(0.0);
        let d = resolve_sampling_defaults(Some(&cfg), &args(), &p);
        assert_eq!(d.min_p, 0.31);
    }
}
