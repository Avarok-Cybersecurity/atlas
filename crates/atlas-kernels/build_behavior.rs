// SPDX-License-Identifier: AGPL-3.0-only
//
// `[behavior]` MODEL.toml parsing for build.rs. Split from build_parse.rs
// (500-LoC cap). Included via `#[path = "build_behavior.rs"] mod build_behavior;`.

/// Parsed `[behavior]` table from a model's MODEL.toml. Field defaults
/// match `ModelBehavior::default()` so an absent table / absent field is
/// behavior-neutral.
#[derive(Clone)]
pub(super) struct ParsedBehavior {
    pub thinking_in_tools: bool,
    pub max_thinking_budget: u32,
    pub thinking_default: bool,
    pub fp8_kv_calibration_tokens: usize,
    pub default_kv_dtype: String,
    pub default_num_drafts: u32,
    pub disable_tool_steering: bool,
    pub tool_call_parser: String,
    pub enable_loop_watchdog: bool,
    pub min_p_floor: f32,
    pub temperature_max: f32,
    pub think_loop_min_repeats: u32,
    pub think_loop_scan_window: u32,
    pub confidence_early_stop: bool,
    pub confidence_run_length: u32,
    pub fuzzy_repeat_tolerance_div: u32,
    pub max_inter_tool_prose: u32,
    pub max_post_think_content_tokens: u32,
    pub tscg: bool,
    pub disable_tool_grammar: bool,
    pub rollback_resteer: bool,
    pub rom_head: String,
    pub tool_retry: bool,
    /// Suppress CUDA decode-graph capture for this model family.
    /// Nemotron-H models crash under graph replay (CUDA 700/716 at
    /// specific prompt lengths) and graphs are a measured no-op on GB10.
    pub no_decode_graphs: bool,
}

impl Default for ParsedBehavior {
    fn default() -> Self {
        Self {
            thinking_in_tools: true,
            max_thinking_budget: 256,
            thinking_default: false,
            fp8_kv_calibration_tokens: 0,
            default_kv_dtype: String::new(),
            default_num_drafts: 0,
            disable_tool_steering: false,
            tool_call_parser: String::new(),
            enable_loop_watchdog: false,
            min_p_floor: 0.0,
            temperature_max: 0.0,
            think_loop_min_repeats: 3,
            think_loop_scan_window: 160,
            confidence_early_stop: true,
            confidence_run_length: 30,
            fuzzy_repeat_tolerance_div: 12,
            max_inter_tool_prose: 384,
            max_post_think_content_tokens: 100_000,
            tscg: false,
            disable_tool_grammar: false,
            rollback_resteer: true,
            rom_head: String::new(),
            tool_retry: true,
            no_decode_graphs: false,
        }
    }
}

/// Parse `[behavior]` from MODEL.toml. Absent table or parse error →
/// `ParsedBehavior::default()`.
pub(super) fn parse_behavior(model_dir: &std::path::Path) -> ParsedBehavior {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return ParsedBehavior::default();
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return ParsedBehavior::default(),
    };
    let b = toml.get("behavior");
    let thinking_in_tools = b
        .and_then(|v| v.get("thinking_in_tools"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let max_thinking_budget = b
        .and_then(|v| v.get("max_thinking_budget"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(256);
    let thinking_default = b
        .and_then(|v| v.get("thinking_default"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let fp8_kv_calibration_tokens = b
        .and_then(|v| v.get("fp8_kv_calibration_tokens"))
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(0);
    let default_kv_dtype = b
        .and_then(|v| v.get("default_kv_dtype"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let default_num_drafts = b
        .and_then(|v| v.get("default_num_drafts"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(0);
    let disable_tool_steering = b
        .and_then(|v| v.get("disable_tool_steering"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let tool_call_parser = b
        .and_then(|v| v.get("tool_call_parser"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let enable_loop_watchdog = b
        .and_then(|v| v.get("enable_loop_watchdog"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let no_decode_graphs = b
        .and_then(|v| v.get("no_decode_graphs"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Server-side sampling safety floor/ceiling (0.0 = disabled). See
    // ModelBehavior::{min_p_floor,temperature_max} for rationale.
    let min_p_floor = b
        .and_then(|v| v.get("min_p_floor"))
        .and_then(|v| v.as_float())
        .unwrap_or(0.0) as f32;
    let temperature_max = b
        .and_then(|v| v.get("temperature_max"))
        .and_then(|v| v.as_float())
        .unwrap_or(0.0) as f32;
    let think_loop_min_repeats = b
        .and_then(|v| v.get("think_loop_min_repeats"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(3);
    let think_loop_scan_window = b
        .and_then(|v| v.get("think_loop_scan_window"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(160);
    let confidence_early_stop = b
        .and_then(|v| v.get("confidence_early_stop"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let confidence_run_length = b
        .and_then(|v| v.get("confidence_run_length"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(30);
    let fuzzy_repeat_tolerance_div = b
        .and_then(|v| v.get("fuzzy_repeat_tolerance_div"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(12);
    let max_inter_tool_prose = b
        .and_then(|v| v.get("max_inter_tool_prose"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(384);
    let max_post_think_content_tokens = b
        .and_then(|v| v.get("max_post_think_content_tokens"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(100_000);
    let tscg = b
        .and_then(|v| v.get("tscg"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let disable_tool_grammar = b
        .and_then(|v| v.get("disable_tool_grammar"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let rollback_resteer = b
        .and_then(|v| v.get("rollback_resteer"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let rom_head = b
        .and_then(|v| v.get("rom_head"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let tool_retry = b
        .and_then(|v| v.get("tool_retry"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    ParsedBehavior {
        thinking_in_tools,
        max_thinking_budget,
        thinking_default,
        fp8_kv_calibration_tokens,
        default_kv_dtype,
        default_num_drafts,
        disable_tool_steering,
        tool_call_parser,
        enable_loop_watchdog,
        min_p_floor,
        temperature_max,
        think_loop_min_repeats,
        think_loop_scan_window,
        confidence_early_stop,
        confidence_run_length,
        fuzzy_repeat_tolerance_div,
        max_inter_tool_prose,
        max_post_think_content_tokens,
        tscg,
        disable_tool_grammar,
        rollback_resteer,
        rom_head,
        tool_retry,
        no_decode_graphs,
    }
}
