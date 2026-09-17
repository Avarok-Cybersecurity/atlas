// SPDX-License-Identifier: AGPL-3.0-only
//
// `[behavior]` MODEL.toml parsing for build.rs. Split from `build_parse.rs`
// (500-LoC cap). Included as a child module of `build_parse`, so `super::`
// reaches its items and `crate::` reaches build.rs types.

// Shared with `src/lib.rs` (which `mod`s the same file) so the parse
// default below and `ModelBehavior::default()` are one literal. A build
// script cannot import the library it builds, and the hand-synced copies
// this replaces drifted for a month (#328: 384 here vs 3072 lib-side).
include!("src/behavior_defaults.rs");

/// Parsed `[behavior]` table from a model's MODEL.toml. Field defaults
/// match `ModelBehavior::default()` so an absent table / absent field is
/// behavior-neutral.
#[derive(Clone)]
pub(crate) struct ParsedBehavior {
    pub thinking_in_tools: bool,
    pub max_thinking_budget: u32,
    /// See `behavior_defaults.rs`: clamp effort levels at the ceiling.
    pub effort_capped_at_ceiling: bool,
    pub thinking_default: bool,
    pub fp8_kv_calibration_tokens: usize,
    pub default_kv_dtype: String,
    pub default_num_drafts: u32,
    pub disable_tool_steering: bool,
    pub disable_cwd_hint_injection: bool,
    pub use_sampling_presets_for_core: bool,
    pub tool_call_parser: String,
    pub enable_loop_watchdog: bool,
    /// Gate for the THINKING-phase token-loop watchdog, which force-injects
    /// `</think>` when it detects a period-4..20 repeat in the reasoning tail.
    /// Defaults TRUE (the behaviour every model had before this flag existed).
    /// Set false for models where it misfires: force-closing a reasoning block
    /// the model did not choose to end leaves it in a state it cannot continue
    /// from, and the post-close content degenerates into token spam.
    pub enable_think_loop_watchdog: bool,
    /// Honor an EOS the model samples INSIDE a `<think>` block by implicitly
    /// closing the block, instead of discarding the token and forcing the
    /// model to keep reasoning.
    ///
    /// Defaults FALSE (the behaviour every model had before this flag
    /// existed): a suppressed mid-think EOS is dropped and the model recovers,
    /// closes `</think>` itself and goes on to emit its tool call.
    ///
    /// ★ Why this is a flag and not unconditional. Landed unconditionally in
    /// the p350 stack, where it fixed a real Laguna-S-2.1 stall. But for a
    /// model that samples a spurious mid-think EOS and would otherwise
    /// recover, closing the block early leaves only the post-think content
    /// guard (POST_THINK_MIN_CONTENT = 16) holding the turn open: the model
    /// emits ~16 tokens of narration, its next EOS is honored, and the turn
    /// ends WITH NO TOOL CALL. Measured on Qwen3.6-35B-A3B-FP8 (thinking on by
    /// default, tools active): agentic runs collapsed to a single announcement
    /// sentence and an empty workspace, 8/10 on three consecutive N=10 tiers,
    /// against 10/10 without the change. So it is opt-in per model.
    pub honor_eos_inside_thinking: bool,
    /// Cap the thinking budget at 90% of the request's `max_tokens` (true), or
    /// let `max_thinking_budget` be the sole cap (false, vLLM single-budget).
    pub cap_thinking_at_max_tokens: bool,
    pub min_p_floor: f32,
    /// A4 POST_THINK_MIN_REASONING floor: suppress `</think>` (-8 logit)
    /// until this many thinking tokens have been emitted. 16 = the
    /// historical constant; 0 disables (models with card-native brief
    /// thinking, e.g. reasoning_effort=low, must not have their close
    /// token suppressed — the turn-ending mass reroutes to im_end/im_start
    /// and sampled runs EOS inside think or simulate template turns).
    pub min_reasoning_floor_tokens: u32,
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
    /// Tri-state `preserve_thinking` chat-template flag. `None` (key absent)
    /// = do not inject the Jinja variable; the model template's own default
    /// applies. See `ModelBehavior::preserve_thinking`.
    pub preserve_thinking: Option<bool>,
}

impl Default for ParsedBehavior {
    fn default() -> Self {
        Self {
            thinking_in_tools: true,
            max_thinking_budget: DEFAULT_MAX_THINKING_BUDGET,
            effort_capped_at_ceiling: DEFAULT_EFFORT_CAPPED_AT_CEILING,
            thinking_default: false,
            fp8_kv_calibration_tokens: 0,
            default_kv_dtype: String::new(),
            default_num_drafts: 0,
            disable_tool_steering: false,
            disable_cwd_hint_injection: false,
            use_sampling_presets_for_core: false,
            tool_call_parser: String::new(),
            enable_loop_watchdog: false,
            enable_think_loop_watchdog: true,
            honor_eos_inside_thinking: false,
            cap_thinking_at_max_tokens: true,
            min_p_floor: 0.0,
            min_reasoning_floor_tokens: 16,
            temperature_max: 0.0,
            think_loop_min_repeats: 3,
            think_loop_scan_window: 160,
            confidence_early_stop: true,
            confidence_run_length: 30,
            fuzzy_repeat_tolerance_div: 12,
            max_inter_tool_prose: DEFAULT_MAX_INTER_TOOL_PROSE,
            max_post_think_content_tokens: 100_000,
            tscg: false,
            disable_tool_grammar: false,
            rollback_resteer: true,
            rom_head: String::new(),
            tool_retry: true,
            preserve_thinking: None,
        }
    }
}

/// Parse `[behavior]` from MODEL.toml. Absent table or parse error →
/// `ParsedBehavior::default()`.
pub(crate) fn parse_behavior(model_dir: &std::path::Path) -> ParsedBehavior {
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
        .unwrap_or(DEFAULT_MAX_THINKING_BUDGET);
    let effort_capped_at_ceiling = b
        .and_then(|v| v.get("effort_capped_at_ceiling"))
        .and_then(|v| v.as_bool())
        .unwrap_or(DEFAULT_EFFORT_CAPPED_AT_CEILING);
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
    let disable_cwd_hint_injection = b
        .and_then(|v| v.get("disable_cwd_hint_injection"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let use_sampling_presets_for_core = b
        .and_then(|v| v.get("use_sampling_presets_for_core"))
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
    let enable_think_loop_watchdog = b
        .and_then(|v| v.get("enable_think_loop_watchdog"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    // Default FALSE: pre-p350 behaviour (discard a mid-think EOS, let the
    // model recover and emit its tool call). Opt in per model.
    let honor_eos_inside_thinking = b
        .and_then(|v| v.get("honor_eos_inside_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Default true: keep the 90%-of-max_tokens thinking cap. Set false for
    // vLLM-style single-budget behavior (only max_thinking_budget caps).
    let cap_thinking_at_max_tokens = b
        .and_then(|v| v.get("cap_thinking_at_max_tokens"))
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    // Server-side sampling safety floor/ceiling (0.0 = disabled). See
    // ModelBehavior::{min_p_floor,temperature_max} for rationale.
    let min_p_floor = b
        .and_then(|v| v.get("min_p_floor"))
        .and_then(|v| v.as_float())
        .unwrap_or(0.0) as f32;
    // A4 floor override; 16 = historical constant, 0 disables. See the
    // struct field for rationale.
    let min_reasoning_floor_tokens = b
        .and_then(|v| v.get("min_reasoning_floor_tokens"))
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(16);
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
        .unwrap_or(DEFAULT_MAX_INTER_TOOL_PROSE);
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
    // Tri-state: absent key stays `None` (template default), no unwrap_or.
    let preserve_thinking = b
        .and_then(|v| v.get("preserve_thinking"))
        .and_then(|v| v.as_bool());
    ParsedBehavior {
        thinking_in_tools,
        max_thinking_budget,
        effort_capped_at_ceiling,
        thinking_default,
        fp8_kv_calibration_tokens,
        default_kv_dtype,
        default_num_drafts,
        disable_tool_steering,
        disable_cwd_hint_injection,
        use_sampling_presets_for_core,
        tool_call_parser,
        enable_loop_watchdog,
        enable_think_loop_watchdog,
        honor_eos_inside_thinking,
        cap_thinking_at_max_tokens,
        min_p_floor,
        min_reasoning_floor_tokens,
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
        preserve_thinking,
    }
}
