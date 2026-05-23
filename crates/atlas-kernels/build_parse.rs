// SPDX-License-Identifier: AGPL-3.0-only
//
// Parse helpers for build.rs. Included via `#[path = "build_parse.rs"] mod build_parse;`
// so types defined in build.rs (`SamplingCat`, `ModelTypeMatch`,
// `DflashRaw`) are reachable via `super::`.

use std::collections::HashMap;

use super::{DflashRaw, ModelTypeMatch, SamplingCat};

pub(super) fn parse_kernel_toml(
    kernel_dir: &std::path::Path,
    vendor: &str,
) -> (Vec<String>, HashMap<String, String>) {
    let kernel_toml_path = kernel_dir.join("KERNEL.toml");
    let kernel_toml: toml::Value = toml::from_str(
        &std::fs::read_to_string(&kernel_toml_path)
            .unwrap_or_else(|e| panic!("{}: {e}", kernel_toml_path.display())),
    )
    .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", kernel_toml_path.display()));
    println!("cargo:rerun-if-changed={}", kernel_toml_path.display());

    // Per-vendor extra flag keys. NVIDIA reads `extra_nvcc_flags`; Apple
    // reads `extra_metal_flags`. KERNEL.toml may declare both — only the
    // vendor-matching list is forwarded so flags don't bleed across
    // toolchains (e.g. nvcc's `--fmad=false` is invalid for xcrun metal).
    let flag_key = match vendor {
        "apple" | "metal" => "extra_metal_flags",
        _ => "extra_nvcc_flags",
    };
    let extra_flags: Vec<String> = kernel_toml
        .get("build")
        .and_then(|b| b.get(flag_key))
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default();

    let module_overrides: HashMap<String, String> = kernel_toml
        .get("modules")
        .and_then(|m| m.as_table())
        .map(|t| {
            t.iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect()
        })
        .unwrap_or_default();

    (extra_flags, module_overrides)
}

/// Parse sampling presets from MODEL.toml `[sampling.*]` sections.
pub(super) fn parse_sampling_presets(
    model_dir: &std::path::Path,
) -> (SamplingCat, SamplingCat, SamplingCat, SamplingCat) {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return (
            SamplingCat::default(),
            SamplingCat::default(),
            SamplingCat::default(),
            SamplingCat::default(),
        );
    }
    println!("cargo:rerun-if-changed={}", model_toml_path.display());
    let content = std::fs::read_to_string(&model_toml_path)
        .unwrap_or_else(|e| panic!("{}: {e}", model_toml_path.display()));
    let toml: toml::Value = toml::from_str(&content)
        .unwrap_or_else(|e| panic!("Bad TOML in {}: {e}", model_toml_path.display()));

    let parse_cat = |key: &str| -> SamplingCat {
        let section = toml.get("sampling").and_then(|s| s.get(key));
        match section {
            Some(v) => SamplingCat {
                temperature: v
                    .get("temperature")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.7) as f32,
                top_p: v.get("top_p").and_then(|t| t.as_float()).unwrap_or(0.95) as f32,
                top_k: v.get("top_k").and_then(|t| t.as_integer()).unwrap_or(20) as u32,
                presence_penalty: v
                    .get("presence_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                frequency_penalty: v
                    .get("frequency_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                repetition_penalty: v
                    .get("repetition_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(1.0) as f32,
                dry_multiplier: v
                    .get("dry_multiplier")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
                dry_base: v.get("dry_base").and_then(|t| t.as_float()).unwrap_or(1.75) as f32,
                dry_allowed_length: v
                    .get("dry_allowed_length")
                    .and_then(|t| t.as_integer())
                    .unwrap_or(2) as u32,
                lz_penalty: v
                    .get("lz_penalty")
                    .and_then(|t| t.as_float())
                    .unwrap_or(0.0) as f32,
            },
            None => SamplingCat::default(),
        }
    };

    (
        parse_cat("thinking_text"),
        parse_cat("thinking_coding"),
        parse_cat("non_thinking"),
        parse_cat("tools"),
    )
}

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
    pub think_loop_min_repeats: u32,
    pub think_loop_scan_window: u32,
    pub confidence_early_stop: bool,
    pub confidence_run_length: u32,
    pub fuzzy_repeat_tolerance_div: u32,
    pub max_inter_tool_prose: u32,
    pub tscg: bool,
    pub disable_tool_grammar: bool,
    pub rollback_resteer: bool,
    pub rom_head: String,
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
            think_loop_min_repeats: 3,
            think_loop_scan_window: 160,
            confidence_early_stop: true,
            confidence_run_length: 30,
            fuzzy_repeat_tolerance_div: 12,
            max_inter_tool_prose: 384,
            tscg: false,
            disable_tool_grammar: false,
            rollback_resteer: true,
            rom_head: String::new(),
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
        think_loop_min_repeats,
        think_loop_scan_window,
        confidence_early_stop,
        confidence_run_length,
        fuzzy_repeat_tolerance_div,
        max_inter_tool_prose,
        tscg,
        disable_tool_grammar,
        rollback_resteer,
        rom_head,
    }
}

/// Parse `[[model_types]]` from MODEL.toml.
///
/// Each entry maps a `(model_type, optional hidden_size)` pair to this kernel target.
/// Missing `hidden_size` = wildcard (matches any hidden_size not caught by a more specific entry).
pub(super) fn parse_model_types(model_dir: &std::path::Path) -> Vec<ModelTypeMatch> {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return Vec::new();
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = match toml::from_str(&content) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(entries) = toml.get("model_types").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let mt = entry.get("model_type")?.as_str()?.to_string();
            let hs = entry
                .get("hidden_size")
                .and_then(|v| v.as_integer())
                .map(|v| v as usize);
            Some(ModelTypeMatch {
                model_type: mt,
                hidden_size: hs,
            })
        })
        .collect()
}

/// Parse `[dflash]` from MODEL.toml. Returns `None` when the section is
/// absent (model has no DFlash drafter pairing). The build emits the parsed
/// values into the static `TargetPtxSet::dflash` field that spark-server
/// reads at runtime to satisfy `--dflash` without an explicit `--draft-model`.
pub(super) fn parse_dflash(model_dir: &std::path::Path) -> Option<DflashRaw> {
    let model_toml_path = model_dir.join("MODEL.toml");
    if !model_toml_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&model_toml_path).unwrap_or_default();
    let toml: toml::Value = toml::from_str(&content).ok()?;
    let dflash = toml.get("dflash")?;
    let draft_model = dflash.get("draft_model")?.as_str()?.to_string();
    let gamma = dflash
        .get("gamma")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(16);
    let window_size = dflash
        .get("window_size")
        .and_then(|v| v.as_integer())
        .map(|v| v as usize)
        .unwrap_or(0);
    let mask_token_id = dflash
        .get("mask_token_id")
        .and_then(|v| v.as_integer())
        .map(|v| v as u32)
        .unwrap_or(0);
    let target_layer_ids: Vec<usize> = dflash
        .get("target_layer_ids")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_integer().map(|x| x as usize))
                .collect()
        })
        .unwrap_or_default();
    Some(DflashRaw {
        draft_model,
        gamma,
        window_size,
        mask_token_id,
        target_layer_ids,
    })
}
