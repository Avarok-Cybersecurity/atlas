// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared type definitions for build.rs. Included via
// `#[path = "build_types.rs"] mod build_types;` and re-exported at the
// build.rs root so the sibling `build_parse` / `build_codegen` modules
// reach them via `super::`.

use std::collections::HashMap;
use std::path::PathBuf;

/// Per-category sampling defaults parsed from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone)]
pub(super) struct SamplingCat {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    // DRY sampler params (see SamplingCategory in atlas-kernels/src/lib.rs
    // for full rationale). Defaults disable DRY; individual MODEL.toml
    // `[sampling.*]` tables opt in when needed.
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    // LZ penalty (arXiv:2504.20131). Frequency-weighted n-gram penalty
    // over the recent token window. 0.0 = disabled. 0.2 is the SGLang
    // reference value; lossless on AIME/GPQA at that strength.
    pub lz_penalty: f32,
}

impl Default for SamplingCat {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
        }
    }
}

/// A `(model_type, optional hidden_size)` pair declaring which models a kernel target supports.
#[derive(Debug, Clone)]
pub(super) struct ModelTypeMatch {
    pub model_type: String,
    pub hidden_size: Option<usize>,
}

/// A resolved (hw, model, quant) compilation target.
pub(super) struct Target {
    pub hw: String,
    pub model: String,
    pub quant: String,
    pub arch: String,
    /// Per-model quant dir (for KERNEL.toml and optional override .cu files).
    pub model_kernel_dir: PathBuf,
    /// Common quant dir (hw_dir/quant/) with shared .cu files.
    pub common_kernel_dir: Option<PathBuf>,
    pub extra_flags: Vec<String>,
    pub module_overrides: HashMap<String, String>,
    pub sampling_thinking_text: SamplingCat,
    pub sampling_thinking_coding: SamplingCat,
    pub sampling_non_thinking: SamplingCat,
    pub sampling_tools: SamplingCat,
    pub behavior_thinking_in_tools: bool,
    pub behavior_max_thinking_budget: u32,
    pub behavior_thinking_default: bool,
    pub behavior_fp8_kv_calibration_tokens: usize,
    pub behavior_default_kv_dtype: String,
    pub behavior_default_num_drafts: u32,
    pub behavior_disable_tool_steering: bool,
    pub behavior_tool_call_parser: String,
    pub behavior_enable_loop_watchdog: bool,
    pub behavior_think_loop_min_repeats: u32,
    pub behavior_think_loop_scan_window: u32,
    pub behavior_confidence_early_stop: bool,
    pub behavior_confidence_run_length: u32,
    pub behavior_fuzzy_repeat_tolerance_div: u32,
    pub behavior_max_inter_tool_prose: u32,
    pub behavior_max_post_think_content_tokens: u32,
    pub behavior_tscg: bool,
    pub behavior_disable_tool_grammar: bool,
    pub behavior_rollback_resteer: bool,
    pub behavior_rom_head: String,
    pub behavior_tool_retry: bool,
    /// Which `(model_type, hidden_size)` pairs this kernel target supports.
    /// Parsed from `[[model_types]]` in MODEL.toml.
    pub model_type_matches: Vec<ModelTypeMatch>,
    /// `[dflash]` section if present in MODEL.toml — drafter pairing for
    /// block-diffusion speculative decoding. `None` when the model has no
    /// associated DFlash drafter checkpoint.
    pub dflash: Option<DflashRaw>,
}

#[derive(Default, Clone)]
pub(super) struct DflashRaw {
    pub draft_model: String,
    pub gamma: usize,
    pub window_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}
