// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]

//! Atlas CUDA kernel PTX modules.
//!
//! Single source of truth for embedded PTX. The `spark-runtime`
//! (pure Rust engine) and benchmarks consume these.
//!
//! PTX modules are grouped by [`KernelTarget`] — each `(H, M_q)`
//! tuple maps to a distinct set of hyperoptimized kernels.
//!
//! Constants, `ptx_modules()`, and `all_ptx_sets()` are auto-generated
//! by `build.rs` from the `kernels/{hw}/{model}/{quant}/` directories.
//! When `ATLAS_TARGET_MODEL=*` or `ATLAS_TARGET_QUANT=*`, multiple
//! targets are compiled and available at runtime.

use atlas_core::target::KernelTarget;

// Auto-generated: per-target PTX constants, ptx_modules() function,
// and all_ptx_sets() for multi-target builds.
include!(concat!(env!("OUT_DIR"), "/target_ptx.rs"));

// ═══════════════════════════════════════════════════════════════════
// Target-aware PTX grouping
// ═══════════════════════════════════════════════════════════════════

/// Per-category sampling defaults from MODEL.toml.
#[derive(Debug, Clone, Copy)]
pub struct SamplingCategory {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// Multiplicative penalty on already-seen tokens (1.0 = disabled).
    /// Populated from MODEL.toml `[sampling.*].repetition_penalty` via build.rs.
    pub repetition_penalty: f32,
    /// DRY (Don't-Repeat-Yourself) sampler parameters. Penalises tokens
    /// that extend repeated n-grams past `dry_allowed_length` with an
    /// exponential `dry_multiplier * dry_base^(match_len - allowed)` —
    /// the targeted fix for phrase-level attractors (e.g. the
    /// ```` ```bash cd … cargo test ``` ```` fence-narration loop
    /// observed in Qwen3.5-35B-A3B-FP8 opencode sessions at turn ≥ 8).
    ///
    /// `presence_penalty` on its own is a FLAT per-unique-token hit
    /// (does not scale with repetition count), so it can't break a
    /// phrase attractor where individual tokens already paid their
    /// penalty once. DRY scales with the repeat-length and is the
    /// published remedy (oobabooga/text-generation-webui#5677, used in
    /// llama.cpp / Aphrodite / TabbyAPI).
    ///
    /// `dry_multiplier = 0.0` disables DRY for this category (default
    /// for every preset unless MODEL.toml sets it explicitly).
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: u32,
    /// LZ penalty (arXiv:2504.20131). Per-extension n-gram penalty
    /// over a 256-token rolling window. Frequency-weighted and length-
    /// scaled, so it correctly distinguishes "phrase loop" from
    /// "legitimate vocabulary reuse" without the flat-per-token
    /// `presence_penalty` regression. 0.0 = disabled. SGLang reference
    /// strength = 0.2 (lossless on AIME/GPQA).
    pub lz_penalty: f32,
}

/// Model-specific sampling presets loaded from MODEL.toml `[sampling.*]`.
#[derive(Debug, Clone, Copy)]
pub struct SamplingPresets {
    pub thinking_text: SamplingCategory,
    pub thinking_coding: SamplingCategory,
    pub non_thinking: SamplingCategory,
    /// Tool-calling preset: model-recommended sampling for agentic tasks.
    /// Qwen3.5 recommends temperature=0.6 (NOT greedy) to avoid repetition loops.
    pub tools: SamplingCategory,
}

impl Default for SamplingPresets {
    fn default() -> Self {
        let default_cat = SamplingCategory {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            // DRY defaults = disabled (multiplier 0.0). Per-MODEL.toml
            // tools presets opt in when the model needs it.
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
        };
        let tools_cat = SamplingCategory {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            lz_penalty: 0.0,
        };
        Self {
            thinking_text: default_cat,
            thinking_coding: default_cat,
            non_thinking: default_cat,
            tools: tools_cat,
        }
    }
}

/// Model-specific behavior flags from MODEL.toml `[behavior]`.
#[derive(Debug, Clone)]
pub struct ModelBehavior {
    /// Allow thinking when tools are active. Default: true.
    pub thinking_in_tools: bool,
    /// Maximum thinking budget (tokens). Default: 256.
    pub max_thinking_budget: u32,
    /// Default thinking state for this model when the client request does not
    /// specify a reasoning_effort / thinking parameter. Typical values:
    /// - thinking-first models (Mistral Small 4, Qwen3.5, …): `true`
    /// - instruct-only models with no `<think>` tokens: `false`
    ///
    /// Overridden per-request by `reasoning_effort`, and globally by the
    /// `--disable-thinking` CLI flag.
    pub thinking_default: bool,
    /// Default FP8 KV calibration tokens (0 = disabled).
    pub fp8_kv_calibration_tokens: usize,
    /// Default KV cache dtype from MODEL.toml (e.g., "bf16", "fp8").
    /// When non-empty, overrides the CLI default for models that need
    /// higher precision. User can still override with explicit --kv-cache-dtype.
    pub default_kv_dtype: &'static str,
    /// Default num_drafts for speculative decoding (0 = use CLI default).
    /// K = num_drafts + 1 (num_drafts=1 → K=2 verifies 2 tokens per step).
    /// Optimal K varies per model; benchmarks sometimes show K=2 beats K=3.
    /// User override with --num-drafts still wins.
    pub default_num_drafts: u32,
    /// Skip the `<tool_call>\n` steering prefix in the chat template's
    /// generation prompt. Some Nemotron variants (Super 120B) weren't
    /// trained on qwen3_coder XML and emit a `<tool_call>` token loop
    /// when the prefix forces them into that structure. Default: false
    /// (keep the existing Nemotron-Nano-correct behavior).
    pub disable_tool_steering: bool,
    /// Per-model tool-call parser override. Empty string = use the
    /// `tool_defaults.toml` mapping for this `model_type`. Set in MODEL.toml
    /// `[behavior].tool_call_parser` when one variant of a model_type needs
    /// a different parser than its siblings (e.g. Nemotron-Super-120B uses
    /// `bare_json` while Nemotron-Nano-30B stays on `qwen3_coder`).
    pub tool_call_parser: &'static str,
    /// Enable the content-loop watchdog (period-N token-repetition detector
    /// at `decode_logits_step.rs:230`). Default: `false` — most models
    /// terminate cleanly via EOS / `max_tokens` without it. Models with a
    /// known prose-attractor failure mode (Qwen3.5-35B-A3B's "Running:```bash
    /// cmd```Executing:" loop, observed during agentic Claude Code sessions)
    /// should set this `true` in MODEL.toml `[behavior]`.
    ///
    /// The watchdog has false-positives on legitimate structured output
    /// (chess board JS init `{color:BLACK,type:'P'},` × 8, HTML tables,
    /// JSON arrays of similar objects, multiplication tables). Enable only
    /// when the model has been observed to need it.
    pub enable_loop_watchdog: bool,
}

impl Default for ModelBehavior {
    fn default() -> Self {
        Self {
            thinking_in_tools: true,
            max_thinking_budget: 256,
            thinking_default: false,
            fp8_kv_calibration_tokens: 0,
            default_kv_dtype: "",
            default_num_drafts: 0,
            disable_tool_steering: false,
            tool_call_parser: "",
            enable_loop_watchdog: false,
        }
    }
}

/// Declares which `(model_type, hidden_size)` pairs a kernel target supports.
/// Parsed from `[[model_types]]` in MODEL.toml at build time.
pub struct ModelTypeMatch {
    pub model_type: &'static str,
    /// `None` = wildcard (matches any hidden_size not caught by a more specific entry).
    pub hidden_size: Option<usize>,
}

/// DFlash speculative-decoding pairing for a target model.
/// Parsed from `[dflash]` in MODEL.toml at build time. `None` when the
/// model has no DFlash drafter associated.
#[derive(Debug, Clone)]
pub struct DflashConfig {
    /// HuggingFace id (or local path) of the drafter checkpoint.
    pub draft_model: &'static str,
    /// Block size γ (parallel draft tokens per step). Defaults to 16.
    pub gamma: usize,
    /// Drafter sliding-window size in tokens. 0 = full attention.
    pub window_size: usize,
    /// Token id used to fill the γ "to-be-predicted" positions during
    /// drafter forward. From the drafter's `dflash_config.mask_token_id`.
    pub mask_token_id: u32,
    /// Target-side layer indices to capture intermediate hidden states from
    /// (shallow-to-deep). The drafter's `fc` projection consumes the stack
    /// of these hiddens. From the drafter's `dflash_config.target_layer_ids`.
    pub target_layer_ids: &'static [usize],
}

/// PTX modules hyperoptimized for a specific (H, M_q) target.
pub struct TargetPtxSet {
    pub target: KernelTarget,
    pub modules: Vec<(&'static str, &'static str)>,
    pub sampling: SamplingPresets,
    pub behavior: ModelBehavior,
    pub model_type_matches: Vec<ModelTypeMatch>,
    /// DFlash drafter pairing for this model. `None` when the MODEL.toml has
    /// no `[dflash]` section. Consumed by spark-server when `--dflash` is
    /// set without an explicit `--draft-model` flag.
    pub dflash: Option<DflashConfig>,
}

/// All compiled kernel targets and their PTX module sets.
///
/// Returns one entry per target compiled at build time.
/// Single-target builds return one entry; wildcard builds return all.
pub fn available_targets() -> Vec<TargetPtxSet> {
    all_ptx_sets()
}

/// Find the PTX module set for a target whose model name contains `needle`.
///
/// Returns `None` if no compiled target matches.
pub fn ptx_for_model(needle: &str) -> Option<TargetPtxSet> {
    all_ptx_sets()
        .into_iter()
        .find(|t| t.target.model.contains(needle))
}

/// Find the PTX module set matching a `(model_type, hidden_size)` pair.
///
/// Matching rules:
/// 1. Exact match on `(model_type, Some(hidden_size))` wins
/// 2. Wildcard match `(model_type, None)` is fallback
/// 3. Returns `None` if no compiled target matches
pub fn ptx_for_config(model_type: &str, hidden_size: usize) -> Option<TargetPtxSet> {
    let targets = all_ptx_sets();
    // Exact match first (specific hidden_size)
    let exact = targets.iter().position(|t| {
        t.model_type_matches
            .iter()
            .any(|m| m.model_type == model_type && m.hidden_size == Some(hidden_size))
    });
    if let Some(idx) = exact {
        return targets.into_iter().nth(idx);
    }
    // Wildcard fallback (hidden_size == None)
    let wildcard = targets.iter().position(|t| {
        t.model_type_matches
            .iter()
            .any(|m| m.model_type == model_type && m.hidden_size.is_none())
    });
    wildcard.and_then(|idx| targets.into_iter().nth(idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_ptx_modules_non_empty() {
        for (name, ptx) in ptx_modules() {
            assert!(
                !ptx.is_empty(),
                "PTX module '{name}' is empty — nvcc compilation may have failed"
            );
            assert!(
                ptx.contains(".version"),
                "PTX module '{name}' doesn't contain .version directive"
            );
        }
    }

    // These tests assert that PTX modules were actually compiled into the
    // crate at build time. They require nvcc + a real CUDA toolchain — the
    // CI host runs with `ATLAS_SKIP_BUILD=1`, which emits an empty stub
    // registry by design (so `cargo check` / `cargo clippy` / `cargo test`
    // can run on hosts without a GPU). Mark them `#[ignore]` so default
    // `cargo test` is green; they're still exercised on a developer
    // machine via `cargo test -p atlas-kernels -- --ignored` after a
    // real PTX build.

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn module_count_matches_cu_files() {
        let count = ptx_modules().len();
        assert!(count >= 31, "Expected at least 31 PTX modules, got {count}");
    }

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn available_targets_non_empty() {
        let targets = available_targets();
        assert!(!targets.is_empty(), "No kernel targets available");
        assert!(
            targets.iter().any(|t| t.target.quant == "nvfp4"),
            "Expected at least one NVFP4 target"
        );
    }

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn all_targets_have_modules() {
        for t in available_targets() {
            assert!(
                t.modules.len() >= 31,
                "Target {} has only {} modules (expected >= 31)",
                t.target,
                t.modules.len()
            );
        }
    }

    #[test]
    #[ignore = "requires nvcc and ATLAS_SKIP_BUILD unset"]
    fn ptx_for_model_lookup() {
        let found = ptx_for_model("qwen3-next-80b");
        assert!(
            found.is_some(),
            "ptx_for_model('qwen3-next-80b') should find the default target"
        );
    }
}
