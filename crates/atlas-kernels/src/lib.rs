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
// NOTE: cargo does NOT track this build-script-generated include! as a
// recompile trigger, so when build.rs regenerates target_ptx.rs (e.g. the
// module set changes) this lib can keep a STALE embedded set. Any edit to
// this file (or `cargo clean -p atlas-kernels`) forces a fresh recompile
// against the current OUT_DIR/target_ptx.rs.
include!(concat!(env!("OUT_DIR"), "/target_ptx.rs"));

/// Content fingerprint of the generated kernel set, emitted by `build.rs` as a
/// `rustc-env`. Referencing it here makes cargo recompile this crate whenever
/// the kernel set changes — closing the `include!`-not-tracked staleness hole
/// that silently embedded a stale module list (the 98-vs-99 regression).
pub const KERNEL_SET_HASH: &str = env!("ATLAS_KERNEL_SET_HASH");

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
    /// Server-side min-p FLOOR (0.0 = disabled). Applied as `min_p.max(floor)`
    /// AFTER request/preset resolution, so it binds even when a client sends
    /// `min_p = 0` (or omits it on a server without `--default-min-p`). On
    /// drift-prone quantized models (FP8 / NVFP4 lm-head) an unfloored tail
    /// lets the degenerate low-probability tail be sampled into repetition
    /// loops + argmax-flip garbling on long generation — the Claude-Code
    /// failure mode. MEASURED 2026-06-07 (nvfp4-head@64k): 0.05 turned 4 loop-
    /// watchdog fires → 0. Set in MODEL.toml `[behavior]`.
    pub min_p_floor: f32,
    /// Server-side temperature CEILING (0.0 = disabled). `temperature.min(max)`
    /// AFTER resolution — defense-in-depth net against a client sending a high
    /// temperature; min_p_floor is the dominant lever. Set in MODEL.toml.
    pub temperature_max: f32,
    /// Thinking-loop watchdog: substring-occurrence count that trips a
    /// forced `</think>`. Default 3 (historical `THINK_LOOP_MIN_REPEATS`).
    pub think_loop_min_repeats: u32,
    /// Thinking-loop watchdog: trailing-token scan window. Default 160.
    pub think_loop_scan_window: u32,
    /// F2 confidence-run early-stop enabled. Default `true`. Set false
    /// for models whose deterministic code drafting trips the heuristic.
    pub confidence_early_stop: bool,
    /// F2 confidence run length before arming forced `</think>`.
    /// Default 30.
    pub confidence_run_length: u32,
    /// Fuzzy-repetition detector Hamming tolerance divisor: a
    /// `pattern_len`-token window tolerates `pattern_len / div`
    /// mismatches. Default 12 (~8%).
    pub fuzzy_repeat_tolerance_div: u32,
    /// Cap on free-text tokens between successive `<tool_call>` opens in
    /// `tool_choice=auto`. Default 384. Agentic coding may want larger.
    pub max_inter_tool_prose: u32,
    /// Unconditional per-generation cap on post-`</think>` content tokens
    /// for tool-active requests (grammar attached). Bounds a runaway where
    /// a grammar-legal-but-never-closing tool value burns to `max_tokens`
    /// (the dominant opencode `webserver_ok` 360s-timeout cause). Default
    /// 100_000 — effectively unbounded, the historical no-op — so a model
    /// that sets nothing is byte-identical to before. Set a small value
    /// (e.g. 1536) per-model to backstop the runaway. Never caps plain
    /// chat: the runtime gate also requires `grammar_state.is_some()`.
    pub max_post_think_content_tokens: u32,
    /// TSCG (Tool-Schema Compilation) enabled — compile tool JSON
    /// schemas to compact function signatures before prompting.
    /// Default `false`; the TAS operator is tokenizer-specific so
    /// enable + verify per model. arXiv:2605.04107.
    pub tscg: bool,
    /// Disable XGrammar tool-call constrained decoding for this model.
    /// Default `false`. Escape hatch for the "structure snowballing"
    /// alignment tax (arXiv:2604.06066) — a few models tool-call more
    /// reliably unconstrained. When `true`, tool calls are parsed but
    /// not grammar-enforced.
    pub disable_tool_grammar: bool,
    /// Phase-C: when a decode-time watchdog (content-loop, fuzzy-repeat,
    /// inter-tool prose) detects degeneration, roll the sequence back to
    /// the last well-formed boundary and let generation re-steer, instead
    /// of hard-stopping the response. Default `true` (recovers responses,
    /// especially mid-tool-call — arXiv:2603.27905 ATLAS-RTC). Set `false`
    /// to keep the legacy hard-stop behavior. Capped at
    /// [`crate::ROLLBACK_RESTEER_CAP`] rollbacks per sequence, after which
    /// the hard-stop fires regardless.
    pub rollback_resteer: bool,
    /// Phase-C ROM (arXiv:2603.22016) scaffold. Path to a trained
    /// repetition-onset detection head artifact. Empty string = no ROM
    /// head; the F2 confidence heuristic stays as the fallback. A trained
    /// artifact can be dropped in later via MODEL.toml
    /// `[behavior].rom_head` without further code changes — the runtime
    /// loads it through the `RomHead` trait seam. The detector
    /// itself is intentionally NOT implemented (no per-model trained head
    /// is available); only the optional hook is wired.
    pub rom_head: &'static str,
    /// Tier 5c (2026-05-26): one-shot tool-call re-roll on hard
    /// validation failure. When `true`, `validate_tool_calls` errors on
    /// the chat path fire a single retry inference with the same
    /// grammar spec + a correction nudge appended to the prompt. If the
    /// retry produces valid tool calls, they replace the failed call
    /// before the response leaves the server. Default `true` — the
    /// blocking-path canonical-probe trace shows a write-→bash recovery
    /// path that's strictly better than the previous "`[atlas]` Tool call
    /// rejected" content fallback. Set `false` per-model when a
    /// specific model is known to ALWAYS get tool args right on the
    /// first attempt (extra inference round-trip cost is wasted there).
    pub tool_retry: bool,
}

/// Phase-C: maximum number of watchdog-triggered rollbacks a single
/// sequence may perform before the watchdog reverts to a hard stop.
/// Bounds the worst case where re-steering re-enters the same attractor
/// — without this a degenerate sequence could rollback indefinitely.
pub const ROLLBACK_RESTEER_CAP: u32 = 2;

/// Phase-C: number of boundary SSM-state snapshots retained per sequence in
/// the decode-rollback ring (hybrid GDN/Mamba models). DECOUPLED from
/// [`ROLLBACK_RESTEER_CAP`]: the cap bounds how many times we re-steer, but
/// the ring must retain enough *boundary* snapshots that a clean PRE-loop
/// boundary survives long enough to roll back to. Sizing it at the old
/// `CAP + 1 = 3` meant a loop spanning ≥3 sentence/newline boundaries evicted
/// the clean boundary before the fuzzy detector (3 repeats) fired, forcing a
/// `NoSsmSnapshot` decline → hard-stop (observed: Claude-Code @ nvfp4-head,
/// 2026-06-07). 8 covers the 3-repeat detector with margin at modest cost
/// (8 × max_batch × per-layer GDN state, allocated once). Pure-attention
/// models ignore this (their ring is 0; they roll back to any boundary).
pub const DECODE_ROLLBACK_RING_SLOTS: usize = 8;

/// Domain salt folded into every decode cold-tier key so a decode-ring blob can
/// never collide with a Marconi prefix-hash key on a shared store/peer.
pub const DECODE_DOMAIN: u64 = 0xD3C0_DE12_A5B6_C7D8;

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
            rom_head: "",
            tool_retry: true,
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

/// Kernel modules hyperoptimized for a specific (H, M_q) target.
///
/// Each blob is the compiled kernel for one module, emitted uniformly as
/// `&'static [u8]` by build.rs (`include_bytes!`). NVIDIA PTX is ASCII
/// text but valid as bytes; SCALE/AMD and Metal produce binary objects.
/// The runtime registry sniffs text-vs-binary per blob at load time.
pub struct TargetPtxSet {
    pub target: KernelTarget,
    pub modules: Vec<(&'static str, &'static [u8])>,
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
        for (name, blob) in ptx_modules() {
            assert!(
                !blob.is_empty(),
                "PTX module '{name}' is empty — nvcc compilation may have failed"
            );
            // Blobs are `&[u8]` (uniform across backends). For the NVIDIA
            // build under test the bytes are ASCII PTX, so decode and check
            // the `.version` directive; on a non-text backend this lossily
            // decodes to "" and the assert would (correctly) not apply.
            let ptx = std::str::from_utf8(blob).unwrap_or("");
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
