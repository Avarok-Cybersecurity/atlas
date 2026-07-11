// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-flag CLI validation (issue #288).
//!
//! `clap` validates each flag in isolation; it cannot catch *combinations* that
//! contradict each other (e.g. `--kv-cache-dtype bf16` together with
//! `--fp8-kv-calibration-tokens 256`, where the calibration can never apply).
//! Such combinations silently do the wrong thing and — because operators copy
//! working command lines around and publish them — the mistake propagates.
//!
//! [`validate_serve_args`] turns every known-contradictory combination into a
//! **hard error** (never a warning), collected and reported together, with a
//! message shaped for both humans and AI agents: each finding states *what* is
//! wrong, *why*, and the concrete *fix*. It runs before the (multi-minute)
//! model load so a bad invocation fails in milliseconds.
//!
//! Add new rules here as flags are added — this is the single place invalid
//! combinations are rejected.

use super::ServeArgs;

/// Enumerated string flags and their allowed values. Kept next to the rules so
/// the "did you mean" listing stays in sync with the parse sites in `serve.rs`.
const LM_HEAD_DTYPES: &[&str] = &["default", "bf16", "nvfp4", "fp8"];
const MTP_QUANTS: &[&str] = &["bf16", "fp8", "nvfp4"];
const SCHEDULING_POLICIES: &[&str] = &["fifo", "slai"];
const TOOL_CALL_PARSERS: &[&str] = &[
    "hermes",
    "qwen3_coder",
    "qwen3_xml",
    "gemma4",
    "mistral",
    "minimax_xml",
    "bare_json",
];

/// One validation failure: what is wrong, why it is wrong, and how to fix it.
struct Violation {
    what: String,
    why: String,
    fix: String,
}

impl Violation {
    fn new(what: impl Into<String>, why: impl Into<String>, fix: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            why: why.into(),
            fix: fix.into(),
        }
    }
}

/// Validate cross-flag combinations. Returns a single formatted, actionable
/// error string listing EVERY violation (so the operator fixes them in one
/// pass), or `Ok(())` when the invocation is self-consistent.
pub fn validate_serve_args(args: &ServeArgs) -> Result<(), String> {
    let mut v: Vec<Violation> = Vec::new();

    // ── Enumerated-value typos (caught here so a typo fails fast, before the
    //    model load, rather than mid-startup at each scattered parse site). ──
    check_enum(
        &mut v,
        "--lm-head-dtype",
        &args.lm_head_dtype,
        LM_HEAD_DTYPES,
    );
    check_enum(
        &mut v,
        "--mtp-quantization",
        &args.mtp_quantization,
        MTP_QUANTS,
    );
    check_enum(
        &mut v,
        "--scheduling-policy",
        &args.scheduling_policy,
        SCHEDULING_POLICIES,
    );
    if let Some(parser) = args.tool_call_parser.as_deref() {
        check_enum(&mut v, "--tool-call-parser", parser, TOOL_CALL_PARSERS);
    }
    // `kv_cache_dtype` has a large TurboQuant-Plus variant set — validate via
    // the runtime's own `FromStr` so this stays in sync automatically.
    if args
        .kv_cache_dtype
        .parse::<spark_runtime::kv_cache::KvCacheDtype>()
        .is_err()
    {
        v.push(Violation::new(
            format!(
                "--kv-cache-dtype '{}' is not a known KV cache dtype.",
                args.kv_cache_dtype
            ),
            "the value does not parse to any supported KV cache format.",
            "use one of: fp8, bf16, nvfp4 (or a turbo* TurboQuant-Plus variant).",
        ));
    }

    // ── FP8 KV calibration only applies to an FP8 KV cache (issue #288 example). ──
    if args.fp8_kv_calibration_tokens > 0 && args.kv_cache_dtype != "fp8" {
        v.push(Violation::new(
            format!(
                "--fp8-kv-calibration-tokens {} has no effect with --kv-cache-dtype {}.",
                args.fp8_kv_calibration_tokens, args.kv_cache_dtype
            ),
            "online FP8 KV-scale calibration only feeds an FP8 KV cache; with a \
             bf16/nvfp4 cache the calibrated scales are never read.",
            "set --kv-cache-dtype fp8, or drop --fp8-kv-calibration-tokens (0 = off).",
        ));
    }
    // NOTE: --kv-high-precision-layers with a bf16 base is redundant (a no-op),
    // but NOT a hard error — the canonical flagship serve recipe passes
    // `--kv-cache-dtype bf16 --kv-high-precision-layers auto` together, so
    // rejecting it would break a real, supported command. Redundant ≠ invalid.

    // ── --require-auth needs at least one token source. ──
    if args.require_auth && args.auth_tokens_file.is_none() && args.auth_token.is_none() {
        v.push(Violation::new(
            "--require-auth is set but no bearer tokens were provided.",
            "with auth enforced and no tokens loaded, EVERY request is rejected 401.",
            "pass --auth-tokens-file <path> (preferred, 0600) or --auth-token <token>.",
        ));
    }

    // ── Speculative-decode draft count needs a speculative method. ──
    let any_spec = args.speculative || args.self_speculative || args.ngram_speculative;
    if args.num_drafts > 1 && !any_spec {
        v.push(Violation::new(
            format!(
                "--num-drafts {} is set but no speculative method is enabled.",
                args.num_drafts
            ),
            "the draft count only applies when speculative decoding proposes drafts; \
             without it the flag is ignored.",
            "add --speculative (MTP), --self-speculative, or --ngram-speculative — or \
             drop --num-drafts.",
        ));
    }

    // ── Thinking budget contradicts disabling thinking. ──
    if args.disable_thinking && args.max_thinking_budget.is_some() {
        v.push(Violation::new(
            "--max-thinking-budget is set together with --disable-thinking.",
            "--disable-thinking strips reasoning entirely, so there is nothing for the \
             budget to cap.",
            "drop one: keep --disable-thinking for no reasoning, or drop it and keep the \
             budget to bound reasoning length.",
        ));
    }

    // ── Distributed topology sanity. ──
    if args.rank >= args.world_size {
        v.push(Violation::new(
            format!(
                "--rank {} is out of range for --world-size {}.",
                args.rank, args.world_size
            ),
            "ranks are 0-indexed, so a valid rank is in 0..world_size.",
            format!(
                "set --rank in 0..={} (or raise --world-size).",
                args.world_size.saturating_sub(1)
            ),
        ));
    }
    if args.ep_size > args.world_size {
        v.push(Violation::new(
            format!(
                "--ep-size {} exceeds --world-size {}.",
                args.ep_size, args.world_size
            ),
            "expert parallelism cannot span more ranks than exist.",
            "raise --world-size to at least --ep-size, or lower --ep-size.",
        ));
    }
    if args.tp_size > args.world_size {
        v.push(Violation::new(
            format!(
                "--tp-size {} exceeds --world-size {}.",
                args.tp_size, args.world_size
            ),
            "tensor parallelism cannot span more ranks than exist.",
            "raise --world-size to at least --tp-size, or lower --tp-size.",
        ));
    }

    // ── High-speed swap sub-options require the feature to be enabled. ──
    if !args.high_speed_swap {
        let mut orphaned: Vec<&str> = Vec::new();
        if args.high_speed_swap_dir.is_some() {
            orphaned.push("--high-speed-swap-dir");
        }
        if args.high_speed_swap_gb.is_some() {
            orphaned.push("--high-speed-swap-gb");
        }
        if args.high_speed_swap_resident_blocks.is_some() {
            orphaned.push("--high-speed-swap-resident-blocks");
        }
        if args.high_speed_swap_graph.is_some() {
            orphaned.push("--high-speed-swap-graph");
        }
        if !orphaned.is_empty() {
            v.push(Violation::new(
                format!("{} set without --high-speed-swap.", orphaned.join(", ")),
                "high-speed-swap tuning options are ignored unless the feature is on.",
                "add --high-speed-swap, or drop the tuning option(s).",
            ));
        }
    }

    // ── GPU memory utilization must be a usable fraction. ──
    if !(args.gpu_memory_utilization > 0.0 && args.gpu_memory_utilization <= 1.0) {
        v.push(Violation::new(
            format!(
                "--gpu-memory-utilization {} is outside (0.0, 1.0].",
                args.gpu_memory_utilization
            ),
            "the value is the fraction of total GPU memory Atlas may claim.",
            "use a fraction in (0.0, 1.0], e.g. 0.90.",
        ));
    }

    if v.is_empty() {
        return Ok(());
    }
    Err(format_violations(&v))
}

/// Push a violation if `value` is not in `allowed`.
fn check_enum(v: &mut Vec<Violation>, flag: &str, value: &str, allowed: &[&str]) {
    if !allowed.contains(&value) {
        v.push(Violation::new(
            format!("{flag} '{value}' is not a valid value."),
            format!("valid values are: {}.", allowed.join(", ")),
            format!("pick one of {}.", allowed.join(", ")),
        ));
    }
}

fn format_violations(v: &[Violation]) -> String {
    let mut out = format!(
        "Atlas CLI: {} invalid flag combination{} — fix before serving:\n",
        v.len(),
        if v.len() == 1 { "" } else { "s" }
    );
    for (i, vio) in v.iter().enumerate() {
        out.push_str(&format!(
            "\n  [{}] {}\n      why: {}\n      fix: {}\n",
            i + 1,
            vio.what,
            vio.why,
            vio.fix
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Parse a `spark serve ...` command line into `ServeArgs` for testing.
    fn parse(extra: &[&str]) -> ServeArgs {
        let mut argv = vec!["spark", "serve", "dummy/model", "--model-name", "dummy"];
        argv.extend_from_slice(extra);
        match super::super::Cli::parse_from(argv).command {
            super::super::Command::Serve(a) => a,
        }
    }

    #[test]
    fn defaults_are_valid() {
        assert!(validate_serve_args(&parse(&[])).is_ok());
    }

    #[test]
    fn fp8_calibration_requires_fp8_kv() {
        let err = validate_serve_args(&parse(&[
            "--kv-cache-dtype",
            "bf16",
            "--fp8-kv-calibration-tokens",
            "256",
        ]))
        .unwrap_err();
        assert!(err.contains("--fp8-kv-calibration-tokens"));
        assert!(err.contains("fix:"));
        // The same flags with an fp8 cache are fine.
        assert!(
            validate_serve_args(&parse(&[
                "--kv-cache-dtype",
                "fp8",
                "--fp8-kv-calibration-tokens",
                "256",
            ]))
            .is_ok()
        );
    }

    #[test]
    fn require_auth_needs_a_token() {
        assert!(validate_serve_args(&parse(&["--require-auth"])).is_err());
        assert!(validate_serve_args(&parse(&["--require-auth", "--auth-token", "sk-x"])).is_ok());
    }

    #[test]
    fn num_drafts_needs_speculative() {
        assert!(validate_serve_args(&parse(&["--num-drafts", "2"])).is_err());
        assert!(validate_serve_args(&parse(&["--num-drafts", "2", "--speculative"])).is_ok());
    }

    #[test]
    fn rank_must_be_below_world_size() {
        assert!(validate_serve_args(&parse(&["--rank", "2", "--world-size", "2"])).is_err());
        assert!(validate_serve_args(&parse(&["--rank", "1", "--world-size", "2"])).is_ok());
    }

    #[test]
    fn ep_size_cannot_exceed_world_size() {
        assert!(validate_serve_args(&parse(&["--ep-size", "2"])).is_err());
        assert!(validate_serve_args(&parse(&["--ep-size", "2", "--world-size", "2"])).is_ok());
    }

    #[test]
    fn disable_thinking_conflicts_with_budget() {
        assert!(
            validate_serve_args(&parse(&[
                "--disable-thinking",
                "--max-thinking-budget",
                "2048"
            ]))
            .is_err()
        );
    }

    #[test]
    fn flagship_recipe_is_accepted() {
        // The canonical 35B flagship serve recipe (PR #278) passes
        // `--kv-cache-dtype bf16 --kv-high-precision-layers auto` together —
        // redundant but valid. The validator must NOT reject it.
        assert!(
            validate_serve_args(&parse(&[
                "--kv-cache-dtype",
                "bf16",
                "--lm-head-dtype",
                "nvfp4",
                "--kv-high-precision-layers",
                "auto",
                "--scheduling-policy",
                "slai",
                "--speculative",
                "--num-drafts",
                "1",
                "--mtp-quantization",
                "bf16",
                "--enable-prefix-caching",
            ]))
            .is_ok()
        );
    }

    #[test]
    fn enum_typos_are_rejected() {
        let err = validate_serve_args(&parse(&["--scheduling-policy", "fifoo"])).unwrap_err();
        assert!(err.contains("--scheduling-policy"));
        assert!(err.contains("fifo, slai"));
    }

    #[test]
    fn multiple_violations_all_reported() {
        let err = validate_serve_args(&parse(&[
            "--require-auth",
            "--num-drafts",
            "3",
            "--rank",
            "5",
            "--world-size",
            "2",
        ]))
        .unwrap_err();
        assert!(err.contains("[1]"));
        assert!(err.contains("[2]"));
        assert!(err.contains("[3]"));
    }

    #[test]
    fn gpu_mem_util_range_enforced() {
        assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "1.5"])).is_err());
        assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "0.0"])).is_err());
        assert!(validate_serve_args(&parse(&["--gpu-memory-utilization", "0.9"])).is_ok());
    }
}
