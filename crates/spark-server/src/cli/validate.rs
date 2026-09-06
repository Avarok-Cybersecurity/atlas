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
// The allowed-value lists live in `flag_values` — one module read by BOTH this
// validator and the dashboard's option picker, so what is offered and what is
// enforced cannot drift apart. Their sync with the parse sites in `serve.rs`
// is pinned by `flag_values_tests`.
use super::flag_values::{
    KvHighPrecisionLayers, LM_HEAD_DTYPES, MTP_GATES, MTP_QUANTS, SCHEDULING_POLICIES,
    SSM_H_DTYPES, TOOL_CALL_PARSERS,
};

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
    if let Some(dtype) = &args.ssm_h_dtype {
        check_enum(&mut v, "--ssm-h-dtype", dtype, SSM_H_DTYPES);
    }
    // Only when it was given: absent means "let ATLAS_MTP_GATE_FORCE decide",
    // and validating an unwritten value would reject nothing but confuse the
    // reader of this list.
    if let Some(gate) = &args.mtp_gate {
        check_enum(&mut v, "--mtp-gate", gate, MTP_GATES);
    }
    // The FP16 h-state twins live ONLY on the fused-norm decode arm. Without
    // it the dispatch lands on an FP32-only kernel pointed at an FP16 pool,
    // which does not fault — it emits fluent garbage. Reject the pair here,
    // in milliseconds, rather than at the first decode step of a benchmark.
    // `!= Some(true)`, not `== Some(false)`: an ABSENT `--gdn-fused-norm` beside
    // an explicit `--ssm-h-dtype f16` publishes fused_norm OFF (the three GDN
    // flags are one cell), so absent is just as dangerous here as explicit off.
    // SSOT with what the server PUBLISHES: `f16-pool` is `f16` plus the narrow
    // pool, so every f16 rule below has to bind on it too. Reading the pair off
    // `ssm_h_dtype_bits` rather than matching the string twice is what stops a
    // new spelling from silently escaping these checks.
    // `.0` only: every rule here binds on "the h-state is FP16", which
    // `f16-pool` also is. Nothing below is specific to the pool WIDTH — the
    // narrowing is transparent to the flag combinations.
    let h_f16 = spark_model::layers::qwen3_ssm::ssm_h_dtype_bits(args.ssm_h_dtype.as_deref()).0;
    if h_f16 && args.gdn_fused_norm != Some(true) {
        v.push(Violation::new(
            "--ssm-h-dtype f16 without --gdn-fused-norm",
            "the FP16 h-state twins exist only on the fused-norm decode arm; the unfused \
             arms (gated_delta_rule_decode, ..._decode_f32_strided) are FP32-only and \
             would read the FP16 pool as FP32 — fluent garbage, not an error",
            "add --gdn-fused-norm, or use --ssm-h-dtype f32",
        ));
    }
    // DFlash verifies gamma+1 = 17 rows, which lands on `gated_delta_rule_wy17`
    // — the one WY family with no FP16 h-state twin AND an explicit
    // FP32-element intermediate stride. Both halves read an FP16 h-state as
    // FP32: fluent garbage, not an error. Applies to plain `f16` as well as
    // `f16-pool`, hence `h_f16`.
    // 2026-08-29 (#812): the wyN family now ships FP16 h-state twins for
    // K=5..16 (gated_delta_rule_wy{5..16}_f16, same module), so DFlash under
    // the f16 pool is supported for --dflash-gamma <= 16 — the whole reachable
    // range on block-8-class drafters (Qwen3.8 DFlash2). Above 16 the verify
    // dispatches gated_delta_rule_wy17, which still has no FP16 twin, and the
    // runtime backstop would refuse at the first verify step; reject the pair
    // here in milliseconds instead. Missing-twin builds are also refused at
    // runtime (hard error, never a silent FP32-over-FP16 fallback).
    // `dflash_gamma` is an Option here (#650 made the drafter's own
    // `dflash_config.block_size` the default), so this rejects only an
    // EXPLICIT over-16 request. An unset gamma resolves from the drafter
    // checkpoint, which is not readable at validate time — that path is
    // covered by the runtime backstop, which hard-errors on a missing twin
    // rather than falling back to FP32-over-FP16.
    // OFF-BY-ONE, corrected 2026-08-30: #817 wrote `> 16` while its message
    // claimed "widths 5..16 are covered". The verify width is K = gamma + 1,
    // so gamma 16 is K=17 — wy17, the one width with no twin — and `> 16`
    // admitted exactly that case. The runtime backstop would have caught it,
    // loudly, after a full model load; this catches it in milliseconds. Both
    // sites now read the same constant instead of a literal.
    if h_f16
        && args.dflash
        && args
            .dflash_gamma
            .is_some_and(|g| g > spark_model::layers::qwen3_ssm::MAX_F16_TWIN_DFLASH_GAMMA)
    {
        v.push(Violation::new(
            "--dflash with a verify width above the FP16 twin range, together with \
             --ssm-h-dtype f16",
            "DFlash verify widths above 16 dispatch gated_delta_rule_wy17, which has \
             no FP16 h-state twin — an FP32 kernel over an FP16 h-state emits fluent \
             garbage. Widths 5..16 are covered by the wyN _f16 twins",
            "use --dflash-gamma <= 15 (verify width 16), drop --dflash, or use \
             --ssm-h-dtype f32",
        ));
    }

    // `--ssm-rollback-mode`: validated through the model-side `FromStr` —
    // the SAME parse `publish_kernel_flags` uses — so what is accepted here
    // and what is published cannot drift (mirrors the kv-cache-dtype rule).
    if let Err(why) = args
        .ssm_rollback_mode
        .parse::<spark_model::ssm_reserve::SsmRollbackMode>()
    {
        v.push(Violation::new(
            format!(
                "--ssm-rollback-mode '{}' is not a valid value.",
                args.ssm_rollback_mode
            ),
            why,
            "use snapshot (default, wired) or replay (experimental scaffold).",
        ));
    }
    // #435: the exact-verify arm's kernels are FP32 readers, so an FP16
    // h-state pool disables it (`GdnFlags::verify_exact_active`). Honouring
    // `--ssm-h-dtype f16` by SILENTLY ignoring an explicit `--exact-verify`
    // would ship the very divergence the operator just opted out of — the
    // exact class of combination this validator exists to refuse.
    if args.exact_verify == Some(true) && h_f16 {
        v.push(Violation::new(
            "--exact-verify together with --ssm-h-dtype f16",
            "the exact MTP-verify chain (issue #435) runs FP32-reader kernels and must \
             never read the FP16 h-state pool, so with f16 the exact request would be \
             silently dropped and spec-on output would NOT equal spec-off",
            "drop --ssm-h-dtype f16 (f32 is the default), or drop --exact-verify",
        ));
    }
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
    // the runtime's own `FromStr` so this stays in sync automatically. Only an
    // explicitly passed value can be checked here: an omitted flag resolves
    // later against MODEL.toml (`resolve_kv_dtype_str`). NOTE the MODEL.toml
    // value is NOT build-validated — `build_parse_behavior.rs` embeds
    // `default_kv_dtype` verbatim (`as_str().unwrap_or("")`), so a typo there
    // surfaces only at load time when the effective string hits this same
    // `FromStr` in `serve_phases/kv_cache.rs`, after GPU init.
    if let Some(kv_dtype) = args.kv_cache_dtype.as_deref()
        && kv_dtype
            .parse::<spark_runtime::kv_cache::KvCacheDtype>()
            .is_err()
    {
        v.push(Violation::new(
            format!("--kv-cache-dtype '{kv_dtype}' is not a known KV cache dtype."),
            "the value does not parse to any supported KV cache format.",
            "use one of: fp8, bf16, nvfp4 (or a turbo* TurboQuant-Plus variant).",
        ));
    }

    // ── FP8 KV headroom: < 1.0 shrinks the frozen scale below the observed
    // absmax, guaranteeing clipping on the very tokens it was measured from. ──
    if args.fp8_kv_headroom < 1.0 {
        v.push(Violation::new(
            format!("--fp8-kv-headroom {} is below 1.0.", args.fp8_kv_headroom),
            "the frozen FP8 KV scale covers headroom× the first-observe absmax; \
             a multiplier under 1.0 clips the very values it was measured from.",
            "use a value ≥ 1.0 (default 2.0).",
        ));
    }

    // ── FP8 KV calibration only applies to an FP8 KV cache (issue #288 example). ──
    // Both flags must be explicit to flag the combination here: an omitted
    // --kv-cache-dtype resolves against MODEL.toml only later, so its
    // effective value is unknown at CLI-validation time.
    if let (Some(calib), Some(kv_dtype)) = (
        args.fp8_kv_calibration_tokens,
        args.kv_cache_dtype.as_deref(),
    ) && calib > 0
        && kv_dtype != "fp8"
    {
        v.push(Violation::new(
            format!(
                "--fp8-kv-calibration-tokens {calib} has no effect with --kv-cache-dtype {kv_dtype}.",
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
    // Only an explicit flag is checked: an omitted --num-drafts resolves
    // against MODEL.toml later, and a model default is inert without a
    // speculative method rather than a user error.
    let any_spec = args.speculative || args.self_speculative || args.ngram_speculative;
    if let Some(num_drafts) = args.num_drafts
        && num_drafts > 1
        && !any_spec
    {
        // --dflash IS a speculative method, but it does not consume
        // --num-drafts either: the drafter's trained block size (γ) decides
        // the draft count (`serve_load` forces num_drafts = γ - 1). The flag
        // is ignored in both arms — what differs is the correct remedy.
        if args.dflash {
            v.push(Violation::new(
                format!("--num-drafts {num_drafts} is ignored under --dflash."),
                "a DFlash serve drafts at the drafter checkpoint's trained block size \
                 (γ); the scheduler overrides --num-drafts with γ - 1.",
                "drop --num-drafts, or use --dflash-gamma to override the drafter's γ \
                 (block-diffusion drafters are trained at ONE block size — expect \
                 acceptance collapse away from it).",
            ));
        } else {
            v.push(Violation::new(
                format!("--num-drafts {num_drafts} is set but no speculative method is enabled.",),
                "the draft count only applies when speculative decoding proposes drafts; \
                 without it the flag is ignored.",
                "add --speculative (MTP), --self-speculative, or --ngram-speculative — or \
                 drop --num-drafts.",
            ));
        }
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

    // ── --kv-high-precision-layers is a free-form string, so a typo used to
    //    reach the resolve site and be swallowed there. ──
    // The resolve site (`serve_phases::kv_cache`) ran `s.parse().unwrap_or(0)`
    // behind a `warn!` — AFTER the weight load, and `0` is not the default but
    // the value that defers to `auto_high_precision_layers`. So `atuo` bought
    // the operator a third configuration, minutes in, in a log line. The parse
    // is now shared with that site (one definition of the accepted forms) and
    // a bad value is refused here, in milliseconds, instead.
    if let Err(why) = args
        .kv_high_precision_layers
        .parse::<KvHighPrecisionLayers>()
    {
        v.push(Violation::new(
            format!(
                "--kv-high-precision-layers '{}' is not a valid value.",
                args.kv_high_precision_layers
            ),
            why,
            "use a layer count (0 = let the KV dtype decide), auto (2, recommended), \
             or max/all (every attention layer).",
        ));
    }

    // ── --warmup-prompt promises a startup prefill that does not exist. ──
    // The flag parses, and NOTHING reads `args.warmup_prompt` — not one site in
    // the workspace. Its own `--help` text states that the server "tokenizes and
    // prefills this prompt" and "eliminates the cold-start TTFT penalty
    // (~196ms)", and QUICKSTART.md, book/src/operations/server.md and
    // docs/GB10_DEPLOYMENT_GUIDE.md each tell the reader to pass it. So an
    // operator following the quickstart passes a path, measures the same cold
    // TTFT, and has no way to tell that the flag is inert rather than
    // ineffective — the diagnosis lands on Atlas being slow.
    //
    // Refused rather than warned, for this module's usual reason: a warning
    // scrolls off, and command lines get copied and published. Refused rather
    // than deleted from the CLI, so the operator gets a sentence explaining
    // what happened instead of clap's bare "unexpected argument".
    if args.warmup_prompt.is_some() {
        v.push(Violation::new(
            "--warmup-prompt is not implemented.",
            "nothing reads it: no prompt is tokenized, no prefill runs and nothing enters \
             the prefix cache, so the cold-start TTFT its help text promises to remove is \
             still paid on the first real request.",
            "drop --warmup-prompt, and warm the server yourself by sending one throwaway \
             request after startup — that populates the prefix cache through the same path \
             a real request does.",
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
#[path = "validate_tests.rs"]
mod tests;
