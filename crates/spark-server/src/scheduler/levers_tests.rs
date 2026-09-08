// SPDX-License-Identifier: AGPL-3.0-only

//! Scheduler lever resolution tests: the polarity of every switch, and the
//! two that must be asserted against `from_env()` rather than `defaults()`.
//!
//! Split out of `levers.rs` to keep it under the repository's 500-LoC cap —
//! the same pattern `model_levers_tests.rs` and `mtp_carry_tests.rs` use.
//! `levers.rs` reached 497 lines when the two turn-termination levers moved
//! in, which is a trap rather than a margin.

use super::*;

#[test]
fn the_five_opt_out_levers_ship_on() {
    // Each of these is spelled as a NEGATIVE env var. Collapsing them into
    // an opt-in resolver would silently disable five shipped behaviours.
    let d = SchedLevers::defaults();
    assert!(d.fast_greedy_grammar, "ATLAS_DISABLE_FAST_GREEDY");
    assert!(d.fast_masked, "ATLAS_DISABLE_FAST_MASKED");
    assert!(d.mtp_minp, "ATLAS_NO_MTP_MINP");
    assert!(d.mtp_verify_sample, "ATLAS_NO_MTP_VERIFY_SAMPLE");
    assert!(d.forced_token_fastpath, "ATLAS_DISABLE_FORCED_TOKEN");
}

/// The two turn-termination levers ship ON, and they accept `false` as
/// well as `0`.
///
/// Pinned in `from_env()`, not just `defaults()`, for the reason
/// `spec_think_is_off_in_the_resolver_the_server_actually_uses` exists:
/// `defaults()` is a hand-written literal and cannot catch a change to
/// what the SERVER resolves. Both of these were per-token env reads
/// before they became fields, and both are root-cause fixes for
/// post-tool-call runaway — defaulting either OFF re-opens a cap-burn
/// that the webserver_ok gate measures.
#[test]
fn the_turn_termination_levers_ship_on_in_the_live_resolver() {
    // SAFETY: single-threaded test process; no other thread reads the env.
    unsafe {
        std::env::remove_var("ATLAS_TOOL_RESPONSE_STOP");
        std::env::remove_var("ATLAS_TOOL_EOS_ESCAPE");
    }
    let live = SchedLevers::from_env();
    assert!(live.tool_response_stop, "ATLAS_TOOL_RESPONSE_STOP");
    assert!(live.tool_eos_escape, "ATLAS_TOOL_EOS_ESCAPE");
    assert!(SchedLevers::defaults().tool_response_stop);
    assert!(SchedLevers::defaults().tool_eos_escape);

    // The rule they resolve through, exercised directly. `!= "1"` would
    // pass the `0` case and silently ignore every other spelling.
    use crate::scheduler::helpers::parse_flag_default_on as f;
    assert!(f(None));
    assert!(f(Some("1")));
    assert!(f(Some("true")));
    assert!(f(Some("junk")), "unknown values keep the shipped behaviour");
    assert!(!f(Some("0")));
    assert!(!f(Some("false")));
    assert!(!f(Some("FALSE")), "case-insensitive");
    assert!(!f(Some("  0  ")), "trimmed");
}

#[test]
fn every_opt_in_lever_ships_off() {
    let d = SchedLevers::defaults();
    assert!(!d.force_temp_zero);
    assert!(!d.dflash_masked_verify && !d.dflash_adaptive && !d.dflash_spec_think);
    assert!(!d.disable_watchdogs);
    assert!(!d.decode_timing && !d.mtp_timing && !d.adadec_diagnostic);
}

/// ★ `defaults()` CANNOT catch a change to what the SERVER resolves.
///
/// It is a hand-written struct literal; `from_env()` is the constructor
/// `spark serve` actually calls. When PR #831 graduated
/// `dflash_spec_think` from `opt_in` to `on_unless_zero`, ONLY `from_env()`
/// changed — `defaults()` still said `false`, so
/// `every_opt_in_lever_ships_off` above stayed green and
/// `cargo test --workspace` passed. The regression reached the GPU gates
/// instead, where it cost a full 11-gate campaign to find.
///
/// This asserts the resolver itself, with no env set. It is deliberately
/// narrow: `dflash_spec_think` is the one lever in this struct whose value
/// escapes the DFlash lane. `mtp_gate::spec_dispatch_eligible` reads it as
///
///     if inside_thinking && !spec_think { return false; }
///
/// for BOTH lanes, so defaulting it on lets speculation enter `<think>` on
/// plain MTP, where batch-K verify is not byte-lossless at T=0. Measured
/// twice with the same signature — the 2026-08-16 bisect, and 2026-09-01
/// on this PR: agentic-webserver 10/10 -> 9/10 webserver_ok and 10/10 ->
/// 7/10 followed_directions, deterministically, plus bfcl-subset-echolp
/// 0.44 below both floors on the same recipe.
#[test]
fn spec_think_is_off_in_the_resolver_the_server_actually_uses() {
    // SAFETY: single-threaded test process; no other thread reads the env.
    unsafe { std::env::remove_var("ATLAS_DFLASH_SPEC_THINK") };
    let live = SchedLevers::from_env();
    assert!(
        !live.dflash_spec_think,
        "ATLAS_DFLASH_SPEC_THINK must stay OPT-IN: from_env() resolved it ON. \
         It is the one lever here that is not gated behind dflash_verify_raw_argmax, \
         so defaulting it on changes plain-MTP serving and deterministically \
         damages agentic trajectories. See mtp_gate::spec_dispatch_eligible."
    );
    // The two levers this PR DID graduate stay graduated: both are
    // additionally gated on `dflash_verify_raw_argmax` (= args.dflash), so
    // they cannot reach a no-drafter serve.
    assert!(
        live.dflash_masked_verify,
        "masked_verify is intentionally default-ON"
    );
    assert!(
        live.dflash_seam_serial,
        "seam_serial is intentionally default-ON"
    );
}

#[test]
fn the_loop_watchdog_is_toggleable_at_runtime() {
    // The one lever with real runtime mutation: the TUI ops REPL flips it
    // mid-run. Modelled as an atomic INSIDE the carried struct rather than
    // as a process global with a setter.
    let d = SchedLevers::defaults();
    assert!(!d.loop_watchdog());
    d.set_loop_watchdog(true);
    assert!(d.loop_watchdog());
    d.set_loop_watchdog(false);
    assert!(!d.loop_watchdog());
}

#[test]
fn an_absent_mtp_gate_flag_leaves_the_legacy_variable_reachable() {
    // The whole of the fix: publishing the clap default sealed
    // `MTP_GATE_FORCE_CLI` on every `spark serve`, so the
    // `ATLAS_MTP_GATE_FORCE` fallback in `mtp_gate_force` could never run
    // even though `--help` documents it. `None` must not seal.
    //
    // ★ The cell is process-global with no reset, so this is the only test
    // in this binary that may write it — a second writer would make both
    // order-dependent.
    for _ in 0..3 {
        set_mtp_gate_force(None);
    }
    set_mtp_gate_force(Some(true));
    assert!(
        mtp_gate_force(),
        "an absent flag must leave the cell open for the next writer"
    );
    assert!(
        SchedLevers::from_env().mtp_gate_force,
        "and the carried levers read the same resolution — one rule, not two"
    );
}

#[test]
fn two_runs_hold_independent_levers() {
    let a = SchedLevers::defaults();
    let b = SchedLevers {
        dflash_adaptive: true,
        ..SchedLevers::defaults()
    };
    assert!(!a.dflash_adaptive && b.dflash_adaptive);
    a.set_loop_watchdog(true);
    assert!(!b.loop_watchdog(), "and independent runtime state");
}

/// ★ THE SCHEDULER THREAD DOES NOT READ THE ENVIRONMENT PER TOKEN.
///
/// `emit_step` and `decode_logits_step` run once per generated token per
/// sequence. Two `env_flag_default_on` readers on that path cost ~320,000
/// `std::env::var` calls in one sweep — each an allocation plus the
/// process-wide environment lock, which serialises this thread against every
/// other reader and whose cost GROWS with concurrency (0.57 us for a 30-var
/// resolve at one thread, 4.00 us at eight). Both are now `SchedLevers`
/// fields, resolved once when the run starts.
///
/// A source-level check because the property is "who may read the
/// environment", which no runtime assertion can observe. The sibling table
/// for `spark-model` lives in `layers/ops/hot_path_env_guards.rs`.
#[test]
fn the_per_token_scheduler_path_does_not_read_the_environment() {
    // (file, functions still allowed to read)
    const GUARDED: [(&str, &[&str]); 3] = [
        ("emit_step.rs", &[]),
        (
            "decode_logits_step.rs",
            &[
                // Both are `OnceLock`ed: the read happens at most once per
                // PROCESS, not once per token, so the lock is paid once. Named
                // individually rather than exempted as a class, because
                // "it is behind a OnceLock" is a claim to be checked per
                // function — and a process-lifetime cache is the weaker
                // pattern (see `ModelLevers`'s module doc on why a static
                // outlives the model whose flags it encodes); these two are
                // scheduler-level rather than model-level, which is the only
                // reason it is acceptable here.
                "think_ended_gpu_argmax_enabled",
                "parallel_sample_enabled",
            ],
        ),
        (
            "helpers.rs",
            &[
                // Its one remaining caller, `grammar_budget_close_enabled`,
                // runs on the grammar-close path — at most once per request.
                "env_flag_default_on",
                // `WatchdogParams::from_behavior` is built once per model load.
                "from_behavior",
            ],
        ),
    ];
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut offenders = Vec::new();
    for (file, allowed) in GUARDED {
        let path = dir.join(file);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} is a guarded path: {e}", path.display()));
        let mut current = "<file scope>".to_string();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            for prefix in ["pub(crate) fn ", "pub(super) fn ", "pub fn ", "fn "] {
                if let Some(rest) = trimmed.strip_prefix(prefix) {
                    current = rest
                        .split(['(', '<'])
                        .next()
                        .unwrap_or("?")
                        .trim()
                        .to_string();
                    break;
                }
            }
            // Comments name these variables throughout; only code counts.
            let code = line.split("//").next().unwrap_or("");
            if code.contains("std::env::var") && !allowed.contains(&current.as_str()) {
                offenders.push(format!("{file}:{} in `{current}`", i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "environment reads on the per-token scheduler path. Resolve the \
         variable ONCE into `SchedLevers` and read `sched.levers` instead — \
         and reuse the parser in `helpers`, do not re-spell the rule: \
         {offenders:?}"
    );
}
