// SPDX-License-Identifier: AGPL-3.0-only

//! The agentic-webserver descriptor (Gate A).

use super::AgenticWebserver;
use crate::benchmark::BenchmarkDescriptor;
use crate::metadata::PluginMetadata;

const SUMMARY: &str = "N agentic runs: build a working Axum server, then verify it";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

/// ★★ THIS BLOCK IS DOCUMENTATION. Nothing executes it.
///
/// Under `--pull-request-gate` the serve is built by `bench_selfstart` from the
/// RECIPE named in `BENCH.toml` — `qwen3.6/qwen3.6-35b-a3b-fp8-bf16head`, which
/// lives in the separate `atlas-recipes` repo and is honoured verbatim. Editing
/// the command below changes what a reader believes, not what runs. It is
/// written down here because it is the shape an operator reproduces by hand,
/// and it must not drift from the recipe.
///
/// ★ `--mtp-gate force` is a DETERMINISM pin, and its absence is the root cause
/// of this gate's intermittent 9/10 on `followed_directions`.
///
/// **IT IS NOT YET IN EFFECT.** The recipe carries `speculative: true` and
/// `mtp_quantization: bf16` and no `mtp_gate` key; `--mtp-gate` is
/// `Option<String>` with no clap default, so absent means `auto`. Closing this
/// requires a PR to `atlas-recipes` adding `mtp_gate: force` to that recipe.
/// Until that lands, the flip described below can still happen.
/// In `auto`, the MTP gate is a bandit arbiter that switches MTP<->serial at
/// runtime on **wall-clock** tok/s EWMAs. Speculation is NOT output-neutral at
/// temperature 0 on Atlas today, so a throughput-timed path switch makes greedy
/// decode depend on how fast the box happened to be.
///
/// ★★ THAT NON-NEUTRALITY IS A BUG, NOT A PROPERTY OF SPECULATION, and an
/// earlier version of this comment read as though it were the latter.
/// Speculative decoding is output-equivalent BY CONSTRUCTION: the drafter
/// proposes, the target verifies, and at temperature 0 the emitted sequence
/// must be bit-identical to plain greedy. Atlas violates it because restoring
/// SSM/conv state after a rejected draft does not reproduce what a fresh
/// prefill of the same tokens would produce — recorded 2026-07-22 as
/// "restore != fresh prefill, diverges ~token 250", with an OPEN workstream to
/// make the restore bit-exact.
///
/// The error scales with ROLLBACK COUNT (drafter context pulls output TOWARD
/// true greedy because higher acceptance means fewer rollbacks), which is why
/// the restore path is the suspect and the verify is not — `mtp_head` is
/// explicit that drafter logits may differ bitwise "because drafts are verified
/// by the main head". Fix the restore and the arbiter can switch freely with
/// nothing downstream noticing; the pin below stops being needed at all.
///
pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "agentic-webserver",
    name: "Agentic Webserver Test",
    summary: SUMMARY,
    detail: "Runs the flagship agentic task N times: the model writes a Rust Axum ping/pong \
             server, tests it, runs it and tears it down, using bash/write_file/read_file tools \
             in a fresh sandbox. Each run is scored on OUTCOME (the scorer builds it and gets a \
             'pong') and on PROCESS (did the agent do all six things the prompt asked?), plus \
             wall time. RUNS MODEL-AUTHORED SHELL inside the sandbox directory.",
    duration_hint: "~5 min per iteration",
    updated: "2026-07-31",
    needs_confirmation: true,
    // Gate A. The webserver_ok thresholds (10/10 and Σ wall ≤ 1000 s) were
    // measured on the 35B MoE flagship and mean nothing against another
    // checkpoint. FP8 and NVFP4 are both the same family and both valid.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "Gate A is defined on the 35B MoE flagship (Qwen3.6-35B-A3B, FP8 or NVFP4). \
               The dense 27B is a different gate (C2/D) with different thresholds, so a \
               run here would produce numbers that compare to nothing.",
    }),
    ctor: || Box::new(AgenticWebserver::default()),
};
