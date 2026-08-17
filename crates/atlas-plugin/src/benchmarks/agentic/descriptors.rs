// SPDX-License-Identifier: AGPL-3.0-only

//! The agentic-webserver descriptor (Gate A).

use super::AgenticWebserver;
use crate::benchmark::BenchmarkDescriptor;
use crate::hardware::Sensitivity;
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
    updated: "2026-08-14",
    needs_confirmation: true,
    // Gate A's thresholds were measured on the 35B MoE flagship, which stays the
    // DEFAULT subject. This benchmark also carries BENCH.toml variants for both
    // dense 27Bs, for different reasons — and the numbers are never comparable
    // ACROSS families: a dense 27B activates every parameter per token where the
    // 35B MoE activates ~3B, so its wall band is roughly 2x and the 35B's
    // ceiling does not transfer. That is exactly why they are separate baseline
    // entries with their own thresholds and serve recipes rather than one bar.
    //
    //   qwen3.6-27b : REGISTERED but UNMEASURED — its entry carries no
    //                 thresholds because none exist to carry, so a run there
    //                 BASELINES rather than gates.
    //   qwen3.8-27b : MEASURED — its entry carries thresholds and a serve
    //                 recipe, so a run there gates against its own bar.
    //
    // FP8 and NVFP4 of one family are both valid.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b", "qwen3.6-27b", "qwen3.8-27b"],
        note: "This benchmark is defined on the 35B MoE flagship (Qwen3.6-35B-A3B, FP8 or \
               NVFP4 — the required Gate A subject) and on both dense 27B variants. \
               Qwen3.6-27B is registered but UNMEASURED: its BENCH.toml entry has no \
               thresholds, so a run there baselines, it does not gate. Qwen3.8-27B is \
               MEASURED and gates against its own thresholds. Each variant carries its own \
               thresholds and serve recipe; any other checkpoint would produce numbers that \
               compare to nothing.",
    }),
    // The run-time Σ-wall verdict reads the SELECTED variant's committed
    // ceiling rather than a schema default one variant would contradict.
    threshold_params: &[("wall_budget_s", "sum_wall_s")],
    // MIXED, and classified by the half that can be corrupted. The
    // webserver_ok / followed_directions halves are correctness; the
    // `wall_budget_s` half is a Σwall bound, and it is the exact number the
    // 2026-08-15 retraction turned on — 692 s (dgx1) vs 1079 s (dgx2) for
    // unmodified `main`, both 10/10 + 10/10. A gate that carries a wall bound
    // is a speed gate for the purposes of this check.
    sensitivity: Sensitivity::Speed,
    ctor: || Box::new(AgenticWebserver::default()),
};
