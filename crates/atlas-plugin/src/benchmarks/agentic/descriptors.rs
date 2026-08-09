// SPDX-License-Identifier: AGPL-3.0-only

//! The agentic-webserver descriptor (Gate A).

use super::AgenticWebserver;
use crate::benchmark::BenchmarkDescriptor;
use crate::metadata::PluginMetadata;

const SUMMARY: &str = "N agentic runs: build a working Axum server, then verify it";
pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

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
    // Gate A. The webserver_ok thresholds (10/10 and Σ wall ≤ 1300 s) were
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
