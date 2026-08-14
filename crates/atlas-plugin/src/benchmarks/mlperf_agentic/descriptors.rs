// SPDX-License-Identifier: AGPL-3.0-only

//! The registered MLPerf Agentic Inference descriptor.
//!
//! One leg, not the two the eventual shape calls for: `mlperf-agentic-subset`
//! is the pinned-draw merge-gate candidate. The full 613-trajectory official
//! pass (`mlperf-agentic-full`) is NOT registered yet on purpose — its draw,
//! concurrency shape and wall time can only be sized from a calibration run,
//! and no calibration can happen until MLCommons publishes the dataset.
//! Registering a leg whose parameters are guesses would pin the guesses.

use super::MlperfAgentic;
use crate::benchmark::BenchmarkDescriptor;
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const SUBSET_SUMMARY: &str =
    "MLPerf Agentic Inference replay, inline-scored — UNRUNNABLE: dataset unpublished";
pub const SUBSET_METADATA: PluginMetadata = PluginMetadata::atlas(SUBSET_SUMMARY);

pub const SUBSET_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "mlperf-agentic-subset",
    name: "MLPerf agentic (subset)",
    summary: SUBSET_SUMMARY,
    detail: "Teacher-forced multi-turn replay of the MLPerf Agentic Inference dataset \
             (mlcommons/endpoints@7935df4): recorded trajectories are replayed single-stream \
             under the official immutable sampling params (temp 1.0, top_k 20, top_p 0.95, \
             presence 1.5, max 8192, preserve_thinking) plus a pinned seed, and scored \
             in-process by a fixture-verified port of the upstream inline scorer — workflow \
             intent-code match plus coding bash-executable multiset IoU. \
             ★ CANNOT RUN TODAY: the official dataset is unpublished (\"MLCommons storage, \
             link TBD\") and this leg refuses proxies, so it fails loudly at provisioning \
             until the file ships. No baseline exists; the first measured run on main \
             becomes one. Reports inline accuracy + OSL only — the SWE-bench Verified leg \
             of the official three-part gate is a separate live-agent workflow, not a leg.",
    duration_hint: "unrunnable — dataset TBD",
    updated: "2026-08-13",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "MLCommons specifies Qwen/Qwen3.6-35B-A3B (BF16); Atlas serves the official \
               FP8 sibling, which is rules-legal quantization but NOT the named checkpoint \
               — every number needs that caveat until the three-part accuracy gate is \
               cleared. Kimi K2.6 (1T) does not fit a GB10.",
    }),
    // No baseline exists yet — the detail above says so explicitly ("the first
    // measured run on main becomes one"), and a threshold param with nothing to
    // substitute from would pin a guess.
    threshold_params: &[],
    // CORRECTNESS, same reasoning as the BFCL legs: this scores intent-code
    // match and bash-executable IoU, and reports OSL rather than a rate. There
    // is no timing number here for a thermal or clock excursion to reach, so
    // gating it on machine state would stop accuracy work for a reason that
    // cannot affect the result.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(MlperfAgentic::new()),
};
