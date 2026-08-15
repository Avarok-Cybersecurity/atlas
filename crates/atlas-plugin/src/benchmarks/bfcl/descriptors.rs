// SPDX-License-Identifier: AGPL-3.0-only

//! The registered BFCL benchmark descriptors.
//!
//! Three descriptors, two of which are gates and differ ONLY in their draw:
//! `bfcl-subset` is the golden n=995 MLPerf-edge draw (the dense 27B's gate)
//! and `bfcl-subset-echolp` is the n=1004 echolp draw (the 35B MoE's gate,
//! because that is the only draw its recorded history is on). Their scores are
//! NOT interchangeable — the category mix alone moves
//! `normalized_single_turn_score` by ~1.8 points while leaving
//! `overall_accuracy` in the same place, which is exactly what makes crossing
//! them easy to miss and impossible to catch after the fact.

use super::{Bfcl, Variant};
use crate::benchmark::BenchmarkDescriptor;
use crate::metadata::PluginMetadata;

const SUBSET_SUMMARY: &str = "The golden n=995 MLPerf-edge draw, AST-scored";
const FULL_SUMMARY: &str = "Every single-turn sample in the three scored categories";
const ECHOLP_SUMMARY: &str = "The echolp n=1004 draw, AST-scored";
pub const SUBSET_METADATA: PluginMetadata = PluginMetadata::atlas(SUBSET_SUMMARY);
pub const FULL_METADATA: PluginMetadata = PluginMetadata::atlas(FULL_SUMMARY);
pub const ECHOLP_METADATA: PluginMetadata = PluginMetadata::atlas(ECHOLP_SUMMARY);

pub const SUBSET_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset",
    name: "BFCL (subset)",
    summary: SUBSET_SUMMARY,
    detail: "Berkeley Function Calling Leaderboard v4, single-turn, on the golden MLPerf-edge \
             draw: categories non_live/live/hallucination at 62/10/10 with a 25-sample floor, \
             which is exactly 995 samples. Reports overall_accuracy and \
             normalized_single_turn_score against the MLPerf-edge floor (83.64 / 85.32). \
             Downloads bfcl-eval into ~/.atlas/artifacts on first run.",
    duration_hint: "~3.5 h",
    updated: "2026-07-31",
    needs_confirmation: false,
    // Gates B and D. B runs on whichever model the PR targets; D on a dense 27B.
    // Qwen3.8-27B joined 2026-08-14 as the incoming dense gate subject, on the
    // same golden draw and a serve recipe byte-identical to the 3.6 one, so the
    // two legs read as a generation-over-generation delta on one axis. It is
    // UNMEASURED — its BENCH.toml entry carries no thresholds, so a run there
    // baselines rather than gates, and it does NOT inherit 3.6's floors or the
    // MLPerf floor: same architecture and draw, different weights.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.8-27b"],
        note: "The BFCL gates are defined on Qwen3.6-27B (dense — the MLPerf-edge floor \
               83.64/85.32 rides on this checkpoint), Qwen3.6-35B-A3B (MoE, gate B), and \
               Qwen3.8-27B (dense, UNMEASURED — a run there baselines rather than gates, \
               and inherits neither 3.6's floors nor the MLPerf floor). Scores on any \
               other checkpoint have no recorded baseline to beat.",
    }),
    threshold_params: &[],
    ctor: || Box::new(Bfcl::new(Variant::Subset)),
};

pub const FULL_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-full",
    name: "BFCL (full)",
    summary: FULL_SUMMARY,
    detail: "The same benchmark with no sampling: every single-turn sample in the three scored \
             categories (~3625). Same composition as the subset draw, so the normalized score \
             stays comparable — it just removes the sampling noise, at roughly 3.6× the wall \
             time.",
    duration_hint: "~12 h",
    updated: "2026-07-31",
    needs_confirmation: false,
    // Gates B and D. B runs on whichever model the PR targets; D on a dense 27B.
    // Qwen3.8-27B joined 2026-08-14 as the incoming dense gate subject, on the
    // same golden draw and a serve recipe byte-identical to the 3.6 one, so the
    // two legs read as a generation-over-generation delta on one axis. It is
    // UNMEASURED — its BENCH.toml entry carries no thresholds, so a run there
    // baselines rather than gates, and it does NOT inherit 3.6's floors or the
    // MLPerf floor: same architecture and draw, different weights.
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-27b", "qwen3.6-35b-a3b", "qwen3.8-27b"],
        note: "The BFCL gates are defined on Qwen3.6-27B (dense — the MLPerf-edge floor \
               83.64/85.32 rides on this checkpoint), Qwen3.6-35B-A3B (MoE, gate B), and \
               Qwen3.8-27B (dense, UNMEASURED — a run there baselines rather than gates, \
               and inherits neither 3.6's floors nor the MLPerf floor). Scores on any \
               other checkpoint have no recorded baseline to beat.",
    }),
    threshold_params: &[],
    ctor: || Box::new(Bfcl::new(Variant::Full)),
};

pub const SUBSET_ECHOLP_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset-echolp",
    name: "BFCL (subset, echolp draw)",
    summary: ECHOLP_SUMMARY,
    detail: "Berkeley Function Calling Leaderboard v4, single-turn, on the echolp draw: \
             categories non_live/live/hallucination at 46/23/12 with a 25-sample floor, which is \
             exactly 1004 samples. This draw weights `live` more than twice as heavily as the \
             golden one, which moves normalized_single_turn_score by ~1.8 points while leaving \
             overall_accuracy in the same place — so its scores are NOT comparable to the golden \
             draw's, and it carries its own baseline. It exists because the 35B's only recorded \
             BFCL history is on this draw.",
    duration_hint: "~3.5 h",
    updated: "2026-08-06",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "The echolp draw is where the 35B MoE's recorded history lives (84.66 / 83.32 \
               high-water). The dense 27B is gated on the golden n=995 draw instead — do not \
               cross the two, the category mix alone moves normalized by ~1.8 points.",
    }),
    threshold_params: &[],
    ctor: || Box::new(Bfcl::new(Variant::SubsetEcholp)),
};
