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
use crate::hardware::Sensitivity;
use crate::metadata::PluginMetadata;

const SUBSET_SUMMARY: &str = "The golden n=995 MLPerf-edge draw, AST-scored";
const FULL_SUMMARY: &str = "Every single-turn sample in the three scored categories";
const ECHOLP_SUMMARY: &str = "The echolp n=1004 draw, AST-scored";
pub const SUBSET_METADATA: PluginMetadata = PluginMetadata::atlas(SUBSET_SUMMARY);
pub const FULL_METADATA: PluginMetadata = PluginMetadata::atlas(FULL_SUMMARY);
pub const ECHOLP_METADATA: PluginMetadata = PluginMetadata::atlas(ECHOLP_SUMMARY);

/// The baseline-coupled verdict params both GATED draws share: under
/// `--pull-request-gate` each is auto-filled from the selected variant's own
/// BENCH.toml `min` bound (`bench_resolve::apply_threshold_params`), so a
/// non-MLPerf checkpoint that clears its committed bars gets a PASS verdict —
/// which the gate machinery requires (`GateRecord::verdict_passes`) — instead
/// of the info verdict that used to read red. Both draws gate on the same two
/// metric names; their BARS differ per variant, which is exactly why these are
/// threshold params and not schema constants. `bfcl-full` is not a gate and
/// deliberately declares none.
const GATE_THRESHOLD_PARAMS: &[(&str, &str)] = &[
    ("min_overall", "overall_accuracy"),
    ("min_normalized", "normalized_single_turn_score"),
];

pub const SUBSET_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "bfcl-subset",
    name: "BFCL (subset)",
    summary: SUBSET_SUMMARY,
    detail: "Berkeley Function Calling Leaderboard v4, single-turn, on the golden MLPerf-edge \
             draw: categories non_live/live/hallucination at 62/10/10 with a 25-sample floor, \
             which is exactly 995 samples. Reports overall_accuracy and \
             normalized_single_turn_score against the MLPerf-edge floor (83.64 / 85.32); \
             the floor VERDICT applies only to the Qwen3.6-27B submission checkpoints — \
             every other checkpoint is judged by its own BENCH.toml thresholds, with the \
             floor kept as table styling for reference. \
             Downloads bfcl-eval into ~/.atlas/artifacts on first run.",
    duration_hint: "~3.5 h",
    updated: "2026-08-15",
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
    // Under --pull-request-gate the run's own verdict is judged against the
    // selected variant's committed BENCH.toml floors (min bounds), not the
    // MLPerf floor — which gates only the submission checkpoints (report.rs).
    threshold_params: GATE_THRESHOLD_PARAMS,
    // Accuracy. A hot box scores the same and takes longer, and blocking a
    // 3.5-hour accuracy run on chassis temperature would stop correctness work
    // for a reason that cannot reach the number.
    sensitivity: Sensitivity::Correctness,
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
    updated: "2026-08-15",
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
    // Accuracy, same as the subsets.
    sensitivity: Sensitivity::Correctness,
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
    updated: "2026-08-15",
    needs_confirmation: false,
    intended_for: Some(crate::benchmark::ModelExpectation {
        families: &["qwen3.6-35b-a3b"],
        note: "The echolp draw is where the 35B MoE's recorded history lives (84.66 / 83.32 \
               high-water). The dense 27B is gated on the golden n=995 draw instead — do not \
               cross the two, the category mix alone moves normalized by ~1.8 points.",
    }),
    // Same two gating metrics as the golden draw (its BENCH.toml entry bounds
    // overall_accuracy and normalized_single_turn_score), different bars.
    threshold_params: GATE_THRESHOLD_PARAMS,
    // Accuracy, same as the golden draw.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(Bfcl::new(Variant::SubsetEcholp)),
};

// ── Shards ───────────────────────────────────────────────────────────────────
//
// Each gate above is also a BENCHMARK GROUP of four shards that can run at the
// same time on different boxes. The group id is unchanged (`bfcl-subset`,
// `bfcl-subset-echolp`), so `coverage::REQUIRED` keeps its eleven entries and
// the BENCH.toml thresholds are untouched — only the way the number is produced
// changes. See `gate::group` for the composition rules and
// `benchmarks::bfcl::aggregate` for why the members' scores are recombined over
// COUNTS rather than averaged.
//
// The members carry NO `threshold_params`: a shard is not a gate and has no
// bars of its own. Its verdict is informational; the group's aggregate is what
// `check_record` judges.
//
// `duration_hint` is the whole draw's ~3.5 h divided four ways. That is the
// entire point: the two BFCL legs are the floor on every campaign's wall-clock,
// and on 2026-09-06 they were 3.5 h each even after being split across two
// boxes.
macro_rules! shard_descriptor {
    ($konst:ident, $id:literal, $name:literal, $variant:expr, $index:literal, $base:ident) => {
        pub const $konst: BenchmarkDescriptor = BenchmarkDescriptor {
            id: $id,
            name: $name,
            summary: $base.summary,
            detail: $base.detail,
            duration_hint: "~55 min (one quarter of the draw)",
            updated: $base.updated,
            needs_confirmation: false,
            intended_for: $base.intended_for,
            // Not a gate: the GROUP owns the thresholds.
            threshold_params: &[],
            sensitivity: Sensitivity::Correctness,
            ctor: || Box::new(Bfcl::sharded($variant, $index, 4)),
        };
    };
}

shard_descriptor!(
    SUBSET_A,
    "bfcl-subset-a",
    "BFCL (subset) shard A",
    Variant::Subset,
    0,
    SUBSET_DESCRIPTOR
);
shard_descriptor!(
    SUBSET_B,
    "bfcl-subset-b",
    "BFCL (subset) shard B",
    Variant::Subset,
    1,
    SUBSET_DESCRIPTOR
);
shard_descriptor!(
    SUBSET_C,
    "bfcl-subset-c",
    "BFCL (subset) shard C",
    Variant::Subset,
    2,
    SUBSET_DESCRIPTOR
);
shard_descriptor!(
    SUBSET_D,
    "bfcl-subset-d",
    "BFCL (subset) shard D",
    Variant::Subset,
    3,
    SUBSET_DESCRIPTOR
);
shard_descriptor!(
    ECHOLP_A,
    "bfcl-subset-echolp-a",
    "BFCL (echolp) shard A",
    Variant::SubsetEcholp,
    0,
    SUBSET_ECHOLP_DESCRIPTOR
);
shard_descriptor!(
    ECHOLP_B,
    "bfcl-subset-echolp-b",
    "BFCL (echolp) shard B",
    Variant::SubsetEcholp,
    1,
    SUBSET_ECHOLP_DESCRIPTOR
);
shard_descriptor!(
    ECHOLP_C,
    "bfcl-subset-echolp-c",
    "BFCL (echolp) shard C",
    Variant::SubsetEcholp,
    2,
    SUBSET_ECHOLP_DESCRIPTOR
);
shard_descriptor!(
    ECHOLP_D,
    "bfcl-subset-echolp-d",
    "BFCL (echolp) shard D",
    Variant::SubsetEcholp,
    3,
    SUBSET_ECHOLP_DESCRIPTOR
);
