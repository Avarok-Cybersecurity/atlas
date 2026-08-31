// SPDX-License-Identifier: AGPL-3.0-only

//! The benchmark suite, in the order the Benchmarks pane lists it.

use crate::benchmark::BenchmarkDescriptor;
use crate::benchmarks::{
    agentic, bfcl, concurrency, contamination, decode_floor, mlperf_agentic, quick_speed,
    serve_matrix, ssm_poison, ttft, video, vision,
};

/// Every benchmark, list order. Cheapest and most-run first.
///
/// STATIC, DELIBERATELY — compile-time data. A table of `&'static`
/// descriptors with no interior mutability and nothing derived from a model
/// or a run; it needs a stable address only so `all()` can hand out slices
/// of it. Registration is a compile-time decision, not a runtime one.
const ALL: &[&BenchmarkDescriptor] = &[
    // The cheapest probe in the suite (~1–3 min) and a measurement tool by
    // design: no baseline, no thresholds, excused from the PR gate set in
    // `gate::coverage::NOT_REQUIRED`.
    &quick_speed::DESCRIPTOR,
    // The gate-shaped counterpart of the probe above: same cost class
    // (~3–6 min), every knob pinned, judged against a BENCH.toml floor.
    // REQUIRED (`gate::coverage::REQUIRED`) since 2026-08-15, promoted on
    // the 12-run sigma calibration of the floor.
    &decode_floor::DESCRIPTOR,
    &concurrency::DESCRIPTOR,
    &concurrency::DFLASH2_DESCRIPTOR,
    &ttft::WARM_DESCRIPTOR,
    &ttft::COLD_DESCRIPTOR,
    &contamination::DESCRIPTOR,
    // Cheap and endpoint-only, and REQUIRED on the vision targets — the
    // per-model constraint lives in BENCH.toml, since gate coverage is
    // path-based and has no per-model dimension. A text-only target has no
    // entry and the gate does not apply to it.
    &vision::DESCRIPTOR,
    // Required where a target's BENCH.toml declares video-fidelity. The
    // per-model applicability lives there because coverage is path-based.
    &video::DESCRIPTOR,
    // Cheaper than the agentic gate (~10 min vs ~17 min) and catches a
    // class the agentic run only surfaces by accident, so it is listed
    // before it.
    &ssm_poison::DESCRIPTOR,
    &agentic::DESCRIPTOR,
    &bfcl::SUBSET_DESCRIPTOR,
    &bfcl::SUBSET_ECHOLP_DESCRIPTOR,
    &bfcl::FULL_DESCRIPTOR,
    // Unrunnable until MLCommons publishes its dataset; listed after the BFCL
    // legs it will eventually sit beside so the pane shows it exists.
    &mlperf_agentic::SUBSET_DESCRIPTOR,
    // Last: the only one that REPLACES the model the box is serving, so it is
    // the one an operator should have to travel furthest to start by accident.
    &serve_matrix::DESCRIPTOR,
];

pub fn all() -> &'static [&'static BenchmarkDescriptor] {
    ALL
}

/// Look one up by its stable id (run history, restart-last-run).
pub fn find(id: &str) -> Option<&'static BenchmarkDescriptor> {
    all().iter().copied().find(|d| d.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_filename_safe() {
        let mut seen = std::collections::BTreeSet::new();
        for d in all() {
            assert!(!d.id.is_empty(), "benchmark ids address history files");
            assert!(seen.insert(d.id), "duplicate benchmark id {}", d.id);
            assert!(
                d.id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{} is not filename-safe",
                d.id
            );
        }
    }

    #[test]
    fn find_round_trips_every_descriptor() {
        for d in all() {
            assert_eq!(find(d.id).unwrap().name, d.name);
        }
        assert!(find("nope").is_none());
    }

    #[test]
    fn every_benchmark_declares_defaults_that_validate() {
        for d in all() {
            let b = d.build();
            let specs = b.parameters();
            let values = crate::params::ParamValues::defaults(&specs);
            values
                .validate_against(&specs)
                .unwrap_or_else(|e| panic!("{}: {e}", d.id));
        }
    }
}
