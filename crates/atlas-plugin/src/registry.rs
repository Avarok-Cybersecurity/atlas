// SPDX-License-Identifier: AGPL-3.0-only

//! The benchmark suite, in the order the Benchmarks pane lists it.

use crate::benchmark::BenchmarkDescriptor;
use crate::benchmarks::{agentic, bfcl, concurrency, serve_matrix, ttft};

/// Every benchmark, list order. Cheapest and most-run first.
///
/// STATIC, DELIBERATELY — compile-time data. A table of `&'static`
/// descriptors with no interior mutability and nothing derived from a model
/// or a run; it needs a stable address only so `all()` can hand out slices
/// of it. Registration is a compile-time decision, not a runtime one.
const ALL: &[&BenchmarkDescriptor] = &[
    &concurrency::DESCRIPTOR,
    &ttft::WARM_DESCRIPTOR,
    &ttft::COLD_DESCRIPTOR,
    &agentic::DESCRIPTOR,
    &bfcl::SUBSET_DESCRIPTOR,
    &bfcl::SUBSET_ECHOLP_DESCRIPTOR,
    &bfcl::FULL_DESCRIPTOR,
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
