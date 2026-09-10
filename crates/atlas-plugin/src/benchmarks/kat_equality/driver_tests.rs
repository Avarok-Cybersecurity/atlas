// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::benchmark::Benchmark;
use crate::benchmarks::kat_equality::driver::KatEquality;
use crate::hardware::Sensitivity;

#[test]
fn the_gate_is_registered_and_findable_by_its_id() {
    let d = crate::registry::find("kat-equality-gate").expect("registered");
    assert_eq!(d.id, DESCRIPTOR.id);
    assert_eq!(d.name, "KAT Equality Gate");
}

/// A busy box can make a run SLOWER; it cannot make one request's output
/// depend on another's. Classifying this as Speed would let a hot box block a
/// correctness finding — and would drag it into the single-signer rule that
/// exists for tok/s comparisons.
#[test]
fn byte_equality_is_a_correctness_finding_not_a_speed_one() {
    assert_eq!(DESCRIPTOR.sensitivity, Sensitivity::Correctness);
}

/// Two orders is the minimum that proves anything: the parameter's floor must
/// not permit a run that compares one order with nothing.
#[test]
fn the_orders_parameter_cannot_be_set_below_two() {
    let specs = KatEquality::default().parameters();
    let orders = specs.iter().find(|s| s.key == "orders").expect("declared");
    match orders.kind {
        crate::params::ParamKind::Int { min, .. } => assert_eq!(
            min, 2,
            "a single-order run cannot prove equality and must not be expressible"
        ),
        ref other => panic!("orders should be an Int, got {other:?}"),
    }
}
