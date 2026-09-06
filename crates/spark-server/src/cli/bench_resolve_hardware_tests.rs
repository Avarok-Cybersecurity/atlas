// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the `--hardware` axis of the gate's resolution: which box-class
//! ids resolve, and which of the two refusals an unresolvable one earns.
//!
//! Split from `bench_resolve_tests.rs` for the 500-LoC cap, and because these
//! read against a different oracle: the box-class REGISTRY
//! (`atlas_plugin::hardware::ids`), not the assembled baseline.

use super::tests::baseline;
use super::*;

/// Oracle: `atlas_plugin::hardware::ids::KNOWN_HARDWARE_IDS` — the registry of
/// box classes Atlas recognises — crossed with the Hopper campaign's premise
/// that no H100 record exists yet.
///
/// The two refusals are DIFFERENT actions for the operator. "Unknown" means the
/// id is wrong and no run will ever fix it; "no record yet" means the id is
/// right and the fix is to run the gate on that box and commit the record. A
/// single message for both sends the Hopper campaign hunting for a typo that is
/// not there.
#[test]
fn a_registered_box_class_with_no_record_says_to_go_measure_it() {
    let b = baseline(&[("gb10", "m", Some("r"))]);
    for hw in ["h100", "h200", "b200"] {
        let err = resolve(&b, "bfcl-subset", Some(hw), None).expect_err("refused");
        let msg = format!("{err:#}");
        assert!(msg.contains(hw), "names what was asked for: {msg}");
        assert!(msg.contains("gb10"), "lists what it has: {msg}");
        assert!(
            msg.contains("no record yet"),
            "names the state, not a typo: {msg}"
        );
        assert!(
            !msg.contains("not a box class Atlas knows"),
            "a registered id must not read as a typo: {msg}"
        );
    }
}

/// Oracle: the same registry, read from the resolver's side. A registered id
/// that HAS a baseline resolves exactly as `gb10` does — registration adds a
/// slot, it does not add a second gate in front of measured entries.
#[test]
fn a_registered_box_class_with_a_record_resolves_normally() {
    let b = baseline(&[
        ("gb10", "a", Some("recipe-a")),
        ("h100", "b", Some("recipe-b")),
    ]);
    let r = resolve(&b, "bfcl-subset", Some("h100"), None).expect("resolved");
    assert_eq!(r.model, "b");
    assert_eq!(r.recipe_id, "recipe-b");
}
