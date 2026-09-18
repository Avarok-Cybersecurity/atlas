// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the control-vector identity and the variant composition.
//!
//! The registry itself needs a GPU to populate (a `ControlVector` owns device
//! memory), so these cover the parts that decide CACHE CORRECTNESS: the id
//! hash and `compose_variant_id`. Those are what partition the prefix cache,
//! and a mistake there is silent.

use super::*;

#[test]
fn a_serve_with_no_control_vector_keys_exactly_as_before() {
    // THE pin: adding this feature must not invalidate a single existing
    // prefix entry, for base or for any adapter.
    assert_eq!(compose_variant_id(0, 0), 0, "base sentinel moved");
    for adapter in [1u64, 42, 0xdead_beef, u64::MAX] {
        assert_eq!(
            compose_variant_id(adapter, 0),
            adapter,
            "adapter {adapter} key moved when no cvec is selected"
        );
    }
}

#[test]
fn a_request_with_a_vector_never_looks_like_base() {
    for name in ["refusal", "style", "a", "verbose-off"] {
        let c = cvec_id_hash(name);
        assert_ne!(c, 0);
        assert_ne!(compose_variant_id(0, c), 0, "{name} aliased base");
    }
}

#[test]
fn different_vectors_give_different_variants() {
    let a = cvec_id_hash("refusal");
    let b = cvec_id_hash("style");
    assert_ne!(a, b);
    assert_ne!(compose_variant_id(0, a), compose_variant_id(0, b));
    // …and the same holds on top of an adapter.
    assert_ne!(compose_variant_id(77, a), compose_variant_id(77, b));
}

#[test]
fn the_same_vector_on_different_adapters_differs() {
    let c = cvec_id_hash("refusal");
    assert_ne!(compose_variant_id(1, c), compose_variant_id(2, c));
}

#[test]
fn selecting_a_vector_changes_the_key_for_the_same_adapter() {
    // The whole point: request A (vector on) and request B (vector off) with
    // an identical token prefix must NOT share cached blocks.
    let c = cvec_id_hash("refusal");
    assert_ne!(compose_variant_id(5, c), compose_variant_id(5, 0));
}

#[test]
fn the_id_hash_is_stable_and_name_derived() {
    // Stability matters: the id outlives a process and is not a slot index,
    // so reordering --control-vector flags must not move it.
    assert_eq!(cvec_id_hash("refusal"), cvec_id_hash("refusal"));
    assert_ne!(cvec_id_hash("refusal"), cvec_id_hash("Refusal"));
}

#[test]
fn id_zero_resolves_to_no_vector() {
    let reg = ControlVectorRegistry::default();
    assert!(reg.is_empty());
    assert!(reg.by_id(0).is_none());
    assert!(reg.by_id(cvec_id_hash("refusal")).is_none());
}
