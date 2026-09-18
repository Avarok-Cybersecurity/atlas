// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the control-vector identity and the variant composition.
//!
//! The registry itself needs a GPU to populate (a `ControlVector` owns device
//! memory), so these cover the parts that decide CACHE CORRECTNESS: the id
//! hash and `compose_variant_id`. Those are what partition the prefix cache,
//! and a mistake there is silent.

use super::*;

/// Stand-in for `ControlVector::config_identity()`. Built by hand because the
/// real one needs a loaded vector and therefore a GPU; the format is mirrored
/// so these tests exercise the same shape of input the loader produces.
fn cfg(sha: &str, mode: &str, scale: f32, lo: usize, hi: usize) -> String {
    format!(
        "sha256={sha} mode={mode} scale=0x{:08x} layers={lo}..={hi}",
        scale.to_bits()
    )
}

/// The configuration used wherever a test only cares about the name.
fn cfg_a() -> String {
    cfg("aaaa", "Project", 1.0, 4, 44)
}

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
        let c = cvec_id_hash(name, &cfg_a());
        assert_ne!(c, 0);
        assert_ne!(compose_variant_id(0, c), 0, "{name} aliased base");
    }
}

#[test]
fn different_vectors_give_different_variants() {
    let a = cvec_id_hash("refusal", &cfg_a());
    let b = cvec_id_hash("style", &cfg_a());
    assert_ne!(a, b);
    assert_ne!(compose_variant_id(0, a), compose_variant_id(0, b));
    // …and the same holds on top of an adapter.
    assert_ne!(compose_variant_id(77, a), compose_variant_id(77, b));
}

#[test]
fn the_same_vector_on_different_adapters_differs() {
    let c = cvec_id_hash("refusal", &cfg_a());
    assert_ne!(compose_variant_id(1, c), compose_variant_id(2, c));
}

#[test]
fn selecting_a_vector_changes_the_key_for_the_same_adapter() {
    // The whole point: request A (vector on) and request B (vector off) with
    // an identical token prefix must NOT share cached blocks.
    let c = cvec_id_hash("refusal", &cfg_a());
    assert_ne!(compose_variant_id(5, c), compose_variant_id(5, 0));
}

#[test]
fn the_id_hash_is_stable_for_one_configuration() {
    // Stability matters: the id is not a slot index, so reordering
    // --control-vector flags must not move it.
    assert_eq!(
        cvec_id_hash("refusal", &cfg_a()),
        cvec_id_hash("refusal", &cfg_a())
    );
    assert_ne!(
        cvec_id_hash("refusal", &cfg_a()),
        cvec_id_hash("Refusal", &cfg_a())
    );
}

/// The regression this whole identity scheme exists for.
///
/// Under EP/TP every rank builds its own registry from its own command line.
/// If the id came from the NAME alone, each of the four divergences below
/// would produce matching ids on both ranks, the worker's "do I have this id?"
/// check would pass, and the two halves of the model would steer differently
/// with nothing anywhere reporting it.
#[test]
fn the_same_name_with_a_different_configuration_is_a_different_id() {
    let base = cvec_id_hash("refusal", &cfg_a());

    let other_file = cvec_id_hash("refusal", &cfg("bbbb", "Project", 1.0, 4, 44));
    assert_ne!(base, other_file, "a different FILE must not share an id");

    let other_mode = cvec_id_hash("refusal", &cfg("aaaa", "Add", 1.0, 4, 44));
    assert_ne!(base, other_mode, "a different MODE must not share an id");

    let other_scale = cvec_id_hash("refusal", &cfg("aaaa", "Project", 0.5, 4, 44));
    assert_ne!(base, other_scale, "a different SCALE must not share an id");

    let other_layers = cvec_id_hash("refusal", &cfg("aaaa", "Project", 1.0, 1, 47));
    assert_ne!(
        base, other_layers,
        "a different LAYER RANGE must not share an id"
    );
}

/// Scale is compared as an exact configuration value, not an approximate one.
#[test]
fn a_barely_different_scale_is_a_different_id() {
    let a = cvec_id_hash("v", &cfg("aaaa", "Add", 1.0, 4, 44));
    let b = cvec_id_hash("v", &cfg("aaaa", "Add", 1.000_000_1, 4, 44));
    assert_ne!(
        a, b,
        "scales that differ at all are different configurations"
    );
}

/// Name and config are separated by a byte that cannot appear in either, so a
/// boundary shift cannot produce the same hash from different inputs.
#[test]
fn the_name_config_boundary_is_unambiguous() {
    assert_ne!(cvec_id_hash("ab", "c"), cvec_id_hash("a", "bc"));
}

#[test]
fn id_zero_resolves_to_no_vector() {
    let reg = ControlVectorRegistry::default();
    assert!(reg.is_empty());
    assert!(reg.by_id(0).is_none());
    assert!(reg.by_id(cvec_id_hash("refusal", &cfg_a())).is_none());
}
