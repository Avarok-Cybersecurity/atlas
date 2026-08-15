// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the baseline's `[benchmarks.param_overrides]` pins — the
//! request-side sibling of `serve_overrides`. Own file (500-LoC cap on
//! `bench_resolve_tests.rs`), same subject: pure resolution, no GPU.

use std::collections::BTreeMap;

use super::*;
use atlas_plugin::gate;

/// An entry carrying only param pins — the shape the concurrency gate's
/// BENCH.toml entry adds.
fn entry_with_pins(pins: &[(&str, &str)]) -> gate::ModelBaseline {
    gate::ModelBaseline {
        recipe: Some("qwen3.8/qwen3.8-27b-nvfp4-unsloth".to_string()),
        label: String::new(),
        note: String::new(),
        metrics: BTreeMap::new(),
        serve_overrides: BTreeMap::new(),
        param_overrides: pins
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
    }
}

/// The instrument-pin path, through the REAL concurrency-sweep wiring: the
/// variant's `[benchmarks.param_overrides]` replace the schema defaults (the
/// gate ladder is C=1/4/8/16 at isl 512 / osl 320, the schema sweeps to 32
/// at osl 128), an explicit `--param` outranks a pin, and unpinned keys keep
/// their schema defaults. All three precedence arms in one place.
#[test]
fn param_overrides_pin_the_instrument_and_yield_to_an_explicit_param() {
    let descriptor = atlas_plugin::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[
        ("concurrencies", "1,4,8,16"),
        ("isls", "512"),
        ("osl", "320"),
    ]);

    // Pinned: the ladder replaces the schema default.
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let applied =
        apply_param_overrides(descriptor, &specs, &mut values, &entry, &[]).expect("applies");
    assert_eq!(applied.len(), 3, "{applied:?}");
    assert_eq!(values.int_list("concurrencies").unwrap(), &[1, 4, 8, 16]);
    assert_eq!(values.int_list("isls").unwrap(), &[512]);
    assert_eq!(values.usize("osl").unwrap(), 320);
    // An unpinned key keeps its schema default.
    assert_eq!(values.usize("warmup").unwrap(), 1);

    // Explicit --param wins untouched.
    let explicit = vec![("osl".to_string(), "512".to_string())];
    let mut values =
        atlas_plugin::ParamValues::from_overrides(&specs, vec![("osl", "512")]).unwrap();
    let applied =
        apply_param_overrides(descriptor, &specs, &mut values, &entry, &explicit).expect("applies");
    assert!(
        applied.iter().all(|(k, _)| k != "osl"),
        "stated intent is never overridden: {applied:?}"
    );
    assert_eq!(values.usize("osl").unwrap(), 512);
    assert_eq!(
        values.int_list("concurrencies").unwrap(),
        &[1, 4, 8, 16],
        "the other pins still apply"
    );
}

/// A pin naming no schema parameter is DRIFT between the BENCH.toml and the
/// driver, and a silently-dropped pin runs the wrong instrument — so it is a
/// loud error naming the key.
#[test]
fn a_param_override_for_an_unknown_key_is_a_loud_error() {
    let descriptor = atlas_plugin::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[("no_such_knob", "7")]);
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_param_overrides(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("must refuse");
    let msg = format!("{err:#}");
    assert!(msg.contains("no_such_knob"), "{msg}");
    assert!(msg.contains("drifted"), "{msg}");
}

/// A pin naming a threshold-coupled parameter is refused: that value is
/// derived from the paired metric's bound, and a second source would fight
/// it silently.
#[test]
fn a_param_override_cannot_name_a_threshold_coupled_param() {
    let descriptor = atlas_plugin::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[("min_c16", "94.0")]);
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_param_overrides(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("must refuse");
    let msg = format!("{err:#}");
    assert!(msg.contains("min_c16"), "{msg}");
    assert!(msg.contains("threshold"), "{msg}");
}

/// An out-of-domain pin fails through the spec's own parser, exactly like a
/// typed --param — the kind's bounds cannot be bypassed by this path.
#[test]
fn a_param_override_goes_through_the_kinds_own_parser() {
    let descriptor = atlas_plugin::registry::find("concurrency-sweep").expect("registered");
    let specs = descriptor.build().parameters();
    let entry = entry_with_pins(&[("osl", "0")]);
    let mut values = atlas_plugin::ParamValues::from_overrides(&specs, vec![]).unwrap();
    let err = apply_param_overrides(descriptor, &specs, &mut values, &entry, &[])
        .expect_err("must refuse");
    assert!(format!("{err:#}").contains("osl=0"), "{err:#}");
}
