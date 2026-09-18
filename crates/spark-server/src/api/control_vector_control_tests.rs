// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for control-vector request resolution.

use super::*;

fn table() -> Vec<(String, u64)> {
    vec![
        ("refusal".to_string(), 0xAAAA),
        ("style".to_string(), 0xBBBB),
    ]
}

#[test]
fn absent_means_no_steering() {
    assert_eq!(resolve_request_cvec_id(&table(), None).unwrap(), 0);
}

#[test]
fn empty_string_means_no_steering() {
    // A client templating JSON from a form field sends "" rather than omitting
    // the key; that is "off", not a typo.
    assert_eq!(resolve_request_cvec_id(&table(), Some("")).unwrap(), 0);
    assert_eq!(resolve_request_cvec_id(&table(), Some("  ")).unwrap(), 0);
}

#[test]
fn a_registered_name_resolves_to_its_id() {
    assert_eq!(
        resolve_request_cvec_id(&table(), Some("refusal")).unwrap(),
        0xAAAA
    );
    assert_eq!(
        resolve_request_cvec_id(&table(), Some("style")).unwrap(),
        0xBBBB
    );
    // Surrounding whitespace is forgiven; the name itself is exact.
    assert_eq!(
        resolve_request_cvec_id(&table(), Some(" refusal ")).unwrap(),
        0xAAAA
    );
}

#[test]
fn an_unknown_name_is_rejected_not_ignored() {
    // Serving unsteered when the operator asked for a vector is the failure
    // this feature must not have, so a typo must be loud.
    let e = resolve_request_cvec_id(&table(), Some("refusl"));
    assert!(e.is_err(), "unknown name was accepted");
}

#[test]
fn an_empty_registry_still_rejects_a_named_vector() {
    let e = resolve_request_cvec_id(&[], Some("refusal"));
    assert!(e.is_err(), "named a vector on a serve with none loaded");
}

#[test]
fn an_empty_registry_accepts_no_selection() {
    // The overwhelmingly common case: no vectors loaded, no field sent.
    assert_eq!(resolve_request_cvec_id(&[], None).unwrap(), 0);
}
