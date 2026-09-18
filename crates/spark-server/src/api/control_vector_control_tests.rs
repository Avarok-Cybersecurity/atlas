// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for control-vector request resolution.

use super::*;

fn table() -> Vec<(String, u64)> {
    vec![
        ("refusal".to_string(), 0xAAAA),
        ("style".to_string(), 0xBBBB),
    ]
}

/// No server default configured — the overwhelmingly common deployment, and
/// the one whose behaviour must not have changed when the default was added.
// Same `Err` type as the function under test, so it carries the same allow.
#[allow(clippy::result_large_err)]
fn no_default(
    control_vectors: &[(String, u64)],
    directive: CvecDirective,
) -> Result<u64, axum::response::Response> {
    resolve_request_cvec_id(control_vectors, &directive, None)
}

#[test]
fn absent_means_no_steering_when_no_default_is_configured() {
    assert_eq!(
        no_default(&table(), CvecDirective::ServerDefault).unwrap(),
        0
    );
}

#[test]
fn explicit_off_means_no_steering() {
    // `null`, `false` and `""` all arrive here as Off. A client templating
    // JSON from a form field sends ""; that is "off", not a typo.
    assert_eq!(no_default(&table(), CvecDirective::Off).unwrap(), 0);
}

#[test]
fn a_registered_name_resolves_to_its_id() {
    assert_eq!(
        no_default(&table(), CvecDirective::Named("refusal".into())).unwrap(),
        0xAAAA
    );
    assert_eq!(
        no_default(&table(), CvecDirective::Named("style".into())).unwrap(),
        0xBBBB
    );
}

#[test]
fn an_unknown_name_is_rejected_not_ignored() {
    // Serving unsteered when the caller asked for a vector is the failure
    // this feature must not have, so a typo must be loud.
    let e = no_default(&table(), CvecDirective::Named("refusl".into()));
    assert!(e.is_err(), "unknown name was accepted");
}

#[test]
fn an_empty_registry_still_rejects_a_named_vector() {
    let e = no_default(&[], CvecDirective::Named("refusal".into()));
    assert!(e.is_err(), "named a vector on a serve with none loaded");
}

#[test]
fn an_empty_registry_accepts_no_selection() {
    assert_eq!(no_default(&[], CvecDirective::ServerDefault).unwrap(), 0);
}

#[test]
fn an_omitted_field_takes_the_server_default() {
    assert_eq!(
        resolve_request_cvec_id(&table(), &CvecDirective::ServerDefault, Some("refusal")).unwrap(),
        0xAAAA
    );
}

#[test]
fn explicit_off_overrides_the_server_default() {
    // THE reason the three states exist. A caller must be able to opt out of a
    // deployment-wide default, or an A/B against it is impossible — and
    // `Option<String>` could not express this, because null and omitted were
    // the same value.
    assert_eq!(
        resolve_request_cvec_id(&table(), &CvecDirective::Off, Some("refusal")).unwrap(),
        0
    );
}

#[test]
fn a_named_vector_overrides_the_server_default() {
    assert_eq!(
        resolve_request_cvec_id(
            &table(),
            &CvecDirective::Named("style".into()),
            Some("refusal")
        )
        .unwrap(),
        0xBBBB
    );
}

#[test]
fn an_unresolvable_server_default_does_not_blame_the_request() {
    // An operator typo surfacing on somebody else's traffic. Reporting
    // "unknown control_vector" would send the caller hunting through a payload
    // for a field they never sent, so this is a 500 that names the flag.
    //
    // Boot-time validation should make this unreachable; it is defended here
    // because "unreachable" and "unhandled" are different claims.
    let e = resolve_request_cvec_id(&table(), &CvecDirective::ServerDefault, Some("ghost"))
        .expect_err("an unregistered default resolved");
    assert_eq!(e.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
}
