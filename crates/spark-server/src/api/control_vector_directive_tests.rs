// SPDX-License-Identifier: AGPL-3.0-only

//! Wire-format tests for the three-state `control_vector` field.
//!
//! These pin an API contract, so they are deliberately written against JSON
//! text rather than against the enum: the thing that must not change is what a
//! given request body means, not how it is represented internally.

use super::*;

#[derive(Deserialize, Debug)]
struct Req {
    #[serde(default, deserialize_with = "deserialize_cvec_directive")]
    control_vector: CvecDirective,
}

fn parse(json: &str) -> CvecDirective {
    serde_json::from_str::<Req>(json).unwrap().control_vector
}

#[test]
fn an_omitted_field_defers_to_the_server() {
    // The whole reason this type exists. Today "defer" resolves to no
    // steering, so this is behaviour-preserving; once a default is configured
    // it is what lets the deployment supply one.
    assert_eq!(parse("{}"), CvecDirective::ServerDefault);
}

#[test]
fn null_is_an_explicit_no() {
    // MUST differ from omitted. A caller sending null is declining the
    // server's default, not failing to mention it.
    assert_eq!(parse(r#"{"control_vector": null}"#), CvecDirective::Off);
}

#[test]
fn false_is_an_explicit_no() {
    assert_eq!(parse(r#"{"control_vector": false}"#), CvecDirective::Off);
}

#[test]
fn an_empty_string_is_an_explicit_no() {
    // A client building JSON from a form field should not have to omit the
    // key to mean "none" — and now that omitting has a different meaning,
    // this reading matters more than it used to.
    assert_eq!(parse(r#"{"control_vector": ""}"#), CvecDirective::Off);
    assert_eq!(parse(r#"{"control_vector": "  "}"#), CvecDirective::Off);
}

#[test]
fn a_name_selects_it() {
    assert_eq!(
        parse(r#"{"control_vector": "refusal"}"#),
        CvecDirective::Named("refusal".into())
    );
}

#[test]
fn surrounding_whitespace_is_trimmed() {
    assert_eq!(
        parse(r#"{"control_vector": " refusal "}"#),
        CvecDirective::Named("refusal".into())
    );
}

#[test]
fn true_is_refused_because_it_does_not_say_which() {
    // A serve can hold many vectors. Picking one would be inventing intent,
    // and picking "the only one" when there happens to be one would make a
    // request's meaning depend on the server's inventory.
    let e = serde_json::from_str::<Req>(r#"{"control_vector": true}"#)
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not say WHICH"), "{e}");
}

#[test]
fn a_number_is_refused() {
    assert!(serde_json::from_str::<Req>(r#"{"control_vector": 3}"#).is_err());
}
