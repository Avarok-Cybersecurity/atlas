// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_no_model_hint_names_the_way_out() {
    let h = hint_for("model_not_loaded").expect("the observed case must have a hint");
    // The complaint was not that the error was wrong, but that it left the
    // reader with nowhere to go. Naming the section is the whole point.
    assert!(h.contains("Library"), "{h}");
    assert!(h.contains("4"), "and how to get there: {h}");
    // A hint is the action, not a second telling of the problem. It is glued
    // onto a message that already stated the condition.
    assert!(
        !h.to_lowercase().contains("no model is loaded"),
        "hint must not restate the message it is appended to: {h}"
    );
}

#[test]
fn unknown_types_get_no_hint_rather_than_a_guess() {
    assert_eq!(hint_for("invalid_request_error"), None);
    assert_eq!(hint_for(""), None);
    // A hint invented for an error we have not thought about is worse than
    // none: it sends the reader somewhere that will not help.
    assert_eq!(hint_for("some_future_type"), None);
}

#[test]
fn message_with_hint_leaves_unhinted_messages_exactly_alone() {
    // Byte-identical, not merely similar — this runs on every error response.
    let m = "context length exceeded";
    assert_eq!(message_with_hint(m, "invalid_request_error"), m);
}

#[test]
fn message_with_hint_appends_for_known_types() {
    let out = message_with_hint("no model is loaded", "model_not_loaded");
    assert!(out.starts_with("no model is loaded"), "{out}");
    assert!(out.contains("Library"), "{out}");
}

#[test]
fn the_hint_survives_a_round_trip_through_a_real_response_body() {
    // The load-bearing path: server appends the hint to `message`, client reads
    // `message` back out. If either half regresses the hint silently vanishes,
    // which is exactly the failure this change exists to prevent — a hint that
    // is emitted but never displayed looks identical to no hint at all.
    let body = serde_json::json!({
        "error": {
            "message": message_with_hint("no model is loaded", "model_not_loaded"),
            "type": "model_not_loaded",
            "hint": hint_for("model_not_loaded"),
        }
    })
    .to_string();

    let seen = atlas_plugin::http::message_from_body(&body).expect("a well-formed body parses");
    assert!(
        seen.contains("Library"),
        "hint must survive the round trip: {seen}"
    );
}
