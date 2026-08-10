// SPDX-License-Identifier: AGPL-3.0-only

//! Readiness must reflect "can this server serve", not "is a model loaded"
//! (issue #429).
//!
//! The bug: a sticky CUDA error destroyed the context, but the model was still
//! *published*, so `/health` answered `200 ready` while every request it
//! admitted died in the driver. The two claims had silently come apart, and
//! readiness means the second one.
//!
//! Each test below was observed RED against the stated mutation.

use crate::api::misc_handlers::readiness;
use axum::http::StatusCode;

const REASON: &str = "kernel launch failed (716), and a no-op synchronize also failed";

/// POSITIVE: the ordinary healthy case still reports ready, with the model
/// named. A fault check that broke this would take down every healthy server.
///
/// PROVEN BY: making `readiness` return the fault branch unconditionally turns
/// this red on the status assertion.
#[test]
fn a_loaded_model_with_no_fault_is_ready() {
    let (code, body) = readiness(Some("Qwen/Qwen3.6-27B-NVFP4"), None);
    assert_eq!(code, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model"], "Qwen/Qwen3.6-27B-NVFP4");
}

/// NEGATIVE: still loading is still 503 — the pre-existing contract, which the
/// fault branch must not disturb.
///
/// PROVEN BY: collapsing the `match model` arms to always return OK turns this
/// red.
#[test]
fn no_model_and_no_fault_is_still_loading() {
    let (code, body) = readiness(None, None);
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["status"], "loading");
}

/// **This is issue #429.** A model IS published and the context IS dead. The
/// fault must win: the pre-fix code reported `200 ready` in exactly this
/// state, and an orchestrator kept routing traffic into a server that could
/// only 500.
///
/// PROVEN BY: deleting the `if let Some(reason) = fault` block — i.e.
/// restoring the pre-fix behaviour — turns this red with `200 ready`, which is
/// verbatim the bug.
#[test]
fn a_fault_outranks_a_published_model() {
    let (code, body) = readiness(Some("Qwen/Qwen3.6-27B-NVFP4"), Some(REASON));
    assert_eq!(
        code,
        StatusCode::SERVICE_UNAVAILABLE,
        "a published model on a dead context is NOT ready"
    );
    assert_eq!(body["status"], "faulted");
}

/// The reason must reach the operator. A bare `faulted` sends them to the
/// logs of a process that is on its way out; the endpoint is often the only
/// surface they can still reach.
///
/// PROVEN BY: dropping `reason` from the fault JSON turns this red.
#[test]
fn the_fault_body_carries_the_reason() {
    let (_, body) = readiness(None, Some(REASON));
    assert_eq!(body["reason"], REASON);
}

/// A fault with no model loaded is reported as a FAULT, not as "loading".
/// The distinction matters operationally: "loading" invites waiting, and this
/// server will never become ready.
///
/// PROVEN BY: reordering `readiness` to check `model` before `fault` turns
/// this red with `loading`.
#[test]
fn a_fault_before_any_model_loaded_is_not_reported_as_loading() {
    let (code, body) = readiness(None, Some(REASON));
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body["status"], "faulted",
        "a permanently dead server must not look like a starting one"
    );
}
