// SPDX-License-Identifier: AGPL-3.0-only

//! Resolving a request's `control_vector` NAME to the registry identity the
//! scheduler and the forward pass key on.
//!
//! Deliberately simpler than [`super::lora_control::resolve_request_adapter_slot`]
//! in one way that matters: there is no "defer to the installed active"
//! fallthrough, and the `model` field never selects a vector. Absent means
//! **no steering**, full stop.
//!
//! That asymmetry is on purpose. The LoRA pool has no selector meaning "base"
//! once an adapter is resident, which makes an adapter-vs-base A/B impossible
//! on one serve. Steering has to be switchable off per request or it cannot be
//! compared against, and comparing against it is most of the point.
//!
//! The name→id table is mirrored into `AppState` at startup, the same way
//! `adapter_names` is, so a handler resolves without reaching for the model.

use axum::http::StatusCode;
use axum::response::Response;

use crate::api::compact::openai_error_response;

/// Resolve `control_vector` against the registered `(name, id)` table.
/// `Ok(0)` = no steering.
///
/// `Err` carries a ready-to-return 400 naming what IS registered, because the
/// alternative — quietly serving unsteered when the operator asked for a
/// vector — is exactly the failure this feature must not have.
// `Err` is an axum `Response`, which is large but is what every handler-edge
// resolver in this crate returns (see `chat/prepare.rs`, `chat_phases.rs`).
#[allow(clippy::result_large_err)]
pub fn resolve_request_cvec_id(
    control_vectors: &[(String, u64)],
    control_vector: Option<&str>,
) -> Result<u64, Response> {
    let Some(name) = control_vector else {
        return Ok(0);
    };
    let name = name.trim();
    // An explicit empty string is "none", not a lookup failure: a client
    // building JSON from a form field should not have to omit the key.
    if name.is_empty() {
        return Ok(0);
    }
    if let Some((_, id)) = control_vectors.iter().find(|(n, _)| n == name) {
        return Ok(*id);
    }
    let known = if control_vectors.is_empty() {
        "none are loaded (start the server with --control-vector NAME=PATH)".to_string()
    } else {
        control_vectors
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(openai_error_response(
        StatusCode::BAD_REQUEST,
        format!("unknown control_vector '{name}'; registered: {known}"),
    ))
}

#[cfg(test)]
#[path = "control_vector_control_tests.rs"]
mod control_vector_control_tests;
