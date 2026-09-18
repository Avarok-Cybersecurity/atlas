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
use crate::api::control_vector_directive::CvecDirective;

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
    directive: &CvecDirective,
    server_default: Option<&str>,
) -> Result<u64, Response> {
    // Resolve the three-state directive to "a name, or nothing".
    //
    // `ServerDefault` is the only state that consults the deployment. With no
    // default configured it means no steering, which is exactly today's
    // behaviour for a request that omits the field — so introducing the
    // default changes nothing for anyone who has not configured one.
    let name = match directive {
        CvecDirective::Off => return Ok(0),
        CvecDirective::Named(n) => n.as_str(),
        CvecDirective::ServerDefault => match server_default {
            Some(d) => d,
            None => return Ok(0),
        },
    };

    if let Some((_, id)) = control_vectors.iter().find(|(n, _)| n == name) {
        return Ok(*id);
    }

    // A server default that does not resolve is an OPERATOR error surfacing on
    // a caller's request, and saying "unknown control_vector" would send them
    // hunting through their own payload for a field they never sent.
    if matches!(directive, CvecDirective::ServerDefault) {
        return Err(openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "server default control vector '{name}' is not registered — \
                 --default-control-vector names a vector that no --control-vector \
                 declares. This is a server misconfiguration, not a problem with \
                 this request."
            ),
        ));
    }

    let known = if control_vectors.is_empty() {
        "none are loaded (start the server with --control-vector NAME=PATH, and \
         check that --disable-control-vectors is not set)"
            .to_string()
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
