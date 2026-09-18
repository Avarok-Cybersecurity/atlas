// SPDX-License-Identifier: AGPL-3.0-only

//! `POST /v1/control_vector/capture` — drive a derivation run.
//!
//! Deriving a control vector is a two-pass corpus job: prefill the positive
//! set, dump; reset; prefill the negative set, dump. The offline step
//! (`scripts/derive_control_vector.py`) subtracts the two means and writes the
//! GGUF.
//!
//! Both actions ride the LoRA rotation channel, which the scheduler drains
//! ONLY at quiescence. That is the point, not a convenience: a reset or dump
//! that landed mid-forward would leave a partial pass in the running sum, and
//! nothing downstream could see that it had — the resulting vector would just
//! be quietly wrong.

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use tokio::sync::oneshot;

use crate::api::compact::openai_error_response;
use crate::main_modules::model_host::CurrentModel;
use crate::scheduler::{LoraAck, LoraCommand};

#[derive(Deserialize)]
pub struct CaptureRequest {
    /// `"reset"` to zero the accumulator, `"dump"` to write it out.
    pub action: String,
    /// Destination for `dump`. Required for that action, ignored otherwise.
    #[serde(default)]
    pub path: Option<String>,
}

pub async fn control_vector_capture(
    CurrentModel(state): CurrentModel,
    body: Result<Json<CaptureRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    let cmd = match req.action.as_str() {
        "reset" => LoraCommand::CaptureReset,
        "dump" => match req.path.as_deref() {
            Some(p) if !p.is_empty() => LoraCommand::CaptureDump(std::path::PathBuf::from(p)),
            _ => {
                return openai_error_response(
                    StatusCode::BAD_REQUEST,
                    "action \"dump\" requires a non-empty \"path\"".to_string(),
                );
            }
        },
        other => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("unknown action {other:?}; expected \"reset\" or \"dump\""),
            );
        }
    };

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "the scheduler command channel is not armed on this serve".to_string(),
        );
    };

    let (ack_tx, ack_rx) = oneshot::channel();
    if tx.send((cmd, ack_tx)).await.is_err() {
        return openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler command channel closed".to_string(),
        );
    }
    // Generous: the channel drains only at quiescence, so a busy serve can
    // legitimately take a while to reach it.
    match tokio::time::timeout(std::time::Duration::from_secs(120), ack_rx).await {
        Err(_) => openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "timed out waiting for scheduler quiescence; retry when idle".to_string(),
        ),
        Ok(Err(_)) => openai_error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "scheduler dropped the capture ack (shutting down?)".to_string(),
        ),
        Ok(Ok(Err(reason))) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Ok(Ok(Ok(ack))) => {
            let tokens = match ack {
                LoraAck::Captured(n) => Some(n),
                _ => None,
            };
            Json(serde_json::json!({
                "action": req.action,
                "path": req.path,
                "tokens": tokens,
            }))
            .into_response()
        }
    }
}
