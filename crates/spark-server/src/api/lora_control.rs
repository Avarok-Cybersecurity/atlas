// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA adapter rotation control plane: `POST /v1/lora/active`.
//!
//! Selects the globally-active resident adapter at runtime (eager-on-rotate).
//! The request is forwarded to the scheduler over the rotation channel and
//! applied at a QUIESCENT point (no in-flight decode), so the re-point never
//! races a live delta read or a CUDA-graph replay. Batch-1 honest: rotation
//! changes the adapter applied to ALL subsequent requests (per-request adapter
//! routing is a future increment).

use std::sync::Arc;

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::AppState;
use crate::api::compact::openai_error_response;

/// Resolve a request's optional `adapter` NAME to a pool slot for M2 routing
/// (`-1` = defer to the installed active adapter), attempting the Task #27
/// on-miss RDMA promotion for STAGEABLE adapters. Shared by the chat and
/// legacy-completions handlers. `Err` carries the ready-to-return response:
/// 400 unknown adapter / 503 pool full (retryable) / 502 peer error.
pub async fn resolve_request_adapter_slot(
    state: &AppState,
    adapter: Option<&str>,
    model: &str,
) -> Result<i32, Response> {
    // Selection precedence: the explicit `adapter` field wins; otherwise the
    // OpenAI `model` field routes when it names a RESIDENT adapter (so a plain
    // `{"model":"demo"}` selects the `demo` pool slot — the standard OpenAI way
    // to pick a served LoRA). A `model` that is the base model (or any name not
    // a resident adapter) falls through to `None` → installed active (`-1`),
    // never an unknown-adapter 400.
    // A cold STAGEABLE name reached via the `model` field must fall through
    // resolve_adapter_slot (which returns None for a non-resident name) into
    // ensure_adapter_hot_opt, so select it here too. This also fixes
    // `model=<peer-stageable>` routing (previously only reachable via `adapter`).
    let selector = match adapter {
        Some(a) => Some(a),
        None if state.adapter_names.iter().any(|n| n == model)
            || state.is_stageable_name(model) =>
        {
            Some(model)
        }
        None => None,
    };
    if let Some(slot) =
        crate::main_modules::app_state::resolve_adapter_slot(&state.adapter_names, selector)
    {
        return Ok(slot);
    }
    // #27: a non-resident name may be STAGEABLE — try an on-miss RDMA
    // promotion into a cache slot. Only 400 when it isn't stageable
    // (byte-identical to today) or the promote fails.
    let name = selector.unwrap_or("");
    match state.ensure_adapter_hot_opt(name).await {
        Ok(Some(slot)) => Ok(slot),
        Ok(None) => Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "unknown adapter '{}'; resident adapters: [{}]",
                name,
                state.adapter_names.join(", ")
            ),
        )),
        Err(crate::main_modules::promotion::PromoteReject::PoolFull(m)) => {
            Err(openai_error_response(StatusCode::SERVICE_UNAVAILABLE, m))
        }
        Err(crate::main_modules::promotion::PromoteReject::Peer(m)) => {
            Err(openai_error_response(StatusCode::BAD_GATEWAY, m))
        }
    }
}

#[derive(Deserialize)]
pub struct SetActiveLoraRequest {
    /// The resident adapter NAME to activate (as advertised by `/v1/models`).
    pub adapter: String,
}

#[derive(Serialize)]
struct SetActiveLoraResponse {
    object: &'static str,
    active: String,
}

/// POST /v1/lora/active  `{"adapter": "NAME"}`
pub async fn set_active_lora(
    State(state): State<Arc<AppState>>,
    body: Result<Json<SetActiveLoraRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no LoRA adapter is loaded (start with --lora-adapter NAME=PATH)".to_string(),
        );
    };

    if !state.adapter_names.iter().any(|n| n == &req.adapter) {
        return openai_error_response(
            StatusCode::NOT_FOUND,
            format!(
                "adapter '{}' is not resident (resident: [{}])",
                req.adapter,
                state.adapter_names.join(", ")
            ),
        );
    }

    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    if tx
        .send((
            crate::scheduler::LoraCommand::Rotate(req.adapter.clone()),
            ack_tx,
        ))
        .await
        .is_err()
    {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler rotation channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => {
            // Optimistic status mirror (the scheduler's model owns the truth).
            if let Ok(mut a) = state.active_adapter.lock() {
                *a = Some(req.adapter.clone());
            }
            Json(SetActiveLoraResponse {
                object: "lora.active",
                active: req.adapter,
            })
            .into_response()
        }
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the rotation ack (shutting down?)".to_string(),
        ),
    }
}

#[derive(Deserialize)]
pub struct LoadLoraRequest {
    /// Name to stamp on the loaded adapter (its `/v1/models` id after the swap).
    pub name: String,
    /// Filesystem path to the PEFT adapter dir (contains adapter_config.json +
    /// adapter_model.safetensors).
    pub path: String,
    /// Pool slot to load into (default 0 — the single slot under --max-loras 1).
    #[serde(default)]
    pub slot: usize,
}

#[derive(Serialize)]
struct LoadLoraResponse {
    object: &'static str,
    loaded: String,
    slot: usize,
}

/// POST /v1/lora/load  `{"name": "vega", "path": "/dir", "slot": 0}`
///
/// Dynamically loads a DIFFERENT adapter from disk into a pool slot at runtime
/// — the pool-size-1 demonstration of per-request weight change: with
/// `--max-loras 1` only one adapter is resident, and this swaps the single
/// slot's contents on demand (needs `ATLAS_LORA_ROTATE=1` so decode is eager).
pub async fn load_lora_into_slot(
    State(state): State<Arc<AppState>>,
    body: Result<Json<LoadLoraRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => {
            return openai_error_response(StatusCode::BAD_REQUEST, format!("invalid body: {e}"));
        }
    };

    // #22 hardening: bound the request inputs before doing any work (the name is
    // stamped onto a pool slot; the path is opened; the slot indexes the pool).
    if req.name.is_empty() || req.name.len() > 256 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "adapter name must be 1..=256 chars".to_string(),
        );
    }
    if req.path.len() > 4096 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "adapter path too long (max 4096 chars)".to_string(),
        );
    }
    if req.slot > 4096 {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("slot {} out of range (max 4096)", req.slot),
        );
    }

    let Some(ref tx) = state.rotation_tx else {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            "no LoRA adapter pool is loaded (start with --lora-adapter NAME=PATH)".to_string(),
        );
    };

    let dir = std::path::PathBuf::from(&req.path);
    if !dir.join("adapter_config.json").exists() {
        return openai_error_response(
            StatusCode::BAD_REQUEST,
            format!("no adapter_config.json under path '{}'", req.path),
        );
    }

    let (ack_tx, ack_rx) = tokio::sync::oneshot::channel();
    let cmd = crate::scheduler::LoraCommand::LoadIntoSlot {
        name: req.name.clone(),
        dir,
        slot: req.slot,
    };
    if tx.send((cmd, ack_tx)).await.is_err() {
        return openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler rotation channel closed".to_string(),
        );
    }
    match ack_rx.await {
        Ok(Ok(_)) => {
            // The swapped slot becomes the served adapter; mirror it.
            if let Ok(mut a) = state.active_adapter.lock() {
                *a = Some(req.name.clone());
            }
            Json(LoadLoraResponse {
                object: "lora.loaded",
                loaded: req.name,
                slot: req.slot,
            })
            .into_response()
        }
        Ok(Err(reason)) => openai_error_response(StatusCode::BAD_REQUEST, reason),
        Err(_) => openai_error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "scheduler dropped the load ack (shutting down?)".to_string(),
        ),
    }
}
