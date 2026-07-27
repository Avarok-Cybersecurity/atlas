// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/models` handlers: list + retrieve served model ids, plus the
//! embeddings auto-probe stub. Split out of `completions.rs` when the M2
//! LoRA advertise path (resident adapters ARE served models) grew it past
//! the 500-LoC cap.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};

use crate::AppState;
use crate::openai::{ModelInfo, ModelListResponse};

use super::compact::openai_error_response;

/// GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    // #22 hardening: bound the advertised set so the pre-sized allocation can
    // never be driven large. adapter_names is startup-bounded by --max-loras
    // today, but capping keeps the allocation independent of that (and clears the
    // CodeQL "allocation size from a user-provided value" alert). A real pool far
    // below the cap is unaffected.
    const MAX_ADVERTISED_MODELS: usize = 1024;
    let advertised = state.adapter_names.len().min(MAX_ADVERTISED_MODELS);
    let mut data = Vec::with_capacity(advertised.saturating_add(1));
    // The resident adapters ARE served models — advertise them first (slot
    // order; data[0] is the default route) so clients can pick a fine-tune.
    for adapter in state.adapter_names.iter().take(MAX_ADVERTISED_MODELS) {
        data.push(ModelInfo {
            id: adapter.clone(),
            object: "model".to_string(),
            created: crate::ids::unix_timestamp(),
            owned_by: "atlas-spark".to_string(),
        });
    }
    // Cold STAGEABLE names (peer- and disk-backed) are selectable via `model`
    // and fault in on first use — advertise them too (before the base model),
    // bounded by the same cap. They are NOT in adapter_names (the resident
    // routing index), so this only affects what /v1/models lists.
    for name in state
        .lora_stageable
        .keys()
        .chain(state.lora_disk_stageable.keys())
        .take(MAX_ADVERTISED_MODELS.saturating_sub(data.len()))
    {
        data.push(ModelInfo {
            id: name.clone(),
            object: "model".to_string(),
            created: crate::ids::unix_timestamp(),
            owned_by: "atlas-spark".to_string(),
        });
    }
    data.push(ModelInfo {
        id: state.model_name.clone(),
        object: "model".to_string(),
        created: crate::ids::unix_timestamp(),
        owned_by: "atlas-spark".to_string(),
    });
    Json(ModelListResponse {
        object: "list".to_string(),
        data,
    })
}

/// GET /v1/models/{model_id} — retrieve a single model (OpenAI SDK `client.models.retrieve()`).
pub async fn get_model(
    State(state): State<Arc<AppState>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    // Any resident adapter is a routable model id (M2): `models.retrieve(name)`
    // must succeed for every adapter advertised by /v1/models, not just slot 0.
    let known = model_id == state.model_name
        || state.adapter_names.iter().any(|n| n == &model_id)
        || state.is_stageable_name(&model_id);
    if known {
        Json(serde_json::json!({
            "id": model_id,
            "object": "model",
            "created": crate::ids::unix_timestamp(),
            "owned_by": "atlas-spark",
        }))
        .into_response()
    } else {
        openai_error_response(
            StatusCode::NOT_FOUND,
            format!("The model '{model_id}' does not exist"),
        )
    }
}

/// POST /v1/embeddings — stub for clients that probe this endpoint during auto-detection.
pub async fn embeddings_stub() -> Response {
    openai_error_response(
        StatusCode::NOT_IMPLEMENTED,
        "Embeddings are not supported by this model. Atlas serves generative (chat/completion) models only.".into(),
    )
}
