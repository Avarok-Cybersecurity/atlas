// SPDX-License-Identifier: AGPL-3.0-only

//! `/v1/models` handlers: list + retrieve served model ids, plus the
//! embeddings auto-probe stub. Split out of `completions.rs` when the M2
//! LoRA advertise path (resident adapters ARE served models) grew it past
//! the 500-LoC cap.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::Serialize;

use crate::main_modules::AppState;
use crate::main_modules::swap_metrics::{
    self, SwapStats,
    atlas::{self, EntryKind, ListXAtlas, Residency, XAtlas},
};
use crate::openai::{ModelInfo, ModelListResponse};

use super::compact::openai_error_response;

/// The `/v1/models` body.
///
/// Untagged, so `Plain` serializes exactly as `ModelListResponse` always has:
/// with the `model_swap_timing` row off the response bytes are unchanged.
#[derive(Serialize)]
#[serde(untagged)]
pub enum ModelListBody {
    Plain(ModelListResponse),
    Atlas(AtlasModelList),
}

/// The list with the port sprint's `x_atlas` objects (row `model_swap_timing`).
#[derive(Serialize)]
pub struct AtlasModelList {
    object: String,
    data: Vec<AtlasModelEntry>,
    x_atlas: ListXAtlas,
}

/// One listed model plus its `x_atlas` residency and swap-cost object.
#[derive(Serialize)]
pub struct AtlasModelEntry {
    #[serde(flatten)]
    info: ModelInfo,
    x_atlas: XAtlas,
}

/// What kind of entry `id` is, from what the loaded state serves.
fn kind_of(model_name: &str, adapter_names: &[String], id: &str) -> EntryKind {
    if id == model_name {
        EntryKind::Base
    } else if adapter_names.iter().any(|n| n == id) {
        EntryKind::Adapter
    } else {
        EntryKind::Staged
    }
}

/// The entries `/v1/models` lists for a loaded model, each with its kind.
fn advertised_entries(state: &AppState) -> Vec<(ModelInfo, EntryKind)> {
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
        data.push((
            ModelInfo::advertise(adapter.clone(), state.max_seq_len),
            EntryKind::Adapter,
        ));
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
        data.push((
            ModelInfo::advertise(name.clone(), state.max_seq_len),
            EntryKind::Staged,
        ));
    }
    data.push((
        ModelInfo::advertise(state.model_name.clone(), state.max_seq_len),
        EntryKind::Base,
    ));
    data
}

/// Assemble the list body. With `on == false` (the `model_swap_timing` row off)
/// this is the plain list, byte for byte what the handler returned before.
fn list_body(
    entries: Vec<(ModelInfo, EntryKind)>,
    residency: Option<&Residency>,
    stats: &SwapStats,
    on: bool,
) -> ModelListBody {
    if !on {
        return ModelListBody::Plain(ModelListResponse {
            object: "list".to_string(),
            data: entries.into_iter().map(|(info, _)| info).collect(),
        });
    }
    let data = entries
        .into_iter()
        .map(|(info, kind)| {
            let x_atlas = atlas::entry_x_atlas(stats, &info.id, kind, residency);
            AtlasModelEntry { info, x_atlas }
        })
        .collect();
    ModelListBody::Atlas(AtlasModelList {
        object: "list".to_string(),
        data,
        x_atlas: atlas::list_x_atlas(stats, residency),
    })
}

/// Add `x_atlas` to a retrieved model body when the row is on.
fn with_x_atlas(
    mut body: serde_json::Value,
    id: &str,
    kind: EntryKind,
    residency: Option<&Residency>,
    stats: &SwapStats,
    on: bool,
) -> serde_json::Value {
    let x_atlas = on
        .then(|| atlas::entry_x_atlas(stats, id, kind, residency))
        .and_then(|x| serde_json::to_value(x).ok());
    if let Some(x) = x_atlas {
        body["x_atlas"] = x;
    }
    body
}

/// GET /v1/models
pub async fn list_models(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
) -> Json<ModelListBody> {
    // An empty list is the honest answer before a model is chosen: better than
    // 503, because the endpoint itself is working and the answer is "none".
    let entries = host
        .current()
        .map(|state| advertised_entries(&state))
        .unwrap_or_default();
    Json(list_body(
        entries,
        host.residency().as_ref(),
        swap_metrics::stats(),
        swap_metrics::enabled(),
    ))
}

/// GET /v1/models/{model_id} — retrieve a single model (OpenAI SDK `client.models.retrieve()`).
pub async fn get_model(
    State(host): State<Arc<crate::main_modules::model_host::ModelHost>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Response {
    // An empty list is the honest answer before a model is chosen — better than
    // 503, because the endpoint itself is working and the answer is "none".
    let Some(state) = host.current() else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": crate::error_hints::message_with_hint(
                        "no model is loaded",
                        "model_not_loaded",
                    ),
                    "type": "model_not_loaded",
                    "hint": crate::error_hints::hint_for("model_not_loaded"),
                }
            })),
        )
            .into_response();
    };
    // Any resident adapter is a routable model id (M2): `models.retrieve(name)`
    // must succeed for every adapter advertised by /v1/models, not just slot 0.
    let known = model_id == state.model_name
        || state.adapter_names.iter().any(|n| n == &model_id)
        || state.is_stageable_name(&model_id);
    if known {
        let body = serde_json::json!({
            "id": model_id,
            "object": "model",
            "created": crate::ids::unix_timestamp(),
            "owned_by": "avarok-spark",
            "max_model_len": state.max_seq_len,
        });
        let kind = kind_of(&state.model_name, &state.adapter_names, &model_id);
        Json(with_x_atlas(
            body,
            &model_id,
            kind,
            host.residency().as_ref(),
            swap_metrics::stats(),
            swap_metrics::enabled(),
        ))
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

#[cfg(test)]
#[path = "models_tests.rs"]
mod tests;
