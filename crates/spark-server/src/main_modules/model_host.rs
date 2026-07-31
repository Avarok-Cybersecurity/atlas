// SPDX-License-Identifier: AGPL-3.0-only

//! The cell every request reads the current model through.
//!
//! A hot-swap replaces the model, and with it everything derived from it:
//! the tokenizer, the served name, the sampling presets and behaviour, the
//! think/tool token ids, and the channel to the scheduler. Those are ~14
//! required fields on [`AppState`] with no neutral value, so "AppState with no
//! model" is not expressible — the swap has to replace the whole thing.
//!
//! Hence one indirection rather than fourteen `Option`s: the router holds a
//! `ModelHost`, each handler takes the current `Arc<AppState>` once, and every
//! field access inside the handler is unchanged. A request that started before
//! a swap keeps the `Arc` it took and finishes against the model it began with,
//! which is what makes draining meaningful — a mid-flight request never sees
//! half of one model and half of another.
//!
//! `RwLock` rather than a lock-free cell: reads are per-request but
//! uncontended (a swap is rare and brief), and `parking_lot`'s read path is a
//! few nanoseconds. Adding an `arc-swap` dependency to save that would be
//! paying a supply-chain cost for a benchmark artefact.

use std::sync::Arc;

use super::app_state::AppState;

pub struct ModelHost {
    current: parking_lot::RwLock<Option<Arc<AppState>>>,
    /// The scheduler thread of the model currently loaded.
    ///
    /// Lives here rather than in `Prepared` because a swap CONSUMES it: the
    /// join is what proves the old model is drained and safe to tear down, so
    /// whoever performs the swap must be able to take it, and that is not the
    /// one call site that first built the server.
    scheduler: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// The argv the live model was loaded from.
    ///
    /// Kept here, not passed to `swap`, because a caller that has to supply it
    /// is a caller that can forget: the first one did, which silently disabled
    /// restore-on-failure. The host always knows what it is running.
    args: parking_lot::Mutex<Option<crate::cli::ServeArgs>>,
}

impl ModelHost {
    /// A host with a model already loaded — the normal `spark serve <MODEL>`.
    pub fn with_model(state: Arc<AppState>) -> Self {
        Self {
            current: parking_lot::RwLock::new(Some(state)),
            scheduler: parking_lot::Mutex::new(None),
            args: parking_lot::Mutex::new(None),
        }
    }

    /// A host with nothing loaded yet — `spark serve` with no arguments, where
    /// the dashboard is the front door.
    pub fn empty() -> Self {
        Self {
            current: parking_lot::RwLock::new(None),
            scheduler: parking_lot::Mutex::new(None),
            args: parking_lot::Mutex::new(None),
        }
    }

    /// The model serving right now, or `None` while none is loaded.
    ///
    /// Returns an owned `Arc` on purpose: the caller must not hold the lock
    /// across an await, and a request that took this keeps serving against the
    /// model it started with even if a swap lands mid-flight.
    pub fn current(&self) -> Option<Arc<AppState>> {
        self.current.read().clone()
    }

    /// Install a newly loaded model. The previous `Arc` stays alive for as long
    /// as any in-flight request still holds it.
    pub fn publish(&self, state: Arc<AppState>) {
        *self.current.write() = Some(state);
    }

    /// Drop the current model, so requests are refused while a swap runs.
    pub fn clear(&self) {
        *self.current.write() = None;
    }

    /// Hand over the scheduler of the model just loaded.
    pub fn set_scheduler(&self, handle: std::thread::JoinHandle<()>) {
        *self.scheduler.lock() = Some(handle);
    }

    /// Take the current scheduler, for a swap to join.
    pub fn take_scheduler(&self) -> Option<std::thread::JoinHandle<()>> {
        self.scheduler.lock().take()
    }

    /// Record what the live model was loaded from, for a restore.
    pub fn set_args(&self, args: crate::cli::ServeArgs) {
        *self.args.lock() = Some(args);
    }

    pub fn args(&self) -> Option<crate::cli::ServeArgs> {
        self.args.lock().clone()
    }

    pub fn is_loaded(&self) -> bool {
        self.current.read().is_some()
    }
}

#[cfg(test)]
#[path = "model_host_tests.rs"]
mod tests;

/// The model currently serving, as an extractor.
///
/// Handlers ask for `CurrentModel(state)` instead of `State(state)` and are
/// otherwise unchanged — every field access inside them still reads
/// `state.tokenizer`, `state.behavior` and so on.
///
/// Doing this as an extractor rather than a `let … else` in each handler is
/// what keeps the change to one line per handler: the 27 call sites have a
/// dozen different return types (`Response`, `Result<Json<_>, _>`, `Sse<_>`,
/// …), so a hand-written 503 path would have to be written differently in each
/// one. A rejection is the framework's own answer to "this request cannot be
/// served", and it is identical everywhere.
pub struct CurrentModel(pub Arc<AppState>);

impl<S> axum::extract::FromRequestParts<S> for CurrentModel
where
    Arc<ModelHost>: axum::extract::FromRef<S>,
    S: Send + Sync,
{
    type Rejection = axum::response::Response;

    async fn from_request_parts(
        _parts: &mut axum::http::request::Parts,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        use axum::extract::FromRef;
        use axum::response::IntoResponse;
        let host = Arc::<ModelHost>::from_ref(state);
        match host.current() {
            Some(state) => Ok(Self(state)),
            // 503, not 500: nothing is broken — a model is being loaded or has
            // not been chosen yet, and the request is worth retrying. The body
            // matches what /health already reports in the same situation.
            None => Err((
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
                    "error": {
                        "message": "no model is loaded",
                        "type": "model_not_loaded",
                    }
                })),
            )
                .into_response()),
        }
    }
}
