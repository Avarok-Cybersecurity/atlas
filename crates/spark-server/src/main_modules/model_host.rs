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
    /// Held for the duration of a swap.
    ///
    /// Without it, N concurrent requests naming an absent model each start
    /// their own multi-minute load on the same GPU — the second one OOMs
    /// against the first, and the failure looks like a bad model rather than a
    /// stampede. Requests queue here and the winner does the work; see
    /// `swap_guard`.
    swapping: parking_lot::Mutex<()>,
    /// The server's Tokio runtime.
    ///
    /// A swap re-runs the load, and the load spawns Tokio tasks (the OOM
    /// watchdog, for one). The TUI drives swaps from a plain `std::thread`
    /// with no runtime in scope, so without this the first Library launch
    /// panics with "there is no reactor running" — after the old model has
    /// already been released. Held by the host because BOTH swap callers need
    /// it and only one of them happens to be inside the runtime already.
    runtime: parking_lot::Mutex<Option<tokio::runtime::Handle>>,
    /// Where the listener actually bound, once it has.
    ///
    /// A swap cannot move the socket — it is bound for the process lifetime —
    /// so a recipe naming a different port would otherwise serve on the old
    /// one with nothing saying so.
    bound: parking_lot::Mutex<Option<(String, u16)>>,
}

impl ModelHost {
    /// A host with a model already loaded — the normal `spark serve <MODEL>`.
    pub fn with_model(state: Arc<AppState>) -> Self {
        Self {
            current: parking_lot::RwLock::new(Some(state)),
            scheduler: parking_lot::Mutex::new(None),
            args: parking_lot::Mutex::new(None),
            swapping: parking_lot::Mutex::new(()),
            runtime: parking_lot::Mutex::new(tokio::runtime::Handle::try_current().ok()),
            bound: parking_lot::Mutex::new(None),
        }
    }

    /// A host with nothing loaded yet — `spark serve` with no arguments, where
    /// the dashboard is the front door.
    pub fn empty() -> Self {
        Self {
            current: parking_lot::RwLock::new(None),
            scheduler: parking_lot::Mutex::new(None),
            args: parking_lot::Mutex::new(None),
            swapping: parking_lot::Mutex::new(()),
            runtime: parking_lot::Mutex::new(tokio::runtime::Handle::try_current().ok()),
            bound: parking_lot::Mutex::new(None),
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

    /// Record where the listener bound.
    pub fn set_bound(&self, addr: String, port: u16) {
        *self.bound.lock() = Some((addr, port));
    }

    /// Where the listener bound, if it has.
    pub fn bound(&self) -> Option<(String, u16)> {
        self.bound.lock().clone()
    }

    /// The runtime a swap must run inside, if one was in scope at construction.
    pub fn runtime(&self) -> Option<tokio::runtime::Handle> {
        self.runtime.lock().clone()
    }

    /// Attach the runtime after the fact, for a host built outside it.
    pub fn set_runtime(&self, handle: tokio::runtime::Handle) {
        *self.runtime.lock() = Some(handle);
    }

    /// Remove the current model and hand it back, so the caller can prove it is
    /// the last owner before dropping it. `clear` discards that proof.
    pub fn take(&self) -> Option<Arc<AppState>> {
        self.current.write().take()
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

    /// Serialise swaps. The caller re-checks what is loaded AFTER acquiring:
    /// by then another request may have loaded exactly what it wanted, and
    /// doing it again would be a second outage for no reason.
    pub fn swap_guard(&self) -> parking_lot::MutexGuard<'_, ()> {
        // parking_lot: no poisoning, so a panic mid-swap cannot wedge every
        // later one and there is no `into_inner` branch to get wrong.
        self.swapping.lock()
    }

    /// The model id currently being served, if any.
    pub fn live_model(&self) -> Option<String> {
        self.current.read().as_ref().map(|s| s.model_name.clone())
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
