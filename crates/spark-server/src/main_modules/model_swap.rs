// SPDX-License-Identifier: AGPL-3.0-only

//! Replacing the running model without restarting the process.
//!
//! The order below is the whole design, and every step exists because skipping
//! it breaks something specific:
//!
//! 1. **Clear the host.** New requests get 503 `model_not_loaded` immediately.
//!    Requests already running keep the `Arc` they took and finish against the
//!    model they started with — see `ModelHost::current`.
//! 2. **Drop the outgoing `AppState`.** That closes `request_tx`, which is the
//!    only way the scheduler learns to stop.
//! 3. **Join the scheduler.** It returns only once fully drained
//!    (`scheduler/mod.rs`: breaks when new/active/prefilling are all empty AND
//!    the channel is closed). This join is what proves nothing is still
//!    touching the weights — without it, teardown races live kernels.
//! 4. **Tear the model down.** Frees ~20 GB. Only safe once (3) has returned:
//!    on GB10 a free interleaved with other allocation traffic corrupts
//!    neighbouring allocations, and a drained, joined scheduler is the
//!    quiescent moment that makes it safe.
//! 5. **Load the new model**, carrying the process-scoped stores forward.
//! 6. **Publish.** Requests resume against the new model.
//!
//! **The cost, and what is done about it.** The swap is committed: by step 4
//! the old model is gone, so a failure in step 5 cannot be undone by simply
//! not proceeding. Three things narrow that window, in order of how much they
//! buy:
//!
//! 1. **Validate before step 1.** A bad flag combination, an absent
//!    checkpoint or a multi-rank deployment never reaches the drain, so the
//!    overwhelmingly common failure costs nothing at all.
//! 2. **Restore on failure.** If the new model fails to load, the previous
//!    argv is reloaded automatically. The memory it needs was just freed by
//!    its own teardown, so the restore is loading a model that demonstrably
//!    fit moments ago — the case with the best odds of succeeding.
//! 3. **Report honestly when both fail.** No model is loaded, `/health` says
//!    so, requests get 503, and the error names BOTH failures — the one that
//!    started it and the one that prevented recovery. A restore that fails
//!    silently is worse than no restore, because the operator then debugs the
//!    wrong model.

use std::sync::Arc;

use anyhow::Result;

use super::model_host::ModelHost;
use super::serve_load::{Carried, load_model};
use crate::cli;

/// What a swap needs to know to undo itself.
#[derive(Debug)]
pub(crate) struct SwapOutcome {
    /// The argv of the model that was replaced, for a restore offer.
    pub previous: Option<cli::ServeArgs>,
}

/// Replace the running model with the one `next` describes.
///
/// Blocking — it loads a model. Call it off the runtime.
pub(crate) fn swap(
    host: &Arc<ModelHost>,
    next: cli::ServeArgs,
    tui_handles_tx: Option<std::sync::mpsc::Sender<crate::tui::RunHandles>>,
) -> Result<SwapOutcome> {
    // Refuse before anything is torn down. A bad flag combination, a missing
    // checkpoint or an impossible VRAM budget must cost nothing — the window
    // where the server has no model is opened only for a config that has
    // already passed everything cheap.
    cli::validate_serve_args(&next).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Multi-rank is out of scope and must fail loudly rather than half-swap:
    // the EP worker takes the model by `Option::take` and only returns when the
    // head exits, so there is no "load a different model" command to send it.
    anyhow::ensure!(
        next.world_size <= 1 && next.rank == 0,
        "hot-swap is single-node only (world_size={}, rank={})",
        next.world_size,
        next.rank
    );

    // Taken from the host, not a parameter: see `ModelHost::args`.
    let previous_args = host.args();

    // 1 + 2. Stop admitting work, and release the state that owns request_tx.
    // The stores are taken FIRST: they must outlive the model being dropped.
    let carried = match host.current() {
        Some(state) => {
            let carried = Carried::from_previous(&state);
            host.clear();
            drop(state);
            carried
        }
        // Nothing loaded — the modelless boot. Nothing to drain, nothing to
        // lose, which is why this path is the safest one to exercise first.
        None => Carried::from_env(),
    };

    // 3. Wait for the scheduler to finish draining.
    if let Some(handle) = host.take_scheduler() {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the scheduler thread panicked while draining"))?;
    }

    // 4 + 5. The model drops as the scheduler thread unwinds, which is where
    // `Model::teardown` frees its pools; then the new one loads.
    let next_args = next.clone();
    let load_err = match load_model(next, tui_handles_tx.clone(), carried.clone()) {
        Ok(Some(prepared)) => {
            // 6.
            host.set_scheduler(prepared.scheduler);
            host.set_args(next_args);
            host.publish(prepared.state);
            return Ok(SwapOutcome {
                previous: previous_args,
            });
        }
        Ok(None) => anyhow::anyhow!("hot-swap reached an EP-worker path on rank 0"),
        Err(e) => e,
    };

    // The new model did not load and the old one is already gone. Put the old
    // one back: its memory was freed by its own teardown moments ago, so this
    // is the load with the best chance of succeeding.
    let Some(previous) = previous_args else {
        return Err(load_err
            .context("the new model failed to load and there was no previous model to restore"));
    };
    tracing::warn!("load failed, restoring the previous model: {load_err:#}");
    match load_model(previous.clone(), tui_handles_tx, carried) {
        Ok(Some(prepared)) => {
            host.set_scheduler(prepared.scheduler);
            host.set_args(previous);
            host.publish(prepared.state);
            // Deliberately an Err: the requested swap did NOT happen, and
            // returning Ok would tell the caller it did.
            Err(load_err.context("the new model failed to load; the previous one was restored"))
        }
        // Both failed. Name both — an operator told only about the restore
        // failure debugs the wrong model.
        Ok(None) => Err(load_err.context("restore reached an EP-worker path")),
        Err(restore_err) => Err(load_err.context(format!(
            "the new model failed to load AND the previous one could not be \
             restored ({restore_err:#}) — no model is loaded"
        ))),
    }
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod tests;
