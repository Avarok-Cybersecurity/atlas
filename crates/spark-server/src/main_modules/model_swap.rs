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
//! **The cost, stated plainly:** the swap is committed, not transactional. By
//! step 4 the old model is gone, so a failure in step 5 leaves the server with
//! nothing loaded rather than with what it had. The host reports that honestly
//! (503, `/health` "loading") and the previous argv is returned so a caller can
//! offer to restore it. Validating the new config BEFORE step 1 is what keeps
//! that window small — a bad recipe never reaches the drain.

use std::sync::Arc;

use anyhow::{Context, Result};

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
    previous_args: Option<cli::ServeArgs>,
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
    let prepared = load_model(next, tui_handles_tx, carried)
        .context("loading the new model")?
        .context("hot-swap reached an EP-worker path on rank 0, which cannot happen")?;

    // 6.
    host.set_scheduler(prepared.scheduler);
    host.publish(prepared.state);
    Ok(SwapOutcome {
        previous: previous_args,
    })
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod tests;
