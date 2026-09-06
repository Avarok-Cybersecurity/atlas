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

/// Do not start a multi-minute load into a process that is on its way out.
///
/// The accept loop stops the moment shutdown is requested, so the model would
/// finish loading with nothing left to serve it — and the release it performs
/// first would take the OUTGOING model down with it, turning a clean drain
/// into an abrupt one.
///
/// Takes the answer rather than reading the global, so the rule is testable
/// without a test mutating a process-wide latch that has no reset and that
/// every other test calling `swap` would then trip over.
fn refuse_if_shutting_down(shutting_down: bool) -> Result<()> {
    anyhow::ensure!(
        !shutting_down,
        "shutdown is in progress — not starting a model load"
    );
    Ok(())
}

/// Refuse a model this binary has no kernels for, BEFORE anything is released.
///
/// The check itself already exists inside the load — but it runs at phase 3,
/// long after the outgoing model has been torn down. Discovering it there cost
/// a live server its model: the 35B was rejected for `model_type
/// 'qwen3_6_moe'`, and by then the 27B was gone and its restore failed on the
/// memory the dead attempt had taken. Reading a JSON file to find that out is
/// cheap; finding out with no model loaded is not.
fn preflight_kernel_target(args: &cli::ServeArgs) -> Result<()> {
    let model_dir = super::serve_phases::resolve_model_dir(args)?;
    let (config, _) = super::serve_phases::load_model_config(&model_dir)?;
    // Same resolution (references + optional pin) the load itself runs at
    // phase 3 — including the ambiguity hard-error, which must ALSO be
    // discovered here, before the outgoing model is torn down.
    let model_dir_str = model_dir.display().to_string();
    let model_refs: Vec<&str> = [
        args.model.as_deref(),
        args.model_name.as_deref(),
        Some(model_dir_str.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect();
    let resolved = atlas_kernels::ptx_for_config(
        &config.model_type,
        config.hidden_size,
        &model_refs,
        args.kernel_target.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{e} — the running model is untouched"))?;
    if resolved.is_none() {
        anyhow::bail!(
            "this build has no compiled kernels for model_type '{}' / hidden_size={} \
             (available: {:?}) — the running model is untouched",
            config.model_type,
            config.hidden_size,
            atlas_kernels::available_targets()
                .iter()
                .map(|t| &t.target.model)
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}

/// Copy the flags that describe the PROCESS, not the model, from the argv that
/// is running onto the argv that is about to.
///
/// A recipe describes a model: its checkpoint, quantization, context, batch
/// shape. It has no business deciding whether this deployment permits
/// request-triggered loading, or which socket the operator bound. Letting a
/// recipe's argv replace those wholesale is not a hypothetical: launching one
/// from the Library dropped `--auto-swap` from a server started with it, so
/// the very next request that should have swapped was quietly served by the
/// old model with nothing logged.
///
/// The socket is the starker case — it is bound for the process lifetime and
/// cannot move, so a recipe's port is unserveable by construction.
fn carry_process_flags(next: &mut cli::ServeArgs, previous: &cli::ServeArgs) {
    next.auto_swap = previous.auto_swap;
    next.no_auto_swap = previous.no_auto_swap;
    next.bind = previous.bind.clone();
    next.port = previous.port;
    // Request dumping is an operator's observability choice, and no recipe
    // sets it. Without this, a swap replaces argv with the recipe's and the
    // dump silently stops — the file stays where it was, simply never written
    // to again, which is the worst way for a diagnostic to fail.
    next.dump = previous.dump.clone();
}

/// How long in-flight requests get to release the outgoing model before the
/// swap gives up and puts it back. Matches the adapter hot-load's quiescence
/// window (`app_state.rs`), which solves the same "wait for readers" problem.
const DRAIN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

/// Block until `state` has no other owners, returning how many remain.
///
/// Split out from `release_state` so the waiting rule is testable without
/// standing up an `AppState`, which needs a loaded model.
fn wait_for_sole_owner<T>(state: &Arc<T>, grace: std::time::Duration) -> usize {
    let deadline = std::time::Instant::now() + grace;
    while Arc::strong_count(state) > 1 && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    Arc::strong_count(state) - 1
}

/// Take the outgoing model out of the host and drop it, carrying the
/// process-scoped stores forward.
///
/// The wait is the point. An `Arc<AppState>` that outlives this window keeps
/// `request_tx` open, so the scheduler never drains and the join below never
/// returns — the swap wedges the server with no model loaded and no way back.
/// That is not hypothetical: a middleware layer bound one for the router's
/// lifetime and did exactly that to a live server. Two kinds of holder reach
/// this point and the wait separates them: an in-flight request finishes on its
/// own, while a structural leak never does and is reported as a refusal, with
/// the model put back and still serving.
fn release_state(host: &Arc<ModelHost>, grace: std::time::Duration) -> Result<Carried> {
    // Nothing loaded — the modelless boot. Nothing to drain, nothing to lose,
    // which is why that path is the safest one to exercise first.
    let Some(state) = host.take() else {
        // The host has carried these since before the listener bound, so the
        // `None` arm is unreachable in a live process. Re-reading the
        // environment can now fail, and surfacing that beats papering over a
        // state that should not exist in the first place.
        return match host.process() {
            Some(carried) => Ok(carried),
            None => Carried::from_env().map_err(|e| anyhow::anyhow!("{e}")),
        };
    };
    let carried = Carried::from_previous(&state);
    let holders = wait_for_sole_owner(&state, grace);
    if holders > 0 {
        // Transactional: the host had it a moment ago and nothing has been
        // freed, so putting it back restores the exact state we started in.
        host.publish(state);
        anyhow::bail!(
            "cannot swap: {holders} reference(s) to the running model outlived \
             the {}s drain window, so it can never be released. The model is \
             still serving. This is a leaked `Arc<AppState>` — most likely one \
             bound into a router layer or a spawned task rather than resolved \
             per request.",
            grace.as_secs()
        );
    }
    drop(state);
    Ok(carried)
}

/// Mark the router/listening phases done and announce ready.
///
/// The listener is bound at boot and a swap never touches it, so `load_model`
/// cannot emit these itself — but a dashboard that never sees them keeps a
/// half-finished checklist and a LOADING pill over a server that is serving.
/// Both the swap and the restore need it, which is why it is a function and
/// not two copies.
fn signal_listener_phases(host: &Arc<ModelHost>) {
    if let Some((bind, port)) = host.bound() {
        spark_runtime::progress::phase(10, "router");
        spark_runtime::progress::phase(11, "listening");
        spark_runtime::progress::ready(port);
        // The same closing line a boot prints, at the same moment of truth:
        // the model was published a breath ago onto a listener that has been
        // serving since boot, so "live and ready" is a statement, not a
        // forecast. Emitted here, not in `load_model`, because the scheduler's
        // last init line ("Swap space: …") lands minutes before the model is
        // reachable and a user curling on it gets 503.
        tracing::info!(
            "{}",
            super::serve_router::ready_line(&bind, port, host.live_model().as_deref())
        );
    }
}

/// Replace the running model with the one `next` describes.
///
/// Blocking — it loads a model. Call it off the runtime.
pub(crate) fn swap(host: &Arc<ModelHost>, next: cli::ServeArgs) -> Result<SwapOutcome> {
    // Serialised HERE, not at one call site. Two swaps at once both call
    // `ModelHost::take`; the second gets `None`, mistakes it for a modelless
    // boot, rebuilds the carried stores from scratch (losing every stored
    // response and conversation) and loads a second model onto a GPU that is
    // already loading one. The TUI's Library launch reached `swap` directly
    // with no guard at all, so pressing `s` twice was enough.
    let _swapping = host.swap_guard();

    // Taken from the host, not a parameter: see `ModelHost::args`.
    let previous_args = host.args();

    let mut next = next;
    if let Some(previous) = previous_args.as_ref() {
        if next.port != previous.port || next.bind != previous.bind {
            tracing::warn!(
                "this recipe asks to bind {}:{}, but the listener is on {}:{} for the process \
                 lifetime — serving the new model there instead",
                next.bind,
                next.port,
                previous.bind,
                previous.port
            );
        }
        carry_process_flags(&mut next, previous);
    }

    // Whoever waited on the guard may be asking for what the winner just
    // loaded. Compared as a WHOLE argv rather than by model id, because
    // switching between two recipes for the SAME checkpoint is a real swap.
    //
    // AFTER carrying, not before. `previous_args` is the LIVE argv and already
    // holds the carried flags; `next` is the recipe's and does not. Comparing
    // them first meant the two could never be equal whenever any process flag
    // was set — so with --auto-swap on, which is exactly when requests queue
    // behind a swap, every queued request redid the load the winner had just
    // finished. That is the stampede this check exists to prevent.
    if previous_args.as_ref() == Some(&next) && host.current().is_some() {
        return Ok(SwapOutcome {
            previous: previous_args,
        });
    }

    // Refuse before anything is torn down. A bad flag combination, a missing
    // checkpoint or an impossible VRAM budget must cost nothing — the window
    // where the server has no model is opened only for a config that has
    // already passed everything cheap.
    cli::validate_serve_args(&next).map_err(|e| anyhow::anyhow!("{e}"))?;

    refuse_if_shutting_down(crate::tui::shutdown::requested())?;

    // The load spawns Tokio tasks. Entering here rather than at each call site
    // means a caller that is already inside the runtime and one that is not
    // (the TUI's plain thread) both work, and neither has to know which it is.
    let runtime = host.runtime();
    let _entered = runtime.as_ref().map(|h| h.enter());

    // Multi-rank is out of scope and must fail loudly rather than half-swap:
    // the EP worker takes the model by `Option::take` and only returns when the
    // head exits, so there is no "load a different model" command to send it.
    //
    // UNTESTED ON HARDWARE as of 2026-08-01, and deliberately so: everything
    // else in this file was exercised on a live GB10, but multi-node needs two
    // boxes and a real EP deployment, which was not available while this was
    // written. What IS verified is that the refusal fires — see
    // `a_multi_rank_deployment_is_refused`. What is NOT verified is the
    // behaviour of a head or worker that reaches this path in a genuine
    // world_size > 1 run.
    //
    // INTENT: test and debug this against a real two-node EP deployment
    // (dgx1 head + dgx2 worker over the RoCE fabric). Two things to establish
    // first, because the refusal above may be too blunt or not blunt enough:
    //
    //   1. Does a WORKER rank ever reach `swap` at all? It should not — the EP
    //      worker path returns `Startup::Worker` and never publishes into the
    //      host — but that has not been observed, only read.
    //   2. If a head swaps while workers are live, the workers keep the old
    //      model. Refusing is the safe answer today; the real fix is a
    //      "load this instead" command on the EP control channel, which does
    //      not exist yet. Until it does, this gate is load-bearing.
    anyhow::ensure!(
        next.world_size <= 1 && next.rank == 0,
        "hot-swap is single-node only (world_size={}, rank={})",
        next.world_size,
        next.rank
    );

    // Cheapest checks first: this one reads the checkpoint's config.json, so it
    // runs after the two that need nothing but the argv.
    preflight_kernel_target(&next)?;

    // 1 + 2. Stop admitting work, and release the state that owns request_tx.
    let carried = release_state(host, DRAIN_GRACE)?;

    // 3. Wait for the scheduler to finish draining.
    if let Some(handle) = host.take_scheduler() {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the scheduler thread panicked while draining"))?;
    }

    // 4 + 5. The model drops as the scheduler thread unwinds, which is where
    // `Model::teardown` frees its pools; then the new one loads.
    let next_args = next.clone();
    // From the HOST: a swap must republish its handles or the dashboard keeps
    // sampling the scheduler it just joined.
    let tui_handles_tx = host.tui_handles();
    let load_err = match load_model(next, tui_handles_tx.clone(), carried.clone()) {
        Ok(Some(prepared)) => {
            // 6.
            host.set_scheduler(prepared.scheduler);
            host.set_args(next_args);
            host.publish(prepared.state);
            signal_listener_phases(host);
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
            // The restored model is serving, so the dashboard must say so.
            // Without this the checklist stays frozen part-way and the pill
            // reads LOADING for a server that is answering requests — the same
            // defect the success path above was fixed for, on the branch
            // nobody looks at until a load has already failed.
            signal_listener_phases(host);
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

#[cfg(test)]
mod drain_tests {
    use super::wait_for_sole_owner;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    #[test]
    fn a_holder_that_lets_go_is_waited_for_rather_than_refused() {
        // An in-flight request is a legitimate holder. Refusing the swap the
        // instant one exists would make swapping impossible under any load.
        let state = Arc::new(0u32);
        let borrowed = state.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            drop(borrowed);
        });
        assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(5)), 0);
    }

    #[test]
    fn a_holder_that_never_lets_go_is_reported_not_waited_on_forever() {
        // The deadlock this exists to prevent: a leaked Arc keeps request_tx
        // open, so joining the scheduler never returns. Bounded wait, then say
        // how many are stuck.
        let state = Arc::new(0u32);
        let _leaked = state.clone();
        let began = Instant::now();
        assert_eq!(wait_for_sole_owner(&state, Duration::from_millis(200)), 1);
        assert!(began.elapsed() < Duration::from_secs(2), "bounded");
    }

    #[test]
    fn an_unshared_state_is_released_without_waiting() {
        let state = Arc::new(0u32);
        let began = Instant::now();
        assert_eq!(wait_for_sole_owner(&state, Duration::from_secs(30)), 0);
        assert!(
            began.elapsed() < Duration::from_millis(50),
            "no sleep at all"
        );
    }
}
