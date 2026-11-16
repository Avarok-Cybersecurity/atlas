// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the seam's two pieces of process state: the one registered
//! selector, and the caller's per-step sequence identities.
//!
//! The selector slot is a `OnceLock`: registration happens once at startup
//! (`port_apply`), before the model is built, and a second registration is a
//! startup error naming both selectors. Reading it is one atomic load, which is
//! the whole cost of the seam on a serve with no approximate row on.
//!
//! Step identities are thread-local: the scheduler thread that calls the
//! model's forward publishes the ids of the sequences in that forward, in row
//! order, and the hooks the forward reaches read them. Nothing is published
//! (and nothing allocated) while no selector is registered.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use anyhow::Result;

use super::KvSelector;

/// The registered-selector slot. `ACTIVE` is the serve's; tests build their own.
pub(crate) struct Registry {
    slot: OnceLock<Arc<dyn KvSelector>>,
}

impl Registry {
    pub(crate) const fn new() -> Self {
        Self {
            slot: OnceLock::new(),
        }
    }

    pub(crate) fn register(&self, sel: Arc<dyn KvSelector>) -> Result<()> {
        let name = sel.name();
        let reserved = sel.reserved_bytes();
        match self.slot.set(sel) {
            Ok(()) => {
                tracing::info!(
                    "sparse attention: selector `{name}` registered ({:.1} MiB device reservation); \
                     approximate attention is ON for the full-attention layers",
                    reserved as f64 / f64::from(1u32 << 20),
                );
                Ok(())
            }
            Err(_) => {
                let current = self.slot.get().map_or("?", |s| s.name());
                anyhow::bail!(
                    "sparse attention: selector `{name}` cannot register because `{current}` is \
                     already active; at most one approximate attention selector runs per serve, \
                     so turn one of the two rows off"
                )
            }
        }
    }

    pub(crate) fn get(&self) -> Option<&dyn KvSelector> {
        self.slot.get().map(|s| s.as_ref())
    }
}

pub(super) static ACTIVE: Registry = Registry::new();

/// What the caller published for the forward in progress.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct StepInfo {
    /// One id per logical sequence, in row order, padding excluded.
    pub(crate) seq_ids: Vec<u64>,
    /// The forward is a prompt's last prefill chunk.
    pub(crate) last_chunk: bool,
}

thread_local! {
    static STEP: RefCell<Option<StepInfo>> = const { RefCell::new(None) };
}

/// Restores the previous publication when dropped (publications nest). Not
/// `Send`: it belongs to the thread that published.
#[must_use = "the publication ends when the guard is dropped"]
pub struct StepGuard {
    prev: Option<Option<StepInfo>>,
    _thread: PhantomData<*const ()>,
}

impl Drop for StepGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            STEP.with(|s| *s.borrow_mut() = prev);
        }
    }
}

/// Publish the sequence ids (row order, padding excluded) and the last-chunk
/// flag of the forward the caller is about to run, for the guard's lifetime.
/// A no-op while no selector is registered.
pub fn publish_step(seq_ids: &[u64], last_chunk: bool) -> StepGuard {
    if super::active().is_none() {
        return StepGuard {
            prev: None,
            _thread: PhantomData,
        };
    }
    push_step(StepInfo {
        seq_ids: seq_ids.to_vec(),
        last_chunk,
    })
}

/// The unconditional half of [`publish_step`] (tests reach it directly).
pub(crate) fn push_step(info: StepInfo) -> StepGuard {
    let prev = STEP.with(|s| s.borrow_mut().replace(info));
    StepGuard {
        prev: Some(prev),
        _thread: PhantomData,
    }
}

/// A copy of the current publication, if any.
pub(crate) fn current_step() -> Option<StepInfo> {
    STEP.with(|s| s.borrow().clone())
}
