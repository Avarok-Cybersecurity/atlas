// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: where the tier's host bytes come from. [`HostArena`] is the small interface
//! the tier allocates page runs through; [`PoolArena`] serves it from lane L25's
//! [`PinnedPool`] (page-locked, tier-owned registrations, so tier DMA never takes the implicit
//! sync of `copy_h2d_async`). Tests use an in-memory arena with the same contract.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::pinned_pool::{PageRun, PinnedPool};

/// One host allocation: `bytes` writable bytes at address `ptr`, valid until freed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostRun {
    pub ptr: usize,
    pub bytes: usize,
    pub id: u64,
}

/// A source of host page runs. `alloc` returns None when the arena cannot place `bytes` now.
/// Runs must stay at a fixed address until `free`, and `free` takes back exactly what `alloc`
/// returned.
pub trait HostArena: Send + Sync {
    fn alloc(&self, bytes: usize) -> Option<HostRun>;
    fn free(&self, run: HostRun);
}

/// [`HostArena`] over the pinned pool.
pub struct PoolArena {
    pool: Arc<PinnedPool>,
    live: Mutex<HashMap<u64, PageRun>>,
}

impl PoolArena {
    pub fn new(pool: Arc<PinnedPool>) -> Self {
        Self {
            pool,
            live: Mutex::new(HashMap::new()),
        }
    }

    pub fn pool(&self) -> &Arc<PinnedPool> {
        &self.pool
    }

    /// Runs handed out and not yet freed.
    pub fn live_runs(&self) -> usize {
        self.live.lock().len()
    }
}

impl HostArena for PoolArena {
    fn alloc(&self, bytes: usize) -> Option<HostRun> {
        let run = self.pool.alloc_run(bytes)?;
        let host = HostRun {
            ptr: run.ptr,
            bytes: run.bytes,
            id: run.id,
        };
        if host.bytes < bytes || host.ptr == 0 {
            // A pool that rounds down or hands out a null run is a contract break; never use it.
            tracing::warn!(
                "kv_host_tier: pinned pool returned {} bytes at {:#x} for a {bytes}-byte request",
                host.bytes,
                host.ptr
            );
            self.pool.free(run);
            return None;
        }
        self.live.lock().insert(host.id, run);
        Some(host)
    }

    fn free(&self, run: HostRun) {
        match self.live.lock().remove(&run.id) {
            Some(r) => self.pool.free(r),
            None => tracing::warn!("kv_host_tier: freeing unknown pinned run {}", run.id),
        }
    }
}
