// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/distributed/l1_manager/base.py (commit 3aefae1, Apache-2.0): a copy is published only after it completed, and the memory it touches is kept until then; no code copied.

//! Responsibility: the window desk's copies (lane L36). [`HostSide`] is the desk's
//! host half: a slab arena of registered host memory and a page API
//! (`spark_runtime::kv_host_tier::pages`) on a copy stream of its own. [`Pending`] is one copy
//! in flight, and [`settle`] applies the finished ones to a sequence's block table:
//!
//! * a finished DEMOTE releases the original GPU blocks (the desk held their reference while
//!   the D2H read them); a failed one puts each block back in the table in place of the hole;
//! * a finished FETCH installs each fresh block in the table in place of the hole (the page is
//!   then on the GPU again); a failed one frees the fresh blocks and the page stays hidden.
//!
//! Every reference stays balanced: the table owns one reference per entry (the hole's count
//! grows by one per hidden entry), the desk owns one per block a copy still touches.

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_host_tier::pages::{PageApi, PageStats, PageTicket, SlabArena};
use spark_runtime::kv_host_tier::{PageDeps, RestoreState};

use super::policy::Slot;

/// Host slab size: 64 MiB, the pinned pool's chunk (at least one page).
pub const SLAB_BYTES: usize = 64 << 20;

/// The desk's host memory and copy engine for one KV cache.
pub(crate) struct HostSide {
    pub(crate) api: PageApi,
    pub(crate) arena: Arc<SlabArena>,
}

impl HostSide {
    /// Pages of `page_bytes` (one KV block, all layers), at most `budget` bytes of host memory.
    pub(crate) fn new(
        gpu: &dyn GpuBackend,
        page_bytes: usize,
        budget: u64,
        stats: Arc<PageStats>,
    ) -> Result<Self> {
        let copy_stream = gpu.create_copy_stream()?;
        let arena = Arc::new(SlabArena::new(
            page_bytes,
            SLAB_BYTES.max(page_bytes),
            budget,
        ));
        let api = PageApi::with_stats(
            PageDeps {
                arena: arena.clone(),
                block_bytes: page_bytes,
                copy_stream,
            },
            stats,
        );
        Ok(Self { api, arena })
    }

    /// Wait for every copy, free every page and unmap the slabs (best effort).
    pub(crate) fn teardown(&self, gpu: &dyn GpuBackend) {
        self.api.quiesce(gpu);
        if let Err(e) = self.arena.release_all(gpu) {
            tracing::debug!("kv window desk: host slabs kept at teardown: {e:#}");
        }
    }
}

/// One copy in flight for one sequence.
pub(crate) enum Pending {
    /// `(logical index, original block)`: the desk holds each original block's reference.
    Demote {
        ticket: PageTicket,
        sources: Vec<(u32, u32)>,
    },
    /// `(logical index, fresh block)`: the desk holds each fresh block's reference.
    Fetch {
        ticket: PageTicket,
        staged: Vec<(u32, u32)>,
    },
}

impl Pending {
    pub(crate) fn ticket(&self) -> &PageTicket {
        match self {
            Self::Demote { ticket, .. } | Self::Fetch { ticket, .. } => ticket,
        }
    }

    /// Logical indices of a running demote (a page whose host copy is not complete yet).
    pub(crate) fn demoting(&self) -> &[(u32, u32)] {
        match self {
            Self::Demote { sources, .. } => sources,
            Self::Fetch { .. } => &[],
        }
    }
}

/// What [`settle`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Settled {
    /// Original blocks released after their demote completed.
    pub freed: usize,
    /// Fetched blocks installed in the table.
    pub installed: usize,
    /// Blocks put back after a failed demote.
    pub reverted: usize,
    /// Fetches that failed (their pages stay hidden).
    pub failed: usize,
}

/// Apply every finished copy of one live sequence to its `table` and `slots`; returns the copies
/// still running. A poll error keeps the copy pending (it is polled again next step).
#[allow(clippy::too_many_arguments)]
pub(crate) fn settle(
    api: &PageApi,
    gpu: &dyn GpuBackend,
    pending: Vec<Pending>,
    table: &mut [u32],
    slots: &mut [Slot],
    hole: u32,
    kv: &mut PagedKvCache,
    out: &mut Settled,
) -> Vec<Pending> {
    let mut running = Vec::new();
    for p in pending {
        let state = match api.poll(gpu, p.ticket()) {
            Ok(s) => s,
            Err(e) => {
                tracing::debug!("kv window desk: polling copy {}: {e:#}", p.ticket().id());
                running.push(p);
                continue;
            }
        };
        if state == RestoreState::Pending {
            running.push(p);
            continue;
        }
        let ok = state == RestoreState::Ready;
        match p {
            Pending::Demote { ticket, sources } => {
                for (i, src) in sources {
                    let i = i as usize;
                    let still_hidden =
                        slots.get(i) == Some(&Slot::Hidden) && table.get(i) == Some(&hole);
                    if !ok && still_hidden {
                        table[i] = src;
                        slots[i] = Slot::Resident;
                        kv.dec_ref(hole);
                        out.reverted += 1;
                    } else {
                        kv.dec_ref(src);
                        out.freed += 1;
                    }
                }
                api.release(ticket);
            }
            Pending::Fetch { ticket, staged } => {
                if !ok {
                    out.failed += 1;
                }
                for (i, blk) in staged {
                    let i = i as usize;
                    let waiting =
                        slots.get(i) == Some(&Slot::Fetching) && table.get(i) == Some(&hole);
                    if ok && waiting {
                        table[i] = blk;
                        slots[i] = Slot::Recalled;
                        kv.dec_ref(hole);
                        out.installed += 1;
                    } else {
                        kv.dec_ref(blk);
                        if waiting {
                            slots[i] = Slot::Hidden;
                        }
                    }
                }
                api.release(ticket);
            }
        }
    }
    running
}

/// Settle copies whose sequence is gone: every block the desk still holds is released once its
/// copy finished (the table that held the hole entries was freed with the sequence).
pub(crate) fn settle_orphans(
    api: &PageApi,
    gpu: &dyn GpuBackend,
    pending: Vec<Pending>,
    kv: &mut PagedKvCache,
) -> Vec<Pending> {
    let mut running = Vec::new();
    for p in pending {
        match api.poll(gpu, p.ticket()) {
            Ok(RestoreState::Pending) | Err(_) => running.push(p),
            Ok(_) => {
                let (ticket, blocks) = match p {
                    Pending::Demote { ticket, sources } => (ticket, sources),
                    Pending::Fetch { ticket, staged } => (ticket, staged),
                };
                for (_, b) in blocks {
                    kv.dec_ref(b);
                }
                api.release(ticket);
            }
        }
    }
    running
}
