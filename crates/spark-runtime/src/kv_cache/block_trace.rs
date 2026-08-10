// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-block refcount event history, for diagnosing KV refcount bugs.
//
// A refcount bug is silent at the moment it happens: an unmatched decrement
// just leaves a block one ref lower than it should be, and the damage only
// surfaces much later — either as `dec_ref on block N with 0 refs` when the
// rightful owner finally frees it, or (worse) as two sequences writing the
// same physical block after it was returned to the free list early. By then
// the guilty call site is long gone from the logs.
//
// With `ATLAS_KV_TRACE=1` every alloc/inc/dec/evict-return on every block is
// appended to a per-block ring together with the caller's file:line (via
// `#[track_caller]`, so no manual tagging at the call sites). When an
// underflow is detected the whole ring for that block is dumped, naming the
// call site that took the extra reference away.
//
// Off by default: the rings cost one `Vec` per block plus a push per refcount
// operation, which is not free on the decode hot path.

use std::panic::Location;
use std::sync::OnceLock;

/// Events retained per block. The interesting window is the last handful of
/// operations before the underflow; older history is dropped.
const RING_LEN: usize = 24;

#[derive(Clone, Copy)]
pub(super) struct BlockEvent {
    op: &'static str,
    count_after: u32,
    file: &'static str,
    line: u32,
}

/// Per-block ring buffer of refcount events. Empty (and free) when disabled.
#[derive(Default)]
pub(super) struct BlockTrace {
    rings: Vec<Vec<BlockEvent>>,
}

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_KV_TRACE").as_deref() == Ok("1"))
}

impl BlockTrace {
    pub(super) fn new(num_blocks: usize) -> Self {
        if enabled() {
            tracing::info!(
                "ATLAS_KV_TRACE=1: recording per-block refcount history \
                 ({RING_LEN} events x {num_blocks} blocks)"
            );
            Self {
                rings: vec![Vec::with_capacity(RING_LEN); num_blocks],
            }
        } else {
            Self::default()
        }
    }

    pub(super) fn is_on(&self) -> bool {
        !self.rings.is_empty()
    }

    pub(super) fn record(
        &mut self,
        idx: usize,
        op: &'static str,
        count_after: u32,
        loc: &'static Location<'static>,
    ) {
        let Some(ring) = self.rings.get_mut(idx) else {
            return;
        };
        if ring.len() == RING_LEN {
            ring.remove(0);
        }
        ring.push(BlockEvent {
            op,
            count_after,
            file: loc.file(),
            line: loc.line(),
        });
    }

    /// Render a block's history oldest-first, one event per ` | ` segment.
    pub(super) fn dump(&self, idx: usize) -> String {
        let Some(ring) = self.rings.get(idx) else {
            return String::from("(no history)");
        };
        ring.iter()
            .map(|e| {
                let file = e.file.rsplit('/').next().unwrap_or(e.file);
                format!("{}->{} @{}:{}", e.op, e.count_after, file, e.line)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }
}
