// SPDX-License-Identifier: AGPL-3.0-only

//! How much of the memory beside the weights the SSM snapshot regions may take.
//!
//! The two regions this sizes, the Phase-C decode-rollback ring
//! (`decode_ring`) and the Marconi prefix-cache snapshots
//! (`ssm_reserve::marconi_snapshot_slots`), both used to be sized from
//! constants that were right for one machine: depth 8 and 16 slots, multiplied
//! by `--max-batch-size` and the per-sequence state blob. On a 120 GB part
//! that is a rounding error. On a 16 GB card carrying a 4B model it is
//! 3,264 MB of ring (8 slots x 8 seqs x 51 MB) and 816 MB of Marconi
//! snapshots next to roughly 6 GB that are left once the weights are in, and
//! neither region is on the serving path: the ring is a safety net for a rare
//! loop re-steer and Marconi only accelerates repeated prefixes.
//!
//! Both regions therefore get a share of the memory BESIDE the weights
//! (device memory that is free before the load, minus the checkpoint), and
//! yield to it:
//!
//! * the ring may take at most `1 / RING_BUDGET_DIVISOR` of it, and gives up
//!   depth down the same `DECODE_RING_FIT_LADDER` the #915 fit uses
//!   (`8, 4, 2, 1, 0`), never batch;
//! * Marconi may take at most `1 / MARCONI_BUDGET_DIVISOR` of it, and gives up
//!   slots.
//!
//! "Keep the GB10 defaults when memory is plentiful" is the design constraint
//! the divisors are chosen against. With ~115 GiB free and a 17 GiB checkpoint
//! the ring budget is ~33 GiB: every in-tree SSM model at `--max-batch-size`
//! up to 27 keeps depth 8 (the 27B's 151.5 MiB blob costs 32.0 GiB at batch
//! 27 and 33.1 GiB at 28), and the Marconi budget is ~12 GiB, about 80 slots
//! of the same blob against the default 16. A serve that pins a depth with
//! `--ssm-decode-ring-slots N` is never shrunk by this.
//!
//! Like the ring depth itself, the Marconi cap is computed by preflight and
//! read by the allocating call site, so the reserve and the allocation cannot
//! disagree (see [`published_marconi_slot_cap`]).

use std::sync::OnceLock;

use super::fit_decode_ring_slots;

/// The decode-rollback ring may take at most `1 / RING_BUDGET_DIVISOR` of the
/// memory beside the weights.
pub const RING_BUDGET_DIVISOR: usize = 3;

/// The Marconi snapshot region may take at most `1 / MARCONI_BUDGET_DIVISOR`
/// of the memory beside the weights.
pub const MARCONI_BUDGET_DIVISOR: usize = 8;

/// Memory beside the weights: free device memory before the load, minus the
/// checkpoint's on-disk size.
///
/// The on-disk size is clamped to half of free memory. It equals the resident
/// size only for a checkpoint that is loaded verbatim; a BF16 checkpoint that
/// is quantized to NVFP4 while it loads is far larger on disk than in memory,
/// and subtracting all of it would zero the budget of a serve that fits
/// comfortably. A clamped checkpoint makes the budget generous, never tight.
pub fn memory_beside_weights(free_mem: usize, checkpoint_bytes: usize) -> usize {
    free_mem - checkpoint_bytes.min(free_mem / 2)
}

/// Bytes the decode-rollback ring may take.
pub fn ring_budget_bytes(beside_weights: usize) -> usize {
    beside_weights / RING_BUDGET_DIVISOR
}

/// Bytes the Marconi snapshot region may take.
pub fn marconi_budget_bytes(beside_weights: usize) -> usize {
    beside_weights / MARCONI_BUDGET_DIVISOR
}

/// Largest [`super::DECODE_RING_FIT_LADDER`] depth `<= start_slots` whose ring
/// fits [`ring_budget_bytes`]. `bytes_per_ring_slot` is one unit of depth,
/// `max_batch_size x` the per-sequence SSM state blob.
///
/// This is [`fit_decode_ring_slots`] with nothing else beside the ring: the
/// budget is a share of memory, not the whole reserve, so the rest of the
/// reserve is not part of this question.
pub fn cap_decode_ring_slots(
    start_slots: usize,
    bytes_per_ring_slot: usize,
    beside_weights: usize,
) -> usize {
    if bytes_per_ring_slot == 0 {
        return start_slots;
    }
    fit_decode_ring_slots(
        start_slots,
        0,
        bytes_per_ring_slot,
        ring_budget_bytes(beside_weights),
    )
}

/// Most Marconi slots `beside_weights` allows when one slot costs
/// `bytes_per_slot` (the per-sequence state blob), capped at `requested`.
pub fn cap_marconi_slots(requested: usize, bytes_per_slot: usize, beside_weights: usize) -> usize {
    if bytes_per_slot == 0 {
        return requested;
    }
    requested.min(marconi_budget_bytes(beside_weights) / bytes_per_slot)
}

/// `slots` under a published cap, or unchanged when none was published.
pub fn apply_marconi_cap(slots: usize, cap: Option<usize>) -> usize {
    cap.map_or(slots, |cap| slots.min(cap))
}

/// The Marconi slot cap preflight decided, published for the allocating call
/// site. First write wins, like the ring depth's cell.
///
/// Absent is NOT a value: a process that never ran preflight (a bench, a unit
/// test, the plugin binary) applies no cap and behaves exactly as before.
static MARCONI_SLOT_CAP: OnceLock<usize> = OnceLock::new();

/// Publish the Marconi slot cap. Returns the cap in force.
pub fn set_marconi_slot_cap(cap: usize) -> usize {
    let _ = MARCONI_SLOT_CAP.set(cap);
    *MARCONI_SLOT_CAP.get().expect("just set")
}

/// The published cap, or `None` when preflight has not run in this process.
pub fn published_marconi_slot_cap() -> Option<usize> {
    MARCONI_SLOT_CAP.get().copied()
}

#[cfg(test)]
#[path = "budget_tests.rs"]
mod budget_tests;
