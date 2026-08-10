// SPDX-License-Identifier: AGPL-3.0-only

//! The wait a model release has to do first, in one testable place.
//!
//! `Model::teardown`'s contract is "after the scheduler has drained AND the
//! stream is synchronised". The draining half was honoured by the scheduler
//! loop; the synchronise half was supplied by nobody, and a device free racing
//! live kernels is an unmapped read, not a benign reuse — on GB10 that surfaces
//! as `Xid 31 … MMU Fault: ENGINE GRAPHICS … FAULT_PTE ACCESS_TYPE_VIRT_READ`,
//! observed on a hot-swap 2026-08-07.
//!
//! Split out the way `wait_for_sole_owner` was: the rule is a few lines and
//! standing up a real `Model` to exercise it is not feasible, so the sync
//! arrives as a closure and the tests assert that EVERY stream is waited on and
//! that a failure is named rather than swallowed.
//!
//! ★ Why this is a separate call from `teardown` rather than one wrapper around
//! both: `Model::synchronize` takes `&self` and `Model::teardown` takes
//! `&mut self`, so a single function holding closures over both cannot borrow-
//! check. The ordering therefore lives at the one call site, on the two lines
//! immediately following each other, rather than being enforced by a type.

use anyhow::Result;

/// Block until every stream has finished, returning the ones that would not.
///
/// The caller releases immediately after. Failures are returned rather than
/// aborting the release: refusing to free because a stream would not
/// synchronise leaks the entire model, and a stream in that state is not one
/// teardown can improve. What must never happen is freeing *before* waiting —
/// which is the bug this exists to prevent.
pub(super) fn quiesce_streams(
    streams: &[(&'static str, u64)],
    mut sync: impl FnMut(u64) -> Result<()>,
) -> Vec<&'static str> {
    streams
        .iter()
        .filter(|(_, stream)| sync(*stream).is_err())
        .map(|(name, _)| *name)
        .collect()
}

#[cfg(test)]
#[path = "teardown_tests.rs"]
mod tests;
