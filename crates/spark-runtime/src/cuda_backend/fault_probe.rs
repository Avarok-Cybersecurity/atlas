// SPDX-License-Identifier: AGPL-3.0-only

//! Detect a destroyed CUDA context after a failed GPU operation (issue #429).
//!
//! This is the one place that turns a driver failure into the process-wide
//! verdict held by [`atlas_core::fault`]. The decision itself is that module's
//! pure [`classify`](atlas_core::fault::classify); everything here is the
//! probe it needs.
//!
//! # The probe
//!
//! `cuStreamSynchronize` on the legacy default stream. On a healthy context
//! with no outstanding work this returns `CUDA_SUCCESS` immediately; on a
//! context destroyed by a sticky error it returns that error, and keeps
//! returning it forever. It is therefore a direct measurement of "is this
//! context still usable", which is exactly the question — and not a guess from
//! the failing call's status code.
//!
//! Synchronizing is also what surfaces *asynchronous* faults. A misaligned
//! access inside a kernel does not fail the launch; it faults the context
//! later, and the next synchronizing call is what reports it. Probing with a
//! sync is what lets an async fault be attributed at all.
//!
//! # Cost
//!
//! One sync, and only on an error path. Once the latch is set the probe is
//! skipped entirely, so a dead context does not pay a sync per doomed call.

use atlas_core::fault::{self, Fatality};

use super::cuStreamSynchronize;

/// The legacy default stream. Synchronizing it is a no-op on a healthy,
/// idle context.
const DEFAULT_STREAM: u64 = 0;

/// Issue the probe. `Ok(())` means the context is still usable.
fn probe() -> Result<(), String> {
    let status = unsafe { cuStreamSynchronize(DEFAULT_STREAM) };
    if status == 0 {
        Ok(())
    } else {
        Err(atlas_core::registry::cuda_error_text(status))
    }
}

/// Record a failed GPU operation and, if it destroyed the context, latch the
/// process-wide fault.
///
/// Call this from every site that observes a nonzero CUDA status. It never
/// changes control flow: the caller still returns its own error, and an
/// isolated failure is still an ordinary per-request error. What it adds is
/// that a *fatal* failure stops being indistinguishable from a recoverable
/// one — which is the whole of #429, where a poisoned context looked like an
/// endless run of unrelated 500s.
pub(super) fn note_failure(op: &str, err: &str) {
    // Already known dead: skip the probe. It would fail (that is the nature of
    // a sticky error) and the latch is first-writer-wins anyway, so the only
    // effect would be a sync per doomed call while the server drains.
    if fault::global().is_faulted() {
        return;
    }
    if let Fatality::ContextLost(reason) = fault::classify(op, err, probe())
        && fault::global().latch(reason.clone())
    {
        // ERROR, not WARN: this is terminal. The server can no longer serve
        // any request, and the log line is the operator's only explanation
        // for the shutdown that follows.
        tracing::error!(target: "atlas::fault", "{reason}");
    }
}
