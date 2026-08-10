// SPDX-License-Identifier: AGPL-3.0-only

//! Process-fatal GPU fault latch.
//!
//! # Why this exists (issue #429)
//!
//! A large class of CUDA errors — `CUDA_ERROR_MISALIGNED_ADDRESS` (716),
//! `CUDA_ERROR_ILLEGAL_ADDRESS` (700), `CUDA_ERROR_LAUNCH_FAILED` (719) — are
//! **sticky**: they do not merely fail the call that produced them, they
//! destroy the CUDA *context*. Every subsequent driver call in the process
//! returns the same status, forever. There is no in-process recovery; the
//! context cannot be re-created while the primary context is retained.
//!
//! Before this module, Atlas treated such a failure as a per-request error.
//! The forward pass returned `Err`, the scheduler failed that batch, the
//! handler emitted a 500 — and then **kept serving**. `/health` still said
//! `ready`, because a model was still published, and every following request
//! died deep in the driver (observed at `cuMemsetD8Async`). The process was
//! alive, advertised itself as healthy, and could only produce errors.
//!
//! # How fatality is decided — probed, never guessed
//!
//! Classification does **not** match on the error code or its text. Sticky-ness
//! is a property of the *context*, so it is measured directly: after a failed
//! operation, issue a call that must succeed on a healthy context (a no-op
//! synchronize). If that also fails, the context is gone.
//!
//! This is why [`classify`] takes a probe *result* rather than an error code.
//! It buys three things a code allowlist cannot:
//!
//! - it covers every sticky status, including ones not yet enumerated;
//! - it does **not** kill the server for a status that merely *looks* fatal —
//!   an isolated `invalid argument` from a bad launch config leaves the
//!   context healthy, and the probe says so;
//! - it is a pure function of the probe, so both verdicts are unit-testable
//!   with no GPU.
//!
//! # Contract
//!
//! The latch is one-shot and first-writer-wins: the *first* fault is the
//! diagnostic one, and everything after it is that fault echoing through the
//! remaining call sites. Reporting the tenth `cuMemsetD8Async` failure instead
//! of the launch that poisoned the context would bury the cause.
//!
//! Both properties come from `OnceLock` rather than from code that maintains
//! them. A flag-plus-reason pair (`AtomicBool` + `Mutex<Option<String>>`) has a
//! window in which the flag is visible and the reason is not, and a health
//! endpoint that lands in it reports "faulted, reason unknown" — the least
//! useful of the three possible answers. A single `OnceLock<String>` makes
//! "is faulted" and "has a reason" the same word, so the window does not
//! exist and `set` supplies first-writer-wins atomically.

use std::sync::OnceLock;

/// The verdict for one failed GPU operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fatality {
    /// The CUDA context is unusable. The process cannot recover and must be
    /// drained and restarted; the payload is the operator-facing reason.
    ContextLost(String),
    /// The operation failed but the context still works. Fail the request,
    /// keep the server.
    Isolated,
}

/// Decide whether a failed GPU operation destroyed the context.
///
/// `probe` is the result of a call that **succeeds on any healthy context**,
/// issued after the failure. `Err` from that probe is the evidence — no
/// inference is drawn from `op` or `err`, which serve only to name the cause
/// in the message.
pub fn classify(op: &str, err: &str, probe: Result<(), String>) -> Fatality {
    match probe {
        Ok(()) => Fatality::Isolated,
        Err(probe_err) => Fatality::ContextLost(format!(
            "{op} failed ({err}), and a no-op synchronize issued afterwards \
             also failed ({probe_err}) — the CUDA context is destroyed. Errors \
             of this class are sticky: every later driver call in this process \
             returns the same status, so no request can be served."
        )),
    }
}

/// A one-shot, first-writer-wins fault flag.
///
/// Constructible in a test so the global is never a shared fixture — a latch
/// is by design irreversible, which would make one global instance a
/// cross-test dependency.
#[derive(Debug, Default)]
pub struct FaultLatch {
    reason: OnceLock<String>,
}

impl FaultLatch {
    pub const fn new() -> Self {
        Self {
            reason: OnceLock::new(),
        }
    }

    /// Record a fatal fault. Returns `true` iff this call was the first — the
    /// caller uses that to log and to trigger shutdown exactly once.
    pub fn latch(&self, reason: impl Into<String>) -> bool {
        self.reason.set(reason.into()).is_ok()
    }

    /// The reason for the fault, or `None` if healthy.
    pub fn fault(&self) -> Option<&str> {
        self.reason.get().map(String::as_str)
    }

    /// Cheap health check — one acquire load.
    pub fn is_faulted(&self) -> bool {
        self.reason.get().is_some()
    }
}

static GLOBAL: FaultLatch = FaultLatch::new();

/// The process-wide latch. A destroyed CUDA context is a property of the
/// process, not of any one backend handle, so this is deliberately global —
/// a per-backend flag would report healthy from a second handle onto the same
/// dead context.
pub fn global() -> &'static FaultLatch {
    &GLOBAL
}

#[cfg(test)]
#[path = "fault_tests.rs"]
mod tests;
