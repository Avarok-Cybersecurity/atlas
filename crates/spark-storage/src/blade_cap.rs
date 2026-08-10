// SPDX-License-Identifier: AGPL-3.0-only
//
// A process-global commit ledger for the RDMA memory blades (`cache_peer`,
// `expert_peer`). Each peer server owns one `CommitLedger` (an `Arc`, cloned
// into every per-connection thread) and, at the RDMA handshake — AFTER the
// requested commit size is known but BEFORE any `mmap`/`reg_mr` pins RAM — a
// connection calls `try_reserve(bytes)`. That either succeeds (returning an
// RAII `Reservation` that releases the bytes on drop) or bails, so a client is
// rejected before any memory is registered when it would push the running
// total past the configured ceiling.
//
// This lives entirely OUTSIDE `cfg(atlas_rdma_verbs)` so the arithmetic is
// unit-testable on the metal/skip build with no RDMA hardware — the verbs
// handshake body that consumes it is gated, but the ledger is not.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};

/// A process-global running total of committed (registered) bytes against a
/// fixed ceiling. `cap == 0` means unlimited (the default — every existing
/// invocation with no `--max-blade-gb` flag behaves exactly as before).
#[derive(Debug)]
pub struct CommitLedger {
    cap: u64,
    committed: AtomicU64,
}

/// An RAII claim on `bytes` of the ledger. Constructed only after a successful
/// reservation; its `Drop` decrements the running total exactly once, so early
/// bail, `reg_mr` failure, and normal hangup all release uniformly.
#[derive(Debug)]
pub struct Reservation {
    ledger: Arc<CommitLedger>,
    bytes: u64,
}

impl CommitLedger {
    /// `cap_bytes == 0` = unlimited.
    pub fn new(cap_bytes: u64) -> Self {
        Self {
            cap: cap_bytes,
            committed: AtomicU64::new(0),
        }
    }

    /// The configured ceiling in bytes (0 = unlimited).
    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// Current committed total in bytes.
    pub fn committed(&self) -> u64 {
        self.committed.load(Ordering::Acquire)
    }

    /// Atomically test-against-cap-and-add `bytes`. A `compare_exchange` loop
    /// keeps the test and the add indivisible, so two concurrent handshakes
    /// cannot both slip past a nearly-full ledger. Returns an RAII `Reservation`
    /// on success; bails (pinning nothing) when the request would cross the cap.
    pub fn try_reserve(self: &Arc<Self>, bytes: u64) -> Result<Reservation> {
        let mut cur = self.committed.load(Ordering::Acquire);
        loop {
            let new = cur
                .checked_add(bytes)
                .context("blade commit total overflow")?;
            if self.cap != 0 && new > self.cap {
                bail!(
                    "blade cap exceeded: request {bytes} B + {cur} B committed > cap {} B",
                    self.cap,
                );
            }
            match self.committed.compare_exchange_weak(
                cur,
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    return Ok(Reservation {
                        ledger: self.clone(),
                        bytes,
                    });
                }
                Err(observed) => cur = observed,
            }
        }
    }
}

impl Reservation {
    /// The number of bytes this reservation holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger
            .committed
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_within_cap_raises_committed() {
        let l = Arc::new(CommitLedger::new(1000));
        let r = l.try_reserve(400).expect("within cap");
        assert_eq!(l.committed(), 400);
        assert_eq!(r.bytes(), 400);
    }

    #[test]
    fn reserve_over_cap_bails_and_leaves_committed_unchanged() {
        let l = Arc::new(CommitLedger::new(1000));
        let _r = l.try_reserve(600).expect("within cap");
        assert_eq!(l.committed(), 600);
        let err = l.try_reserve(500);
        assert!(err.is_err(), "600 + 500 > 1000 must be rejected");
        // The rejected request pinned nothing: the total is unchanged.
        assert_eq!(l.committed(), 600);
    }

    #[test]
    fn dropping_a_reservation_restores_committed() {
        let l = Arc::new(CommitLedger::new(1000));
        {
            let _r = l.try_reserve(700).expect("within cap");
            assert_eq!(l.committed(), 700);
        }
        assert_eq!(l.committed(), 0);
        // And a subsequent full-cap request now fits.
        let _r2 = l.try_reserve(1000).expect("fits after release");
        assert_eq!(l.committed(), 1000);
    }

    #[test]
    fn two_reservations_aggregate_reject_then_accept_after_drop() {
        let l = Arc::new(CommitLedger::new(1000));
        let r1 = l.try_reserve(700).expect("first fits");
        // Second crosses the cap while the first is held.
        assert!(l.try_reserve(400).is_err(), "700 + 400 > 1000");
        assert_eq!(l.committed(), 700);
        drop(r1);
        // Now it fits.
        let _r2 = l.try_reserve(400).expect("fits after first drops");
        assert_eq!(l.committed(), 400);
    }

    #[test]
    fn cap_zero_accepts_arbitrarily_large() {
        let l = Arc::new(CommitLedger::new(0));
        let _r = l.try_reserve(u64::MAX / 2).expect("unlimited");
        let _r2 = l.try_reserve(1 << 40).expect("still unlimited");
        assert_eq!(l.committed(), (u64::MAX / 2) + (1 << 40));
    }

    #[test]
    fn exact_fit_is_accepted() {
        let l = Arc::new(CommitLedger::new(1000));
        let _r = l.try_reserve(1000).expect("new == cap is allowed");
        assert_eq!(l.committed(), 1000);
        // But one more byte is not.
        assert!(l.try_reserve(1).is_err());
    }

    #[test]
    fn overflow_is_rejected_not_wrapped() {
        // cap == 0 (unlimited) but the running total must not wrap around.
        let l = Arc::new(CommitLedger::new(0));
        let _r = l.try_reserve(u64::MAX - 10).expect("fits");
        assert!(
            l.try_reserve(100).is_err(),
            "checked_add must reject wraparound"
        );
    }
}
