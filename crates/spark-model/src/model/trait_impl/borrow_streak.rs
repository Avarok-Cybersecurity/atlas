// SPDX-License-Identifier: AGPL-3.0-only

//! Steady-state guard for drain-tail CUDA-graph borrowing (`graph_borrow.rs`).
//!
//! A borrow replays a WIDER captured graph for a batch whose exact key missed,
//! padding the tail rows. It exists for drains, where every composition is new
//! and short-lived. But it never captures the narrow key, so a batch that
//! SETTLES at a width below an earlier peak keeps replaying the wide graph for
//! as long as it runs. Measured on dgx1 (Qwen3.8-27B NVFP4, 2026-09-23): after
//! one 32-wide burst, a steady C=16 verified through the captured 32-seq graph
//! (64 rows, the tile/MMQ kernels instead of the 32-row GEMV). Acceptance was
//! unchanged (tok_step 1.70), but each step was about 35% slower: 184 tok/s
//! against 223-243 tok/s for the same C=16 with no burst before it.
//!
//! The guard: after [`STEADY_BORROW_LIMIT`] CONSECUTIVE borrows for one exact
//! key, the borrow is declined. The step then captures the exact key, and every
//! later step replays at its own width. Drain compositions that change within
//! the limit are unaffected. Kill switch: `AVAROK_NO_BORROW_STREAK_LIMIT`
//! (any non-empty value disables the guard, `0` included; a gate record
//! discloses it in `PERF_CONTROLS` with default `unset`).

use std::sync::Mutex;

/// Consecutive borrows of one exact key before that key gets its own capture.
/// A borrowed step costs its padded rows (up to 2x) on every replay. A capture
/// costs one eager step. Eight steps is well past the break-even at the widths
/// that borrow.
pub(super) const STEADY_BORROW_LIMIT: u32 = 8;

/// The `AVAROK_NO_BORROW_STREAK_LIMIT` rule over the looked-up kill switch:
/// the guard is ON unless it is set to a non-empty value.
pub fn streak_limit_from(kill: Option<&std::ffi::OsStr>) -> bool {
    kill.is_none_or(|v| v.is_empty())
}

fn guard_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        streak_limit_from(std::env::var_os("AVAROK_NO_BORROW_STREAK_LIMIT").as_deref())
    })
}

/// Consecutive-borrow counter for one graph family (decode or verify).
pub(super) struct BorrowStreak(Mutex<(u64, u32)>);

impl BorrowStreak {
    pub(super) const fn new() -> Self {
        Self(Mutex::new((0, 0)))
    }

    /// Record one borrow the caller is about to make for `exact_key`. Returns
    /// false once this key has borrowed [`STEADY_BORROW_LIMIT`] times in a
    /// row: the caller then declines the borrow and captures the exact key.
    pub(super) fn allow(&self, exact_key: &[u32]) -> bool {
        self.allow_with(exact_key, guard_enabled())
    }

    fn allow_with(&self, exact_key: &[u32], enabled: bool) -> bool {
        // FNV-1a over length + values. 0 means "no key yet".
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for v in
            std::iter::once(exact_key.len() as u64).chain(exact_key.iter().map(|&v| u64::from(v)))
        {
            h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let h = h.max(1);
        let mut g = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if g.0 == h {
            g.1 = g.1.saturating_add(1);
        } else {
            *g = (h, 1);
        }
        !enabled || g.1 <= STEADY_BORROW_LIMIT
    }
}

pub(super) static DECODE_BORROW_STREAK: BorrowStreak = BorrowStreak::new();
pub(super) static VERIFY_BORROW_STREAK: BorrowStreak = BorrowStreak::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_steady_key_stops_borrowing_after_the_limit() {
        let s = BorrowStreak::new();
        let k = [0u32, 1, 1, 1, 2, 1, 9];
        for i in 1..=STEADY_BORROW_LIMIT {
            assert!(s.allow_with(&k, true), "borrow {i} is within the limit");
        }
        assert!(!s.allow_with(&k, true), "borrow limit+1 must be declined");
        assert!(
            !s.allow_with(&k, true),
            "and stays declined while the key persists"
        );
    }

    #[test]
    fn a_changing_drain_key_keeps_borrowing() {
        let s = BorrowStreak::new();
        for i in 0..(4 * STEADY_BORROW_LIMIT) {
            // A drain: the composition changes before the limit is reached.
            let n = 2 + (i / (STEADY_BORROW_LIMIT - 1));
            let k: Vec<u32> = (0..n).collect();
            assert!(
                s.allow_with(&k, true),
                "step {i}: a new composition restarts the count"
            );
        }
    }

    #[test]
    fn keys_that_differ_only_at_the_pair_boundary_are_distinct() {
        let s = BorrowStreak::new();
        for _ in 0..STEADY_BORROW_LIMIT {
            assert!(s.allow_with(&[1, 2, 3], true));
        }
        assert!(
            s.allow_with(&[1, 2], true),
            "a different key resets the streak"
        );
    }

    #[test]
    fn the_kill_switch_rule_is_presence_of_a_non_empty_value() {
        use std::ffi::OsStr;
        assert!(streak_limit_from(None), "unset: guard on");
        assert!(
            streak_limit_from(Some(OsStr::new(""))),
            "exported empty: guard on"
        );
        assert!(!streak_limit_from(Some(OsStr::new("1"))));
        assert!(!streak_limit_from(Some(OsStr::new("0"))), "0 is NOT on");
    }

    #[test]
    fn the_kill_switch_always_allows() {
        let s = BorrowStreak::new();
        for _ in 0..(3 * STEADY_BORROW_LIMIT) {
            assert!(s.allow_with(&[4, 5, 6], false));
        }
    }
}
