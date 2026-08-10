// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the GPU fault latch (issue #429).
//!
//! Every behaviour has a POSITIVE and a NEGATIVE case, because the failure
//! mode this module guards against is symmetric and both halves are damaging:
//! failing to latch leaves a dead server advertising itself as healthy;
//! latching too eagerly kills a healthy server over a recoverable request.
//!
//! Each test was run against a mutated `fault.rs` and observed RED before
//! being kept — see the `PROVEN BY` note on each.

use super::*;

/// The exact text the driver produces for the sticky error in #429. Used to
/// prove that classification does NOT key off it.
const STICKY_716: &str = "CUDA_ERROR_MISALIGNED_ADDRESS (716): misaligned address";

// ---------------------------------------------------------------------------
// classify — the probe decides, and nothing else does
// ---------------------------------------------------------------------------

/// POSITIVE: a failed probe means the context is gone.
///
/// PROVEN BY: swapping the `classify` match arms (`Ok` → ContextLost,
/// `Err` → Isolated) turns this red.
#[test]
fn failed_probe_means_context_lost() {
    let v = classify(
        "w4a16_gemm_t launch",
        STICKY_716,
        Err("cuStreamSynchronize returned 716".into()),
    );
    match v {
        Fatality::ContextLost(reason) => {
            // The message must name BOTH the originating op and the probe, or
            // an operator reading only the log cannot tell which call died.
            assert!(reason.contains("w4a16_gemm_t launch"), "reason: {reason}");
            assert!(reason.contains("cuStreamSynchronize"), "reason: {reason}");
        }
        Fatality::Isolated => panic!("a failed probe must be fatal"),
    }
}

/// NEGATIVE, and the one that matters most: the SAME sticky-looking 716 text,
/// but the probe succeeded. The context is alive, so the server must live.
///
/// This is the test that forbids re-implementing classification as a
/// string/code match. Any such implementation returns ContextLost here.
///
/// PROVEN BY: replacing the body of `classify` with
/// `if err.contains("716") { ContextLost } else { Isolated }` — the
/// error-code allowlist an author would naturally reach for — turns this red
/// while leaving every other test in this file green.
#[test]
fn scary_error_text_with_a_healthy_probe_is_not_fatal() {
    assert_eq!(
        classify("some launch", STICKY_716, Ok(())),
        Fatality::Isolated,
    );
}

/// NEGATIVE: an ordinary recoverable failure is never fatal.
///
/// PROVEN BY: the same match-arm swap as the positive case.
#[test]
fn isolated_failure_with_healthy_probe_is_not_fatal() {
    assert_eq!(
        classify("cuMemAlloc", "CUDA_ERROR_OUT_OF_MEMORY (2)", Ok(())),
        Fatality::Isolated,
    );
}

// ---------------------------------------------------------------------------
// FaultLatch — one-shot, first-writer-wins
// ---------------------------------------------------------------------------

/// NEGATIVE: a fresh latch reports healthy and carries no reason.
///
/// PROVEN BY: making `is_faulted` return `true` unconditionally turns this red.
#[test]
fn fresh_latch_is_healthy() {
    let l = FaultLatch::new();
    assert!(!l.is_faulted());
    assert_eq!(l.fault(), None);
}

/// POSITIVE: after latching, both readers agree and the reason survives.
///
/// PROVEN BY: making `latch` a no-op (`return false` first) turns this red.
#[test]
fn latching_records_the_reason() {
    let l = FaultLatch::new();
    assert!(l.latch("context destroyed by 716"));
    assert!(l.is_faulted());
    assert_eq!(l.fault(), Some("context destroyed by 716"));
}

/// The FIRST fault is the diagnostic one — later calls must not overwrite it.
/// After a context dies, every subsequent driver call fails too, so a
/// last-writer-wins latch would reliably report a downstream `cuMemsetD8Async`
/// instead of the launch that caused it.
///
/// PROVEN BY: replacing `set(..).is_ok()` with an unconditional
/// overwrite-and-return-`true` turns this red on BOTH assertions (the reason
/// becomes the second one, and the return becomes `true`).
#[test]
fn latch_is_first_writer_wins() {
    let l = FaultLatch::new();
    assert!(l.latch("first: the launch that poisoned the context"));
    assert!(
        !l.latch("second: a downstream cuMemsetD8Async echo"),
        "a second latch must report that it was not first"
    );
    assert_eq!(
        l.fault(),
        Some("first: the launch that poisoned the context"),
        "the diagnostic (first) fault must survive the echoes"
    );
}

/// Concurrency: the scheduler and every in-flight request hit the latch at
/// once when a context dies. Exactly one caller may be told it was first —
/// that is what gates "log once, shut down once".
///
/// PROVEN BY: replacing `self.reason.set(..).is_ok()` with a
/// get-then-set-then-`true` sequence (i.e. every caller returns `true`) turns
/// this red with `winners = 8`.
#[test]
fn exactly_one_caller_wins_the_race() {
    use std::sync::Arc;
    let l = Arc::new(FaultLatch::new());
    let winners: usize = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let l = Arc::clone(&l);
                s.spawn(move || usize::from(l.latch(format!("thread {i}"))))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    assert_eq!(winners, 1, "exactly one latch call may report first");
    assert!(l.fault().is_some(), "the winner's reason must be present");
}

// NOT A TEST, deliberately: "a visible fault always carries a reason."
//
// A flag-plus-reason latch has a window where `is_faulted()` is true and
// `fault()` is still `None`, and a health endpoint landing in it reports
// "faulted, reason unknown". I wrote a threaded test for that window; it
// SURVIVED the mutation that opens it (store the flag first, then yield, then
// write the reason) because the reader thread must win a race it almost never
// wins. A test that cannot be made to fail is decoration.
//
// So the window was removed instead of tested: `FaultLatch` is one
// `OnceLock<String>`, which makes "is faulted" and "has a reason" the same
// word. The property now holds by construction, and there is no mutation of
// `latch`/`fault`/`is_faulted` that violates it without deleting the field.
// See the module docs on `fault.rs`.

/// The global exists and starts healthy in a process that has not faulted.
/// Deliberately the ONLY test touching the global: the latch is irreversible,
/// so a test that latched it would silently order-couple every other test.
///
/// PROVEN BY: seeding `GLOBAL`'s `OnceLock` at construction turns this red.
#[test]
fn global_starts_healthy() {
    assert!(!global().is_faulted());
    assert_eq!(global().fault(), None);
}
