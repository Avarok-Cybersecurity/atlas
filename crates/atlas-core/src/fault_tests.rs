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
            // The message must preserve the originating operation and error,
            // plus the probe failure. An operator reading only the log needs
            // all three to distinguish the first failure from its sticky echo.
            assert!(reason.contains("w4a16_gemm_t launch"), "reason: {reason}");
            assert!(reason.contains(STICKY_716), "reason: {reason}");
            assert!(
                reason.contains("cuStreamSynchronize returned 716"),
                "reason: {reason}"
            );
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
/// PROVEN BY: adding a pre-match guard that returns `ContextLost` when a
/// healthy probe accompanies error text containing `716` turns only this
/// partition red. The failed-probe and ordinary-OOM partitions stay green.
#[test]
fn scary_error_text_with_a_healthy_probe_is_not_fatal() {
    assert_eq!(
        classify("some launch", STICKY_716, Ok(())),
        Fatality::Isolated,
    );
}

/// NEGATIVE: an ordinary recoverable failure is never fatal.
///
/// PROVEN BY: treating `OUT_OF_MEMORY` as fatal even when the probe is healthy
/// turns this red while the sticky-716 and failed-probe partitions stay green.
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
/// PROVEN BY: replacing the atomic set result with `get().is_none()`, then a
/// yield, set, and unconditional `true` lets multiple callers report first.
/// The sequential first-writer test stays green while this test turns red.
#[test]
fn exactly_one_caller_wins_the_race() {
    use std::sync::{Arc, Barrier};
    let l = Arc::new(FaultLatch::new());
    let start = Arc::new(Barrier::new(9));
    let winners: usize = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let l = Arc::clone(&l);
                let start = Arc::clone(&start);
                s.spawn(move || {
                    start.wait();
                    usize::from(l.latch(format!("thread {i}")))
                })
            })
            .collect();
        start.wait();
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

// ---------------------------------------------------------------------------
// exit status — the last mile of #429
// ---------------------------------------------------------------------------
//
// The latch, the 503s and the drain all worked, and the server STILL stayed
// down: a faulted shutdown is a *clean* one, so `main` returned `Ok` and the
// process exited 0. `restart: on-failure` does not restart an exit-0 process,
// so the endpoint never came back. Symmetric to the latch itself — exiting 0
// on a fault leaves a dead endpoint, and exiting nonzero on a clean stop puts
// a healthy server into a restart loop — so both halves are tested.

/// POSITIVE: a clean drain that followed a fault must NOT look like success.
///
/// PROVEN BY: returning `0` for the `Some` arm turns this red on the first
/// assert (`assertion `left != right` failed: 0 != 0`).
#[test]
fn a_faulted_shutdown_exits_nonzero() {
    assert_ne!(exit_code(true, Some("context destroyed")), 0);
    assert_eq!(exit_code(true, Some("context destroyed")), EXIT_GPU_FAULT);
}

/// POSITIVE: the fault outranks the run's own status, so the operator is told
/// the cause rather than the symptom.
///
/// PROVEN BY: ordering the match so `(false, _)` wins turns this red.
#[test]
fn a_fault_outranks_a_failed_run() {
    assert_eq!(exit_code(false, Some("context destroyed")), EXIT_GPU_FAULT);
}

/// NEGATIVE: an ordinary `docker stop` still exits 0.
///
/// Without this, "make it nonzero" could be satisfied by making EVERY exit
/// nonzero — which would restart-loop every healthy server that was asked to
/// stop. That is a worse outage than the one being fixed.
///
/// PROVEN BY: returning `EXIT_GPU_FAULT` unconditionally turns this red.
#[test]
fn a_clean_shutdown_without_a_fault_still_exits_zero() {
    assert_eq!(exit_code(true, None), 0);
}

/// NEGATIVE: an ordinary failure keeps the generic status and is not
/// mislabelled as a GPU fault — otherwise the distinct code stops meaning
/// anything and an operator cannot trust it.
///
/// PROVEN BY: returning `EXIT_GPU_FAULT` for any `!run_succeeded` turns this
/// red on the second assert.
#[test]
fn an_ordinary_failure_is_not_reported_as_a_gpu_fault() {
    assert_eq!(exit_code(false, None), 1);
    assert_ne!(exit_code(false, None), EXIT_GPU_FAULT);
}
