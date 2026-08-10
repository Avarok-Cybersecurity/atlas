// SPDX-License-Identifier: AGPL-3.0-only

//! `PHASE` is process-global and the request flag has **no reset**, so nothing
//! here may assume it runs first: other tests in this binary trip the latch
//! (Ctrl+C in `app`, `/quit` in `commands`) and cargo runs them as threads in
//! ONE process. Every case below asserts from whatever state it finds.

use super::*;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a current-thread runtime")
}

/// The latch: it goes up once and never comes down.
///
/// The promise is load-bearing beyond shutdown itself — `bench_selfstart`
/// refuses a second self-start in one process BECAUSE this never clears, and
/// `model_swap` refuses to load once it is set. A reset would turn both
/// refusals into a hang.
#[test]
fn a_shutdown_request_is_a_one_way_latch_with_no_reset() {
    let rt = runtime();

    // The transition is only observable from an untripped process, so it is
    // asserted only when this test gets there first. Whichever branch runs, the
    // idempotency below is asserted either way.
    if !requested() {
        rt.block_on(async {
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(50), wait())
                    .await
                    .is_err(),
                "wait() must not resolve before anything has been requested"
            );
        });
        request("SIGINT");
        assert!(requested(), "a request trips the latch");
    }

    // Idempotent: further requests neither panic on the spent escape sender nor
    // clear anything.
    for _ in 0..3 {
        request("Ctrl+C");
        assert!(requested(), "the latch has no reset");
    }

    // And a waiter registered after the fact resolves at once rather than
    // parking on a notification that has already been sent.
    rt.block_on(async {
        tokio::time::timeout(std::time::Duration::from_millis(500), wait())
            .await
            .expect("wait() resolves immediately once requested");
    });
}

/// The startup escape: taken while `serve()` is still loading, parked for good
/// once the accept loop owns shutdown.
///
/// Independent of the latch — `request` hands the reason over whenever the
/// escape is armed, tripped or not — so this holds wherever it is scheduled.
/// It is also the only caller of `disarm_startup_escape` in this binary, and it
/// does its own disarming, in order.
#[test]
fn the_startup_escape_is_taken_while_loading_and_parked_once_disarmed() {
    let (tx, mut rx) = oneshot::channel();
    arm_startup_escape(tx);
    request("escape-armed");
    // Receipt, not the wording: a trigger from a concurrent test can be the one
    // that takes the sender, and whose reason arrives is not the invariant.
    let reason = rx
        .try_recv()
        .expect("a request while in startup hands its reason to main's one-shot");
    assert!(!reason.is_empty());

    disarm_startup_escape();

    // From here the escape is neither taken NOR dropped: dropping the sender
    // resolves main's receiver with `RecvError`, which is indistinguishable
    // from a shutdown and would exit a healthy server the instant it came up.
    //
    // Retried because a request that read `in_startup` before the disarm can
    // still take the sender it finds armed afterwards — the assertion needs one
    // quiet window, not a quiet process.
    let mut parked = false;
    for _ in 0..10 {
        let (tx, mut rx) = oneshot::channel();
        arm_startup_escape(tx);
        request("post-disarm");
        if matches!(rx.try_recv(), Err(oneshot::error::TryRecvError::Empty)) {
            parked = true;
            break;
        }
    }
    assert!(
        parked,
        "the parked sender must stay parked and its channel must stay open"
    );
    assert!(requested(), "and shutdown is still requested");
}

/// STATIC, DELIBERATELY — `REQUESTS_ACTIVE` is a process-global gauge, so the
/// two drain cases cannot run at once without reading each other's requests.
static DRAIN: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Park the gauge back where it was, even if an assertion unwinds.
struct GaugeGuard(i64);

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        crate::metrics::REQUESTS_ACTIVE.sub(self.0);
    }
}

fn hold_requests(n: i64) -> GaugeGuard {
    crate::metrics::REQUESTS_ACTIVE.add(n);
    GaugeGuard(n)
}

#[test]
fn an_idle_server_drains_immediately() {
    let _serial = DRAIN.lock();
    assert_eq!(
        crate::metrics::REQUESTS_ACTIVE.get(),
        0,
        "nothing in flight"
    );
    let start = std::time::Instant::now();
    runtime().block_on(drain_in_flight(std::time::Duration::from_secs(30)));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "a 30s grace must not be waited out when there is nothing to wait for"
    );
}

#[test]
fn a_stuck_request_gives_up_at_the_grace_window_rather_than_hanging() {
    // "Drained" is honest about what it means: the grace expires and the
    // process exits anyway.
    let _serial = DRAIN.lock();
    let _held = hold_requests(1);
    let start = std::time::Instant::now();
    runtime().block_on(drain_in_flight(std::time::Duration::ZERO));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "an expired grace returns instead of waiting on the request"
    );
    assert_eq!(
        crate::metrics::REQUESTS_ACTIVE.get(),
        1,
        "draining does not touch the gauge"
    );
}

/// A negative gauge — the shape a double-decrement leaves behind — must read as
/// drained, not as a request that can never finish.
#[test]
fn a_gauge_below_zero_still_drains() {
    let _serial = DRAIN.lock();
    let _held = hold_requests(-1);
    let start = std::time::Instant::now();
    runtime().block_on(drain_in_flight(std::time::Duration::from_secs(30)));
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
}

/// Not a test of `shutdown` itself: it pins the assumption `drain_in_flight`
/// makes about the gauge it reads, which lives in another module.
#[test]
fn the_gauge_the_drain_reads_is_the_one_requests_move() {
    let _serial = DRAIN.lock();
    let before = crate::metrics::REQUESTS_ACTIVE.get();
    let guard = crate::metrics::ActiveRequestGuard::new();
    assert_eq!(crate::metrics::REQUESTS_ACTIVE.get(), before + 1);
    drop(guard);
    assert_eq!(crate::metrics::REQUESTS_ACTIVE.get(), before);
}
