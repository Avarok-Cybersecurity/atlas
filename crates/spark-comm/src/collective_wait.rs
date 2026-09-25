// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded polling for synchronous completion. Does not bound an NCCL submit
//! call or a wedged driver call; process supervision still owns job teardown.
use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Which primitive [`AdaptiveBackoff::pause`] should invoke next. Exposed
/// mainly so the decision logic (`next_step`) can be unit tested without
/// performing real spins/yields/sleeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackoffStep {
    /// Busy-spin: the wait is expected to resolve within microseconds
    /// (e.g. a small cross-node broadcast that is already in flight).
    Spin,
    /// Give up the rest of the scheduler quantum without going to sleep.
    Yield,
    /// Sleep for the given duration. Escalates (capped) the longer the wait
    /// continues, so a wait that turns out to be long doesn't keep a core
    /// hot forever (see [`AdaptiveBackoff`] docs).
    Sleep(Duration),
}

/// Adaptive backoff for [`poll_completion`] / [`poll_idle_command`]'s `pause`
/// callback (A141): a fixed 1ms sleep on every non-ready poll cost +10.6
/// ms/step on the decode hot path, because a small cross-node broadcast is
/// essentially never ready on the very first query but usually completes
/// within microseconds. This spins briefly (checking readiness every
/// iteration via the caller's poll loop), then yields the scheduler quantum,
/// then falls back to sleeping with increasing (capped) granularity — so a
/// short wait never sleeps at all, and a genuinely long/idle wait still
/// gives up the core instead of spinning it forever.
///
/// The "clock" is injected as an `elapsed: FnMut() -> Duration` returning
/// time-since-backoff-created, matching the convention `poll_completion`
/// already uses for its own `elapsed` parameter — this keeps the decision
/// logic (`next_step`) deterministic and testable without real sleeping.
pub struct AdaptiveBackoff<E: FnMut() -> Duration> {
    elapsed: E,
    spin_until: Duration,
    yield_until: Duration,
    max_sleep: Duration,
    current_sleep: Duration,
}

impl<E: FnMut() -> Duration> AdaptiveBackoff<E> {
    /// Default windows: spin for ~200us, then yield_now up to ~2ms total,
    /// then sleep starting at 50us doubling up to a 1ms cap. These are
    /// starting points (per the A141 fix request), not a measured optimum.
    pub fn new(elapsed: E) -> Self {
        Self::with_windows(
            elapsed,
            Duration::from_micros(200),
            Duration::from_millis(2),
            Duration::from_micros(50),
            Duration::from_millis(1),
        )
    }

    /// `spin_window`: how long (from creation) to busy-spin.
    /// `yield_total`: how long (from creation, inclusive of the spin window)
    /// to yield_now instead of sleeping. Past this, every step sleeps.
    /// `min_sleep`/`max_sleep`: first sleep duration and the cap it doubles
    /// toward.
    pub fn with_windows(
        elapsed: E,
        spin_window: Duration,
        yield_total: Duration,
        min_sleep: Duration,
        max_sleep: Duration,
    ) -> Self {
        Self {
            elapsed,
            spin_until: spin_window,
            yield_until: yield_total,
            max_sleep,
            current_sleep: min_sleep,
        }
    }

    /// Decide the next step without performing it. Pure given the injected
    /// clock, so it is exercised directly in unit tests below.
    pub fn next_step(&mut self) -> BackoffStep {
        let elapsed = (self.elapsed)();
        if elapsed < self.spin_until {
            BackoffStep::Spin
        } else if elapsed < self.yield_until {
            BackoffStep::Yield
        } else {
            let step = self.current_sleep;
            self.current_sleep = (self.current_sleep * 2).min(self.max_sleep);
            BackoffStep::Sleep(step)
        }
    }

    /// Perform one backoff step using real OS/CPU primitives. This is what
    /// production callers pass as the `pause` argument to `poll_completion`
    /// / `poll_idle_command`.
    pub fn pause(&mut self) {
        match self.next_step() {
            BackoffStep::Spin => std::hint::spin_loop(),
            BackoffStep::Yield => std::thread::yield_now(),
            BackoffStep::Sleep(d) => std::thread::sleep(d),
        }
    }
}

/// Convenience constructor: a real-time-clocked `AdaptiveBackoff` boxed as a
/// bare `FnMut()`, ready to pass directly as `poll_completion`'s or
/// `poll_idle_command`'s `pause` argument. One instance must be created per
/// call (its internal clock starts at construction time), not shared/reused
/// across separate waits.
pub fn adaptive_pause() -> impl FnMut() {
    let start = Instant::now();
    let mut backoff = AdaptiveBackoff::new(move || start.elapsed());
    move || backoff.pause()
}

pub fn poll_completion(
    timeout: Duration,
    mut elapsed: impl FnMut() -> Duration,
    mut ready: impl FnMut() -> Result<bool>,
    mut pause: impl FnMut(),
) -> Result<()> {
    loop {
        if ready()? {
            return Ok(());
        }
        if elapsed() >= timeout {
            bail!(
                "collective completion deadline exceeded after {} ms",
                timeout.as_millis()
            );
        }
        pause();
    }
}

/// Wait for the first word of a worker command while no request is active.
/// Idle time is unbounded, but the callback must still check transport errors.
/// The supervisor owns shutdown if an idle peer disappears without an error.
pub fn poll_idle_command(
    mut ready: impl FnMut() -> Result<bool>,
    mut pause: impl FnMut(),
) -> Result<()> {
    loop {
        if ready()? {
            return Ok(());
        }
        pause();
    }
}

/// A completion failure prevents subsequent submissions on this communicator.
pub fn poison_on_error(result: Result<()>, unhealthy: &AtomicBool) -> Result<()> {
    if result.is_err() {
        unhealthy.store(true, Ordering::Release);
    }
    result
}

pub fn ensure_healthy(unhealthy: &AtomicBool, rank: usize, world: usize, op: &str) -> Result<()> {
    anyhow::ensure!(
        !unhealthy.load(Ordering::Acquire),
        "NCCL rank={rank} world_size={world} op={op}: communicator unhealthy; stop all ranks before retrying"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn a_peer_that_never_completes_returns_at_the_deadline() {
        let ticks = Cell::new(0);
        let err = poll_completion(
            Duration::from_millis(3),
            || Duration::from_millis(ticks.get()),
            || Ok(false),
            || ticks.set(ticks.get() + 1),
        )
        .unwrap_err();
        assert_eq!(ticks.get(), 3);
        assert!(err.to_string().contains("deadline exceeded"));
    }

    #[test]
    fn idle_command_can_arrive_after_long_idle_but_payload_still_times_out() {
        let ticks = Cell::new(0);
        poll_idle_command(|| Ok(ticks.get() == 90), || ticks.set(ticks.get() + 1)).unwrap();
        assert_eq!(ticks.get(), 90);
        ticks.set(0);
        assert!(
            poll_completion(
                Duration::from_secs(30),
                || Duration::from_secs(ticks.get()),
                || Ok(ticks.get() == 90),
                || ticks.set(ticks.get() + 1),
            )
            .is_err()
        );
        assert_eq!(ticks.get(), 30);
    }

    #[test]
    fn idle_command_still_checks_errors_and_poisons_the_communicator() {
        let ticks = Cell::new(0);
        let unhealthy = AtomicBool::new(false);
        let result = poll_idle_command(
            || {
                if ticks.get() == 90 {
                    anyhow::bail!("peer lost during idle");
                }
                Ok(false)
            },
            || ticks.set(ticks.get() + 1),
        );
        assert!(poison_on_error(result, &unhealthy).is_err());
        assert!(ensure_healthy(&unhealthy, 1, 2, "broadcast").is_err());
        assert_eq!(ticks.get(), 90);
    }

    #[test]
    fn completion_failure_poison_blocks_later_submissions() {
        let unhealthy = AtomicBool::new(false);
        assert!(poison_on_error(Ok(()), &unhealthy).is_ok());
        assert!(ensure_healthy(&unhealthy, 3, 8, "all_reduce").is_ok());
        let failure = poll_completion(
            Duration::ZERO,
            || Duration::ZERO,
            || Ok(false),
            || panic!("deadline already reached"),
        );
        assert!(poison_on_error(failure, &unhealthy).is_err());
        let msg = ensure_healthy(&unhealthy, 3, 8, "all_reduce")
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("rank=3") && msg.contains("op=all_reduce") && msg.contains("unhealthy")
        );
        // A later success cannot silently clear the poison.
        assert!(poison_on_error(Ok(()), &unhealthy).is_ok());
        assert!(unhealthy.load(Ordering::Acquire));
    }

    #[test]
    fn completion_and_driver_errors_do_not_keep_polling() {
        assert!(
            poll_completion(
                Duration::ZERO,
                || Duration::ZERO,
                || Ok(true),
                || panic!("already complete")
            )
            .is_ok()
        );
        let err = poll_completion(
            Duration::from_secs(30),
            || Duration::ZERO,
            || anyhow::bail!("peer lost"),
            || panic!("already failed"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("peer lost"));
    }

    // --- AdaptiveBackoff (A141) ---

    #[test]
    fn ready_immediately_never_invokes_the_backoff() {
        // poll_completion checks `ready()` before ever calling `pause()`, so
        // a wait that is already satisfied must never spin, yield, or sleep.
        let pause_calls = Cell::new(0u32);
        poll_completion(
            Duration::from_secs(1),
            || Duration::ZERO,
            || Ok(true),
            || pause_calls.set(pause_calls.get() + 1),
        )
        .unwrap();
        assert_eq!(pause_calls.get(), 0);
    }

    #[test]
    fn steps_within_the_spin_window_never_sleep_or_yield() {
        let micros = Cell::new(0u64);
        let mut backoff = AdaptiveBackoff::with_windows(
            || Duration::from_micros(micros.get()),
            Duration::from_micros(200),
            Duration::from_millis(2),
            Duration::from_micros(50),
            Duration::from_millis(1),
        );
        // 20 polls, each 5us apart: still well under the 200us spin window.
        for i in 0..20u64 {
            micros.set(i * 5);
            assert_eq!(
                backoff.next_step(),
                BackoffStep::Spin,
                "poll {i} at {}us should still be spinning",
                i * 5
            );
        }
    }

    #[test]
    fn timeout_still_fires_through_an_adaptive_backoff() {
        // The backoff's own clock is independent of poll_completion's
        // deadline clock in production (both read the real Instant, but the
        // contract is that pause() never overrides poll_completion's
        // deadline check). Give the backoff a huge spin window so pause()
        // stays a cheap spin_loop() and the test runs instantly.
        let ticks = Cell::new(0u64);
        let mut backoff = AdaptiveBackoff::with_windows(
            || Duration::from_micros(ticks.get()),
            Duration::from_secs(1),
            Duration::from_secs(1),
            Duration::from_micros(50),
            Duration::from_millis(1),
        );
        let err = poll_completion(
            Duration::from_millis(3),
            || {
                let v = ticks.get();
                ticks.set(v + 1);
                Duration::from_millis(v)
            },
            || Ok(false),
            || backoff.pause(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("deadline exceeded"));
        // `elapsed()` (which also drives `ticks`) is called once more than
        // `pause()` on the final iteration: it observes 3ms and bails before
        // `pause()` runs, so the last increment (to 4) is never paired with
        // a pause.
        assert_eq!(ticks.get(), 4);
    }

    #[test]
    fn falls_back_to_capped_sleep_after_the_window_and_never_reverts() {
        let micros = Cell::new(0u64);
        let mut backoff = AdaptiveBackoff::with_windows(
            || Duration::from_micros(micros.get()),
            Duration::from_micros(200),
            Duration::from_micros(500),
            Duration::from_micros(50),
            Duration::from_micros(200),
        );
        micros.set(50);
        assert_eq!(backoff.next_step(), BackoffStep::Spin);
        micros.set(300);
        assert_eq!(backoff.next_step(), BackoffStep::Yield);
        // Past the yield window: sleeps, starting at min_sleep and doubling.
        micros.set(600);
        assert_eq!(
            backoff.next_step(),
            BackoffStep::Sleep(Duration::from_micros(50))
        );
        micros.set(700);
        assert_eq!(
            backoff.next_step(),
            BackoffStep::Sleep(Duration::from_micros(100))
        );
        micros.set(800);
        assert_eq!(
            backoff.next_step(),
            BackoffStep::Sleep(Duration::from_micros(200)),
            "should be capped at max_sleep"
        );
        micros.set(900);
        assert_eq!(
            backoff.next_step(),
            BackoffStep::Sleep(Duration::from_micros(200)),
            "must stay capped, never revert to Spin/Yield, once past the window"
        );
    }

    #[test]
    fn idle_command_wired_to_the_backoff_still_completes() {
        // End-to-end: poll_idle_command driven by a real AdaptiveBackoff
        // (real spin_loop/yield_now/sleep primitives via `pause()`), proving
        // the wiring works and the idle wait still resolves once ready.
        // Windows are kept tiny so this test stays fast.
        let micros = Cell::new(0u64);
        let mut backoff = AdaptiveBackoff::with_windows(
            || Duration::from_micros(micros.get()),
            Duration::from_micros(10),
            Duration::from_micros(20),
            Duration::from_micros(1),
            Duration::from_micros(5),
        );
        let polls = Cell::new(0u32);
        poll_idle_command(
            || {
                let n = polls.get();
                polls.set(n + 1);
                micros.set(micros.get() + 5);
                Ok(n == 10)
            },
            || backoff.pause(),
        )
        .unwrap();
        assert_eq!(polls.get(), 11);
    }

    #[test]
    fn adaptive_pause_constructs_with_default_windows_and_does_not_panic() {
        // Exercises the real-clock convenience constructor production code
        // uses (adaptive_pause -> AdaptiveBackoff::new). Called once right
        // after construction, elapsed is ~0 so this is just a spin_loop().
        let mut pause = adaptive_pause();
        pause();
    }
}
