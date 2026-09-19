// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded polling for synchronous completion. Does not bound an NCCL submit
//! call or a wedged driver call; process supervision still owns job teardown.
use anyhow::{Result, bail};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

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
}
