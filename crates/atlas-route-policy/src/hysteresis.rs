// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: swap hysteresis. The swap history the caller keeps, and the two limits every swap obeys: a
//! minimum dwell time after the last swap (default 10 minutes) and a cap on swaps inside any rolling hour. The
//! caller passes `now`; nothing here reads a clock.

use std::time::{Duration, Instant};

use crate::swap::{SwapDenial, SwapPolicy};

/// The rolling window of the hourly cap.
pub const HOUR: Duration = Duration::from_secs(3600);

/// When past swaps happened. Entries older than an hour (relative to the newest record) are pruned on record;
/// the newest swap is always kept, so the dwell check works for any dwell.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SwapHistory {
    swaps: Vec<Instant>,
}

impl SwapHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// A history holding exactly these swap times (for restoring caller state and for tests).
    pub fn from_swaps(swaps: impl IntoIterator<Item = Instant>) -> Self {
        Self {
            swaps: swaps.into_iter().collect(),
        }
    }

    /// Records a swap performed at `now` and prunes swaps more than an hour before it.
    pub fn record(&mut self, now: Instant) {
        self.swaps.push(now);
        self.swaps
            .retain(|t| now.saturating_duration_since(*t) < HOUR);
    }

    /// The most recent swap.
    pub fn last(&self) -> Option<Instant> {
        self.swaps.iter().copied().max()
    }

    /// Swaps less than `window` before `now` (swaps at or after `now` count too).
    pub fn count_within(&self, now: Instant, window: Duration) -> usize {
        self.swaps
            .iter()
            .filter(|t| now.saturating_duration_since(**t) < window)
            .count()
    }

    pub fn swaps(&self) -> &[Instant] {
        &self.swaps
    }

    pub fn is_empty(&self) -> bool {
        self.swaps.is_empty()
    }

    pub fn len(&self) -> usize {
        self.swaps.len()
    }
}

/// Whether a swap at `now` respects the dwell time and the hourly cap. A clock that went backwards reads as
/// "inside the dwell", which denies (fails closed).
pub fn check(history: &SwapHistory, policy: &SwapPolicy, now: Instant) -> Result<(), SwapDenial> {
    let in_dwell = history
        .last()
        .is_some_and(|last| now.saturating_duration_since(last) < policy.dwell);
    if in_dwell {
        return Err(SwapDenial::Dwell);
    }
    if history.count_within(now, HOUR) >= policy.hourly_cap as usize {
        return Err(SwapDenial::HourlyCap);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SwapPolicy {
        SwapPolicy {
            allow: true,
            hourly_cap: 3,
            ..SwapPolicy::default()
        }
    }

    #[test]
    fn dwell_blocks_until_it_has_passed() {
        let t0 = Instant::now();
        let mut h = SwapHistory::new();
        assert_eq!(check(&h, &policy(), t0), Ok(()));
        h.record(t0);
        let dwell = policy().dwell;
        assert_eq!(
            check(&h, &policy(), t0 + dwell - Duration::from_secs(1)),
            Err(SwapDenial::Dwell)
        );
        assert_eq!(check(&h, &policy(), t0 + dwell), Ok(()));
    }

    #[test]
    fn hourly_cap_counts_a_rolling_hour() {
        let t0 = Instant::now();
        let step = Duration::from_secs(15 * 60);
        let mut h = SwapHistory::new();
        for i in 0..3 {
            h.record(t0 + step * i);
        }
        // Three swaps at 0, 15 and 30 minutes: at 45 minutes the cap of 3 is reached.
        assert_eq!(
            check(&h, &policy(), t0 + step * 3),
            Err(SwapDenial::HourlyCap)
        );
        // At 60 minutes the first swap is an hour old and no longer counts.
        assert_eq!(check(&h, &policy(), t0 + step * 4), Ok(()));
    }

    #[test]
    fn record_prunes_old_entries_but_keeps_the_newest() {
        let t0 = Instant::now();
        let mut h = SwapHistory::new();
        h.record(t0);
        h.record(t0 + Duration::from_secs(2 * 3600));
        assert_eq!(h.len(), 1);
        assert_eq!(h.last(), Some(t0 + Duration::from_secs(2 * 3600)));
    }

    #[test]
    fn backwards_clock_fails_closed() {
        let t0 = Instant::now() + Duration::from_secs(3600);
        let h = SwapHistory::from_swaps([t0]);
        assert_eq!(
            check(&h, &policy(), t0 - Duration::from_secs(30 * 60)),
            Err(SwapDenial::Dwell)
        );
    }
}
