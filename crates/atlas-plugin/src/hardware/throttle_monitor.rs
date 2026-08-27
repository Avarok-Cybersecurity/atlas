// SPDX-License-Identifier: AGPL-3.0-only

//! How hard the thermal/power throttles are actually working, over time.
//!
//! [`ThrottleCounters`] are CUMULATIVE microseconds since driver load. A single
//! reading answers "how much has this box ever throttled", which is nearly
//! useless live: a box that spent an hour throttling last week looks identical
//! to one throttling right now. Differencing two readings answers the question
//! people actually ask — *what fraction of the last few seconds did the clocks
//! spend held down* — and that is what this module produces.
//!
//! # Thrashing, specifically
//!
//! Sustained throttling and THRASHING are different failures and want different
//! responses. A part pinned at 100% thermal throttle is simply too hot: it runs
//! slower, predictably. A part oscillating in and out of throttle many times a
//! second has unstable clocks, and the damage shows up as latency VARIANCE —
//! p90 TTFT drifting away from median while the median looks fine. So
//! [`ThrottleWindow`] reports both the fraction and the transition count.
//!
//! # SW power cap is NOT throttling here
//!
//! `state::ThrottleActive::thermal` already documents why, from a measurement on
//! this hardware: SW power cap is asserted for 16,130 s of an 11.2-day uptime on
//! a HEALTHY GB10. It is the normal steady state of a power-limited part. Rolling
//! it into a "throttling" number would light the indicator permanently and train
//! everyone to ignore it. It is reported in its own field, never in
//! `thermal_frac`.

use super::state::ThrottleCounters;

/// One differenced observation of the throttle counters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrottleWindow {
    /// Wall time the window covers.
    pub window_ms: u64,
    /// Fraction of the window (0.0..=1.0) the clocks were held by a THERMAL
    /// reason — SW thermal, HW thermal, or HW power brake.
    ///
    /// This is a LOWER BOUND: the three counters advance independently and can
    /// overlap, so the true union is somewhere between the largest single
    /// counter and their sum. The max is reported because an over-reported
    /// throttle figure is the kind that gets a real regression dismissed as "the
    /// box was hot", and this number exists to be trusted.
    pub thermal_frac: Option<f64>,
    /// Fraction of the window spent under the SW power cap, reported SEPARATELY
    /// because on this hardware it is the normal steady state — see module docs.
    pub power_cap_frac: Option<f64>,
    /// True when a thermal reason advanced at all in this window.
    pub thermal_active: bool,
}

/// Rolling detector over successive counter reads.
///
/// Holds the previous sample; each [`observe`](Self::observe) returns the window
/// between the two. The first call after construction returns `None` — one
/// sample cannot be differenced, and inventing a zero would read as "no
/// throttling" rather than "not known yet".
#[derive(Debug, Default, Clone)]
pub struct ThrottleMonitor {
    prev: Option<(u64, ThrottleCounters)>,
    /// Recent thermal-active flags, oldest first, for the transition count.
    history: Vec<bool>,
}

/// How many samples of thermal on/off history to keep for thrash detection.
const HISTORY: usize = 32;

impl ThrottleMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Difference `counters` (read at `now_ms`) against the previous sample.
    ///
    /// Returns `None` on the first sample, on a non-advancing clock, and on any
    /// counter that went BACKWARDS — the latter means the driver reloaded or the
    /// GPU was reset, and carrying the stale baseline across that would report a
    /// wildly negative or saturated fraction. Fail to "unknown", not to a
    /// plausible wrong number.
    pub fn observe(&mut self, now_ms: u64, counters: ThrottleCounters) -> Option<ThrottleWindow> {
        let out = match &self.prev {
            None => None,
            Some((prev_ms, prev)) => {
                let window_ms = now_ms.checked_sub(*prev_ms).filter(|d| *d > 0)?;
                // `Ok(None)`  = this counter was not reported at all.
                // `Ok(Some)`   = a usable delta.
                // `Err(())`    = it went BACKWARDS: driver reload or GPU reset.
                //
                // The three cases must stay distinct. Collapsing "went
                // backwards" into `None` and then taking a max over the
                // survivors reports Some(0.0) — "not throttling" — from a reset,
                // which is the most misleading answer available.
                type Delta = Result<Option<u64>, ()>;
                let delta = |now: Option<u64>, before: Option<u64>| -> Delta {
                    match (now, before) {
                        (Some(n), Some(b)) => n.checked_sub(b).map(Some).ok_or(()),
                        _ => Ok(None),
                    }
                };
                let window_us = (window_ms as f64) * 1000.0;
                let frac = |us: Option<u64>| us.map(|us| (us as f64 / window_us).clamp(0.0, 1.0));

                let sw_t = delta(counters.sw_thermal_us, prev.sw_thermal_us);
                let hw_t = delta(counters.hw_thermal_us, prev.hw_thermal_us);
                let brake = delta(counters.hw_power_brake_us, prev.hw_power_brake_us);
                let cap = delta(counters.sw_power_cap_us, prev.sw_power_cap_us);

                // Any thermal counter going backwards makes the whole thermal
                // reading unknown — see the `delta` note above.
                let thermal_us: Option<u64> = match (sw_t, hw_t, brake) {
                    (Err(()), _, _) | (_, Err(()), _) | (_, _, Err(())) => None,
                    (Ok(a), Ok(b), Ok(c)) => {
                        // Max, not sum — the three overlap and a sum over-reports.
                        [a, b, c].into_iter().flatten().max()
                    }
                };
                let thermal_reset = matches!((sw_t, hw_t, brake), (Err(()), _, _))
                    || matches!((sw_t, hw_t, brake), (_, Err(()), _))
                    || matches!((sw_t, hw_t, brake), (_, _, Err(())));
                Some(ThrottleWindow {
                    window_ms,
                    thermal_frac: if thermal_reset {
                        None
                    } else {
                        frac(thermal_us)
                    },
                    power_cap_frac: frac(cap.unwrap_or(None)),
                    thermal_active: !thermal_reset && thermal_us.is_some_and(|us| us > 0),
                })
            }
        };
        if let Some(w) = out {
            self.history.push(w.thermal_active);
            if self.history.len() > HISTORY {
                self.history.remove(0);
            }
        }
        self.prev = Some((now_ms, counters));
        out
    }

    /// Times the thermal state flipped across the retained history.
    ///
    /// This is the THRASH signal, and it is deliberately separate from the
    /// fraction: a part pinned at 100% throttle has a high fraction and ZERO
    /// transitions, and is a different problem from one flapping every sample.
    /// Flapping is what turns up as p90 latency drifting away from median.
    pub fn transitions(&self) -> usize {
        self.history.windows(2).filter(|w| w[0] != w[1]).count()
    }

    /// Samples retained for [`transitions`](Self::transitions), so a caller can
    /// tell "0 transitions over 30 samples" from "0 transitions over 1".
    pub fn samples(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(sw_t: u64, hw_t: u64, brake: u64, cap: u64) -> ThrottleCounters {
        ThrottleCounters {
            sw_thermal_us: Some(sw_t),
            hw_thermal_us: Some(hw_t),
            hw_power_brake_us: Some(brake),
            sw_power_cap_us: Some(cap),
            sync_boost_us: None,
        }
    }

    /// One sample cannot be differenced. Reporting 0.0 would read as "not
    /// throttling" when the truth is "not known yet", and that is the reading
    /// someone would quote to dismiss a regression.
    #[test]
    fn the_first_sample_is_unknown_not_zero() {
        let mut m = ThrottleMonitor::new();
        assert_eq!(m.observe(1_000, counters(0, 0, 0, 0)), None);
    }

    #[test]
    fn half_a_window_of_hw_thermal_reads_as_half() {
        let mut m = ThrottleMonitor::new();
        m.observe(1_000, counters(0, 0, 0, 0));
        let w = m.observe(2_000, counters(0, 500_000, 0, 0)).unwrap();
        assert_eq!(w.window_ms, 1_000);
        assert_eq!(w.thermal_frac, Some(0.5));
        assert!(w.thermal_active);
    }

    /// The three thermal counters advance independently and can overlap, so the
    /// union is between max and sum. Max is reported: over-reporting a throttle
    /// is what gets a real regression waved away as "the box was hot".
    #[test]
    fn overlapping_thermal_reasons_take_the_max_not_the_sum() {
        let mut m = ThrottleMonitor::new();
        m.observe(0, counters(0, 0, 0, 0));
        let w = m
            .observe(1_000, counters(400_000, 300_000, 200_000, 0))
            .unwrap();
        // Sum would be 0.9 and exceed the real union; max is the honest bound.
        assert_eq!(w.thermal_frac, Some(0.4));
    }

    /// SW power cap is the normal steady state on this hardware (16,130 s of an
    /// 11.2-day uptime on a HEALTHY GB10). Folding it into the thermal number
    /// would pin the indicator on and train everyone to ignore it.
    #[test]
    fn sw_power_cap_never_counts_as_thermal() {
        let mut m = ThrottleMonitor::new();
        m.observe(0, counters(0, 0, 0, 0));
        let w = m.observe(1_000, counters(0, 0, 0, 1_000_000)).unwrap();
        assert_eq!(w.thermal_frac, Some(0.0));
        assert!(!w.thermal_active);
        assert_eq!(w.power_cap_frac, Some(1.0));
    }

    /// A driver reload or GPU reset resets the counters. Carrying the stale
    /// baseline across that would underflow into a saturated fraction.
    #[test]
    fn counters_going_backwards_report_unknown_rather_than_garbage() {
        let mut m = ThrottleMonitor::new();
        m.observe(0, counters(0, 900_000, 0, 0));
        let w = m.observe(1_000, counters(0, 10_000, 0, 0)).unwrap();
        assert_eq!(w.thermal_frac, None, "must not saturate on a counter reset");
        assert!(!w.thermal_active);
    }

    #[test]
    fn a_non_advancing_clock_yields_no_window() {
        let mut m = ThrottleMonitor::new();
        m.observe(5_000, counters(0, 0, 0, 0));
        assert_eq!(m.observe(5_000, counters(0, 100, 0, 0)), None);
    }

    /// THE THRASH SIGNAL. A part pinned at full throttle and a part flapping
    /// both look bad by fraction, but only the second has unstable clocks — and
    /// that is the one that shows up as p90 latency drifting from median.
    #[test]
    fn pinned_throttling_and_flapping_are_told_apart() {
        let mut pinned = ThrottleMonitor::new();
        let mut flapping = ThrottleMonitor::new();
        let (mut p_us, mut f_us) = (0u64, 0u64);
        for i in 0..10u64 {
            p_us += 1_000_000; // always throttled
            pinned.observe(i * 1_000, counters(0, p_us, 0, 0));
            if i % 2 == 0 {
                f_us += 1_000_000; // on/off/on/off
            }
            flapping.observe(i * 1_000, counters(0, f_us, 0, 0));
        }
        assert_eq!(pinned.transitions(), 0, "pinned throttling does not flap");
        assert!(
            flapping.transitions() >= 6,
            "flapping must register transitions, got {}",
            flapping.transitions()
        );
    }

    /// `transitions()` alone is ambiguous without the denominator: zero over one
    /// sample means nothing, zero over thirty means stable.
    #[test]
    fn sample_count_is_exposed_so_zero_transitions_is_interpretable() {
        let mut m = ThrottleMonitor::new();
        assert_eq!(m.samples(), 0);
        m.observe(0, counters(0, 0, 0, 0));
        assert_eq!(
            m.samples(),
            0,
            "the undifferenced first sample is not history"
        );
        m.observe(1_000, counters(0, 0, 0, 0));
        assert_eq!(m.samples(), 1);
    }
}
