// SPDX-License-Identifier: AGPL-3.0-only

//! Background thermal/throttle sampling for the Stats section.
//!
//! # Why a thread
//!
//! The throttle counters come from `nvidia-smi -q -d PERFORMANCE`, which costs
//! 100-200 ms to spawn and parse. The TUI renders every 100 ms and samples
//! metrics every 1 s, so doing this inline would drop frames on every sample —
//! a stutter caused by the monitoring, in the tool people watch to decide
//! whether the machine is healthy. One detached thread samples on its own
//! cadence and publishes the latest snapshot; the render thread only ever reads
//! a mutex it never blocks on for long.
//!
//! # Why 2 s
//!
//! The counters are cumulative microseconds, so a longer window loses nothing —
//! it just averages over more time. 2 s keeps the `nvidia-smi` spawn rate low
//! while still resolving a throttle episode that lasts a few seconds, which is
//! the timescale a person watching the TUI can act on.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use atlas_plugin::hardware::throttle_monitor::{ThrottleMonitor, ThrottleWindow};

/// How often the background thread re-reads the counters.
const SAMPLE_EVERY: Duration = Duration::from_secs(2);

/// What the renderer reads. `None` fields mean "not known", never "fine" —
/// a thermal indicator that shows healthy when it has no data is worse than
/// one that shows nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThermalSnapshot {
    pub window: Option<ThrottleWindow>,
    /// Flips of the thermal state across the retained history.
    pub transitions: usize,
    /// Denominator for `transitions` — 0 transitions over 1 sample says nothing.
    pub samples: usize,
    pub gpu_temp_c: Option<f64>,
    pub sm_clock_mhz: Option<f64>,
    pub sm_clock_max_mhz: Option<f64>,
    /// Set once the first sample lands, so the UI can distinguish "starting up"
    /// from "this box reports no thermal data at all".
    pub have_data: bool,
}

impl ThermalSnapshot {
    /// Clocks as a fraction of the part's maximum, when both are known.
    pub fn clock_frac(&self) -> Option<f64> {
        match (self.sm_clock_mhz, self.sm_clock_max_mhz) {
            (Some(now), Some(max)) if max > 0.0 => Some((now / max).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

/// Shared handle. Cloning shares the same snapshot.
#[derive(Debug, Clone, Default)]
pub struct ThermalProbe {
    inner: Arc<Mutex<ThermalSnapshot>>,
}

impl ThermalProbe {
    /// Start the sampler. Returns immediately; the thread is detached and ends
    /// with the process.
    pub fn spawn() -> Self {
        let probe = Self::default();
        let sink = Arc::clone(&probe.inner);
        std::thread::Builder::new()
            .name("atlas-thermal".into())
            .spawn(move || {
                let mut monitor = ThrottleMonitor::new();
                loop {
                    let started = Instant::now();
                    let state = atlas_plugin::hardware::collect::collect();
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    let window = monitor.observe(now_ms, state.throttle_counters);
                    if let Ok(mut slot) = sink.lock() {
                        // Keep the previous window when this read produced none
                        // (the first sample, or a counter reset): blanking the
                        // panel on every driver hiccup reads as "recovered".
                        if window.is_some() {
                            slot.window = window;
                        }
                        slot.transitions = monitor.transitions();
                        slot.samples = monitor.samples();
                        slot.gpu_temp_c = state.gpu_temp_c;
                        slot.sm_clock_mhz = state.sm_clock_mhz;
                        slot.sm_clock_max_mhz = state.sm_clock_max_mhz;
                        slot.have_data = true;
                    }
                    std::thread::sleep(SAMPLE_EVERY.saturating_sub(started.elapsed()));
                }
            })
            .ok();
        probe
    }

    /// Latest snapshot. Returns the default (all-unknown) if the sampler thread
    /// panicked and poisoned the mutex — the TUI must not die with it.
    pub fn snapshot(&self) -> ThermalSnapshot {
        self.inner
            .lock()
            .map(|s| *s)
            .unwrap_or_else(|p| *p.into_inner())
    }
}

/// Severity the header indicator reflects.
///
/// Ordered by urgency so a caller can compare. `Ok` deliberately means "we have
/// data and it is fine", never "we have no data" — those are different, and the
/// header must not show a green state for a box it cannot read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalAlert {
    /// No usable reading yet, or this box reports no throttle counters.
    Unknown,
    Ok,
    /// Clocks are being held down, steadily.
    Throttling,
    /// Clocks are oscillating in and out of throttle. Distinct from
    /// `Throttling` because it damages latency VARIANCE rather than throughput:
    /// p90 drifts away from median while the median still looks healthy.
    Thrashing,
}

/// Fraction of a window under thermal throttle before the header speaks up.
///
/// 0.20 rather than something tighter: brief thermal excursions are normal on a
/// power-limited part under load, and an indicator that lights on every one is
/// an indicator people learn to ignore.
pub const THROTTLE_WARN_FRAC: f64 = 0.20;

/// Thermal-state flips, within the retained history, that count as thrashing.
///
/// The monitor keeps 32 samples at 2 s each, so this is 6 flips in ~64 s. One
/// excursion produces two transitions (in, then out); six means the clocks are
/// genuinely unsettled rather than responding to a single burst of work.
pub const THRASH_TRANSITIONS: usize = 6;

/// Minimum history before thrashing can be claimed at all.
///
/// Without it, the first few samples after startup can trip the transition
/// count from a cold-to-loaded ramp, and a false alarm in the first seconds is
/// exactly what makes an indicator untrustworthy.
pub const THRASH_MIN_SAMPLES: usize = 8;

impl ThermalSnapshot {
    /// What the header should show.
    pub fn alert(&self) -> ThermalAlert {
        let Some(w) = self.window else {
            return ThermalAlert::Unknown;
        };
        let Some(frac) = w.thermal_frac else {
            return ThermalAlert::Unknown;
        };
        if self.samples >= THRASH_MIN_SAMPLES && self.transitions >= THRASH_TRANSITIONS {
            return ThermalAlert::Thrashing;
        }
        if frac >= THROTTLE_WARN_FRAC {
            return ThermalAlert::Throttling;
        }
        ThermalAlert::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(frac: Option<f64>, transitions: usize, samples: usize) -> ThermalSnapshot {
        ThermalSnapshot {
            window: Some(ThrottleWindow {
                window_ms: 2_000,
                thermal_frac: frac,
                power_cap_frac: Some(1.0),
                thermal_active: frac.is_some_and(|f| f > 0.0),
            }),
            transitions,
            samples,
            have_data: true,
            ..Default::default()
        }
    }

    /// "No data" must never render as healthy — the whole point of the
    /// indicator is that someone can trust a calm header.
    #[test]
    fn absent_data_is_unknown_not_ok() {
        assert_eq!(ThermalSnapshot::default().alert(), ThermalAlert::Unknown);
        assert_eq!(snap(None, 0, 20).alert(), ThermalAlert::Unknown);
    }

    #[test]
    fn a_quiet_box_is_ok() {
        assert_eq!(snap(Some(0.01), 0, 20).alert(), ThermalAlert::Ok);
    }

    #[test]
    fn sustained_throttling_warns() {
        assert_eq!(snap(Some(0.55), 0, 20).alert(), ThermalAlert::Throttling);
    }

    /// Thrashing outranks throttling: unstable clocks are the worse diagnosis
    /// even when the fraction is modest, because the damage lands in p90.
    #[test]
    fn flapping_outranks_a_steady_hold() {
        let a = snap(Some(0.05), THRASH_TRANSITIONS, 20).alert();
        assert_eq!(a, ThermalAlert::Thrashing);
        assert!(ThermalAlert::Thrashing > ThermalAlert::Throttling);
    }

    /// A cold-to-loaded ramp in the first seconds can flip the state a few
    /// times; claiming "thrashing" then teaches people to ignore the header.
    #[test]
    fn thrashing_needs_enough_history_to_be_believable() {
        assert_ne!(
            snap(Some(0.05), THRASH_TRANSITIONS, THRASH_MIN_SAMPLES - 1).alert(),
            ThermalAlert::Thrashing
        );
    }

    /// SW power cap runs at ~100% on a healthy GB10 and must not colour the
    /// verdict — see `throttle_monitor`'s module docs.
    #[test]
    fn a_pinned_power_cap_alone_is_still_ok() {
        assert_eq!(snap(Some(0.0), 0, 20).alert(), ThermalAlert::Ok);
    }
}
