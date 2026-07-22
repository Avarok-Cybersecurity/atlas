// SPDX-License-Identifier: AGPL-3.0-only

//! Throughput-arbitrated MTP runtime gate.
//!
//! Chooses between MTP speculative decode and plain serial decode by
//! comparing DELIVERED throughput (emitted tokens / wall) measured over
//! whole step windows in each mode — never by comparing component step
//! timings. The previous gate compared `verify_wall / decode_wall` against
//! expected accepted tokens; on the 35B MoE that arithmetic disabled MTP
//! (multiplier 2.07–2.23 ≥ effective 1.75–2.0) while an always-on control
//! measured 18% FASTER end-to-end decode (webserver_ok A/B, 2026-07-20:
//! Σ1028s/10-10 always-on vs Σ1846s/9-10 gated). Component walls miss
//! per-token costs outside the timed step and amortization effects, so the
//! arbiter now measures exactly the quantity being optimized.
//!
//! Policy (bandit-style greedy with scheduled exploration — cf. TapOut
//! arXiv:2511.02017, GammaTune-style hysteresis):
//! - Run the currently-faster mode; accumulate (tokens, wall) into a
//!   fixed-size step window; on window close update that mode's tok/s EWMA
//!   and a deviation EWMA.
//! - Switch modes only when the other mode's EWMA is faster by more than a
//!   noise margin (hysteresis) for [`SWITCH_DWELL_WINDOWS`] consecutive
//!   windows — the old gate flipped ENABLED→DISABLED within 6s on
//!   measurement noise (multiplier 1.35→1.78), each flip costing a
//!   draft-head resync.
//! - While in Serial, re-probe MTP after [`reprobe_tokens`] emitted tokens.
//!   While in Mtp, refresh the serial baseline after
//!   [`serial_refresh_tokens`] (one window ≈ ≤0.3% overhead bound).
//! - A depth-regime change (factor [`REMEASURE_DEPTH_FACTOR`]) marks the
//!   OTHER mode's baseline stale and schedules a refresh probe instead of
//!   wiping all state.
//!
//! `ATLAS_MTP_GATE_FORCE=1` (existing) bypasses the gate entirely.

use std::time::Duration;

/// Depth factor that marks baselines stale (economics are depth-dependent:
/// weight-bound at short context vs KV/SSM-bound at depth).
const REMEASURE_DEPTH_FACTOR: usize = 2;
/// Floor for the regime comparison (below this all contexts are "shallow").
const REMEASURE_DEPTH_FLOOR: usize = 512;
/// Steps per throughput window. 16 ≥ the 12-step acceptance window the
/// proven `adaptive_spec` suspend policy uses, and long enough that one
/// window amortizes bootstrap/propose transients (≥16 serial tokens,
/// ~28 MTP tokens at the measured 0.75 acceptance).
const WINDOW_STEPS: usize = 16;
/// Consecutive out-of-margin windows required before switching mode.
const SWITCH_DWELL_WINDOWS: usize = 2;
/// EWMA smoothing for per-mode tok/s (responds within ~3 windows).
const TPS_ALPHA: f64 = 0.3;
/// Relative noise floor for the switch margin. Derived from the observed
/// window-to-window jitter of the step walls this gate consumes (the old
/// gate's multiplier swung 1.35→1.78 ≈ ±14% within seconds; half the
/// deviation EWMA is added on top of this floor).
const MARGIN_REL_FLOOR: f64 = 0.05;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Serial tokens between MTP re-probes while in Serial mode. Default
/// matches the proven `ATLAS_DFLASH_ADAPTIVE_REPROBE` policy (256).
fn reprobe_tokens() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| env_usize("ATLAS_MTP_GATE_REPROBE", 256))
}

/// MTP tokens between serial-baseline refreshes while in Mtp mode. One
/// 16-step window per 1024 tokens bounds refresh overhead at ≤0.3% even if
/// serial were 18% slower.
fn serial_refresh_tokens() -> usize {
    static C: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *C.get_or_init(|| env_usize("ATLAS_MTP_GATE_REFRESH", 1024))
}

/// What the gate wants the scheduler to run for the NEXT step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStep {
    /// Plain single-token decode step (Serial mode or baseline refresh).
    MeasureDecode,
    /// MTP verify step (Mtp mode or re-probe).
    MeasureVerify,
}

/// Mode-transition signal for the scheduler's one-time bookkeeping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Switched to Mtp: nothing to do (bootstrap happens naturally).
    KeepMtp,
    /// Switched to Serial: clear pending drafts + draft-head resync.
    DisableMtp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Mtp,
    Serial,
}

#[derive(Default)]
struct ModeStats {
    /// Delivered-throughput EWMA (tokens/sec), `None` until first window.
    tps: Option<f64>,
    /// EWMA of |window tps − tps| (deviation, for the noise margin).
    dev: f64,
    /// Stale after a depth-regime change; refreshed by the next probe.
    stale: bool,
}

impl ModeStats {
    /// Fold one closed window into the estimate. `replace=true` (probe
    /// windows and post-regime-change windows) REPLACES the estimate: a
    /// sparse probe is a fresh look at a baseline that may have drifted
    /// arbitrarily since it was last run, and blending it against the stale
    /// value both lags the estimate and pollutes `dev` with the shift
    /// magnitude (inflating the hysteresis margin and delaying recovery).
    /// Continuous same-mode windows blend (EWMA) so `dev` tracks
    /// steady-state noise only.
    fn update(&mut self, window_tps: f64, replace: bool) {
        match (self.tps, replace) {
            (None, _) | (_, true) => {
                self.tps = Some(window_tps);
                self.dev *= 0.5; // decay: fresh baseline, keep a noise memory
            }
            (Some(prev), false) => {
                let next = (1.0 - TPS_ALPHA) * prev + TPS_ALPHA * window_tps;
                self.dev = (1.0 - TPS_ALPHA) * self.dev + TPS_ALPHA * (window_tps - next).abs();
                self.tps = Some(next);
            }
        }
        self.stale = false;
    }
}

/// Per-serve, single-instance gate. Lives on the scheduler thread; every
/// single-sequence decode/verify step is timed and reported, so arbitration
/// runs continuously with zero dedicated measurement phases.
pub struct MtpGate {
    mode: Mode,
    /// True while the gate is running a short window of the OTHER mode
    /// (re-probe from Serial, baseline refresh from Mtp).
    probing: bool,
    /// Windows remaining in the current probe.
    probe_windows_left: usize,
    mtp: ModeStats,
    serial: ModeStats,
    // Current-window accumulators (for whichever mode the steps ran in).
    win_tokens: f64,
    win_wall: f64,
    win_steps: usize,
    /// Consecutive closed windows where the other mode beat this one by
    /// more than the margin.
    losing_windows: usize,
    /// Emitted tokens since the last probe/refresh event in this mode.
    tokens_since_event: usize,
    observed_depth: usize,
    measured_at_depth: usize,
    fresh: Option<GateDecision>,
}

impl MtpGate {
    /// `num_drafts` is retained for construction-site compatibility and
    /// logging; arbitration is measurement-driven and does not model K.
    pub fn new(num_drafts: usize) -> Self {
        tracing::info!(
            "MTP gate: throughput-arbitrated (K={num_drafts}); window={WINDOW_STEPS} steps, \
             dwell={SWITCH_DWELL_WINDOWS}, reprobe={} tok, refresh={} tok",
            reprobe_tokens(),
            serial_refresh_tokens(),
        );
        Self {
            mode: Mode::Mtp,
            probing: false,
            probe_windows_left: 0,
            mtp: ModeStats::default(),
            serial: ModeStats::default(),
            win_tokens: 0.0,
            win_wall: 0.0,
            win_steps: 0,
            losing_windows: 0,
            tokens_since_event: 0,
            observed_depth: 0,
            measured_at_depth: 0,
            fresh: None,
        }
    }

    pub fn note_depth(&mut self, depth: usize) {
        self.observed_depth = depth;
    }

    /// Depth-regime change: mark BOTH baselines stale (economics moved) and
    /// let the normal probe cadence refresh them — no state wipe, no forced
    /// serial phase.
    pub fn maybe_remeasure(&mut self, current_depth: usize) {
        let measured = self.measured_at_depth.max(REMEASURE_DEPTH_FLOOR);
        let live = current_depth.max(REMEASURE_DEPTH_FLOOR);
        if live >= measured * REMEASURE_DEPTH_FACTOR || measured >= live * REMEASURE_DEPTH_FACTOR {
            tracing::info!(
                "MTP gate: depth regime changed ({} -> {} tokens); baselines stale, \
                 will re-probe on cadence",
                self.measured_at_depth,
                current_depth,
            );
            self.mtp.stale = true;
            self.serial.stale = true;
            self.measured_at_depth = current_depth;
            // Refresh the off-mode soon rather than waiting a full interval.
            self.tokens_since_event = self.tokens_since_event.max(self.event_interval());
        }
    }

    /// One-shot handoff of a fresh mode switch for scheduler bookkeeping.
    pub fn take_fresh_decision(&mut self) -> Option<GateDecision> {
        self.fresh.take()
    }

    /// Which step type the scheduler should run next.
    pub fn next_step(&self) -> GateStep {
        let effective = if self.probing {
            Self::other(self.mode)
        } else {
            self.mode
        };
        match effective {
            Mode::Mtp => GateStep::MeasureVerify,
            Mode::Serial => GateStep::MeasureDecode,
        }
    }

    /// Record one plain decode step (1 emitted token).
    pub fn record_decode(&mut self, wall: Duration) {
        self.record_step(wall, 1);
    }

    /// Record one MTP-path step: `emitted` tokens actually committed (a
    /// bootstrap step emits 1; a verify step emits 1 + accepted). Bootstrap
    /// and propose cost are charged to Mtp mode — they are part of what MTP
    /// costs to run.
    pub fn record_verify_step(&mut self, wall: Duration, emitted: usize) {
        self.record_step(wall, emitted.max(1));
    }

    fn other(m: Mode) -> Mode {
        match m {
            Mode::Mtp => Mode::Serial,
            Mode::Serial => Mode::Mtp,
        }
    }

    fn event_interval(&self) -> usize {
        match self.mode {
            Mode::Mtp => serial_refresh_tokens(),
            Mode::Serial => reprobe_tokens(),
        }
    }

    fn stats_mut(&mut self, m: Mode) -> &mut ModeStats {
        match m {
            Mode::Mtp => &mut self.mtp,
            Mode::Serial => &mut self.serial,
        }
    }

    fn record_step(&mut self, wall: Duration, tokens: usize) {
        self.win_tokens += tokens as f64;
        self.win_wall += wall.as_secs_f64();
        self.win_steps += 1;
        if !self.probing {
            self.tokens_since_event += tokens;
        }
        if self.win_steps >= WINDOW_STEPS {
            self.close_window();
        } else if !self.probing && self.tokens_since_event >= self.event_interval() {
            // Time to look at the other mode: finish the current window
            // early so the probe starts on a clean accumulator.
            self.close_window();
        }
    }

    fn close_window(&mut self) {
        let ran = if self.probing {
            Self::other(self.mode)
        } else {
            self.mode
        };
        if self.win_wall > 0.0 && self.win_steps > 0 {
            let window_tps = self.win_tokens / self.win_wall;
            let replace = self.probing || self.stats_mut(ran).stale;
            self.stats_mut(ran).update(window_tps, replace);
        }
        self.win_tokens = 0.0;
        self.win_wall = 0.0;
        self.win_steps = 0;

        if self.probing {
            self.probe_windows_left = self.probe_windows_left.saturating_sub(1);
            if self.probe_windows_left == 0 {
                self.probing = false;
                self.arbitrate();
                self.tokens_since_event = 0;
            }
            return;
        }

        // Scheduled exploration of the other mode.
        if self.tokens_since_event >= self.event_interval() {
            self.probing = true;
            self.probe_windows_left = 1;
            return;
        }
        self.arbitrate();
    }

    /// Compare mode EWMAs with a hysteresis margin; switch after dwell.
    fn arbitrate(&mut self) {
        let (Some(mtp), Some(serial)) = (self.mtp.tps, self.serial.tps) else {
            return; // need both baselines before any switch
        };
        let (cur, other, other_dev) = match self.mode {
            Mode::Mtp => (mtp, serial, self.serial.dev),
            Mode::Serial => (serial, mtp, self.mtp.dev),
        };
        let margin = (MARGIN_REL_FLOOR * cur).max(0.5 * (self.dev_of(self.mode) + other_dev));
        if other > cur + margin {
            self.losing_windows += 1;
            if self.losing_windows >= SWITCH_DWELL_WINDOWS {
                let to = Self::other(self.mode);
                tracing::info!(
                    "MTP gate: switching {:?} -> {:?} (current {cur:.1} tok/s vs other \
                     {other:.1} tok/s, margin {margin:.1}, depth={})",
                    self.mode,
                    to,
                    self.observed_depth,
                );
                self.mode = to;
                self.losing_windows = 0;
                self.tokens_since_event = 0;
                self.measured_at_depth = self.observed_depth;
                self.fresh = Some(match to {
                    Mode::Mtp => GateDecision::KeepMtp,
                    Mode::Serial => GateDecision::DisableMtp,
                });
            }
        } else {
            self.losing_windows = 0;
        }
    }

    fn dev_of(&self, m: Mode) -> f64 {
        match m {
            Mode::Mtp => self.mtp.dev,
            Mode::Serial => self.serial.dev,
        }
    }

    /// Debug/test accessors.
    pub fn mtp_tps_debug(&self) -> Option<f64> {
        self.mtp.tps
    }
    pub fn serial_tps_debug(&self) -> Option<f64> {
        self.serial.tps
    }
    pub fn in_serial_mode(&self) -> bool {
        self.mode == Mode::Serial
    }
}

#[cfg(test)]
mod tests;
