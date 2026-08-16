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
//! - A depth-regime change (factor [`REMEASURE_DEPTH_FACTOR`] = 2, floor
//!   [`REMEASURE_DEPTH_FLOOR`] = 512) marks both baselines stale and
//!   pulls the next probe forward (one [`WINDOW_STEPS`] window). A stale
//!   other-mode EWMA cannot win a switch. This is the shipped gate
//!   (#337 / #344 / #242 / d6171c4), not a second gate.
//! - Steps are BATCH steps: a plain decode step over n active sequences
//!   emits n tokens and is charged as n (2026-08-15 fix — it was charged as
//!   1, under-reading the serial EWMA ~n× at batch n while the verify side
//!   was already multi-seq summed, which made DisableMtp unreachable under
//!   concurrency). Tokens/sec at different batch widths are different
//!   economies, so EWMAs never compare across width regimes: a change of
//!   power-of-two width bucket discards the partial window, marks both
//!   baselines stale and pulls the next probe forward — the same policy as
//!   a depth-regime change.
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
    env_usize("ATLAS_MTP_GATE_REPROBE", 256)
}

/// MTP tokens between serial-baseline refreshes while in Mtp mode. One
/// 16-step window per 1024 tokens bounds refresh overhead at ≤0.3% even if
/// serial were 18% slower.
fn serial_refresh_tokens() -> usize {
    env_usize("ATLAS_MTP_GATE_REFRESH", 1024)
}

/// Spec-entry verify pin, in post-`</think>` tokens
/// (`ATLAS_SPEC_ENTRY_PIN`; `0` disables). While a speculating sequence is
/// within this window the scheduler runs the MTP verify path even when the
/// gate's throughput arbitration says Serial.
///
/// Why the answer opening must not depend on the gate's mode: the serial
/// (M=1) and verify (batch-K) forwards sit on the batch-K numerics floor —
/// at T=0 every observed flip between them fires within ~7 tokens of spec
/// ENTRY (2026-07-07/08 calibration, the same measurement behind
/// `ATLAS_DFLASH_RESUME_GUARD`). The gate arbitrates on WALL-CLOCK
/// throughput, so which path serves an answer opening otherwise depends on
/// how fast the binary happens to be — measured 2026-08-14 (bfcl-subset
/// echolp, 134 samples): one build's gate dwelt in Serial across requests
/// #89–#101 and exactly the three `live_irrelevance` samples inside that
/// window flipped from a prose decline to a fabricated weather tool call,
/// while the reference build served the same requests in Mtp mode and
/// declined. Pinning the entry window to the verify path makes the opening
/// trajectory a property of the model, not of the gate's stopwatch.
///
/// Default 8: covers the measured ≤7-token flip window with one token of
/// margin. Interaction with `ATLAS_DFLASH_RESUME_GUARD` (the serial-entry
/// mirror of this pin): the resume guard is enforced UPSTREAM of the gate
/// dispatch, so for post-think tokens `< guard` the sequence never reaches
/// the gate arm and the pin is moot; a guard ≥ the pin disables it wholesale.
pub(crate) fn parse_entry_pin_tokens(env: Option<&str>) -> u32 {
    env.and_then(|v| v.parse().ok()).unwrap_or(8)
}

fn entry_pin_tokens() -> u32 {
    static CACHED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        parse_entry_pin_tokens(std::env::var("ATLAS_SPEC_ENTRY_PIN").ok().as_deref())
    })
}

/// Whether the spec-entry pin overrides a Serial gate decision for this
/// step. `min_post_think_emitted` is the minimum over the active batch, so
/// one entering sequence pins the whole (already spec-eligible) batch.
pub fn entry_pin_forces_verify(min_post_think_emitted: u32) -> bool {
    min_post_think_emitted < entry_pin_tokens()
}

/// Existing scheduler dispatch predicate for the throughput gate — not a
/// second gate. Standard MTP verifies during `<think>` (ForcedThinkEnd
/// stays on that path). DFlash raw-argmax stays serial-in-think unless
/// `ATLAS_DFLASH_SPEC_THINK=1`.
pub fn spec_dispatch_eligible(
    inside_thinking: bool,
    post_think_emitted: u32,
    output_len: u32,
    suppress_tool_call: bool,
    disable_mtp: bool,
    spec_think: bool,
    resume_guard: u32,
    dflash_raw_argmax: bool,
) -> bool {
    if suppress_tool_call || disable_mtp {
        return false;
    }
    // Speculation never enters `<think>` without the ATLAS_DFLASH_SPEC_THINK
    // opt-in, for BOTH lanes: batch-K verify is not byte-lossless at T=0 (the
    // numerics floor can flip a low-margin token mid-reasoning), and the
    // agentic-webserver gate measured the damage as deterministic 8-9/10
    // trajectory failures (2026-08-16 bisect: main+this-hunk fails, main
    // without it passes 10/10).
    if inside_thinking && !spec_think {
        return false;
    }
    if dflash_raw_argmax && !spec_think {
        return post_think_emitted >= resume_guard;
    }
    if inside_thinking {
        output_len >= resume_guard
    } else {
        post_think_emitted >= resume_guard
    }
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
/// decode/verify BATCH step (whatever its width) is timed and reported, so
/// arbitration runs continuously with zero dedicated measurement phases.
pub struct MtpGate {
    /// Serial tokens between MTP re-probes while in Serial mode.
    reprobe: usize,
    /// MTP tokens between serial-baseline refreshes while in Mtp mode.
    refresh: usize,
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
    /// Power-of-two batch-width bucket the current baselines were measured
    /// in (0 = nothing recorded yet). See [`Self::note_width`].
    width_regime: usize,
    fresh: Option<GateDecision>,
    /// Depth-regime changes this serve (Done-line / tests).
    regime_reprobes: usize,
}

impl MtpGate {
    /// Observability accessor for the scheduler snapshot: current mode and
    /// the delivered-throughput EWMA of that mode (0.0 until measured).
    pub fn observe(&self) -> (super::snapshot::MtpModeSnap, f32) {
        use super::snapshot::MtpModeSnap;
        let mode = if self.probing {
            MtpModeSnap::Probing
        } else {
            match self.mode {
                Mode::Mtp => MtpModeSnap::Mtp,
                Mode::Serial => MtpModeSnap::Serial,
            }
        };
        let stats = match self.mode {
            Mode::Mtp => &self.mtp,
            Mode::Serial => &self.serial,
        };
        (mode, stats.tps.unwrap_or(0.0) as f32)
    }

    /// `num_drafts` is retained for construction-site compatibility and
    /// logging; arbitration is measurement-driven and does not model K.
    pub fn new(num_drafts: usize) -> Self {
        // Resolved once per gate rather than cached in two statics: the gate
        // belongs to the run, and `event_interval` reads these on every decode
        // step, which is too hot for a per-call getenv.
        let reprobe = reprobe_tokens();
        let refresh = serial_refresh_tokens();
        tracing::info!(
            "MTP gate: throughput-arbitrated (K={num_drafts}); window={WINDOW_STEPS} steps, \
             dwell={SWITCH_DWELL_WINDOWS}, reprobe={reprobe} tok, refresh={refresh} tok",
        );
        Self {
            reprobe,
            refresh,
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
            width_regime: 0,
            fresh: None,
            regime_reprobes: 0,
        }
    }

    pub fn note_depth(&mut self, depth: usize) {
        self.observed_depth = depth;
    }

    /// Depth-regime change: mark BOTH baselines stale (economics moved) and
    /// pull the next probe forward — no state wipe. Returns true when a
    /// regime change fired. The early probe is one [`WINDOW_STEPS`] window
    /// (existing test); it must not dominate a 300–1000 token think.
    pub fn maybe_remeasure(&mut self, current_depth: usize) -> bool {
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
            self.regime_reprobes = self.regime_reprobes.saturating_add(1);
            true
        } else {
            false
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

    /// Record one plain decode step over a batch of `width` active
    /// sequences — each emits exactly one token, so the step delivered
    /// `width` tokens. Charging 1 regardless of width (the pre-2026-08-15
    /// bug) under-read the serial EWMA ~n× at batch n while the verify side
    /// was already multi-seq summed, so under concurrency the arbiter
    /// compared a full-batch MTP rate against a one-sequence serial rate
    /// and DisableMtp was unreachable.
    pub fn record_decode(&mut self, wall: Duration, width: usize) {
        self.note_width(width.max(1));
        self.record_step(wall, width.max(1));
    }

    /// Record one MTP-path step: `emitted` tokens actually committed,
    /// summed over ALL `width` sequences in the batch (a bootstrap step
    /// emits 1 per sequence; a verify step 1 + accepted). Bootstrap and
    /// propose cost are charged to Mtp mode — they are part of what MTP
    /// costs to run.
    pub fn record_verify_step(&mut self, wall: Duration, emitted: usize, width: usize) {
        self.note_width(width.max(1));
        self.record_step(wall, emitted.max(1));
    }

    /// Width-regime tracking. Tokens/sec measured at batch width 1 and at
    /// width 8 describe different economies (weight reuse across the batch
    /// changes both modes' per-step cost), so an EWMA taken in one regime
    /// must not arbitrate against a window taken in another. Regime =
    /// power-of-two bucket (`next_power_of_two`), mirroring the factor-2
    /// depth regime: per-step ±1 churn inside a bucket blends as ordinary
    /// window noise, a bucket change discards the partial (mixed-width)
    /// window, marks BOTH baselines stale and pulls the next probe forward
    /// — exactly the [`Self::maybe_remeasure`] policy. A stale other-mode
    /// EWMA cannot win a switch, so the gate re-measures before it re-arbitrates.
    fn note_width(&mut self, width: usize) {
        let regime = width.next_power_of_two();
        if regime == self.width_regime {
            return;
        }
        if self.width_regime != 0 {
            tracing::info!(
                "MTP gate: batch-width regime changed ({} -> {regime}); partial window \
                 discarded, baselines stale, will re-probe on cadence",
                self.width_regime,
            );
            self.mtp.stale = true;
            self.serial.stale = true;
            // The partial window mixes widths — it describes neither regime.
            self.win_tokens = 0.0;
            self.win_wall = 0.0;
            self.win_steps = 0;
            self.losing_windows = 0;
            self.tokens_since_event = self.tokens_since_event.max(self.event_interval());
        }
        self.width_regime = regime;
    }

    fn other(m: Mode) -> Mode {
        match m {
            Mode::Mtp => Mode::Serial,
            Mode::Serial => Mode::Mtp,
        }
    }

    fn event_interval(&self) -> usize {
        match self.mode {
            Mode::Mtp => self.refresh,
            Mode::Serial => self.reprobe,
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
        // A stale other-mode EWMA is a measurement from a different depth
        // regime. Switching onto it dumps a long think into serial for a
        // full reprobe interval — that is the costly failure, not the
        // one-window early probe `maybe_remeasure` already schedules.
        let other_stale = match self.mode {
            Mode::Mtp => self.serial.stale,
            Mode::Serial => self.mtp.stale,
        };
        if other_stale {
            return;
        }
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
    pub fn regime_reprobe_count(&self) -> usize {
        self.regime_reprobes
    }
    pub fn is_probing(&self) -> bool {
        self.probing
    }
}

#[cfg(test)]
mod tests;
