// SPDX-License-Identifier: AGPL-3.0-only

//! Throughput-aware MTP runtime gate.
//!
//! Measures `m = verify_step_wall / decode_step_wall` over the first decode
//! steps, then disables MTP when net-negative. The disable threshold uses the
//! MEASURED expected tokens (`1 + mean_accepted`), not the theoretical `1 + K`
//! max — so MTP is disabled at the ACTUAL acceptance rate, not just the
//! impossible 100% worst case. `m` is depth-dependent (weight-bound at short
//! context, KV/SSM-bound at depth), so the gate re-measures when the live
//! depth leaves the measured regime (see [`REMEASURE_DEPTH_FACTOR`]). Ongoing
//! adaptation: if the acceptance EMA drops below break-even, re-opens
//! measurement (see `should_reconsider`).

use std::time::Duration;

/// Factor that triggers re-measurement when live depth leaves the measured
/// regime (either direction). Factor 2 re-checks at ~2× depth.
const REMEASURE_DEPTH_FACTOR: usize = 2;

/// Floor for the regime comparison (below this, all contexts are "shallow").
const REMEASURE_DEPTH_FLOOR: usize = 512;

/// Number of leading samples of each step type discarded as graph-capture /
/// cache warmup before timing begins. The first verify step and the first
/// decode step each trigger one-time CUDA-graph capture and cold weight
/// fetches whose wall time is not representative of steady state.
///
/// Derivation, not a magic default: CUDA-graph capture is a strictly
/// one-time event per step type (verify-graphed vs decode-batch graphs are
/// captured on first invocation), so a single discarded sample per type is
/// the minimum that excludes it. We discard 2 for a margin against the first
/// post-capture replay still touching cold instruction/constant caches.
const WARMUP_SAMPLES: usize = 2;

/// Number of timed samples collected per step type after warmup. The
/// multiplier uses the MEDIAN of these to reject scheduler-thread jitter
/// (occasional condvar wakeups / pending-queue drains between steps). An odd
/// count gives an unambiguous median.
const TIMED_SAMPLES: usize = 5;

/// What the gate wants the scheduler to do for the NEXT step while it is
/// still collecting samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStep {
    /// Run a plain single-token decode step and report its wall time.
    MeasureDecode,
    /// Run an MTP verify step and report its wall time.
    MeasureVerify,
}

/// The terminal decision once enough samples are collected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Keep MTP enabled: `m < 1 + num_drafts`.
    KeepMtp,
    /// Disable MTP: `m >= 1 + num_drafts` (net-negative at any acceptance).
    DisableMtp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Collecting decode-step samples (warmup then timed).
    Decode,
    /// Collecting verify-step samples (warmup then timed).
    Verify,
    /// Done; `decide()` has produced a result.
    Done,
}

/// Per-serve, single-instance throughput-aware MTP gate. Lives on the
/// scheduler thread; drives a short measurement phase the first time a lone
/// sequence is decoding with MTP requested, then yields a decision that
/// holds for the CURRENT depth regime. The decision is not permanent: the
/// multiplier is depth-dependent (see module docs), so the gate re-opens
/// measurement when the live depth leaves the measured regime
/// ([`Self::maybe_remeasure`]).
pub struct MtpGate {
    /// `1 + num_drafts`: the max effective tokens a verify step can advance,
    /// and the provable net-negative threshold for the multiplier. Derived
    /// from the scheduler's `num_drafts` (K=2 verify ⇒ num_drafts=1 ⇒ 2).
    max_effective: f64,
    phase: Phase,
    decode_samples: Vec<Duration>,
    verify_samples: Vec<Duration>,
    /// Acceptance samples (num_accepted per verify step) collected during the
    /// Verify measurement phase, parallel to `verify_samples`. Used to compute
    /// the MEASURED expected tokens per step (`1 + mean_accepted`) for the
    /// adaptive disable threshold — strictly tighter than the theoretical
    /// `max_effective = 1 + K` (which assumes 100% acceptance).
    acceptance_samples: Vec<usize>,
    /// Exponential moving average of num_accepted, updated after the initial
    /// decision for ongoing adaptation. If it drops below the break-even
    /// (`1 + ema < measured_multiplier`), the gate re-opens measurement.
    acceptance_ema: f64,
    /// The measured verify/decode multiplier at the last `finalize`. Used by
    /// [`Self::should_reconsider`] for the ongoing acceptance-drop check.
    measured_multiplier: f64,
    decision: Option<GateDecision>,
    /// Sequence depth (tokens) most recently observed while sampling; frozen
    /// into `measured_at_depth` when a decision is reached.
    observed_depth: usize,
    /// Depth regime the current `decision` was measured in. Compared against
    /// the live depth by [`Self::maybe_remeasure`].
    measured_at_depth: usize,
    /// Set by `finalize`, taken exactly once by the scheduler to run the
    /// one-time transition work for a fresh decision (e.g. clearing pending
    /// drafts + draft-head resync when MTP turns off).
    fresh_decision: Option<GateDecision>,
}

impl MtpGate {
    /// `num_drafts`: drafts proposed per verify step (scheduler SSOT; K=2 ⇒ 1).
    pub fn new(num_drafts: usize) -> Self {
        Self {
            max_effective: 1.0 + num_drafts as f64,
            phase: Phase::Decode,
            decode_samples: Vec::with_capacity(WARMUP_SAMPLES + TIMED_SAMPLES),
            verify_samples: Vec::with_capacity(WARMUP_SAMPLES + TIMED_SAMPLES),
            acceptance_samples: Vec::with_capacity(WARMUP_SAMPLES + TIMED_SAMPLES),
            acceptance_ema: 0.0,
            measured_multiplier: 0.0,
            decision: None,
            observed_depth: 0,
            measured_at_depth: 0,
            fresh_decision: None,
        }
    }

    /// Note the current sequence depth while measuring. The last value seen
    /// before `finalize` becomes the decision's `measured_at_depth`.
    pub fn note_depth(&mut self, depth: usize) {
        self.observed_depth = depth;
    }

    /// Re-open measurement when the live depth has left the regime the
    /// current decision was measured in (factor-[`REMEASURE_DEPTH_FACTOR`]
    /// crossing in either direction, floored at [`REMEASURE_DEPTH_FLOOR`]).
    /// No-op while a measurement is already in flight.
    pub fn maybe_remeasure(&mut self, current_depth: usize) {
        if self.phase != Phase::Done {
            return;
        }
        let measured = self.measured_at_depth.max(REMEASURE_DEPTH_FLOOR);
        let live = current_depth.max(REMEASURE_DEPTH_FLOOR);
        if live >= measured * REMEASURE_DEPTH_FACTOR || measured >= live * REMEASURE_DEPTH_FACTOR {
            tracing::info!(
                "MTP gate: depth regime changed ({} -> {} tokens); re-measuring \
                 verify/decode economics",
                self.measured_at_depth,
                current_depth,
            );
            self.phase = Phase::Decode;
            self.decode_samples.clear();
            self.verify_samples.clear();
            self.acceptance_samples.clear();
            self.decision = None;
            self.fresh_decision = None;
        }
    }

    /// One-shot handoff of a freshly-reached decision, for the scheduler's
    /// transition bookkeeping. Returns `Some` exactly once per `finalize`.
    pub fn take_fresh_decision(&mut self) -> Option<GateDecision> {
        self.fresh_decision.take()
    }

    /// Whether the gate still needs to drive measurement steps. False once a
    /// decision has been reached.
    pub fn is_measuring(&self) -> bool {
        self.phase != Phase::Done
    }

    /// Which step type the scheduler should run next to advance measurement.
    /// Decode samples are collected first (they need no draft bootstrap),
    /// then verify samples.
    pub fn next_step(&self) -> GateStep {
        match self.phase {
            Phase::Decode => GateStep::MeasureDecode,
            // During the Verify phase we still issue MTP steps; the first
            // such step bootstraps a draft (no verify yet) and is naturally
            // absorbed by WARMUP_SAMPLES.
            Phase::Verify => GateStep::MeasureVerify,
            Phase::Done => GateStep::MeasureDecode, // unreachable while measuring
        }
    }

    /// Record one timed decode-step sample. Caller times only the decode-step
    /// wall (D2H + sample included, identically to the verify path).
    pub fn record_decode(&mut self, wall: Duration) {
        if self.phase != Phase::Decode {
            return;
        }
        self.decode_samples.push(wall);
        if self.decode_samples.len() >= WARMUP_SAMPLES + TIMED_SAMPLES {
            self.phase = Phase::Verify;
        }
    }

    /// Record one timed verify-step sample. Only steps that actually ran a
    /// verify forward (not a bootstrap-only step) should be reported.
    pub fn record_verify(&mut self, wall: Duration) {
        if self.phase != Phase::Verify {
            return;
        }
        self.verify_samples.push(wall);
        if self.verify_samples.len() >= WARMUP_SAMPLES + TIMED_SAMPLES {
            self.finalize();
        }
    }

    /// Record the number of drafts accepted in a verify step. During the
    /// measurement phase this populates `acceptance_samples` (parallel to
    /// `verify_samples`). After the decision, it updates the rolling EMA for
    /// ongoing adaptation — if acceptance drops below the break-even
    /// (`1 + ema < measured_multiplier`), [`Self::should_reconsider`] flags
    /// it so the scheduler can re-measure or disable.
    pub fn record_acceptance(&mut self, num_accepted: usize) {
        if self.phase == Phase::Verify {
            self.acceptance_samples.push(num_accepted);
        } else if self.phase == Phase::Done {
            // EMA update (alpha=0.2 — responds within ~5 steps to a shift).
            self.acceptance_ema = 0.8 * self.acceptance_ema + 0.2 * num_accepted as f64;
        }
    }

    /// Ongoing adaptation check: after the initial decision, if the rolling
    /// acceptance EMA has dropped below the break-even for the measured
    /// multiplier, MTP is now net-negative at the current acceptance even
    /// though it was profitable when measured. The scheduler should call
    /// this and, if true, re-open measurement (which may disable MTP).
    pub fn should_reconsider(&self) -> bool {
        self.phase == Phase::Done
            && self.decision == Some(GateDecision::KeepMtp)
            && self.measured_multiplier > 0.0
            && 1.0 + self.acceptance_ema < self.measured_multiplier
    }

    /// Unconditionally re-open measurement (called by the scheduler when
    /// [`Self::should_reconsider`] fires — the acceptance drop is a
    /// workload-shift signal, not a depth-regime change).
    pub fn force_remeasure(&mut self) {
        if self.phase != Phase::Done {
            return;
        }
        self.phase = Phase::Decode;
        self.decode_samples.clear();
        self.verify_samples.clear();
        self.acceptance_samples.clear();
        self.decision = None;
        self.fresh_decision = None;
    }

    /// Debug accessors for the reconsider log line.
    pub fn acceptance_ema_debug(&self) -> f64 {
        self.acceptance_ema
    }
    pub fn measured_multiplier_debug(&self) -> f64 {
        self.measured_multiplier
    }

    /// Median of the post-warmup samples for a step type, in seconds.
    fn median_secs(samples: &[Duration]) -> f64 {
        let mut timed: Vec<f64> = samples
            .iter()
            .skip(WARMUP_SAMPLES)
            .map(Duration::as_secs_f64)
            .collect();
        timed.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        timed[timed.len() / 2]
    }

    fn finalize(&mut self) {
        let decode_s = Self::median_secs(&self.decode_samples);
        let verify_s = Self::median_secs(&self.verify_samples);
        // decode_s is a real measured decode step; it cannot be zero in
        // practice, but guard against a degenerate timer to avoid div-by-zero.
        let multiplier = if decode_s > 0.0 {
            verify_s / decode_s
        } else {
            f64::INFINITY
        };
        // Adaptive threshold: use MEASURED expected tokens (1 + mean_accepted)
        // instead of theoretical max (1 + K). Disables MTP at the ACTUAL
        // acceptance rate, not the impossible 100% worst case.
        let mean_accepted = if self.acceptance_samples.len() > WARMUP_SAMPLES {
            let timed: Vec<usize> = self
                .acceptance_samples
                .iter()
                .copied()
                .skip(WARMUP_SAMPLES)
                .collect();
            timed.iter().map(|&x| x as f64).sum::<f64>() / timed.len().max(1) as f64
        } else {
            self.max_effective - 1.0 // no acceptance data → theoretical max fallback
        };
        let effective_tokens = 1.0 + mean_accepted;
        let decision = if multiplier >= self.max_effective || multiplier >= effective_tokens {
            GateDecision::DisableMtp
        } else {
            GateDecision::KeepMtp
        };
        match decision {
            GateDecision::DisableMtp => tracing::info!(
                "MTP gate: verify_multiplier={multiplier:.2}, max_effective={:.1}, \
                 measured_effective={effective_tokens:.2} (mean_accepted={mean_accepted:.2}, \
                 decode={:.2}ms verify={:.2}ms, depth={}) => DISABLED for this depth \
                 regime (net-negative at current acceptance; re-measures on regime change)",
                self.max_effective,
                decode_s * 1000.0,
                verify_s * 1000.0,
                self.observed_depth,
            ),
            GateDecision::KeepMtp => tracing::info!(
                "MTP gate: verify_multiplier={multiplier:.2}, max_effective={:.1}, \
                 measured_effective={effective_tokens:.2} (mean_accepted={mean_accepted:.2}, \
                 decode={:.2}ms verify={:.2}ms, depth={}) => ENABLED",
                self.max_effective,
                decode_s * 1000.0,
                verify_s * 1000.0,
                self.observed_depth,
            ),
        }
        self.measured_multiplier = multiplier;
        self.acceptance_ema = mean_accepted;
        self.measured_at_depth = self.observed_depth;
        self.decision = Some(decision);
        self.fresh_decision = Some(decision);
        self.phase = Phase::Done;
    }

    /// The operative decision for the current depth regime, available once
    /// `is_measuring()` is false. `None` while (re-)measuring.
    pub fn decision(&self) -> Option<GateDecision> {
        self.decision
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(x: u64) -> Duration {
        Duration::from_micros(x * 1000)
    }

    /// Drive the gate through a full decode-then-verify measurement with the
    /// given per-step medians (warmup samples are deliberately skewed to prove
    /// they are discarded).
    fn run_gate(num_drafts: usize, decode_ms: u64, verify_ms: u64) -> GateDecision {
        let mut g = MtpGate::new(num_drafts);
        // Decode phase: 2 warmup (huge, must be discarded) + 5 timed.
        for i in 0..(WARMUP_SAMPLES + TIMED_SAMPLES) {
            assert_eq!(g.next_step(), GateStep::MeasureDecode);
            let w = if i < WARMUP_SAMPLES {
                ms(9999)
            } else {
                ms(decode_ms)
            };
            g.record_decode(w);
        }
        // Verify phase.
        for i in 0..(WARMUP_SAMPLES + TIMED_SAMPLES) {
            assert_eq!(g.next_step(), GateStep::MeasureVerify);
            let w = if i < WARMUP_SAMPLES {
                ms(9999)
            } else {
                ms(verify_ms)
            };
            g.record_verify(w);
        }
        assert!(!g.is_measuring());
        g.decision().expect("decided")
    }

    #[test]
    fn fp8_like_multiplier_disables_k2() {
        // verify 23ms vs decode 10ms => m=2.3 >= 2 (num_drafts=1) => DISABLE.
        assert_eq!(run_gate(1, 10, 23), GateDecision::DisableMtp);
    }

    #[test]
    fn nvfp4_like_multiplier_keeps_k2() {
        // verify 11ms vs decode 10ms => m=1.1 < 2 => KEEP.
        assert_eq!(run_gate(1, 10, 11), GateDecision::KeepMtp);
    }

    #[test]
    fn exact_threshold_disables() {
        // m == 1 + num_drafts is net-negative (no per-token gain at 100%).
        assert_eq!(run_gate(1, 10, 20), GateDecision::DisableMtp);
    }

    #[test]
    fn k3_raises_threshold() {
        // num_drafts=2 => max_effective=3; m=2.3 now KEEPS (can win >65% acc).
        assert_eq!(run_gate(2, 10, 23), GateDecision::KeepMtp);
    }

    #[test]
    fn warmup_samples_are_discarded() {
        // If warmup (9999ms) leaked into the median the multiplier would be
        // astronomically off; the clean KEEP proves they are skipped.
        assert_eq!(run_gate(1, 10, 11), GateDecision::KeepMtp);
    }

    #[test]
    fn phase_progression() {
        let mut g = MtpGate::new(1);
        assert_eq!(g.next_step(), GateStep::MeasureDecode);
        for _ in 0..(WARMUP_SAMPLES + TIMED_SAMPLES) {
            g.record_decode(ms(10));
        }
        assert_eq!(g.next_step(), GateStep::MeasureVerify);
        assert!(g.is_measuring());
        for _ in 0..(WARMUP_SAMPLES + TIMED_SAMPLES) {
            g.record_verify(ms(11));
        }
        assert!(!g.is_measuring());
    }

    /// Drive a full measurement on an existing gate at a given depth.
    fn drive(g: &mut MtpGate, depth: usize, decode_ms: u64, verify_ms: u64) {
        assert!(g.is_measuring());
        g.note_depth(depth);
        for _ in 0..(WARMUP_SAMPLES + TIMED_SAMPLES) {
            g.record_decode(ms(decode_ms));
        }
        for _ in 0..(WARMUP_SAMPLES + TIMED_SAMPLES) {
            g.record_verify(ms(verify_ms));
        }
        assert!(!g.is_measuring());
    }

    /// The core depth-regime scenario: a short-context measurement disables
    /// MTP (weight-bound, m≈2.3); the session then goes deep, the gate
    /// re-measures, and the KV-bound economics (m≈1.1) re-enable it.
    #[test]
    fn remeasure_on_depth_regime_change_reenables() {
        let mut g = MtpGate::new(1);
        drive(&mut g, 100, 10, 23); // short ctx: m=2.3 => DISABLE
        assert_eq!(g.decision(), Some(GateDecision::DisableMtp));
        assert_eq!(g.take_fresh_decision(), Some(GateDecision::DisableMtp));

        // Same regime (100 floored to 512; 600 < 512*4): decision holds.
        g.maybe_remeasure(600);
        assert!(!g.is_measuring());
        assert_eq!(g.decision(), Some(GateDecision::DisableMtp));

        // Depth crosses the factor boundary: measurement re-opens.
        g.maybe_remeasure(REMEASURE_DEPTH_FLOOR * REMEASURE_DEPTH_FACTOR);
        assert!(g.is_measuring());
        assert_eq!(g.decision(), None);

        // Deep regime: verify shares the KV pass, m=1.1 => KEEP.
        drive(&mut g, 2048, 10, 11);
        assert_eq!(g.decision(), Some(GateDecision::KeepMtp));
        assert_eq!(g.take_fresh_decision(), Some(GateDecision::KeepMtp));

        // And back down: a fresh short session re-opens measurement again.
        g.maybe_remeasure(100);
        assert!(g.is_measuring());
    }

    #[test]
    fn fresh_decision_taken_exactly_once() {
        let mut g = MtpGate::new(1);
        drive(&mut g, 100, 10, 23);
        assert_eq!(g.take_fresh_decision(), Some(GateDecision::DisableMtp));
        assert_eq!(g.take_fresh_decision(), None);
        // The operative decision is still readable.
        assert_eq!(g.decision(), Some(GateDecision::DisableMtp));
    }

    #[test]
    fn no_remeasure_within_floored_short_regime() {
        let mut g = MtpGate::new(1);
        drive(&mut g, 32, 10, 23);
        // 32 and 400 both floor to 512 — same regime, no re-measure.
        g.maybe_remeasure(400);
        assert!(!g.is_measuring());
        // Just below the factor boundary: still the same regime.
        g.maybe_remeasure(REMEASURE_DEPTH_FLOOR * REMEASURE_DEPTH_FACTOR - 1);
        assert!(!g.is_measuring());
    }

    #[test]
    fn maybe_remeasure_noop_while_measuring() {
        let mut g = MtpGate::new(1);
        assert!(g.is_measuring());
        g.maybe_remeasure(1_000_000);
        // Still in the ORIGINAL measurement (no reset mid-flight).
        assert_eq!(g.next_step(), GateStep::MeasureDecode);
        assert!(g.is_measuring());
    }
}
