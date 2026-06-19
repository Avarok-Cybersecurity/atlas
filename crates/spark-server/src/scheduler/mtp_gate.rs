// SPDX-License-Identifier: AGPL-3.0-only

//! Throughput-aware MTP runtime gate.
//!
//! MTP speculative decode is economical only when a single verify step
//! (which advances up to `1 + num_drafts` tokens) is cheaper per produced
//! token than running that many plain single-token decode steps. Whether it
//! pays off is a property of the *measured* per-step cost on this exact
//! model + quantization + hardware combination — NOT of the weight format.
//! For example, on a hybrid SSM model the verify pass re-runs the full
//! layer / MoE / lm_head stack, and on FP8 weights that makes one verify
//! step cost ~2.3x a plain decode step even though draft acceptance is
//! healthy (~80%); on NVFP4 weights the 4-bit weight traffic makes the same
//! verify step cost only ~1.1x. An acceptance-only gate is therefore wrong:
//! acceptance is healthy in the FP8 case yet MTP still loses.
//!
//! This gate measures the verify-cost multiplier
//!   `m = verify_step_wall / decode_step_wall`
//! over the first decode steps of a single-sequence serving session, then
//! applies a PROVABLE bound:
//!
//! For `K_drafts = num_drafts` drafts per step, a verify step advances at
//! MOST `1 + K_drafts` tokens (perfect acceptance). So its best-case
//! tokens-per-wall is `(1 + K_drafts) / m` (in units of decode steps). If
//!   `m >= 1 + K_drafts`
//! then even at 100% acceptance the verify step produces no more tokens per
//! unit wall than `1 + K_drafts` plain decode steps would — MTP is
//! net-negative at ANY acceptance and is disabled. No acceptance estimate is
//! needed for this decision; it is a hard upper bound.
//!
//! When `m < 1 + K_drafts`, MTP *can* win, and does so once acceptance clears
//! the break-even `m - 1` drafts-accepted-per-step. We keep MTP on in that
//! regime (the live K-summary logging in the verify steps already surfaces
//! acceptance for observability).

use std::time::Duration;

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
/// sequence is decoding with MTP requested, then yields a permanent decision.
pub struct MtpGate {
    /// `1 + num_drafts`: the max effective tokens a verify step can advance,
    /// and the provable net-negative threshold for the multiplier. Derived
    /// from the scheduler's `num_drafts` (K=2 verify ⇒ num_drafts=1 ⇒ 2).
    max_effective: f64,
    phase: Phase,
    decode_samples: Vec<Duration>,
    verify_samples: Vec<Duration>,
    decision: Option<GateDecision>,
}

impl MtpGate {
    /// `num_drafts`: drafts proposed per verify step (scheduler SSOT; K=2 ⇒ 1).
    pub fn new(num_drafts: usize) -> Self {
        Self {
            max_effective: 1.0 + num_drafts as f64,
            phase: Phase::Decode,
            decode_samples: Vec::with_capacity(WARMUP_SAMPLES + TIMED_SAMPLES),
            verify_samples: Vec::with_capacity(WARMUP_SAMPLES + TIMED_SAMPLES),
            decision: None,
        }
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
        let decision = if multiplier >= self.max_effective {
            GateDecision::DisableMtp
        } else {
            GateDecision::KeepMtp
        };
        match decision {
            GateDecision::DisableMtp => tracing::info!(
                "MTP gate: verify_multiplier={multiplier:.2}, max_effective={:.1} \
                 (decode={:.2}ms verify={:.2}ms) => DISABLED (net-negative at any acceptance)",
                self.max_effective,
                decode_s * 1000.0,
                verify_s * 1000.0,
            ),
            GateDecision::KeepMtp => tracing::info!(
                "MTP gate: verify_multiplier={multiplier:.2}, max_effective={:.1} \
                 (decode={:.2}ms verify={:.2}ms) => ENABLED",
                self.max_effective,
                decode_s * 1000.0,
                verify_s * 1000.0,
            ),
        }
        self.decision = Some(decision);
        self.phase = Phase::Done;
    }

    /// The terminal decision, available once `is_measuring()` is false.
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
}
