// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request MTP speculation accounting and its once-per-request log line.
//!
//! # Why this exists
//!
//! The periodic `K2 summary` / `K4 summary` lines fire every 100 verify steps
//! across the whole run, so a short request (a 256-token answer is about 200
//! verify passes at one draft) may finish without one, and none of them says
//! what a request spent in each stage. This module counts, per request:
//!
//! * verify passes (steps that verified at least one pending draft) and
//!   bootstrap steps (MTP steps with no pending draft: one decode + propose),
//! * drafts proposed (the pending drafts each verify pass checked) and drafts
//!   accepted, so `accept_rate = accepted / proposed`,
//! * tokens committed by verify passes, so `tokens_per_pass` is the quantity
//!   the speed-up is made of (`1 + accepted` per pass at one draft),
//! * wall time in the draft (propose + its grammar mask), verify (the verify
//!   forward) and commit (the SSM prefix commit) stages, plus the whole MTP
//!   step wall,
//! * the throughput gate's decisions while the request was active, and the
//!   gate's latest delivered-throughput estimates for both modes.
//!
//! # Cost
//!
//! Plain integer adds on the scheduler thread. Stage times reuse the
//! `Instant`s the verify paths already take for `AVAROK_MTP_TIMING`: the
//! [`StageClock`] is fed from `RunTiming::record`, which now performs one
//! clock read for the four stage phases even when the diagnostic is
//! disarmed. A verify step is milliseconds; four clock reads are tens of
//! nanoseconds.
//!
//! # Batches
//!
//! Stage times are per STEP. When several sequences share one MTP step each
//! is charged the step's stage time, which is the wall time that request
//! spent waiting on that stage. At one stream the numbers are exact.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::mtp_timing::Phase;

/// Stage a timed phase belongs to, or `None` for phases this view ignores.
fn stage_of(phase: Phase) -> Option<usize> {
    match phase {
        Phase::Propose | Phase::ProposeMask => Some(0),
        Phase::VerifyForward => Some(1),
        Phase::Commit => Some(2),
        _ => None,
    }
}

/// Always-on per-step stage accumulators (microseconds), drained once per MTP
/// step by [`account_step`]. Lives on `RunTiming` so every existing
/// `record(Phase::..)` call site feeds it with no new plumbing.
#[derive(Debug, Default)]
pub struct StageClock {
    us: [AtomicU64; 3],
}

/// One step's stage times, in microseconds.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StageTimes {
    pub draft_us: u64,
    pub verify_us: u64,
    pub commit_us: u64,
}

impl StageClock {
    /// Charge the time since `since` to `phase`'s stage (no-op for other
    /// phases).
    #[inline]
    pub(crate) fn note(&self, phase: Phase, since: Instant) {
        if let Some(i) = stage_of(phase) {
            let us = u64::try_from(since.elapsed().as_micros()).unwrap_or(u64::MAX);
            self.add_us(i, us);
        }
    }

    #[inline]
    fn add_us(&self, i: usize, us: u64) {
        self.us[i].fetch_add(us, Ordering::Relaxed);
    }

    /// Read and reset the accumulators.
    pub fn take(&self) -> StageTimes {
        StageTimes {
            draft_us: self.us[0].swap(0, Ordering::Relaxed),
            verify_us: self.us[1].swap(0, Ordering::Relaxed),
            commit_us: self.us[2].swap(0, Ordering::Relaxed),
        }
    }
}

/// Per-request MTP counters. `Copy` because it rides inside
/// [`super::mtp_accept_debug::RequestAccept`], which spill and restore copy.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct MtpRequestStats {
    pub verify_passes: u64,
    pub bootstrap_steps: u64,
    pub proposed: u64,
    pub accepted: u64,
    /// Tokens committed by verify passes (bootstrap steps excluded).
    pub committed: u64,
    pub draft_us: u64,
    pub verify_us: u64,
    pub commit_us: u64,
    /// Whole MTP step wall (bootstrap steps included).
    pub step_us: u64,
    pub gate_disables: u64,
    pub gate_enables: u64,
    /// Latest gate estimates while this request was active (tok/s, 0 = none).
    pub gate_mtp_tps: f64,
    pub gate_serial_tps: f64,
}

impl MtpRequestStats {
    /// Account one MTP step for this sequence. `proposed` is the number of
    /// pending drafts the step verified (0 = bootstrap step) and `committed`
    /// the tokens the step appended.
    pub fn record_step(&mut self, proposed: usize, committed: usize, st: StageTimes, wall_us: u64) {
        self.step_us = self.step_us.saturating_add(wall_us);
        self.draft_us = self.draft_us.saturating_add(st.draft_us);
        self.verify_us = self.verify_us.saturating_add(st.verify_us);
        self.commit_us = self.commit_us.saturating_add(st.commit_us);
        if proposed == 0 {
            self.bootstrap_steps = self.bootstrap_steps.saturating_add(1);
            return;
        }
        let proposed = proposed as u64;
        // A verify pass commits the verified prefix plus one corrected or
        // bonus token, so accepted = committed - 1, capped at the drafts.
        let accepted = (committed as u64).saturating_sub(1).min(proposed);
        self.verify_passes = self.verify_passes.saturating_add(1);
        self.proposed = self.proposed.saturating_add(proposed);
        self.accepted = self.accepted.saturating_add(accepted);
        self.committed = self.committed.saturating_add(committed as u64);
    }

    pub fn note_gate_decision(&mut self, disabled: bool) {
        if disabled {
            self.gate_disables = self.gate_disables.saturating_add(1);
        } else {
            self.gate_enables = self.gate_enables.saturating_add(1);
        }
    }

    pub fn note_gate_estimates(&mut self, mtp_tps: Option<f64>, serial_tps: Option<f64>) {
        if let Some(t) = mtp_tps {
            self.gate_mtp_tps = t;
        }
        if let Some(t) = serial_tps {
            self.gate_serial_tps = t;
        }
    }

    /// True when the request ran at least one MTP step.
    pub fn speculated(&self) -> bool {
        self.verify_passes > 0 || self.bootstrap_steps > 0
    }

    pub fn accept_rate(&self) -> f64 {
        ratio(self.accepted, self.proposed)
    }

    pub fn tokens_per_pass(&self) -> f64 {
        ratio(self.committed, self.verify_passes)
    }

    /// Verify-step cost relative to a plain decode step, implied by the
    /// gate's two throughput estimates and this request's tokens per MTP
    /// step: `mult = tokens_per_step * serial_tps / mtp_tps`. 0 when the gate
    /// has not measured both modes.
    pub fn gate_multiplier(&self) -> f64 {
        if self.gate_mtp_tps <= 0.0 || self.gate_serial_tps <= 0.0 {
            return 0.0;
        }
        let steps = self.verify_passes + self.bootstrap_steps;
        let tok_per_step = ratio(self.committed + self.bootstrap_steps, steps);
        tok_per_step * self.gate_serial_tps / self.gate_mtp_tps
    }

    /// The once-per-request line, or `None` when speculation never ran.
    pub fn summary_line(&self) -> Option<String> {
        if !self.speculated() {
            return None;
        }
        let ms = |us: u64| us as f64 / 1000.0;
        Some(format!(
            "MTP stats: verify_passes={} bootstrap_steps={} proposed={} accepted={} \
             accept_rate={:.1}% tokens_per_pass={:.3} draft_ms={:.1} verify_ms={:.1} \
             commit_ms={:.1} step_ms={:.1} gate_disable={} gate_enable={} \
             gate_mtp_tps={:.1} gate_serial_tps={:.1} gate_mult={:.3}",
            self.verify_passes,
            self.bootstrap_steps,
            self.proposed,
            self.accepted,
            100.0 * self.accept_rate(),
            self.tokens_per_pass(),
            ms(self.draft_us),
            ms(self.verify_us),
            ms(self.commit_us),
            ms(self.step_us),
            self.gate_disables,
            self.gate_enables,
            self.gate_mtp_tps,
            self.gate_serial_tps,
            self.gate_multiplier(),
        ))
    }
}

fn ratio(n: u64, d: u64) -> f64 {
    if d == 0 { 0.0 } else { n as f64 / d as f64 }
}

/// Add one step's totals to the process-wide Prometheus counters.
fn export_step(proposed: u64, accepted: u64, committed: u64, passes: u64, st: StageTimes) {
    use crate::metrics as m;
    if passes > 0 {
        m::MTP_VERIFY_PASSES.inc_by(passes);
        m::MTP_DRAFTS_PROPOSED.inc_by(proposed);
        m::MTP_DRAFTS_ACCEPTED.inc_by(accepted);
        m::MTP_TOKENS_COMMITTED.inc_by(committed);
    }
    for (stage, us) in [
        ("draft", st.draft_us),
        ("verify", st.verify_us),
        ("commit", st.commit_us),
    ] {
        if us > 0 {
            m::MTP_STAGE_MICROS.with_label_values(&[stage]).inc_by(us);
        }
    }
}

/// Run one MTP step under accounting: snapshots each sequence's pending
/// draft count and length, runs `step`, then charges the step to every
/// sequence. Returns the step wall and the tokens committed across the
/// batch, which the throughput gate consumes.
pub(super) fn account_step(
    active: &mut [super::ActiveSeq],
    timing: &super::mtp_timing::RunTiming,
    step: impl FnOnce(&mut [super::ActiveSeq]),
) -> (std::time::Duration, usize) {
    let before: Vec<(usize, usize)> = active
        .iter()
        .map(|a| (a.seq.seq_len, a.pending_drafts.len()))
        .collect();
    // Discard stage time recorded outside an MTP step (none today; cheap).
    let _ = timing.stages.take();
    let t0 = Instant::now();
    step(active);
    let wall = t0.elapsed();
    let st = timing.stages.take();
    let wall_us = u64::try_from(wall.as_micros()).unwrap_or(u64::MAX);
    let (mut tot_committed, mut prop, mut acc, mut com, mut passes) = (0usize, 0, 0, 0, 0);
    for (a, &(len0, proposed)) in active.iter_mut().zip(before.iter()) {
        let committed = a.seq.seq_len.saturating_sub(len0);
        tot_committed += committed;
        a.mtp_acct.record_verify_emitted(committed);
        let s = &mut a.mtp_acct.spec;
        let acc0 = s.accepted;
        s.record_step(proposed, committed, st, wall_us);
        if proposed > 0 {
            passes += 1;
            prop += proposed as u64;
            com += committed as u64;
            acc += s.accepted - acc0;
        }
    }
    export_step(prop, acc, com, passes, st);
    (wall, tot_committed)
}

/// Record a fresh gate decision against every active sequence and count it.
pub(super) fn note_gate_decision(active: &mut [super::ActiveSeq], disabled: bool) {
    let label = if disabled { "disable" } else { "enable" };
    crate::metrics::MTP_GATE_DECISIONS
        .with_label_values(&[label])
        .inc();
    for a in active.iter_mut() {
        a.mtp_acct.spec.note_gate_decision(disabled);
    }
}

/// Copy the gate's current estimates onto every active sequence and the
/// process gauges.
pub(super) fn note_gate_estimates(
    active: &mut [super::ActiveSeq],
    mtp_tps: Option<f64>,
    serial_tps: Option<f64>,
) {
    if let Some(t) = mtp_tps {
        crate::metrics::MTP_GATE_TPS
            .with_label_values(&["mtp"])
            .set(t);
    }
    if let Some(t) = serial_tps {
        crate::metrics::MTP_GATE_TPS
            .with_label_values(&["serial"])
            .set(t);
    }
    for a in active.iter_mut() {
        a.mtp_acct.spec.note_gate_estimates(mtp_tps, serial_tps);
    }
}

#[cfg(test)]
#[path = "mtp_stats_tests.rs"]
mod tests;
