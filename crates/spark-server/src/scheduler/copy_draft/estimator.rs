// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: src/models/qwen3_5/program/speculative/prompt_lookup.cpp:9-23 (estimator constants), :128-152 (observe, log_probability), prompt_lookup.h:32-35 (confident, kConfident). Changes: translated from C++ to Rust; one estimate per window of Atlas's four windows; the proposal is passed in rather than held, by Atlas contributors.
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license). Applies to `YieldGate` below.

//! Responsibility: the two copy-draft estimators.
//!
//! [`WindowEstimates`] (Cinference): per window length, a decayed count of
//! how many proposed tokens were committed and how often a proposal broke
//! before its end. It yields a per-token probability that the next proposed
//! token is committed given the previous one was, raised by a long verbatim
//! match, and [`WindowEstimates::confident`] (`>= 0.96`) is the rule for a
//! lookup round (verify the copy alone, drafter's chain discarded).
//!
//! [`YieldGate`] (the KVMem yield-gate idea): per request, an exponential
//! average of the accepted fraction of copy chains actually verified, and of
//! the MTP drafts accepted per round. A copy chain replaces the MTP chain
//! only when `width * copy_avg > advantage * mtp_avg`; the first two copy
//! rounds are unconditional (there is nothing to average yet) and after
//! eight consecutive refusals one probe round refreshes the average. All
//! state is per sequence, so it resets per request.

use super::index::WINDOWS;

/// Per observed round, older evidence decays (Cinference `kDecay`).
pub(crate) const DECAY: f64 = 0.9;
pub(crate) const PRIOR_HITS: f64 = 1.0;
pub(crate) const PRIOR_MISSES: f64 = 3.0;
pub(crate) const P_MIN: f64 = 0.01;
pub(crate) const P_MAX: f64 = 0.97;
/// Matched tokens beyond this floor count as prior hits (Cinference
/// `kMatchFloor`), one each (`kMatchWeight`).
pub(crate) const MATCH_FLOOR: usize = 8;
pub(crate) const MATCH_WEIGHT: f64 = 1.0;
/// Per-token estimate a lookup round needs (Cinference `kConfident`).
pub(crate) const CONFIDENT: f64 = 0.96;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Estimate {
    hits: f64,
    misses: f64,
}

/// One [`Estimate`] per window of [`WINDOWS`].
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct WindowEstimates {
    est: [Estimate; WINDOWS.len()],
}

/// Length of the common prefix of `proposal` and `committed`.
pub(crate) fn common_prefix(proposal: &[u32], committed: &[u32]) -> usize {
    proposal
        .iter()
        .zip(committed)
        .take_while(|(a, b)| a == b)
        .count()
}

impl WindowEstimates {
    pub fn new() -> Self {
        Self::default()
    }

    /// Score a proposal from `window` against the tokens committed after it
    /// (Cinference `observe`): hits = common prefix, and a miss when the
    /// proposal broke before `min(proposal, committed)` ran out.
    pub fn observe(&mut self, window: usize, proposal: &[u32], committed: &[u32]) {
        let limit = proposal.len().min(committed.len());
        let hits = common_prefix(proposal, committed);
        let e = &mut self.est[window];
        e.hits = DECAY * e.hits + hits as f64;
        e.misses = DECAY * e.misses + if hits < limit { 1.0 } else { 0.0 };
    }

    /// Per-token probability for a proposal from `window` whose source
    /// matched `match_len` tokens, clamped to `[P_MIN, P_MAX]`.
    pub fn probability(&self, window: usize, match_len: usize) -> f64 {
        let e = &self.est[window];
        let matched = MATCH_WEIGHT * match_len.saturating_sub(MATCH_FLOOR) as f64;
        let p = (e.hits + PRIOR_HITS + matched)
            / (e.hits + e.misses + PRIOR_HITS + PRIOR_MISSES + matched);
        p.clamp(P_MIN, P_MAX)
    }

    /// Natural log of [`Self::probability`] (Cinference `log_probability`).
    pub fn log_probability(&self, window: usize, match_len: usize) -> f64 {
        self.probability(window, match_len).ln()
    }

    /// The lookup-round rule (Cinference `confident`).
    pub fn confident(&self, window: usize, match_len: usize) -> bool {
        self.log_probability(window, match_len) >= CONFIDENT.ln()
    }
}

/// EWMA weight of the newest round (KVMem description: alpha 0.25).
pub(crate) const GATE_ALPHA: f32 = 0.25;
/// Copy rounds taken without consulting the averages.
pub(crate) const GATE_COLD_ROUNDS: u32 = 2;
/// Consecutive refusals after which one probe round is taken.
pub(crate) const GATE_PROBE_AFTER: u32 = 8;
/// MTP drafts accepted per round assumed before the first MTP observation.
pub(crate) const GATE_MTP_PRIOR: f32 = 1.0;

/// What the yield gate decided for one qualifying copy source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GateVerdict {
    /// The copy chain wins (or is in its cold start).
    Take,
    /// Refused, but the refusal streak earned one probe round: take it.
    Probe,
    /// Refused: keep the MTP chain.
    Skip,
}

impl GateVerdict {
    pub fn takes(self) -> bool {
        !matches!(self, GateVerdict::Skip)
    }
}

/// The per-request copy-versus-MTP yield gate. See the module docs.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct YieldGate {
    copy_avg: Option<f32>,
    mtp_avg: Option<f32>,
    taken: u32,
    skip_streak: u32,
}

fn ewma(prev: Option<f32>, x: f32) -> f32 {
    match prev {
        None => x,
        Some(p) => p + GATE_ALPHA * (x - p),
    }
}

impl YieldGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Decide for a copy chain of `width` tokens against the MTP chain.
    pub fn decide(&mut self, width: usize, advantage: f32) -> GateVerdict {
        if self.taken < GATE_COLD_ROUNDS {
            self.taken += 1;
            self.skip_streak = 0;
            return GateVerdict::Take;
        }
        let copy = width as f32 * self.copy_avg.unwrap_or(0.0);
        let mtp = self.mtp_avg.unwrap_or(GATE_MTP_PRIOR);
        if copy > advantage * mtp {
            self.taken += 1;
            self.skip_streak = 0;
            return GateVerdict::Take;
        }
        if self.skip_streak >= GATE_PROBE_AFTER {
            self.taken += 1;
            self.skip_streak = 0;
            return GateVerdict::Probe;
        }
        self.skip_streak += 1;
        GateVerdict::Skip
    }

    /// A verified copy chain: `accepted` of `drafted` tokens committed.
    pub fn observe_copy(&mut self, accepted: usize, drafted: usize) {
        if drafted == 0 {
            return;
        }
        let ratio = accepted.min(drafted) as f32 / drafted as f32;
        self.copy_avg = Some(ewma(self.copy_avg, ratio));
    }

    /// A verified MTP chain: `accepted` drafts committed.
    pub fn observe_mtp(&mut self, accepted: usize) {
        self.mtp_avg = Some(ewma(self.mtp_avg, accepted as f32));
    }

    pub fn copy_avg(&self) -> Option<f32> {
        self.copy_avg
    }

    pub fn mtp_avg(&self) -> Option<f32> {
        self.mtp_avg
    }

    pub fn skip_streak(&self) -> u32 {
        self.skip_streak
    }
}
