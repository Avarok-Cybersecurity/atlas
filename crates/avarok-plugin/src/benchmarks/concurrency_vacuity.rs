// SPDX-License-Identifier: AGPL-3.0-only

//! The vacuity rule: is a cell's tok/s a throughput measurement at all?
//!
//! What it was written for (2026-08-15, see the module header of
//! `concurrency.rs`): the counting prompt produced C=1 cells of 49-token
//! bursts against a 512-token budget and C≥4 cells of 0–1 output tokens —
//! E2E==TTFT, TPOT unmeasurable, the aggregate DECREASING with C. Those cells
//! measured nothing, and the rule's job is to say so.
//!
//! What it must NOT do (2026-09-19, decided from the published run's own
//! receipts): void a cell over one natural end-of-sequence. The instrument on
//! the website is ISL 128 / OSL 1024 (`bench/ladder38/l38_r{7..11}_c*.json`),
//! and in every one of those five rounds rep 2 of C=128 contains a request
//! that stops at 691 tokens and rep 2 of C=64 one that stops at 753 —
//! deterministic (temperature 0, seed 42), 0.25% of the cell's tokens, and a
//! DOWNWARD bias on the aggregate. The old rule voided a cell when ANY request
//! delivered under 80% of the budget, so it voided the run behind the
//! published headline, reproducibly, and cleared an eight-rung sweep only
//! about half the time. A rule that fails the instrument it exists to
//! certify is not a rule about the instrument.
//!
//! ## The rule
//!
//! A cell is vacuous unless BOTH hold, with n = its successful requests:
//!
//! 1. **Occupancy** — the cell delivered at least `VACUITY_FLOOR` of its total
//!    budget: `Σ completion_tokens ≥ 0.8 × n × osl`. The aggregate is
//!    `Σ tokens / wall`; this ratio is the batch's mean fullness over the
//!    budget, so it is exactly the quantity that decides whether the aggregate
//!    measured throughput at C or at something narrower.
//! 2. **Majority** — at least half of the requests each delivered at least
//!    `VACUITY_FLOOR × osl`. The reported p50s are nearest-rank
//!    (`stats::percentile`), so a short minority sorts to the fast end and
//!    can move neither the p50 nor the p90/p99 of TTFT / TPOT / E2E.
//!
//! ## What a cell may now contain and still pass — stated so nobody has to
//! discover it
//!
//! * A MINORITY of its requests anywhere under 80% of the budget, down to
//!   zero tokens, as long as the cell's total shortfall stays within 20%.
//!   Worst shapes at C=128 / OSL 1024: 25 requests returning nothing beside
//!   103 full ones (80.5% delivered), or 64 requests stopping at 60% beside
//!   64 full ones. The old rule refused both. Every such shape LOWERS the
//!   aggregate — fewer tokens over a wall the full requests still set — and
//!   leaves the p50 drawn from a request that ran ≥ 80% of the budget. The
//!   one way it can read high: on a curve already FALLING at C (past
//!   saturation) the cell measures a batch ~80% of C wide, and a narrower
//!   batch can be faster there; the published curve rises through C=128.
//! * Every request stopping at exactly 80% — the old rule admitted this too;
//!   the aggregate is unchanged (fewer tokens over a proportionally shorter
//!   wall, minus TTFT amortisation) and E2E p50 reads up to 20% low.
//!
//! ## What is still caught
//!
//! * A cell delivering under 80% of its budget, however the shortfall is
//!   spread: the 49-token burst (9.6%), the 0–1-token cells (≈0%), a serve
//!   that silently returns empty completions for more than a fifth of a
//!   batch.
//! * A cell where half or more of the requests stop under 80%, however much
//!   the rest deliver — the per-request bar still applies to the median.
//! * At C=1 the rule is the old rule: the cell IS the request.
//!
//! The rule is unchanged in the direction that matters for gaming: no
//! distribution of under-delivery can raise `Σ tokens / wall` on a rising
//! curve, so nothing this admits can inflate a floor comparison.

use super::RequestEvidence;

/// Fraction of the output budget a cell must deliver in total, and the
/// fraction of the per-request budget its median request must clear. 80%: a
/// natural stop a few hundred tokens early is evidence; a 49-token burst
/// against a 512-token budget is not.
pub(super) const VACUITY_FLOOR: f64 = 0.8;

/// What a cell delivered against what it was asked for — the two counts the
/// rule is decided on, kept so the warn line can print the same numbers the
/// verdict used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Delivery {
    /// Successful requests in the cell.
    pub(super) requests: usize,
    /// `Σ completion_tokens` over those requests.
    pub(super) delivered: usize,
    /// `requests × osl`.
    pub(super) budget: usize,
    /// Requests that individually delivered ≥ `VACUITY_FLOOR × osl`.
    pub(super) cleared: usize,
    /// The shortest completion, for the evidence line.
    pub(super) min_completion: usize,
}

impl Delivery {
    pub(super) fn of(requests: &[RequestEvidence], osl: usize) -> Self {
        let bar = VACUITY_FLOOR * osl as f64;
        Self {
            requests: requests.len(),
            delivered: requests.iter().map(|r| r.completion_tokens).sum(),
            budget: requests.len() * osl,
            cleared: requests
                .iter()
                .filter(|r| r.completion_tokens as f64 >= bar)
                .count(),
            min_completion: requests
                .iter()
                .map(|r| r.completion_tokens)
                .min()
                .unwrap_or(0),
        }
    }

    pub(super) fn delivered_pct(&self) -> f64 {
        if self.budget == 0 {
            0.0
        } else {
            self.delivered as f64 / self.budget as f64 * 100.0
        }
    }

    /// Both clauses, as documented in the module header. An empty cell is
    /// not vacuous: there is nothing to judge, and its error count already
    /// voids it.
    pub(super) fn is_vacuous(&self) -> bool {
        if self.requests == 0 {
            return false;
        }
        let occupancy_short = (self.delivered as f64) < VACUITY_FLOOR * self.budget as f64;
        let majority_short = self.cleared * 2 < self.requests;
        occupancy_short || majority_short
    }

    /// The evidence behind a vacuous verdict, in the units the rule is
    /// written in.
    pub(super) fn describe(&self, osl: usize) -> String {
        format!(
            "delivered {:.1}% of its {}×{osl}-token budget and {}/{} request(s) cleared \
             {:.0}% of it (min {} tok)",
            self.delivered_pct(),
            self.requests,
            self.cleared,
            self.requests,
            VACUITY_FLOOR * 100.0,
            self.min_completion,
        )
    }
}
