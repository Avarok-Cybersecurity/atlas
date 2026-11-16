// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the speculative round plan seam and
//! the serial-round glue `step_mtp` calls around each per-sequence verify.
//!
//! [`RoundPlan`] names what one round verifies; [`RoundOutcome`] is what it
//! produced. `step_mtp`'s serial loop calls [`begin_serial`] after the grammar
//! truncation and [`end_serial`] after the verify dispatch:
//!
//! * the copy drafter (`copy_draft`, row `copy_draft`) may replace the MTP
//!   chain in place (`CopyChain`, same width, same dispatch) or run a
//!   `LookupRound` itself through `step_verify_dflash` (then `begin_serial`
//!   returns `None` and the caller skips its own dispatch);
//! * with the D-Cut v2 row (`mtp_dcut_cost_model`) on, a single
//!   sequence's MTP chain is cut to `mtp_dcut::cost::budget`, and every serial
//!   round feeds `observe_verify` (all plans) and `observe_outcome` (MTP
//!   rounds with drafter confidences).
//!
//! With both rows off `begin_serial` returns an inert round at its first
//! line and `end_serial` returns at its first line: the dispatch is exactly
//! today's. `Tree` is not built in this sprint: it is routed to `Mtp` with a
//! one-time log, and wave 2 (W01) replaces that arm with the tree verify.

use super::*;

/// What one speculative round verifies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoundPlan {
    /// The MTP drafter's chain (`drafts` of them; 0 = no speculative step).
    Mtp { drafts: usize },
    /// A copy continuation filling the MTP verify chain (same width).
    CopyChain { tokens: Vec<u32> },
    /// A confident copy verified alone (up to 15 tokens), drafter skipped.
    LookupRound { tokens: Vec<u32> },
    /// A draft tree (wave 2); routed to `Mtp` in this sprint.
    Tree { shape_id: u32 },
}

/// What one verified round produced.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RoundOutcome {
    /// Draft tokens verified.
    pub proposed: usize,
    /// Draft tokens committed (the bonus token is not counted).
    pub accepted: usize,
    /// Verify rows (`proposed + 1`).
    pub rows: usize,
    /// Host wall time of the verify step, microseconds.
    pub verify_us: u64,
    /// The drafter's top-1 LOG-probability per verified draft (the
    /// `pending_draft_conf` convention `mtp_dcut::cost::observe_outcome`
    /// reads); MTP rounds with confidences only, empty for copy plans, which
    /// have no drafter.
    pub draft_conf: Vec<f32>,
}

/// Lane L06's D-Cut v2 row (`mtp_dcut_cost_model`), as its cost module reads it.
#[inline]
fn dcut_v2_on() -> bool {
    super::mtp_dcut::cost::enabled()
}

/// Feed one serial round to L06's online cost model. `mtp_stats::account_step`
/// feeds only batched rounds (two or more planned sequences), so a serial
/// round is fed here and nowhere else. Copy rounds have no drafter
/// confidences, so they only time the verify.
fn record(o: &RoundOutcome) {
    super::mtp_dcut::cost::observe_verify(o.rows, o.verify_us);
    if !o.draft_conf.is_empty() {
        let n = o.draft_conf.len().min(o.proposed);
        super::mtp_dcut::cost::observe_outcome(&o.draft_conf[..n], o.accepted);
    }
}

/// Prefix-product survival (descending) of a chain from the drafter's top-1
/// LOG-probabilities, the shape `mtp_dcut::cost::budget` ranks.
pub(super) fn survival_desc(conf_log: &[f32]) -> Vec<f32> {
    let mut acc = 0.0f32;
    conf_log
        .iter()
        .map(|&l| {
            acc += l.min(0.0);
            acc.exp()
        })
        .collect()
}

/// L06's budget for one sequence's chain, clamped to `1..=drafts`.
fn budgeted(conf_log: &[f32], drafts: usize) -> usize {
    if conf_log.len() < drafts || drafts < 2 {
        return drafts;
    }
    let s = survival_desc(&conf_log[..drafts]);
    match super::mtp_dcut::cost::budget(&s, 1) {
        Some(b) => b.clamp(1, drafts),
        None => drafts,
    }
}

/// `Tree` has no verify in this sprint: verify the MTP chain instead.
fn tree_fallback(shape_id: u32) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::info!(
            "round plan: tree verify not built (shape {shape_id}); verifying the MTP chain"
        );
    });
}

/// One serial round in flight between [`begin_serial`] and [`end_serial`].
pub(super) struct SerialRound {
    /// `None`: inert (every row off), nothing to record.
    t0: Option<Instant>,
    seq_len0: usize,
    proposed: usize,
    conf: Vec<f32>,
}

impl SerialRound {
    fn inert() -> Self {
        Self {
            t0: None,
            seq_len0: 0,
            proposed: 0,
            conf: Vec::new(),
        }
    }
}

/// Plan one serial round over `drafts` (already grammar-truncated) with the
/// drafter confidences `draft_conf` (top-1 log-probabilities, possibly
/// empty). May rewrite `drafts` in place. Returns `None` when the round was
/// already run here (a lookup round); the caller then skips its dispatch.
#[allow(clippy::too_many_arguments)]
pub(super) fn begin_serial(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &mut Vec<u32>,
    draft_conf: &[f32],
    num_drafts: usize,
    n_active: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) -> Option<SerialRound> {
    let dcut = dcut_v2_on();
    if !dcut && !super::copy_draft::active() {
        return Some(SerialRound::inert());
    }
    let plan = super::copy_draft::plan_serial(model, a, drafts, n_active, dflash_verify_raw_argmax);
    let plan = match plan {
        RoundPlan::Tree { shape_id } => {
            tree_fallback(shape_id);
            RoundPlan::Mtp {
                drafts: drafts.len(),
            }
        }
        p => p,
    };
    let mut conf: Vec<f32> = Vec::new();
    match plan {
        RoundPlan::LookupRound { tokens } => {
            let seq_len0 = a.seq.seq_len;
            let t0 = Instant::now();
            step_verify_dflash(model, a, sched, &tokens, num_drafts, verify_ctx, false);
            if dcut {
                end_serial(
                    a,
                    SerialRound {
                        t0: Some(t0),
                        seq_len0,
                        proposed: tokens.len(),
                        conf: Vec::new(),
                    },
                );
            }
            return None;
        }
        RoundPlan::CopyChain { tokens } => {
            debug_assert_eq!(tokens.len(), drafts.len());
            *drafts = tokens;
        }
        RoundPlan::Mtp { drafts: 0 } => {
            // `seq.port.no_drafts` rows never propose, so they never reach a
            // verify; a stale chain is dropped rather than verified.
            drafts.clear();
            return None;
        }
        RoundPlan::Mtp { .. } | RoundPlan::Tree { .. } => {
            if dcut && !dflash_verify_raw_argmax && a.grammar_state.is_none() {
                let keep = budgeted(draft_conf, drafts.len());
                drafts.truncate(keep);
            }
            if draft_conf.len() >= drafts.len() {
                conf = draft_conf[..drafts.len()].to_vec();
            }
        }
    }
    if !dcut {
        return Some(SerialRound::inert());
    }
    Some(SerialRound {
        t0: Some(Instant::now()),
        seq_len0: a.seq.seq_len,
        proposed: drafts.len(),
        conf,
    })
}

/// Close a serial round: time it and record its outcome (rows on only).
pub(super) fn end_serial(a: &ActiveSeq, round: SerialRound) {
    let Some(t0) = round.t0 else {
        return;
    };
    if a.error.is_some() || round.proposed == 0 {
        return;
    }
    // The verify advanced seq_len by proposed + 1 and the verdict rewound the
    // rejected tail: what is left past the anchor is accepted + 1.
    let accepted = a
        .seq
        .seq_len
        .saturating_sub(round.seq_len0)
        .saturating_sub(1)
        .min(round.proposed);
    record(&RoundOutcome {
        proposed: round.proposed,
        accepted,
        rows: round.proposed + 1,
        verify_us: t0.elapsed().as_micros() as u64,
        draft_conf: round.conf,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn survival_is_the_descending_prefix_product() {
        let s = survival_desc(&[(0.9f32).ln(), (0.5f32).ln(), (0.8f32).ln()]);
        let want = [0.9f32, 0.45, 0.36];
        for (a, b) in s.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{s:?}");
        }
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
        // A positive log-probability (numerical noise) never raises survival.
        let s = survival_desc(&[0.01, -0.1]);
        assert!(s[0] <= 1.0 && s[1] <= s[0]);
    }

    #[test]
    fn plans_compare_by_value() {
        assert_eq!(
            RoundPlan::CopyChain { tokens: vec![1, 2] },
            RoundPlan::CopyChain { tokens: vec![1, 2] }
        );
        assert_ne!(
            RoundPlan::Mtp { drafts: 3 },
            RoundPlan::LookupRound { tokens: vec![] }
        );
        assert_eq!(RoundOutcome::default().rows, 0);
    }
}
