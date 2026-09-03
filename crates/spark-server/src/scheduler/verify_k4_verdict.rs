// SPDX-License-Identifier: AGPL-3.0-only

//! K-row verify verdict application: accept/rewind/emit/re-propose.
//!
//! Extracted VERBATIM from `verify_k4_step.rs` (behavior-identical refactor,
//! batched-MTP E9) so the single-seq `step_verify_k4` and the batched
//! `step_verify_k4_batched` share ONE copy of the accept logic — rewind
//! arithmetic, `trim_proposer_state`, `commit_accepted_prefix`, emit order,
//! and `k4_record_outcome` are the existing machinery unchanged. The K-vs-
//! batch ladder generalized the four hardcoded K=4 accept branches into the
//! single formula they all instantiated (emit `drafts[0..na]` + the
//! correction/bonus `v[na]`, rewind `nd-na`, `trim_proposer_state(na)`,
//! `commit_accepted_prefix(na+1, k_rows)`) — behavior-identical at K=4 by
//! construction, and now valid for the ladder's K=2/3 batched verifies.
//! The only parameterization is WHERE the accepted-row hidden for the next
//! propose comes from ([`K4Hidden`]): the live verify row (single-seq path,
//! `save_hidden_for_mtp`) or the pre-propose stash slot (batched path, whose
//! phase-3 proposes have already clobbered the live rows).

use super::*;
use super::verify_k2_step::commit_verify_aux_or_finish;

/// Source of the accepted-position hidden fed to `run_mtp_propose_multi`.
#[derive(Clone, Copy)]
pub(super) enum K4Hidden {
    /// Read the live verify forward's row `num_accepted` (single-seq path).
    VerifyRow,
    /// Read stash slot `i` written by `stash_verify_hidden_rows` BEFORE any
    /// propose ran (batched path).
    Stash(usize),
    /// Batched-propose deferral: apply the verdict (emit / rewind / trim /
    /// commit) but skip BOTH the hidden save and the re-propose — the caller
    /// runs ONE cross-sequence batched propose afterwards (stash slot `i`),
    /// falling back to per-seq save+propose when unsupported.
    DeferPropose,
}

#[inline]
fn save_hidden(model: &dyn Model, hidden: K4Hidden, na: usize) -> anyhow::Result<()> {
    match hidden {
        K4Hidden::VerifyRow => model.save_hidden_for_mtp(na, 0),
        K4Hidden::Stash(i) => model.save_hidden_for_mtp_from_stash(i, 0),
        // Deferred mode never saves inline (the batched propose reads the
        // stash rows directly; the per-seq fallback re-saves per sequence).
        K4Hidden::DeferPropose => Ok(()),
    }
}

/// Apply a K-row verify verdict to one sequence: emit the accepted prefix +
/// correction/bonus token, rewind `seq_len`/`tokens` for rejected drafts,
/// roll back proposer + SSM state, save the accepted-position hidden, and
/// re-propose.
///
/// `drafts.len()` = nd (the drafts verified this step), `v.len()` = nd + 1
/// (per-row picks incl. the bonus row), `num_accepted` in 0..=nd. At nd=3
/// this is the verbatim four-branch tail of the pre-refactor
/// `step_verify_k4`, branch-collapsed; the phase ORDER of each original
/// branch is preserved exactly (full accept: emit → commit → save → trim →
/// propose; partial/reject: rewind → trim → commit → emit → save → propose).
#[allow(clippy::too_many_arguments)]
pub(super) fn k4_apply_verdict(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    v: &[u32],
    verify_lps: Vec<crate::api::TokenLogprobs>,
    num_drafts: usize,
    num_accepted: usize,
    hidden: K4Hidden,
    verify_us: u128,
) {
    let defer = matches!(hidden, K4Hidden::DeferPropose);
    // ★ The rewind arithmetic below is sized by the FORWARD, not the draft
    // list. `v` has one pick per verified row, so `v.len()` is ground truth
    // for how many rows the forward committed (`seq_len += k_rows` happened
    // in the model call); `drafts` is whatever the caller carried and CAN be
    // longer. The single-seq K=4 step verifies only `drafts[0..3]` of a
    // sequence that may hold DFlash γ=7 pending drafts — passing the full
    // slice here made `nd = 7`, and `seq_len -= nd - na` rewound 4 rows the
    // 4-row forward never added: the verdict EMITTED its accepted tokens and
    // then erased them from the sequence (seq 46 -> 42, tokens popped back to
    // the prompt, last_token left dangling on the emitted bonus). The model
    // then continued from a history missing its own output — measured
    // 2026-08-21 as the prefix-cache/concurrency "class Minimizing"
    // derailment, though the trigger is any surplus-draft dispatch, cache or
    // not. The old `debug_assert_eq!(v.len(), k_rows)` knew the contract and
    // compiled out in release; the clamp below enforces it.
    let k_rows = v.len();
    debug_assert!(k_rows >= 1, "verdict needs at least the bonus row");
    let nd = drafts.len().min(k_rows.saturating_sub(1));
    if drafts.len() > nd {
        // Surplus drafts were never verified: they are dropped here (the
        // caller already took them out of `pending_drafts`), and the
        // drafter's own row accounting is handled by `trim_proposer_state`,
        // which trims from the drafter's internal `last_num_drafted` — not
        // from this slice's length.
        tracing::debug!(
            "k4 verdict: {} drafts carried into a {}-row verify — verifying the \
             first {nd}, dropping the surplus",
            drafts.len(),
            k_rows,
        );
    }
    let drafts = &drafts[..nd];
    let na = num_accepted.min(nd);

    if na == nd {
        // ── Full accept: every draft matched; v[nd] is the free bonus. ──
        for j in 0..nd {
            emit_token(a, drafts[j], verify_lps.get(j).cloned(), sched);
            if a.finished {
                return;
            }
        }
        emit_token(a, v[nd], verify_lps.get(nd).cloned(), sched);
        if a.finished {
            return;
        }
        a.last_token = v[nd];

        // Item #2 (STree-style in-place verify commit). Full accept
        // (num_accepted == k): the verify kernel already wrote the canonical
        // h_state, so the commit is a no-op.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, k_rows, k_rows) {
            // SSM state is no longer trustworthy — terminate, do not continue.
            tracing::error!("commit_accepted_prefix (K={k_rows} accept-{k_rows}): {e:#}");
            a.finished = true;
            return;
        }
        // `decode_verify_graphed_k4` routes to the mHC mini-prefill too, so this
        // path owes the auxiliary carries the same commit K=2/K=3 owe them.
        if !commit_verify_aux_or_finish(model, a, k_rows, k_rows) {
            return;
        }
    } else {
        // ── Partial accept / reject: rewind the rejected tail. ──
        a.seq.seq_len -= nd - na;
        for _ in 0..(nd - na) {
            a.seq.tokens.pop();
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2: rewind live h_state to intermediate[na] (state after the
        // last accepted row — the correction token v[na] is row na).
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, na + 1, k_rows) {
            tracing::error!(
                "commit_accepted_prefix (K={k_rows} accept-{}): {e:#}",
                na + 1
            );
            a.finished = true;
            return;
        }
        if !commit_verify_aux_or_finish(model, a, na + 1, k_rows) {
            return;
        }
        for j in 0..na {
            emit_token(a, drafts[j], verify_lps.get(j).cloned(), sched);
            if a.finished {
                return;
            }
        }
        emit_token(a, v[na], verify_lps.get(na).cloned(), sched);
        if a.finished {
            return;
        }
        a.last_token = v[na];
    }

    if !defer && let Err(e) = save_hidden(model, hidden, na) {
        tracing::error!("save_hidden_for_mtp({na}): {e:#}");
        return;
    }
    if na == nd {
        // Full-accept branch trims AFTER the hidden save (original order).
        if let Err(e) = model.trim_proposer_state(&mut a.seq, na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
    }
    if !defer {
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            a.last_token,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        tracing::debug!(
            "K{k_rows} ACCEPT-{na}: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
    }
    crate::scheduler::verify_k4_step::stats::k4_record_outcome(sched, na, a.seq.seq_len);
}
