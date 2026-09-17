// SPDX-License-Identifier: AGPL-3.0-only

//! `rollback_to_boundary` itself, split out of `rollback.rs` to keep it
//! under the 500-line cap.

use super::*;

/// Roll an [`ActiveSeq`] back to the last well-formed boundary and
/// re-steer, instead of hard-stopping.
///
/// Steps:
/// 1. Honor the `[behavior].rollback_resteer` flag and the per-sequence
///    [`avarok_kernels::ROLLBACK_RESTEER_CAP`].
/// 2. Find the last boundary token in `output_tokens`
///    ([`find_last_boundary`]); decline if none.
/// 3. Truncate `output_tokens` back to and including that boundary.
/// 4. Rewind `seq.tokens` and `seq.seq_len` by the same count — this is
///    the attention-KV rewind (paged slots beyond `seq_len` are
///    overwritten by the next decode; see the module doc).
/// 5. Restore `remaining` so the recovered budget is not lost, reset
///    `last_token` to the boundary token, rewind the grammar FSM
///    (`GrammarState::rollback`) by the dropped-token count so
///    constrained decoding stays in sync, and clear the watchdog
///    accumulators that drove the trigger (so the same window does not
///    instantly re-fire).
/// 6. Bump `rollback_count`.
///
/// The "re-steer cue" is the rollback itself: with the degenerate tail
/// removed, the model resumes from a clean boundary and — because the
/// repeated suffix is gone and the sampler state advances — naturally
/// picks a different continuation. No synthetic context tokens are
/// injected (that would require a tokenizer-specific cue string and
/// risk corrupting tool-call structure); the boundary truncation is the
/// minimal, structure-safe steering signal.
///
/// `min_keep` is the minimum number of trailing tokens the rollback
/// must discard — it must be large enough to escape the detected
/// attractor's last period.
///
/// ## Hybrid-model SSM rewind
///
/// When `model.has_ssm_layers()` and the sequence's decode ring is
/// enabled, boundary selection is restricted to a boundary with a live
/// SSM snapshot ([`find_last_boundary_with_snapshot`]); the rollback
/// then restores the recurrent state through
/// `Model::restore_decode_ssm_snapshot`. If no boundary has a snapshot,
/// the rollback is declined with [`RollbackFallback::NoSsmSnapshot`] —
/// the caller hard-stops. Pure-attention models keep the original
/// any-boundary behavior. A model-side restore failure is also surfaced
/// as a decline (the caller hard-stops cleanly rather than continuing on
/// corrupt SSM state).
pub fn rollback_to_boundary(
    a: &mut ActiveSeq,
    min_keep: usize,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> RollbackOutcome {
    if !sched.watchdog.rollback_resteer {
        return RollbackOutcome::Fallback(RollbackFallback::Disabled);
    }
    // Streaming requests flush every token to the client as it is
    // sampled — a re-steer cannot retract them (see StreamUnsafe doc).
    // `cancel_flag` is Some exactly for streaming requests.
    if a.cancel_flag.is_some() {
        return RollbackOutcome::Fallback(RollbackFallback::StreamUnsafe);
    }
    if a.rollback_count >= avarok_kernels::ROLLBACK_RESTEER_CAP {
        return RollbackOutcome::Fallback(RollbackFallback::CapReached);
    }
    let mask = match sched.masks.boundary.as_ref() {
        Some(m) => m.clone(),
        None => return RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
    };

    // A hybrid model needs the SSM state rewound too — restrict boundary
    // selection to one with a live snapshot. A DISABLED ring is a DECLINE,
    // not the pure-attention any-boundary path: depth 0 (spec on, watchdogs
    // off, or the #915 auto-fit floor) means no snapshot exists, and rewinding
    // tokens while the recurrent state stays conditioned on the discarded tail
    // is silent corruption. This is the fail-open the ring's SSOT documents.
    let hybrid = model.has_ssm_layers();
    if hybrid && !a.ssm_rollback_ring.is_enabled() {
        return RollbackOutcome::Fallback(RollbackFallback::NoSsmSnapshot);
    }
    let (boundary_idx, ssm_slot) = if hybrid {
        match find_last_boundary_with_snapshot(
            &a.output_tokens,
            &mask,
            min_keep,
            &a.ssm_rollback_ring,
        ) {
            Some(i) => {
                // Snapshot slot is guaranteed present by the search.
                let slot = a.ssm_rollback_ring.slot_for_position(i + 1);
                (i, slot)
            }
            None => {
                // A plain boundary may still exist; distinguish "no
                // boundary at all" from "boundary without snapshot" so
                // the operator log is precise.
                let reason = if find_last_boundary(&a.output_tokens, &mask, min_keep).is_some() {
                    RollbackFallback::NoSsmSnapshot
                } else {
                    RollbackFallback::NoBoundary
                };
                return RollbackOutcome::Fallback(reason);
            }
        }
    } else {
        match find_last_boundary(&a.output_tokens, &mask, min_keep) {
            Some(i) => (i, None),
            None => return RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
        }
    };

    // Tokens to drop = everything strictly after the boundary token.
    let keep_len = boundary_idx + 1;
    let dropped = a.output_tokens.len() - keep_len;
    debug_assert!(dropped >= min_keep);

    // Restore the SSM recurrent state BEFORE truncating the buffers, so
    // that on a model-side failure we decline without having mutated the
    // token buffers (the sequence stays in a consistent state for the
    // caller's hard-stop fallback).
    if let Some(slot) = ssm_slot {
        if let Err(e) = model.restore_decode_ssm_snapshot(&a.seq, slot) {
            tracing::error!(
                error = %e,
                ring_slot = slot,
                keep_len,
                "SSM decode-snapshot restore failed; declining rollback"
            );
            return RollbackOutcome::Fallback(RollbackFallback::NoSsmSnapshot);
        }
        // AUX companion (QSA cursor / PLE history) — restored from the
        // SAME ring slot, and BEFORE any buffer truncation for the same
        // reason as the SSM half: a failure must leave the sequence
        // untouched for the caller's hard-stop. A model with no aux state
        // returns Ok(()) here, so this is inert for pure-attention and
        // plain-SSM models.
        if model.requires_aux_state()
            && let Err(e) = model.restore_decode_aux_snapshot(&mut a.seq, slot)
        {
            tracing::error!(
                error = %e,
                ring_slot = slot,
                keep_len,
                "aux decode-snapshot restore failed; declining rollback \
                 (rolling back without it desyncs the QSA indexer)"
            );
            return RollbackOutcome::Fallback(RollbackFallback::NoAuxSnapshot);
        }
        // The degenerate tail's snapshots are now stale — drop them so
        // their ring slots are reusable. The boundary snapshot itself is
        // kept (generation resumes from it).
        a.ssm_rollback_ring.truncate_after(keep_len);
    }

    apply_rollback(a, keep_len, dropped);
    a.rollback_count = a.rollback_count.saturating_add(1);
    RollbackOutcome::RolledBack { dropped }
}
