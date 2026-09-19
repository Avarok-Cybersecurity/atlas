// SPDX-License-Identifier: AGPL-3.0-only

//! The K-position pick loop of `verify_pick_all_with_pipeline`, split out so
//! it can be driven over host logits without a model.

use super::verify_pick_with_pipeline;
use crate::scheduler::logit_processors::LogitsContext;
use crate::scheduler::types::ActiveSeq;

/// Run the pre-sample pipeline over `k` host-resident logits rows and return
/// the processed pick per position.
///
/// Positions after the first are masked against the matcher state that the
/// EARLIER picks would leave behind (speculative `accept_token`, rolled back
/// at the end so `emit_token` re-advances from a clean matcher).
///
/// Reasoning boundary (2026-09-06): a row may cross `</think>`. The
/// sequence's `inside_thinking` is its state at step START, so without
/// tracking the close every later position was picked free-run and the
/// unconstrained token was then fed to a pristine `required`/schema grammar
/// by `emit_token` — a guaranteed disengage on the first content token. A
/// `</think>` pick now flips the thinking flags for the REST of this loop
/// (mirroring what `emit_token` will do for real), so the first post-think
/// position is picked under the pristine grammar; drafts that disagree are
/// simply rejected by the verifier. The flags are restored on exit — this
/// loop only picks, it never commits.
pub(super) fn pick_positions_from_host(
    buf: &[u8],
    vocab: usize,
    elem_bytes: usize,
    k: usize,
    a: &mut ActiveSeq,
    ctx: &LogitsContext,
) -> Vec<u32> {
    let mut picks: Vec<u32> = Vec::with_capacity(k);
    // Snapshot the matcher's history depth BEFORE speculative advances so we
    // roll back exactly the ACTUAL advances afterward. BUG#3 (2026-06-02):
    // stop/EOS and terminated tokens return true from `accept_token` WITHOUT
    // advancing the matcher, so a count of `accept_token`→true calls would
    // over-rewind. `emit_token` (run after this helper) re-advances from the
    // restored, clean state.
    let grammar_steps_before = a.grammar_state.as_ref().map(|gs| gs.num_history_steps());
    let think_flags_before = (a.inside_thinking, a.think_ended, a.think_just_ended);

    for i in 0..k {
        let slice = &buf[i * vocab * elem_bytes..(i + 1) * vocab * elem_bytes];
        // P1-3 (2026-07-09): `i` threads the verify-position index down for
        // the per-position seed offset of the temp>0 sampling branch.
        let pick = verify_pick_with_pipeline(slice, false, vocab, a, ctx, i);
        picks.push(pick);

        // `</think>` closes the span for every later position. It is never
        // fed to the matcher (the grammar only sees content tokens), so no
        // speculative advance here either.
        if a.inside_thinking && ctx.think_end_token == Some(pick) {
            a.inside_thinking = false;
            a.think_ended = true;
            a.think_just_ended = true;
            continue;
        }

        // Speculatively advance the matcher with `pick[i]` so the next
        // position's bitmask reflects post-emit state. Skip on the last
        // position (no next position to mask) and when the seq has no
        // grammar (nothing to advance).
        if i + 1 < k
            && let Some(ref mut gs) = a.grammar_state
            && !a.inside_thinking
        {
            // Matcher advance can fail if `pick` is not in the current
            // bitmask. If our pipeline correctly applied the bitmask,
            // pick is the argmax over masked logits → MUST be in the
            // bitmask → advance MUST succeed. The defensive check
            // exists for forced-token fast-path returns where the
            // grammar may have terminated; those legitimately can't
            // advance further.
            if !gs.accept_token(pick) {
                tracing::debug!(
                    pick,
                    i,
                    "verify_pick: grammar speculative advance refused — pipeline picked a token outside the current bitmask. \
                     This indicates a stale bitmask in the pipeline or a forced-token fastpath that terminated grammar. \
                     Stopping speculation here; the real `accept_token` in emit_token will fail and end the response."
                );
                break;
            }
            // accept_token advanced the matcher as a side effect; the rollback
            // below counts the ACTUAL advances from matcher history (BUG#3).
        }
    }

    // Roll back exactly the ACTUAL speculative advances (history delta) so the
    // matcher returns to its pre-call state; `emit_token` then re-advances it
    // normally. BUG#3: counting from accept_token→true calls over-rewinds when
    // a stop/EOS/terminated token (which returns true WITHOUT advancing) lands
    // in the verified span.
    if let (Some(before), Some(gs)) = (grammar_steps_before, a.grammar_state.as_mut()) {
        let advanced = gs.num_history_steps().saturating_sub(before);
        if advanced > 0 {
            gs.rollback(advanced);
        }
    }
    // Same discipline for the thinking flags: `emit_token` owns the real
    // `</think>` transition on the accept path.
    (a.inside_thinking, a.think_ended, a.think_just_ended) = think_flags_before;

    picks
}
