// SPDX-License-Identifier: AGPL-3.0-only

//! Phase-C: mid-step rollback + re-steer for decode-time watchdogs.
//!
//! Atlas's degeneration watchdogs (content-phase loop, fuzzy-repetition,
//! inter-tool prose budget) historically *hard-stopped* a sequence —
//! `finished = true` — which kills the response, often mid-tool-call.
//!
//! Per arXiv:2603.27905 (ATLAS-RTC) and ROM boundary-truncation
//! (arXiv:2603.22016), the principled recovery is to **roll back to the
//! last well-formed boundary and let generation re-steer**, rather than
//! discarding the whole turn. [`rollback_to_boundary`] implements that.
//!
//! ## KV-cache rewind: what is and is not feasible mid-decode
//!
//! Atlas uses a **paged** attention KV cache. A decode step writes K/V
//! into the slot at `seq.seq_len`, then advances `seq.seq_len`. So
//! "rewinding" attention is simply lowering `seq.seq_len`: the stale K/V
//! slots beyond the new length are overwritten by the next decode and
//! never read in the interim (attention only reads `[0, seq_len)`). The
//! paged `block_table` is left intact — its physical blocks are reused.
//! This is exactly the rewind the self-speculative path already performs
//! (`spec_step.rs`: `seq.seq_len -= 1; seq.tokens.pop()`), so we reuse
//! that mechanism rather than inventing a new one.
//!
//! **SSM / Mamba layer state CANNOT be rewound an arbitrary N tokens
//! mid-decode.** The recurrent `h_state` / `conv_state` is updated
//! in-place every token; `rollback_ssm_states` can only restore to a
//! checkpoint, and per-token checkpoints (`h_state_intermediates`) are
//! populated *only* during MTP verify, not normal decode. There is no
//! safe way to undo M recurrent updates without those intermediates.
//!
//! Consequence for **hybrid models** (e.g. Qwen3.6-A3B with GDN layers):
//! after a rollback the attention KV is correct but the SSM state still
//! reflects the dropped tail tokens. We accept this as a bounded
//! approximation — the SSM state conditions on a token prefix that is
//! still ~95% valid (we only drop the short degenerate tail), and the
//! re-steer cue plus a fresh sampling trajectory pulls generation out of
//! the attractor. This is the same risk profile the self-spec path runs
//! under for SSM models, and the whole feature is gated behind the
//! `[behavior].rollback_resteer` MODEL.toml flag so it can be disabled
//! for any model where the approximation proves unacceptable. We do NOT
//! fake an SSM rewind — the SSM state is deliberately left untouched and
//! this limitation is logged.

use super::*;

/// Outcome of a [`rollback_to_boundary`] attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollbackOutcome {
    /// The sequence was rewound to a boundary; generation should continue.
    RolledBack {
        /// Number of trailing tokens dropped from `output_tokens`.
        dropped: usize,
    },
    /// Rollback did not happen — the caller should fall back to the
    /// legacy hard stop (`finished = true`).
    Fallback(RollbackFallback),
}

/// Why a rollback was declined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollbackFallback {
    /// `[behavior].rollback_resteer = false`.
    Disabled,
    /// The per-sequence rollback cap was already reached.
    CapReached,
    /// No well-formed boundary was found in the generated tail.
    NoBoundary,
}

/// Find the index (into `output_tokens`) of the last well-formed
/// boundary token, searching backwards but stopping `min_keep` tokens
/// before the end so the rollback always discards at least `min_keep`
/// degenerate tokens.
///
/// A "boundary" is any token whose `mask[id]` is set — built at startup
/// from the tokenizer to mark tokens decoding to a newline or
/// sentence-ending punctuation (see `helpers::set_boundary_token_mask`).
///
/// Returns the boundary token's index. Generation resumes *after* it
/// (the boundary token itself is kept). `None` when no boundary exists
/// in the searchable region.
pub fn find_last_boundary(output_tokens: &[u32], mask: &[bool], min_keep: usize) -> Option<usize> {
    let n = output_tokens.len();
    if n <= min_keep {
        return None;
    }
    // Scan backwards from `n - 1 - min_keep` (inclusive) to 0.
    let search_end = n - 1 - min_keep;
    for idx in (0..=search_end).rev() {
        let id = output_tokens[idx] as usize;
        if id < mask.len() && mask[id] {
            return Some(idx);
        }
    }
    None
}

/// Roll an [`ActiveSeq`] back to the last well-formed boundary and
/// re-steer, instead of hard-stopping.
///
/// Steps:
/// 1. Honor the `[behavior].rollback_resteer` flag and the per-sequence
///    [`atlas_kernels::ROLLBACK_RESTEER_CAP`].
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
pub fn rollback_to_boundary(a: &mut ActiveSeq, min_keep: usize) -> RollbackOutcome {
    if !watchdog_params().rollback_resteer {
        return RollbackOutcome::Fallback(RollbackFallback::Disabled);
    }
    if a.rollback_count >= atlas_kernels::ROLLBACK_RESTEER_CAP {
        return RollbackOutcome::Fallback(RollbackFallback::CapReached);
    }
    let mask = match boundary_token_mask() {
        Some(m) => m,
        None => return RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
    };
    let boundary_idx = match find_last_boundary(&a.output_tokens, &mask, min_keep) {
        Some(i) => i,
        None => return RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
    };

    // Tokens to drop = everything strictly after the boundary token.
    let keep_len = boundary_idx + 1;
    let dropped = a.output_tokens.len() - keep_len;
    debug_assert!(dropped >= min_keep);

    apply_rollback(a, keep_len, dropped);
    a.rollback_count = a.rollback_count.saturating_add(1);
    RollbackOutcome::RolledBack { dropped }
}

/// Token-buffer rewind applied to a generated-token buffer and the
/// paged-attention sequence buffers. Pure over plain `Vec`s / scalars so
/// it is unit-testable without an [`ActiveSeq`] (which carries channels,
/// `Instant`s and a `SequenceState`). This is the load-bearing KV-rewind
/// step — see the module doc on why lowering `seq_len` *is* the
/// attention rewind.
///
/// Returns the new `seq_len`.
pub fn rewind_buffers(
    output_tokens: &mut Vec<u32>,
    seq_tokens: &mut Vec<u32>,
    seq_len: usize,
    keep_len: usize,
) -> usize {
    let dropped = output_tokens.len().saturating_sub(keep_len);
    output_tokens.truncate(keep_len);
    let mut new_seq_len = seq_len;
    for _ in 0..dropped {
        if seq_tokens.pop().is_some() {
            new_seq_len = new_seq_len.saturating_sub(1);
        }
    }
    new_seq_len
}

/// Apply the truncation + KV/position rewind + watchdog-state reset to a
/// live [`ActiveSeq`]. Delegates the buffer rewind to [`rewind_buffers`].
fn apply_rollback(a: &mut ActiveSeq, keep_len: usize, dropped: usize) {
    // 1+2. Truncate the generated-token buffer and rewind the
    //       attention-KV cursor (`seq.tokens` + `seq_len`).
    a.seq.seq_len = rewind_buffers(
        &mut a.output_tokens,
        &mut a.seq.tokens,
        a.seq.seq_len,
        keep_len,
    );

    // 3. Restore the generation budget that the dropped tokens consumed
    //    (only content tokens decrement `remaining`; thinking is free,
    //    and the watchdogs that call this only fire post-`</think>`).
    a.remaining = a.remaining.saturating_add(dropped);
    a.content_tokens = a.content_tokens.saturating_sub(dropped as u32);

    // 4. Re-point the decode cursor at the boundary token.
    if let Some(&last) = a.output_tokens.last() {
        a.last_token = last;
    }

    // 5. Rewind the grammar FSM by the same token count so the
    //    constrained-decoding matcher stays in sync with the truncated
    //    token stream. Every dropped token is a post-`</think>` content
    //    token (the watchdogs that call this fire after thinking has
    //    closed) and was therefore fed to `grammar_state.accept_token`,
    //    so `rollback(dropped)` is exact. Reuses the existing
    //    spec-decode grammar-rewind path (`GrammarState::rollback`).
    if let Some(ref mut gs) = a.grammar_state {
        gs.rollback(dropped);
    }

    // 6. Reset the watchdog accumulators so the just-cleared window does
    //    not immediately re-trigger before fresh tokens arrive.
    a.prose_tokens_since_last_tool = 0;
    a.consecutive_confident = 0;
}

// ── Phase-C ROM (arXiv:2603.22016) scaffold — OPTIONAL hook ──────────
//
// The principled replacement for the F2 confidence early-stop heuristic
// is a trained Repetition-Onset-Model detection head. We do NOT ship a
// ROM model: it needs a per-model trained head that Atlas does not have,
// and a heuristic stand-in would not be the principled detector ROM
// describes. So the items below are an intentional forward-looking SEAM,
// not yet consumed by any call site — `#[allow(dead_code)]` is therefore
// deliberate. A future revision adds the artifact loader: it reads
// MODEL.toml `[behavior].rom_head`, builds a `RomHead`, and installs it
// via `set_rom_head`; F2 in `decode_logits_seq.rs` would then consult
// `rom_head()` and defer to the trained head when present. Until then
// `rom_head()` returns `None` and F2 stays the unchanged fallback.

/// Trained Repetition-Onset-Model detection head (arXiv:2603.22016).
/// See the module's ROM-scaffold comment block. Intentionally unused
/// until the artifact loader lands.
#[allow(dead_code)]
pub trait RomHead: Send + Sync {
    /// Score the probability `[0.0, 1.0]` that the sequence whose recent
    /// output is `recent_tokens` (most recent last) has entered a
    /// repetition-onset state. A trained head consumes hidden states; an
    /// artifact-backed implementation would receive them through an
    /// extended signature in a future revision — this minimal seam keeps
    /// the absent-head path zero-cost.
    fn repetition_onset_score(&self, recent_tokens: &[u32]) -> f32;
}

static ROM_HEAD: std::sync::OnceLock<std::sync::Arc<dyn RomHead>> = std::sync::OnceLock::new();

/// Install a trained ROM detection head. Idempotent. Called at startup
/// only when `[behavior].rom_head` names a loadable artifact. When never
/// called, [`rom_head`] returns `None` and F2 remains the fallback.
/// Intentionally unused until the artifact loader lands (see the
/// ROM-scaffold comment block above).
#[allow(dead_code)]
pub fn set_rom_head(head: std::sync::Arc<dyn RomHead>) {
    let _ = ROM_HEAD.set(head);
}

/// Read the installed ROM head, if any. `None` until [`set_rom_head`]
/// runs — callers MUST treat `None` as "use the F2 fallback".
/// Intentionally unused until the artifact loader lands.
#[allow(dead_code)]
pub fn rom_head() -> Option<std::sync::Arc<dyn RomHead>> {
    ROM_HEAD.get().cloned()
}

#[cfg(test)]
#[path = "rollback_tests.rs"]
mod rollback_tests;
