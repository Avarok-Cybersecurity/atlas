// SPDX-License-Identifier: AGPL-3.0-only

//! Context-vs-emit ledger for the speculative verify paths.
//!
//! # The invariant
//!
//! Every accept/emit branch of a K-row verify must keep one property: the
//! token stream the MODEL ingested (`seq.tokens`) is, past the prompt, the
//! same stream the CLIENT received (`output_tokens`) — modulo the ids
//! `emit_token` deliberately swallows (`<think>`, a stray `</think>`, a
//! suppressed EOS) and the one token still pending in `a.last_token`.
//!
//! When the two disagree the scheduler committed a different token to the
//! model than it emitted, which is the substitution/duplication failure class
//! (an answer whose function signature says `s` and whose body says `string`).
//!
//! # Why it compares from the END
//!
//! `ActiveSeq` carries no prompt length, and the prefix cache can make
//! `seq.tokens` start anywhere. Comparing the two tails sidesteps both: at the
//! accept decision the model has ingested `[.., last_token, draft…]` while the
//! client has received `[.., last_token]`, so after dropping the `k - 1`
//! still-unverified draft rows the overlapping tails must agree token for
//! token.
//!
//! Diagnostic only — `RUST_LOG=spark::scheduler::verify_ledger=debug`. It was
//! written to answer "is the MTP substitution defect an accept/emit
//! bookkeeping bug?" and it answers that question in one run: 602/602 K=2
//! verify steps on qwen4_exp reported `rev_diverge=None`, which exonerates
//! this half of the pipeline and points at the verify forward instead.

use crate::scheduler::ActiveSeq;

/// Tokens of context/emit tail printed with each line.
const TAIL: usize = 10;

/// Distance from the END at which the two streams first disagree (0 = the last
/// element), or `None` when the overlapping tails agree.
///
/// Pure over slices so the contract is unit-testable with no GPU.
pub(super) fn rev_divergence(ingested: &[u32], emitted: &[u32]) -> Option<usize> {
    ingested
        .iter()
        .rev()
        .zip(emitted.iter().rev())
        .position(|(a, b)| a != b)
}

fn tail(v: &[u32], n: usize) -> &[u32] {
    &v[v.len().saturating_sub(n)..]
}

/// Log this verify step's computed / accepted / emitted triple plus the
/// context-vs-emit comparison.
///
/// `k_rows` is the VERIFY WIDTH (drafts + 1), i.e. how many rows the forward
/// appended to `seq.tokens`; `k_rows - 1` of them are drafts this step has not
/// committed yet and are excluded from the comparison.
pub(super) fn trace_ctx_vs_emit(
    tag: &str,
    a: &ActiveSeq,
    k_rows: usize,
    computed: &[u32],
    drafts: &[u32],
    num_accepted: usize,
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let n_draft = k_rows.saturating_sub(1);
    let ing = &a.seq.tokens[..a.seq.tokens.len().saturating_sub(n_draft)];
    let out: &[u32] = &a.output_tokens;
    let div = rev_divergence(ing, out);
    tracing::debug!(
        "{tag} LEDGER: k={k_rows} computed={computed:?} drafts={drafts:?} na={num_accepted} \
         last_token={} seq_len={} n_ing={} n_out={} rev_diverge={div:?} ing_tail={:?} out_tail={:?}",
        a.last_token,
        a.seq.seq_len,
        ing.len(),
        out.len(),
        tail(ing, TAIL),
        tail(out, TAIL),
    );
}

#[cfg(test)]
mod tests {
    use super::rev_divergence;

    #[test]
    fn agreeing_tails_do_not_diverge() {
        // prompt [9,9] then generated [1,2,3]; the client saw [1,2,3].
        assert_eq!(rev_divergence(&[9, 9, 1, 2, 3], &[1, 2, 3]), None);
        // The client is one token ahead (`last_token` is emitted, not ingested).
        assert_eq!(rev_divergence(&[9, 9, 1, 2], &[1, 2, 3]), Some(0));
    }

    #[test]
    fn substitution_is_reported_from_the_end() {
        // The model ingested 7 where the client received 3.
        assert_eq!(rev_divergence(&[9, 9, 1, 2, 7], &[1, 2, 3]), Some(0));
        assert_eq!(rev_divergence(&[9, 9, 1, 7, 3], &[1, 2, 3]), Some(1));
    }

    #[test]
    fn empty_streams_never_diverge() {
        assert_eq!(rev_divergence(&[], &[1, 2, 3]), None);
        assert_eq!(rev_divergence(&[1, 2, 3], &[]), None);
    }
}
