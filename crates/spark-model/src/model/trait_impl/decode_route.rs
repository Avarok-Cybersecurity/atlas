// SPDX-License-Identifier: AGPL-3.0-only

//! The batched-decode routing gate: batched multi-seq step, or the legacy
//! per-sequence staging loop?
//!
//! Split out of `decode_a2.rs` so the decision is a pure function with unit
//! tests that need no GPU. `decode_batch_dispatch` calls
//! [`hc_perseq_fallback`] and nothing else decides this.

/// `true` when an mHC (hyper-connection) batch must fall back to the
/// per-sequence staging loop in `decode_batch_dispatch` — one full model
/// forward per sequence per step plus a full-vocab D2H/H2D logits round
/// trip, with CUDA graphs suppressed.
///
/// ## `qsa_active` is RETIRED as a fallback trigger
///
/// It used to force the fallback: the batched multi-seq attention path ran
/// `QsaIndexer::decode_select` per row for ingest continuity but had no
/// consumer for a `Some(QsaSelection)`, so it refused. Any batch holding
/// one sequence past the indexer's inert bound (`index_topk +
/// index_compress_ratio - 1` = 2051 on qwen3.8-flash-next) therefore
/// abandoned batching ENTIRELY — measured at C=4 / ISL 4096: 4.0 tok/s
/// aggregate versus 23.4 at ISL 1024, with zero `ATLAS_DECODE_BATCH` lines
/// in the whole run.
///
/// `layers/qwen3_attention/trait_impl/multi_seq/qsa.rs` now consumes a
/// per-row selection (select row `i`, attend row `i` against its selected
/// KV scratch, then move on — the indexer's scratch is shared, so it must
/// be consumed before the next row selects). The argument is kept so this
/// gate documents — and the tests pin — that a long sequence no longer
/// takes a batch off the batched path.
///
/// `perseq_env` is `ATLAS_HC_PERSEQ_DECODE=1`, the kill switch that
/// restores the old loop; the MLA escape hatch
/// (`ATLAS_MLA_PERSEQ_FALLBACK`) is a separate term at the call site and is
/// untouched.
///
/// ## `multi_rank` still forces the fallback
///
/// Retiring `qsa_active` makes EP/TP + QSA-active + batched multi-seq decode
/// REACHABLE FOR THE FIRST TIME, and a rank that disagrees with its peer
/// about which collectives to issue does not produce a wrong answer — it
/// hangs both ranks. Single-node is measured (ISL 4096, C=4: 3.0 -> 6.2
/// tok/s per sequence); multi-rank is not, so it keeps the proven per-seq
/// route until a two-node run says otherwise. Drop this term — do not add
/// another env var — once EP=2 has been validated at long context.
pub(crate) fn hc_perseq_fallback(
    hc_mult: usize,
    qsa_active: bool,
    perseq_env: bool,
    multi_rank: bool,
) -> bool {
    hc_mult > 0 && (perseq_env || (qsa_active && multi_rank))
}

#[cfg(test)]
mod tests {
    use super::hc_perseq_fallback;

    /// The regression this change exists for: on qwen3.8-flash-next
    /// (`hc_mult` 4) a batch containing a sequence past the QSA inert bound
    /// takes the BATCHED dispatch, not the per-seq staging loop.
    #[test]
    fn qsa_active_batch_stays_on_the_batched_path() {
        assert!(!hc_perseq_fallback(4, true, false, false));
    }

    /// ATLAS_HC_PERSEQ_DECODE=1 still restores the old loop, QSA or not.
    #[test]
    fn kill_switch_restores_the_per_seq_loop() {
        assert!(hc_perseq_fallback(4, true, true, false));
        assert!(hc_perseq_fallback(4, false, true, false));
    }

    /// Non-hc models never take this fallback, even with the kill switch —
    /// their batched path predates the highway and is the only one wired.
    #[test]
    fn non_hc_models_are_never_routed_per_seq() {
        assert!(!hc_perseq_fallback(0, true, true, false));
        assert!(!hc_perseq_fallback(0, false, false, false));
    }

    /// A short-context hc batch was already batched and stays batched.
    #[test]
    fn short_context_hc_batch_stays_batched() {
        assert!(!hc_perseq_fallback(4, false, false, false));
    }
    /// Multi-rank (EP/TP) keeps QSA-active batches on the proven per-seq
    /// route: the batched path is unmeasured across ranks and a collective
    /// disagreement hangs the pair rather than returning a wrong answer.
    /// Short-context batches are unaffected — they never took the fallback.
    #[test]
    fn multi_rank_keeps_qsa_active_batches_per_seq() {
        assert!(hc_perseq_fallback(4, true, false, true));
        assert!(!hc_perseq_fallback(4, false, false, true));
        // Single node is the measured path and stays batched.
        assert!(!hc_perseq_fallback(4, true, false, false));
    }
}
