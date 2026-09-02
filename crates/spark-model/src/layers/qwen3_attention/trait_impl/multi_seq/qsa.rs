// SPDX-License-Identifier: AGPL-3.0-only

//! QSA (Qwen3.8-Flash-Next sparse-attention indexer) on the BATCHED
//! multi-sequence decode path.
//!
//! Before this module the batched path could only *ingest*: it ran
//! `QsaIndexer::decode_select` per row for continuity and then refused
//! (`ensure!(sel.is_none())`) if any row's selection actually became
//! ACTIVE, because there was no code to consume a `QsaSelection` here.
//! The dispatch gate compensated by throwing the whole batch onto the
//! per-sequence staging loop (`model/trait_impl/decode_a2.rs`) — one full
//! 48-layer forward per sequence per step plus a full-vocab D2H/H2D round
//! trip, i.e. the entire batched step was abandoned above 2051 context
//! tokens.
//!
//! ## Shape: per-row select-then-attend (option (a))
//!
//! `QsaIndexer` owns ONE set of selection scratch buffers
//! (`k_scratch` / `v_scratch` / `sel_dev` / `table_dev` / `seq_len_dev`,
//! see `layers/qsa.rs`) shared across calls. A "select every row, then
//! attend every row" shape would therefore have row `i+1`'s gather
//! overwrite row `i`'s scratch before row `i` ever read it. So each row is
//! selected and *immediately consumed* on the same stream — the scratch is
//! dead by the time the next `decode_select` touches it, and stream order
//! guarantees that on the device too.
//!
//! Everything else in the step (GDN layers, PLE, the MoE FFN,
//! `hc_post_site`, lm_head, sampling) stays batched, and there is no
//! logits round trip — which is the whole win versus the staging loop.
//! Only the `nq`-head attention GEMV of the attention layers is issued per
//! row, and only while selection is actually active.
//!
//! Fully-batched selection (option (b): `n`-row scratch inside the util
//! pledge + one attention call with per-row tables) is a strict follow-up;
//! it needs `QsaIndexer` to grow per-row scratch at construction time from
//! `max_num_seqs`, which is a separate allocation-budget change.
//!
//! ## Mixed batches
//!
//! A batch may mix rows above and below the inert bound. `decode_select`
//! returns `None` for an inert row (its dense attention over the paged
//! cache is exact), and this module runs that row's normal single-row
//! `run_paged_decode` against the shared paged cache. Both kinds are
//! correct in the same step.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::ctx::MultiSeqCtx;
use crate::layer::{AttnMetadataDev, LayerState};
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Which rows of this batch would have an ACTIVE QSA selection?
///
/// Mirrors `QsaIndexer::inert_bound()` exactly: `pos` (the PRE-append
/// `seq_len`) at or above `budget + ratio - 1` is the first step whose
/// visible prefix (`pos + 1`) exceeds the selection budget, which is
/// precisely when `decode_select` stops returning `None`.
///
/// Pure so it can be unit-tested without a GPU — the per-row flags this
/// derives are the invariant the batched path relies on.
pub(super) fn selection_active_rows(bound: usize, seq_lens: &[usize], n: usize) -> Vec<bool> {
    seq_lens.iter().take(n).map(|&l| l >= bound).collect()
}

/// `true` when any row of the batch needs the per-row select+attend path.
pub(super) fn any_selection_active(bound: usize, seq_lens: &[usize], n: usize) -> bool {
    seq_lens.iter().take(n).any(|&l| l >= bound)
}

impl Qwen3AttentionLayer {
    /// `true` when this layer carries an indexer AND at least one row of the
    /// batch is past the inert bound. `false` keeps the batch on the plain
    /// batched attention + ingest-only loop (bit-identical to before).
    pub(super) fn ms_qsa_selection_active(&self, seq_lens: &[usize], n: usize) -> bool {
        self.qsa
            .as_ref()
            .is_some_and(|q| any_selection_active(q.inert_bound(), seq_lens, n))
    }

    /// QSA ingest-only sweep for a batch entirely inside the inert bound.
    ///
    /// Every row must still be ingested every step or the indexer's raw-key
    /// cache loses sync with the sequence (`decode_select` asserts
    /// `pos == st.ingested`). Selection is provably all-visible here, so a
    /// `Some` would mean this function and [`Self::ms_qsa_selection_active`]
    /// disagree — refuse loudly rather than serve dense-past-budget, which
    /// is NOT the reference model.
    pub(super) fn ms_qsa_ingest_only<'a, 'b: 'a>(
        &self,
        c: &MultiSeqCtx<'_>,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let Some(qsa) = self.qsa.as_ref() else {
            return Ok(());
        };
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            bs,
            bf16,
            normed,
            ..
        } = *c;
        for (i, state) in states.iter_mut().enumerate().take(n) {
            // `n` is the PADDED dispatch width; rows the scheduler never
            // committed carry `seq_len == 0` (decode_a2's padding builder).
            // Skip them BEFORE `qsa_seq_state`, which lazily allocates this
            // layer's indexer carry (~10 MB at the default
            // ATLAS_QSA_MAX_TOKENS) — a padding row would allocate one per
            // attention layer per step and drop it unreleased.
            if seq_lens[i] == 0 {
                continue;
            }
            let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, *state, fwd.gpu)?;
            let sel = qsa.decode_select(
                st,
                normed.offset(i * h * bf16),
                seq_lens[i],
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta.block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                bs,
                fwd.gpu,
                stream,
            )?;
            anyhow::ensure!(
                sel.is_none(),
                "QSA selection active for seq {i} on the ingest-only batched ms \
                 path (seq_len {}, inert bound {}); ms_qsa_selection_active and \
                 decode_select disagree",
                seq_lens[i],
                qsa.inert_bound()
            );
        }
        Ok(())
    }

    /// Phase 5 (QSA-active variant): per row, select then immediately attend.
    ///
    /// Must be called AFTER `ms_phase_cache_write` (the gather has to see the
    /// token being decoded) and while `c.normed` still holds the ATTENTION
    /// block's normed hidden (the indexer projects that, exactly like the
    /// single-sequence path in `decode/attention_forward.rs`).
    ///
    /// Writes the standard contiguous `[n, nq*hd]` `attn_output()` buffer, so
    /// `ms_phase_o_proj` downstream is unchanged.
    pub(super) fn ms_qsa_phase_paged_decode<'a, 'b: 'a>(
        &self,
        c: &MultiSeqCtx<'_>,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let qsa = self
            .qsa
            .as_ref()
            .expect("ms_qsa_selection_active gated this call");
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            normed,
            ..
        } = *c;

        // Loud refusals for the shapes deliberately NOT handled here. Serving
        // dense-past-budget instead would silently diverge from the reference
        // model, which is the exact failure the old `ensure!` existed to stop.
        anyhow::ensure!(
            self.mla.is_none(),
            "QSA selection + MLA on the batched multi-seq decode path is not \
             wired (the absorbed-MLA batched kernel has no selection hook); \
             serve with ATLAS_HC_PERSEQ_DECODE=1"
        );
        anyhow::ensure!(
            !self.k_eq_v,
            "QSA selection + a shared K=V cache is not wired (the gather copies \
             distinct raw K and V NHD rows); serve with ATLAS_HC_PERSEQ_DECODE=1"
        );
        anyhow::ensure!(
            matches!(self.kv_dtype.kv_pair().0, KvCacheDtype::Bf16)
                && matches!(self.kv_dtype.kv_pair().1, KvCacheDtype::Bf16),
            "QSA selection requires a plain BF16 KV cache (the gather copies raw \
             NHD rows); serve with --kv-cache-dtype bf16"
        );
        anyhow::ensure!(
            !self.high_speed_swap_engaged(kv_cache),
            "QSA + --high-speed-swap is not wired (the gather reads the HBM pool)"
        );

        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
        let mbps = meta.max_blocks_per_seq;
        // `QsaIndexer::new` pins `block_topk = budget / ratio`, so
        // `pos >= budget + ratio - 1` holds EXACTLY when `decode_select`
        // stops returning `None`. Deriving the flags up-front turns that
        // equivalence into a checked invariant per row below.
        let expect_sel = selection_active_rows(qsa.inert_bound(), seq_lens, n);

        for (i, state) in states.iter_mut().enumerate().take(n) {
            // Padding rows (`seq_len == 0`) are skipped before the indexer
            // carry is lazily allocated — see the ingest loop above. Their
            // attention output is never committed by the scheduler.
            if seq_lens[i] == 0 {
                continue;
            }
            // Q for this row lives inside the interleaved [Q|K|V|gate] block,
            // `per_seq_qkv` bytes apart — the same addressing
            // `ms_phase_paged_decode`'s in-place arm uses. One row per call, so
            // `q_stride` is never actually stepped.
            let q_i = qkv_buf.offset(i * per_seq_qkv);
            let out_i = attn_out.offset(i * q_dim as usize * bf16);
            let table_i = meta.block_table.offset(i * mbps as usize * 4);

            let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, *state, fwd.gpu)?;
            let sel = qsa.decode_select(
                st,
                normed.offset(i * h * bf16),
                seq_lens[i],
                k_pool,
                v_pool,
                table_i,
                bs,
                fwd.gpu,
                stream,
            )?;

            // A row past the bound that came back with no selection would be
            // served DENSE past the budget — a different model from the
            // reference, and precisely the divergence the old `ensure!` on
            // this path existed to prevent. Refuse instead.
            anyhow::ensure!(
                !(expect_sel[i] && sel.is_none()),
                "QSA seq {i} is past the inert bound (seq_len {}, bound {}) but \
                 decode_select returned no selection; refusing to serve \
                 dense-past-budget",
                seq_lens[i],
                qsa.inert_bound()
            );
            match sel {
                // Attention over ONLY the selected tokens: the gathered
                // contiguous NHD scratch IS a valid paged cache when read
                // through the indexer's identity block table, so this is the
                // same BF16 kernel the dense path uses. Consumed HERE, before
                // the next row's `decode_select` overwrites the shared scratch.
                Some(s) => ops::paged_decode_attn_bf16(
                    fwd.gpu,
                    self.paged_decode_k,
                    q_i,
                    s.k_scratch,
                    s.v_scratch,
                    out_i,
                    s.table_dev,
                    s.seq_len_dev,
                    s.max_blocks,
                    1,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    0,
                    stream,
                )?,
                // Row still inside the inert bound: dense over the shared
                // paged cache is exact. `meta.seq_len` is `[n]` i32.
                None => self.run_paged_decode(
                    fwd.gpu,
                    q_i,
                    kv_cache,
                    out_i,
                    table_i,
                    meta.seq_len.offset(i * 4),
                    mbps,
                    1,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    fwd.buffers.splitk_workspace(),
                    fwd.levers.max_decode_seqs,
                    stream,
                )?,
            }
        }
        Ok(attn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::{any_selection_active, selection_active_rows};

    // qwen3.8-flash-next: index_topk (budget) 2048, index_compress_ratio 4
    // → inert bound 2051. `seq_len` is the PRE-append position.
    const BOUND: usize = 2048 + 4 - 1;

    #[test]
    fn bound_is_exclusive_below_and_inclusive_at() {
        assert!(!any_selection_active(BOUND, &[2050], 1));
        assert!(any_selection_active(BOUND, &[2051], 1));
    }

    #[test]
    fn a_single_long_row_activates_the_batch() {
        // The exact batch that used to fall off the batched path entirely.
        let lens = [12, 4096, 30, 7];
        assert!(any_selection_active(BOUND, &lens, 4));
    }

    #[test]
    fn mixed_batch_flags_only_the_rows_past_the_bound() {
        let lens = [12, 4096, 2051, 2050];
        assert_eq!(
            selection_active_rows(BOUND, &lens, 4),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn padding_rows_beyond_n_are_ignored() {
        // `seq_lens` is sized for the padded dispatch width; only the first
        // `n` entries are real sequences.
        let lens = [10, 20, 999_999];
        assert!(!any_selection_active(BOUND, &lens, 2));
        assert_eq!(selection_active_rows(BOUND, &lens, 2), vec![false, false]);
    }

    #[test]
    fn an_all_inert_batch_stays_on_the_batched_attention() {
        assert!(!any_selection_active(BOUND, &[0, 1, 2047, 2050], 4));
    }
}
