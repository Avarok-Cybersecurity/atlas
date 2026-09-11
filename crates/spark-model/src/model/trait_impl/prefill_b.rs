// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_chunk_dispatch` orchestrator.
//!
//! Refactor wave-4e split a 1000-LoC monolith into Pattern-B phase fns
//! (siblings under `prefill_b/`). The MutexGuard on `kv_cache` is
//! acquired here once and threaded through each phase as `&mut`.
//!
//! Phases (by section comment in original):
//!   1+1b → embed_chunk     (token embed + vision-pad overlay)
//!   2    → prefix_lookup   (prefix-cache hit + EP-sync + Marconi)
//!   2b   → proc_range      (recompute proc_start/count after skip; may early-return)
//!   3    → upload_meta     (positions + MRoPE + slots staging upload)
//!   3b   → upload_paged    (paged-prefill block_table + seq_len upload)
//!   4    → forward_layers  (per-layer prefill/decode + diagnostics)
//!   5-8  → finalize_last   (final norm + lm_head + snapshot save) — last chunk
//!   9    → save_intermediate_checkpoint — non-last chunk

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_chunk_zero_tests;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
mod embed_chunk;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod midchunk_capture;
mod prefix_lookup;
mod proc_range;
mod prompt_logprobs;
mod save_checkpoint;
mod stage_batched;
mod upload_meta;
mod upload_paged;

impl TransformerModel {
    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        assert!(
            chunk_start + chunk_len <= total,
            "chunk_start({chunk_start}) + chunk_len({chunk_len}) > total({total})"
        );

        // Tail-checkpoint split (issue #15 follow-up, 2026-07-02): a warm
        // multi-turn hit matches the radix at BLOCK granularity, and the
        // divergence point sits at/near the previous prompt end (the chat
        // template's generation-only suffix — e.g. Qwen's forced empty
        // <think> block — is absent from the re-rendered history), so the
        // next turn's `matched` lands at floor(divergence/bs)*bs, which is
        // the prompt's last full-block boundary OR one block below it (when
        // the template suffix crosses that boundary; measured: both occur).
        // Snapshot eligibility requires snap_tok <= matched — the leaf
        // snapshot (at `total`) is PAST both, making warm turns recompute the
        // full SSM state (or fall back an entire turn to the previous tail,
        // measured 1.3-3.2k-token replays). Split the final chunk ONCE, one
        // block below the last block boundary under `total`: that position is
        // <= both possible match points, so the snapshot
        // `prefill_b_save_checkpoint` saves there (independent of
        // --ssm-checkpoint-interval) is always eligible and the warm replay
        // is <= 2 blocks, folded into the suffix prefill pass. A single cut
        // costs one extra small pass at save time (a cut at the boundary
        // itself would need a second pass and is redundant — measured
        // +~160ms/turn for two cuts vs <=31-token replay for one).
        //
        // The extra pass costs ~150ms on this class of MoE model (a tiny-M
        // pass still sweeps most activated expert weights), which is -7% on a
        // cold 2k prefill. Every request with prefix caching active pays it,
        // including single-shot ones.
        //
        // ★ THIS PARAGRAPH USED TO SAY THE OPPOSITE and was left behind when
        // the split was made unconditional. It read: "on single-GPU the split
        // only fires when the radix already holds a prefix of this prompt
        // (peek is read-only): single-shot requests never pay; conversations
        // pay from turn 2 onward" — describing the `ep_active ||
        // peek_matched_tokens(..) > 0` guard that the block below explains was
        // REMOVED, precisely because making the shape depend on radix contents
        // produced a temp-0 argmax flip. Two comments in one function
        // disagreeing about whether behaviour is conditional is how that class
        // of bug survives review, so it is corrected rather than deleted.
        //
        // Known residual: turn 2 of a conversation still recomputes the full
        // SSM state (its cold turn 1 saved no tail checkpoint). On EP>1 the
        // same unconditional rule applies for an additional reason: rank-local
        // radix contents diverge, and chunk sequences must be deterministic on
        // (tokens, config) across ranks (bug #33 invariant). Skipped for vision
        // prompts (pad runs must not straddle chunk boundaries) and non-SSM
        // models (KV-only cache hits need no snapshot).
        if is_last_chunk
            && self.config.num_ssm_layers() > 0
            && self.ssm_snapshots.is_enabled()
            && self.prefix_cache.is_active()
            && !self.tokens_have_vision_pad(tokens)
        {
            let bs = self.kv_cache.lock().block_size();
            // One block below the last block boundary strictly under `total`.
            // SSOT: `tail_split_cut` — `Model::prefill_tail_cut` hands the same
            // number to the scheduler so the batched path can pre-split and
            // keep this shape (#927).
            let cut = tail_split_cut(total, bs);
            // UNCONDITIONAL. This used to additionally require
            // `ep_active || peek_matched_tokens(..) > 0`, i.e. it split only on a
            // WARM request (radix already populated) — which made the prompt take a
            // DIFFERENT SHAPE cold vs warm: one N-token pass when cold, two passes
            // [0..cut) + [cut..N) when warm. BF16 accumulation is not associative,
            // so the two shapes produce different hidden states, and at temperature 0
            // a near-tied argmax flips. Same prompt, same seed, different answer.
            //
            // MEASURED on Puzzle-75B (fresh container, temp 0, exact-hit shortcut
            // bypassed so this split is the ONLY cold/warm difference):
            //   34-token prompt (cut=16, split fires warm) : cold "17 barrels"
            //                                                warm "15 barrels"  DIVERGE
            //   16-token prompt (cut<=0, split IMPOSSIBLE) : cold == warm       MATCH
            // The cut threshold predicts the divergence exactly.
            //
            // The invariant is already stated for EP>1 in the comment above —
            // "chunk sequences must be deterministic on (tokens, config)" — it was
            // just never enforced on a single rank. Splitting unconditionally makes
            // cold and warm identical by construction, at the cost of one extra pass
            // on cold prefills that cross the cut.
            //
            // `ATLAS_NO_TAIL_SPLIT=1` disables the split entirely (same-binary A/B).
            // That is the OTHER way to satisfy the invariant — always one pass — and
            // it keeps the single-pass numerics, at the cost of the warm-turn tail
            // checkpoint this split exists to create.
            let split_disabled = tail_split_disabled();
            if !split_disabled && cut > chunk_start && cut < total {
                self.prefill_chunk_dispatch(
                    tokens,
                    seq,
                    chunk_start,
                    cut - chunk_start,
                    false,
                    stream,
                )?;
                return self.prefill_chunk_dispatch(tokens, seq, cut, total - cut, true, stream);
            }
        }

        // Guard: chunk_len must not exceed buffer arena capacity.
        // Exceeding this causes CUDA illegal memory access (error 700)
        // which permanently corrupts GPU state.
        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        // Use the caller-provided stream for compute-copy overlap, unless
        // a multi-rank world is active (EP or pure TP — NCCL collectives
        // must stay stream-ordered with the cmd broadcasts, which run on
        // the default stream).
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // EP=2: zero ALL buffers on every chunk (NCCL defense-in-depth).
        // EP=1, first chunk (chunk_start==0): zero only buffers whose stale
        // contents can affect prefill; the remaining scratch buffers are
        // overwritten before read by embedding + layer forward.
        // EP=1, subsequent chunks: skip zeroing — buffers are overwritten by embedding
        // + layer forward before read. Saves 7 memsets × (chunks-1) per prefill.
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();

        // ── Phase 1+1b: embed chunk + vision pad overlay ──
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

        // ── Phase 2: prefix-cache lookup + EP sync + Marconi snapshot restore ──
        let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
            tokens,
            seq,
            chunk_start,
            total,
            &mut kv_cache,
            stream,
            None,
        )?;

        if std::env::var("ATLAS_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("chunk_entry start={chunk_start} len={chunk_len} kvws={kv_write_start}"),
            );
        }

        // Allocate blocks needed through end of this chunk.
        let bs = kv_cache.block_size();
        let end_pos = chunk_start + chunk_len;
        let blocks_needed = (end_pos - 1) / bs + 1;
        super::super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        // ── Phase 2b: compute effective processing range (may early-return) ──
        let (proc_start, proc_count, effective_seq_len_start) = match self.prefill_b_proc_range(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            kv_write_start,
            marconi_skip,
            // Single-stream: hidden lives at offset 0 ⇒ pass base (byte-identical).
            self.buffers.hidden_states(),
            stream,
        )? {
            proc_range::ProcRange::Compute {
                proc_start,
                proc_count,
                effective_seq_len_start,
            } => (proc_start, proc_count, effective_seq_len_start),
            proc_range::ProcRange::EarlyReturn(ptr) => {
                // #155 ROOT CAUSE (warm-turn phantom snapshots): fully-cached
                // chunks skipped compute but ALSO skipped the Phase-5 token
                // append, leaving seq.tokens a SUFFIX (short by k*4096) on
                // every warm turn. Every consumer keyed on seq.tokens —
                // decode-ckpt/finish-leaf registration (hashed over a
                // mid-conversation window → unreachable phantom entries that
                // flood the snapshot pool), the radix insert at retire
                // (suffix tokens paired with the full block_table → polluted
                // token→block branches + refcount leaks), and rep-penalty
                // context — operated on the wrong sequence. Cached chunks
                // must record their tokens like any other chunk.
                seq.tokens
                    .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                seq.seq_len = chunk_start + chunk_len;
                seq.last_decode_ckpt_block = seq.tokens.len() / bs;
                return Ok(ptr);
            }
        };

        // ── Phase 3: upload positions + MRoPE + slot metadata ──
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;

        // ── Phase 3b: paged metadata (block_table + seq_len) ──
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }

        // Force H2D metadata copy to complete before layer forward.
        // On DGX Spark SM121, the DMA engine may not properly serialize
        // pinned H2D copy with subsequent compute on the same stream,
        // causing CUDA 700 at >9K tokens. This sync adds ~5μs overhead
        // per chunk but prevents the illegal memory access.
        self.gpu.synchronize(stream)?;

        // ── Mid-chunk tail SSM capture (opt-in): plan BEFORE the forward
        // pass so SSM layers split their h/conv kernels at `tb` in-pass.
        // `None` (flag off or pass doesn't span `tb`) => no split. ──
        let midcap_plan = self.prepare_midchunk_capture(
            tokens,
            seq,
            &mut kv_cache,
            proc_start,
            proc_count,
            stream,
        );

        // ── Phase 4: forward through all layers ──
        self.prefill_b_forward_layers(
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            midcap_plan.as_ref(),
            stream,
        )?;

        // Register the reserved slot as the session tail once the full pass has
        // captured the @tb state into it (no-op when no capture was planned).
        if let Some(plan) = midcap_plan.as_ref() {
            self.finalize_midchunk_capture(tokens, seq, plan);
        }

        // ── Phase 5: update sequence state incrementally ──
        // Always add chunk tokens exactly once. The early-return path for
        // fully cached non-last chunks doesn't add tokens, so this is the
        // single insertion point for all chunks that reach here.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;
        // #155: prime the decode-checkpoint cadence gate; the last chunk
        // leaves it at the prompt's complete-block count (see prefill_a).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // ── Legacy echo+logprobs: project prompt positions while this
        // chunk's hidden rows are live, BEFORE finalize_last (which
        // re-derives norm_output + logits for the first sampled token).
        // No-op unless seq.collect_prompt_logprobs is set.
        self.collect_prompt_logprobs_chunk(
            tokens,
            seq,
            chunk_start,
            proc_start,
            proc_count,
            stream,
        )?;

        if is_last_chunk {
            // ── Phase 6+7+8: final norm, lm_head, prefix-cache + snapshot save ──
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )
        } else {
            // ── Phase 9: intermediate Marconi checkpoint ──
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            Ok(DevicePtr::NULL)
        }
    }
}

/// The tail-checkpoint cut for a prompt of `total` tokens at KV block size
/// `bs`: one block below the last block boundary strictly under `total`.
///
/// Free function so `prefill_chunk_dispatch` (which performs the split) and
/// `TransformerModel::prefill_tail_cut` (which reports it to the scheduler)
/// cannot drift apart — two call sites computing this expression separately is
/// exactly the class of divergence that produced #927 cell E one module over.
pub(in crate::model) fn tail_split_cut(total: usize, bs: usize) -> usize {
    if bs == 0 {
        return 0;
    }
    ((total.saturating_sub(1) / bs) * bs).saturating_sub(bs)
}

/// `ATLAS_NO_TAIL_SPLIT=1` — disable the tail-checkpoint split entirely
/// (same-binary A/B; keeps the single-pass numerics and loses the warm-turn
/// tail checkpoint).
pub(in crate::model) fn tail_split_disabled() -> bool {
    std::env::var("ATLAS_NO_TAIL_SPLIT").as_deref() == Ok("1")
}

#[cfg(test)]
mod tail_split_cut_tests {
    use super::tail_split_cut;

    #[test]
    fn the_cut_is_one_block_below_the_last_boundary_under_the_prompt() {
        // The #927 shape: a 1193-token prompt at block size 16 splits
        // 1168 + 25. ((1193-1)/16)*16 = 1184, minus one block = 1168.
        assert_eq!(tail_split_cut(1193, 16), 1168);
        assert_eq!(1193 - tail_split_cut(1193, 16), 25);
    }

    #[test]
    fn a_prompt_on_a_boundary_still_cuts_a_block_below_it() {
        // 1024 is itself a boundary; the last boundary STRICTLY under it is
        // 1008, so the cut is 992 and the tail is 32 — never zero-length.
        assert_eq!(tail_split_cut(1024, 16), 992);
    }

    #[test]
    fn short_prompts_saturate_to_zero_and_disable_the_split() {
        // `prefill_chunk_dispatch` only splits when `cut > chunk_start`, so a
        // zero cut means "one pass" for a chunk starting at 0.
        assert_eq!(tail_split_cut(16, 16), 0);
        assert_eq!(tail_split_cut(0, 16), 0);
        assert_eq!(tail_split_cut(1, 16), 0);
    }

    #[test]
    fn a_zero_block_size_is_not_a_division() {
        assert_eq!(tail_split_cut(1193, 0), 0);
    }
}
