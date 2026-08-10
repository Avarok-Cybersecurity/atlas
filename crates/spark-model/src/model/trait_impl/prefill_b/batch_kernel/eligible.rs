// SPDX-License-Identifier: AGPL-3.0-only

//! Eligibility gating for the Q12 Path B kernel-batched prefill.
//!
//! Extracted from `batch_kernel.rs` to keep each file under the 500-LoC
//! file-size cap. Holds the env-flag predicates (`first_chunk_batched_enabled`,
//! `varlen_prefill_enabled`), the pure-data eligibility check
//! (`check_kernel_batched_eligible`, unit-tested in `batch_kernel_tests.rs`),
//! and the `TransformerModel::kernel_batched_eligible` wrapper the dispatcher
//! calls upfront.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use super::super::super::super::types::TransformerModel;
use crate::traits::PrefillSlice;
use spark_runtime::prefix_cache::PrefixMatch;

impl TransformerModel {
    /// Returns true when the batched-kernel path is viable for these
    /// streams. Cheap upfront check — caller (dispatch) falls back to
    /// per-stream when false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        // Routed-MoE prefill is a flat [total_tokens] sort/grouped-GEMM/scatter
        // with no notion of stream boundaries (layers/moe/** contains no
        // chunk_len / batch_size / stream_idx), so it needs no ragged
        // awareness. The C=4 heterogeneous illegal access that motivated the
        // original `num_experts == 0` restriction had two causes, both now
        // fixed: the scratch SSOT divergence (charged n*max_chunk_len at
        // runtime vs sum(chunk_len) at admission — fixed in the same commit
        // that added the restriction) and the batched paged-attention kernel
        // indexing Q/O at `b * max_len` on a buffer packed by sum(len), with a
        // scalar kv_len at the max. The kernels are now cu_seqlens-aware.
        let varlen = varlen_prefill_enabled();
        // Effective staged length per stream. `peek_matched_tokens` is a
        // read-only probe: no refs taken, no LRU touch, no hit/miss counted, so
        // it is safe to call from a pure pre-flight check.
        let bs = self.kv_cache.lock().block_size();
        let effective_arena = effective_arena_charge_enabled();
        let eff = |s: &PrefillSlice<'_>| -> usize {
            if !effective_arena {
                return s.chunk_len;
            }
            let matched =
                self.prefix_cache
                    .peek_matched_tokens(s.prompt_tokens, bs, s.seq.adapter_id);
            let skip = matched.saturating_sub(s.chunk_start).min(s.chunk_len);
            // A fully-cached LAST chunk still stages one token so the LM head
            // has a row to read (`proc_range` re-embeds it). A fully-cached
            // MIDDLE chunk stages nothing — reported as 0 so the caller can
            // reject the batch (see the zero-length guard below); it must never
            // be admitted, because a zero-token stream is degenerate in the
            // packed cu_seqlens layout (empty segment, and `running_proc_off +=
            // 0` leaves it sharing an offset with the next stream).
            match s.chunk_len - skip {
                0 if s.is_last_chunk => 1,
                n => n,
            }
        };
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, eff(s), s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            &self.config.model_type,
            self.config.head_dim,
            self.buffers.scratch_bytes(),
            self.config.num_experts_per_tok,
            self.config.mrope_interleaved,
            // VARLEN v1 batches chunk-0 (fresh K/V) through FlashInfer ragged.
            crate::layers::ops::prefill_batched_first_chunk_enabled() || varlen,
            varlen,
        )
    }
}

/// Whether reserved prefix matches can share one stacked attention forward.
///
/// This is deliberately narrower than the single-stream prefix-cache path.
/// A batched cache hit is admitted only when every request has identical
/// processing geometry and needs neither an SSM restore nor disk-cache work.
/// The caller owns the reservation/rollback protocol; this predicate is pure
/// so the safety envelope remains unit-testable.
pub(in crate::model) fn cache_batch_matches_compatible(
    matches: &[PrefixMatch],
    chunk_len: usize,
) -> bool {
    let Some(first) = matches.first() else {
        return false;
    };
    let matched = first.matched_tokens;
    // Full-chunk hits use the single-token logits/early-return special cases;
    // keep those on the established sequential path in v1.
    if matched >= chunk_len {
        return false;
    }
    matches.iter().all(|m| {
        m.matched_tokens == matched
            && m.matched_blocks.len() == first.matched_blocks.len()
            && m.matched_disk_block_ids.is_empty()
            && m.ssm_snapshot.is_none()
            && m.ssm_snapshot_tokens == 0
            && m.ssm_snapshot_tier_key.is_none()
            && m.ssm_snapshot_tier_tokens == 0
    })
}

impl TransformerModel {
    /// DIAG: detect cross-stream physical-block sharing (co-dispatch KV
    /// double-issue hypothesis for the n>=5 decode-bleed bug). Gated behind
    /// `ATLAS_CODISPATCH_BTCHECK=1`; no-op otherwise.
    pub(super) fn codispatch_btcheck(&self, streams: &[PrefillSlice<'_>], n: usize) {
        if std::env::var("ATLAS_CODISPATCH_BTCHECK").ok().as_deref() != Some("1") {
            return;
        }
        let mut owner: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        let mut slot_owner: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut dump: Vec<(usize, usize, Option<usize>, usize, u32)> = Vec::new();
        for (b, slice) in streams.iter().enumerate() {
            let bt = slice.seq.block_table.clone();
            let slot = slice.seq.slot_idx;
            // Authoritative owned slot from the RAII guard (slot_idx may be
            // stale post-compaction); plus prompt length + first token to
            // prove two DIFFERENT prompts share a slot.
            let guard_slot = slice.seq.ssm_slot.as_ref().and_then(|g| g.idx());
            let ptoks = slice.prompt_tokens.len();
            let tok0 = slice.prompt_tokens.first().copied().unwrap_or(0);
            if let Some(gs) = guard_slot {
                if let Some(&prev) = slot_owner.get(&gs) {
                    tracing::warn!(
                        "ATLAS_GUARDSHARE n={n}: GUARD slot {gs} SHARED by stream {prev} and {b}"
                    );
                } else {
                    slot_owner.insert(gs, b);
                }
            }
            for &blk in &bt {
                if let Some(&prev) = owner.get(&blk) {
                    tracing::warn!(
                        "ATLAS_BTSHARE n={n}: KV block {blk} SHARED by stream {prev} and {b}"
                    );
                } else {
                    owner.insert(blk, b);
                }
            }
            dump.push((b, slot, guard_slot, ptoks, tok0));
        }
        tracing::warn!("ATLAS_BTDUMP n={n} (stream,slot_idx,guard_slot,ptoks,tok0): {dump:?}");
    }
}

/// VARLEN batched prefill enabled? (`ATLAS_PREFILL_VARLEN=1`). Co-admits
/// varied-length concurrent prefills into one forward (cu_seqlens geometry,
/// FlashInfer ragged attention). Requires a FLASHINFER_HOME build.
pub(in crate::model) fn varlen_prefill_enabled() -> bool {
    // SSOT with the batched-attention layer's chunk-0 guard.
    crate::layers::ops::prefill_varlen_enabled()
}

/// Pure-data predicate extracted from [`TransformerModel::kernel_batched_eligible`]
/// so the gating rules are unit-testable without a real `TransformerModel`.
/// Caller materialises per-stream tuples `(chunk_len, chunk_start, is_last_chunk)`.
#[allow(clippy::too_many_arguments)]
pub(in crate::model) fn check_kernel_batched_eligible<I>(
    streams: I,
    n: usize,
    arena_cap: usize,
    model_type: &str,
    head_dim: usize,
    scratch_cap: usize,
    top_k: usize,
    mrope: bool,
    allow_chunk_zero: bool,
    varlen: bool,
) -> bool
where
    I: IntoIterator<Item = (usize, usize, usize, bool)>,
{
    if n < 2 {
        return false;
    }
    // No MLA layers in stack (batched attention doesn't support MLA).
    // Conservatively check via model_type — mistral is the only MLA
    // model in Atlas today.
    if model_type == "mistral" {
        return false;
    }
    // No HDIM=512 layers (Gemma-4 long-attention).
    if head_dim > 256 {
        return false;
    }
    let mut first: Option<(usize, usize, bool)> = None;
    let mut total = 0usize;
    let mut max_chunk_len = 0usize;
    for (chunk_len, eff_len, chunk_start, is_last) in streams {
        // `chunk_start` and `is_last_chunk` must match across streams (different
        // `chunk_start` → different `effective_seq_len_start`; mixing `is_last`
        // can't dispatch finalize_last + save_checkpoint together). `chunk_len`
        // must ALSO match in the legacy path; the VARLEN path allows differing
        // lengths (cu_seqlens geometry + FlashInfer ragged attention).
        match first {
            None => first = Some((chunk_len, chunk_start, is_last)),
            Some((cl, cs, il)) => {
                if (!varlen && chunk_len != cl) || chunk_start != cs || is_last != il {
                    return false;
                }
            }
        }
        // Charge the arena by the tokens that will actually be STAGED, not the
        // raw chunk. The packed layout advances by `proc_count`
        // (batch_kernel.rs `running_proc_off += proc_count`), and a prefix hit
        // collapses proc_count to the uncached suffix — `proc_range` re-embeds
        // exactly that span, so a warm stream occupies a fraction of its chunk.
        // Summing raw `chunk_len` charged for tokens the cache means we never
        // compute, which on an 8192-chunk / 8200-arena config made N>=2 stacking
        // arithmetically impossible no matter how warm the cache was.
        // `eff_len == chunk_len` when the caller cannot prove a hit, so this is
        // never more permissive than the old bound without evidence.
        // Zero-token stream: degenerate in the packed layout. Reject the whole
        // batch and let it run per-stream, where a fully-cached middle chunk is
        // handled correctly by `proc_range`'s EarlyReturn.
        if eff_len == 0 {
            return false;
        }
        total += eff_len;
        max_chunk_len = max_chunk_len.max(eff_len);
    }
    let Some((_chunk_len, chunk_start, _)) = first else {
        return false;
    };
    // Batched attention is paged-only today; chunk 0 uses the non-paged
    // cache-skip path and must stay on the single-stream dispatcher.
    if chunk_start == 0 && !allow_chunk_zero {
        return false;
    }
    // Total stacked tokens fit in the token arena (hidden_states buffer).
    if total > arena_cap {
        return false;
    }
    // #110: the kernel-batched staging footprint must fit in scratch. PURE
    // pre-flight — runs before any stream mutation, so a false routes to the
    // per-stream path from a clean state (a mid-dispatch overrun would leave
    // streams dirty and the fallback would re-run setup → corruption).
    // VARLEN: size the scratch pre-flight by the worst-case per-stream length.
    let scratch_needed = if varlen {
        spark_runtime::buffers::q12_batched_scratch_bytes_varlen(
            n,
            total,
            max_chunk_len,
            top_k,
            mrope,
        )
    } else {
        spark_runtime::buffers::q12_batched_scratch_bytes(n, max_chunk_len, top_k, mrope)
    };
    scratch_needed <= scratch_cap
}

/// `ATLAS_Q12_EFFECTIVE_ARENA=1` — charge the Q12 batched-prefill arena budget
/// by the tokens each stream will actually stage (chunk minus the cached
/// prefix) instead of its raw chunk length.
///
/// Default OFF. The permissive bound is only sound while the staged span really
/// is the uncached suffix; the staging loop asserts that per stream and bails to
/// the per-stream path if a mid-batch eviction shrinks a match after the probe.
pub(in crate::model) fn effective_arena_charge_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_Q12_EFFECTIVE_ARENA").as_deref() == Ok("1"))
}
