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

/// Whether chunk-0 streams may use the batched (paged) prefill path. Enabled by
/// `ATLAS_Q12_BATCHED_FIRST_CHUNK=1` or `ATLAS_PREFILL_CODISPATCH=1` (the latter
/// is the single end-to-end flag for cross-request co-dispatch of fresh prompts,
/// whose every stream starts at chunk_start==0).
pub(super) fn first_chunk_batched_enabled() -> bool {
    ["ATLAS_Q12_BATCHED_FIRST_CHUNK", "ATLAS_PREFILL_CODISPATCH"]
        .iter()
        .any(|k| {
            std::env::var(k)
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        })
}

impl TransformerModel {
    /// Returns true when the batched-kernel path is viable for these
    /// streams. Cheap upfront check — caller (dispatch) falls back to
    /// per-stream when false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        // Fix #4 (mixed-length cache + co-dispatch silent failure): when the
        // prefix cache is active a co-dispatched batch can contain streams with
        // DIFFERENT cache-hit depths (the first arrival recomputes →
        // effective_seq_len_start=0; later arrivals restore the just-saved
        // snapshot → effective_seq_len_start>0). The kernel-batched PHASE A
        // mutates each stream IN ORDER (snapshot restore into the SSM pool slot,
        // KV block alloc, kv_valid_tokens/seq_len) BEFORE it discovers the
        // effective_seq_len_start mismatch and bails Err — leaving streams
        // 0..b partially mutated. The dispatch then re-runs the per-stream loop
        // on those dirty seqs (double snapshot-restore / double block-alloc),
        // and any surfaced Err drops ALL streams in the scheduler
        // (run_batched_prefill.rs: every stream marked failed → client sees a
        // connection reset, server survives). Route cache-possible batches
        // STRAIGHT to the per-stream loop (batch.rs:199) on pristine seqs — that
        // loop is structurally equivalent to the proven single-stream cache path
        // (prefill_chunk_dispatch) and already handles hits correctly, with
        // per-stream logits rows (fix #1). NoPrefixCaching::is_active()==false,
        // so no-cache co-dispatch (the +35% PHASE-C scaling) and the cold path
        // stay byte-identical — the gate only fires when a real radix cache holds
        // refs. Trade: cold requests under active caching lose kernel-batched
        // co-dispatch, but with caching on most requests hit and a hit's
        // processed suffix is tiny, so the co-dispatch scaling is moot.
        if self.prefix_cache.is_active() {
            return false;
        }
        let varlen = varlen_prefill_enabled();
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            &self.config.model_type,
            self.config.head_dim,
            self.buffers.scratch_bytes(),
            self.config.num_experts_per_tok,
            self.config.mrope_interleaved,
            // VARLEN v1 batches chunk-0 (fresh K/V) through FlashInfer ragged.
            first_chunk_batched_enabled() || varlen,
            varlen,
        )
    }
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
    std::env::var("ATLAS_PREFILL_VARLEN")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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
    I: IntoIterator<Item = (usize, usize, bool)>,
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
    for (chunk_len, chunk_start, is_last) in streams {
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
        total += chunk_len;
        max_chunk_len = max_chunk_len.max(chunk_len);
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
    spark_runtime::buffers::q12_batched_scratch_bytes(n, max_chunk_len, top_k, mrope) <= scratch_cap
}
