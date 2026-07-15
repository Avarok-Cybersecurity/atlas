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

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::super::types::TransformerModel;
use crate::traits::PrefillSlice;

/// Whether chunk-0 streams may use the batched (paged) prefill path. Enabled by
/// `ATLAS_Q12_BATCHED_FIRST_CHUNK=1` or `ATLAS_PREFILL_CODISPATCH=1` (the latter
/// is the single end-to-end flag for cross-request co-dispatch of fresh prompts,
/// whose every stream starts at chunk_start==0).
///
/// Delegates to the crate-canonical predicate in `crate::layer` so this
/// eligibility gate and the attention layers (`qwen3_attention`) admit the
/// EXACT same set of chunk-0 streams — see the correctness note there.
pub(super) use crate::layer::first_chunk_batched_enabled;

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
    /// EXPERIMENTAL Lever 1: co-dispatch under an ACTIVE prefix cache, gated by
    /// `ATLAS_Q12_CACHE_CODISPATCH=1` (default OFF). When unset every method
    /// below is dormant and `kernel_batched_eligible`'s blanket cache bail
    /// governs, so behaviour is byte-identical to today.
    pub(in crate::model) fn cache_codispatch_enabled() -> bool {
        std::env::var("ATLAS_Q12_CACHE_CODISPATCH")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Read-only projection of the `effective_seq_len_start` that Phase A WOULD
    /// derive for one chunk-0 stream, WITHOUT mutating any state (no inc_ref, no
    /// snapshot restore/fault-in). Mirrors `prefill_b_prefix_lookup`'s
    /// `skip_tokens` decision followed by `prefill_b_proc_range`'s
    /// `effective_seq_len_start` mapping, using the side-effect-free
    /// `PrefixCache::peek_match` probe.
    ///
    /// Returns `Some(eff_seq_len_start)` when the batched path can represent the
    /// stream, or `None` when it MUST be single-streamed:
    ///   * `chunk_start != 0` (depth carried in already-mutated `marconi_skip_to`),
    ///   * a whole-chunk cache hit on a non-last chunk (`ProcRange::EarlyReturn`),
    ///   * the `marconi_exact_snap` fixup (full exact-leaf hit — the batched
    ///     PHASE-C does not replicate `finalize_last`'s re-restore), or
    ///   * a SPILLED (tiered) anchor whose fault-in depth we don't model.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn preflight_eff_seq_len_start(
        &self,
        tokens: &[u32],
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        session_hash: u64,
        adapter_id: u64,
        collect_logprobs: bool,
        bs: usize,
    ) -> Option<usize> {
        // Batched path only represents chunk-0 co-dispatch; later chunks carry
        // depth in the already-mutated `seq.marconi_skip_to` → single-stream.
        if chunk_start != 0 {
            return None;
        }
        // Mirror prefix_lookup's forced-full-recompute guards: these make the
        // real lookup return matched==0 ⇒ effective_seq_len_start==0.
        if collect_logprobs || self.tokens_have_vision_pad(tokens) {
            return Some(0);
        }
        let total = tokens.len();
        let has_ssm = self.config.num_ssm_layers() > 0;
        let pk = self
            .prefix_cache
            .peek_match(tokens, bs, session_hash, adapter_id);
        let matched = pk.matched_tokens;
        if matched == 0 {
            // Cold / all-fresh: proc_range cold branch ⇒ eff_start == 0.
            return Some(0);
        }
        // Spilled anchor (Phase 1b tier): the real lookup would fault it in,
        // whose resulting depth we don't model read-only ⇒ single-stream
        // conservatively. Inert on the default path (ATLAS_SSM_TIER off).
        if pk.tiered {
            return None;
        }
        let eff_snapshot = pk.ssm_snapshot;
        let snap_tok = pk.ssm_snapshot_tokens;
        // Read-only twin of prefix_lookup's `skip` decision.
        let mut skip = if let Some(snap_id) = eff_snapshot {
            let exact_without_hidden = snap_tok == matched
                && matched == total
                && !self.ssm_snapshots.has_hidden(snap_id);
            snap_tok > 0
                && matched <= total
                && !exact_without_hidden
                && self.ssm_snapshots.session_matches(snap_id, session_hash)
        } else {
            false
        };
        // ATLAS_NO_MARCONI_EXACT parity: a full exact-leaf hit is forced to
        // recompute (skip=false) under that probe.
        if skip
            && snap_tok == matched
            && matched == total
            && std::env::var("ATLAS_NO_MARCONI_EXACT").as_deref() == Ok("1")
        {
            skip = false;
        }
        // F82 non-SSM cache-hit skip path.
        if matched > 0 && !skip && !has_ssm {
            skip = true;
        }
        if !skip {
            // Prefix hit but full recompute (SSM model, no usable snapshot) ⇒
            // proc_range cold branch ⇒ eff_start == 0.
            return Some(0);
        }
        // marconi_exact_snap fixup (finalize_last re-restore + stashed-hidden
        // first token) is NOT replicated by the batched PHASE-C ⇒ single-stream
        // this stream. This is the `snap_tok >= matched && matched == total`
        // branch that sets `seq.marconi_exact_snap` in prefix_lookup.
        if matched == total && snap_tok >= matched {
            return None;
        }
        let skip_tokens = if !has_ssm { matched } else { snap_tok };
        // proc_range mapping (chunk_start==0, marconi_skip=true,
        // kv_write_start=skip_tokens).
        if skip_tokens == 0 {
            return Some(0);
        }
        let skip_in_chunk = skip_tokens.min(chunk_len);
        if skip_in_chunk >= chunk_len {
            // Whole chunk cached.
            if is_last_chunk {
                Some(chunk_start + chunk_len - 1) // proc_count == 1
            } else {
                None // ProcRange::EarlyReturn — not representable in the batched path
            }
        } else {
            Some(chunk_start + skip_in_chunk) // == skip_tokens
        }
    }

    /// EXPERIMENTAL Lever 1 entry (UNVALIDATED — needs GPU A/B before default-on).
    ///
    /// Under an ACTIVE prefix cache, pre-flights each stream's effective depth
    /// (read-only `preflight_eff_seq_len_start`), partitions the largest
    /// uniform-depth subset (size >= 2) that also passes the structural
    /// `check_kernel_batched_eligible` predicate, kernel-batches it, and
    /// single-streams the rest — then scatters logits back to the caller's
    /// stream order.
    ///
    /// Returns `Ok(Some(logits))` when it handled the whole batch, or `Ok(None)`
    /// when no worthwhile uniform subset exists (caller falls through to the
    /// existing per-stream path on PRISTINE seqs — the pre-flight is read-only,
    /// so no stream was mutated).
    ///
    /// CORRECTNESS ASSUMPTIONS (must hold — see report):
    ///   * SINGLE-THREADED, SINGLE-NODE scheduler: no cache insert interleaves
    ///     between the read-only `peek_match` here and the real `lookup` inside
    ///     Phase A, so the probed depth deterministically equals Phase A's.
    ///     Caller gates on `!multi_rank_protocol_active()` (peek is head-local).
    ///   * `marconi_exact_snap` streams are excluded by the pre-flight (→ None →
    ///     single-stream), because the batched PHASE-C does not re-restore state.
    pub(in crate::model) fn try_cache_codispatch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Option<Vec<DevicePtr>>> {
        let n = streams.len();
        let bs = self.kv_cache.lock().block_size();
        let varlen = varlen_prefill_enabled();

        // 1. Read-only per-stream effective depth (None ⇒ must single-stream).
        let depths: Vec<Option<usize>> = streams
            .iter()
            .map(|s| {
                let sq = &*s.seq;
                self.preflight_eff_seq_len_start(
                    s.prompt_tokens,
                    s.chunk_start,
                    s.chunk_len,
                    s.is_last_chunk,
                    sq.session_hash,
                    sq.adapter_id,
                    sq.collect_prompt_logprobs.is_some(),
                    bs,
                )
            })
            .collect();

        // 2. Group indices by (depth, chunk_len, chunk_start, is_last); pick the
        //    largest group of size >= 2 that also passes the structural predicate.
        //    Grouping by depth guarantees uniform effective_seq_len_start AND
        //    uniform proc_count (proc_count is a function of chunk_len − depth for
        //    a shared chunk_len), so the kernel's cross-stream checks cannot bail.
        let mut groups: std::collections::HashMap<(usize, usize, usize, bool), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, d) in depths.iter().enumerate() {
            if let Some(depth) = d {
                let s = &streams[i];
                groups
                    .entry((*depth, s.chunk_len, s.chunk_start, s.is_last_chunk))
                    .or_default()
                    .push(i);
            }
        }
        let best = groups
            .into_iter()
            .filter(|(_, idxs)| idxs.len() >= 2)
            .filter(|(_, idxs)| {
                check_kernel_batched_eligible(
                    idxs.iter().map(|&i| {
                        (
                            streams[i].chunk_len,
                            streams[i].chunk_start,
                            streams[i].is_last_chunk,
                        )
                    }),
                    idxs.len(),
                    self.buffers.max_batch_tokens(),
                    &self.config.model_type,
                    self.config.head_dim,
                    self.buffers.scratch_bytes(),
                    self.config.num_experts_per_tok,
                    self.config.mrope_interleaved,
                    first_chunk_batched_enabled() || varlen,
                    varlen,
                )
            })
            .max_by_key(|(_, idxs)| idxs.len());
        let (key, group) = match best {
            Some(g) => g,
            None => return Ok(None),
        };
        let k = group.len();

        // Fast path: the whole batch is one uniform-depth group — no partition.
        if k == n {
            tracing::debug!(
                target: "atlas::q12",
                n_streams = n,
                subset_size = k,
                depth = key.0,
                "Q12 cache-codispatch: whole-batch uniform-depth subset (no partition)"
            );
            return Ok(Some(self.prefill_batch_chunk_kernel_batched(streams, stream)?));
        }

        tracing::debug!(
            target: "atlas::q12",
            n_streams = n,
            subset_size = k,
            depth = key.0,
            "Q12 cache-codispatch: partitioning uniform-depth subset (rest single-streamed)"
        );

        // 3. Approach (A): in-place reorder so the group is the prefix [0..k),
        //    remaining streams the suffix — preserving original relative order in
        //    each partition (stable). `desired[new_pos] = original_index`.
        let mut in_group = vec![false; n];
        for &i in &group {
            in_group[i] = true;
        }
        let mut desired: Vec<usize> = Vec::with_capacity(n);
        desired.extend(group.iter().copied());
        for (i, &g) in in_group.iter().enumerate() {
            if !g {
                desired.push(i);
            }
        }
        // Physically permute `streams` to `desired` order via swaps (PrefillSlice
        // is not Clone/Copy — it holds `&mut seq` — so move via `slice.swap`).
        // `pos_of_orig[orig]` = current physical slot of original element `orig`;
        // `orig_at_pos[p]` = original index currently occupying slot `p`.
        let mut pos_of_orig: Vec<usize> = (0..n).collect();
        let mut orig_at_pos: Vec<usize> = (0..n).collect();
        for target_pos in 0..n {
            let want_orig = desired[target_pos];
            let cur = pos_of_orig[want_orig];
            if cur != target_pos {
                let other_orig = orig_at_pos[target_pos];
                streams.swap(target_pos, cur);
                orig_at_pos.swap(target_pos, cur);
                pos_of_orig.swap(want_orig, other_orig);
            }
        }

        // 4. Kernel-batch the head; single-stream the tail via the proven
        //    single-stream dispatch (identical to the n==1 fast path).
        let (head, tail) = streams.split_at_mut(k);
        let head_logits = self.prefill_batch_chunk_kernel_batched(head, stream)?;
        let mut new_logits: Vec<DevicePtr> = Vec::with_capacity(n);
        new_logits.extend(head_logits);
        for s in tail.iter_mut() {
            let l = self.prefill_chunk_dispatch(
                s.prompt_tokens,
                s.seq,
                s.chunk_start,
                s.chunk_len,
                s.is_last_chunk,
                stream,
            )?;
            new_logits.push(l);
        }

        // 5. Scatter logits back to the caller's original stream order.
        let mut logits_out = vec![DevicePtr::NULL; n];
        for (new_pos, &orig) in desired.iter().enumerate() {
            logits_out[orig] = new_logits[new_pos];
        }
        Ok(Some(logits_out))
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
