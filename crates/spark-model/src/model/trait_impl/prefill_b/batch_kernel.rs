// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Path B kernel-batched orchestration.
//!
//! `prefill_batch_chunk_kernel_batched` is the outer-layer-loop dispatch
//! that uses the model-level per-layer batched dispatchers
//! (`prefill_attn_batched_layer`, `prefill_ssm_batched_layer`,
//! `prefill_dense_batched_layer`). It mirrors the per-stream Phase 1-3
//! setup but lays out per-stream data at stacked offsets in the shared
//! buffers, then runs ONE outer layer loop calling the right per-layer
//! batched dispatcher.
//!
//! Eligibility check (`kernel_batched_eligible`) is called upfront by
//! `prefill_batch_chunk_dispatch` before any state mutation. When
//! ineligible, the dispatcher falls through to the existing per-stream
//! body (commit baa16fa). When eligible, this function runs.
//!
//! Constraints encoded:
//!   - N ≥ 2 streams
//!   - All streams share `chunk_len`, nonzero `seq_len_start` (q_offset), and
//!     `is_last_chunk` flag
//!   - Total stacked tokens fits in buffer arena
//!   - No MLA / HDIM=512 / HSS-engaged layer in the model
//!   - All batched kernel handles loaded
//!
//! Validation: hardware-validated (#110 — conc repro 80/80, sanitizer-clean).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use super::proc_range::ProcRange;
use super::stage_batched::PerStreamStageInfo;
use super::upload_meta::MetaLayout;

mod eligible;

// Re-exports so `batch_kernel::check_kernel_batched_eligible` (used by
// `batch_kernel_tests.rs`) and the env-flag predicates resolve unchanged
// after the eligibility cluster moved into the `eligible` submodule.
use eligible::first_chunk_batched_enabled;
pub(in crate::model) use eligible::{
    batched_reserve_hybrid_ssm_ok, cache_batch_matches_compatible, check_kernel_batched_eligible,
    config_is_mla, varlen_prefill_enabled,
};

use crate::layer::{
    BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState, TransformerLayer,
};
use crate::traits::{Model, PrefillSlice, SequenceState};

pub(in crate::model) enum KernelBatchResult {
    Completed(Vec<DevicePtr>),
    NotAdmitted,
}

impl TransformerModel {
    /// Q12 Path B: full kernel-batched prefill orchestration.
    ///
    /// Caller (prefill_batch_chunk_dispatch) MUST have verified
    /// `kernel_batched_eligible` before calling this; if a per-stream
    /// constraint is later detected here (e.g. proc_count mismatch from
    /// differing prefix-cache hits), this function bails Err.
    ///
    /// `row_base` shifts each stream's logits row clear of the decode lanes
    /// in a mixed step; the caller bounds-checks it against the arena.
    pub(in crate::model) fn prefill_batch_chunk_kernel_batched(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<KernelBatchResult> {
        let n = streams.len();
        let chunk_len = streams[0].chunk_len;
        let is_last_chunk = streams[0].is_last_chunk;
        let h = self.config.hidden_size;
        let dtype_bytes = 2usize;
        let varlen = varlen_prefill_enabled();
        // DEFECT 2 fix: the stacked hidden buffer MUST be packed by the same
        // cu_seqlens SSOT (Σ proc_count) that Phase1/GDN/Phase3/attn + the staged
        // BatchedAttnMetadata (stage_batched.rs) and GdnPrefillBuffers.total_len
        // consume — NOT by cu_off = Σ chunk_len. On a partial prefix-cache hit
        // proc_count < chunk_len so the two prefix-sums diverge and downstream
        // reads land on the wrong stream's hidden region. proc_count is only
        // known AFTER proc_range, so proc_off is built as a running prefix-sum
        // inside the PHASE-A loop below; proc_off[b] = Σ_{j<b} proc_count[j] is
        // fully known before stream b is processed. (Cold / no-cache:
        // proc_count == chunk_len ⇒ proc_off == old cu_off ⇒ byte-identical.)
        let mut running_proc_off = 0usize;
        let arena_cap_tokens = self.buffers.max_batch_tokens();
        // Largest per-stream chunk (VARLEN). All per-stream scratch slots must be
        // sized for this, not streams[0].chunk_len, or a longer stream's meta /
        // MoE topk staging overruns its slot (CUDA 700).
        let max_chunk_len = streams
            .iter()
            .map(|s| s.chunk_len)
            .max()
            .unwrap_or(chunk_len);

        // Multi-rank world (EP or pure TP) → NCCL needs the default stream.
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // Lock KV cache once.
        let mut kv_cache = self.kv_cache.lock();

        // ── Allocation pre-flight (PURE) ──────────────────────────────────
        // Phase A allocates blocks per stream as it walks
        // (`ensure_blocks_through_prefill`), so a stream that runs out of KV
        // midway returns Err — and an Err from an ADMITTED batch fails ALL N
        // requests (batch.rs: the scheduler pushes `(i, None)` for every
        // stream). One request's exhaustion would take its whole cohort down,
        // and KV exhaustion is a live failure mode under concurrency: the
        // single-stream path handles it by preempting and retrying, this path
        // has 4-8x the blast radius and no such recovery.
        //
        // So establish capacity for the WHOLE cohort before touching anything,
        // mirroring the scratch pre-flight's rule (eligible.rs): "runs before
        // any stream mutation, so a false routes to the per-stream path from a
        // clean state". Declining here costs a per-stream fallback, where each
        // request gets the single-stream preempt-and-retry it deserves.
        //
        // Runs BEFORE the cache reservation deliberately: nothing has been
        // acquired yet, so a decline needs no unwinding. Block counts use the
        // read-only `peek_matched_tokens` probe rather than the reservation.
        {
            let bs = kv_cache.block_size();
            let mut needed = 0usize;
            for s in streams.iter() {
                let through = (s.chunk_start + s.chunk_len).div_ceil(bs);
                // Blocks this stream already has, plus the ones its prefix match
                // will hand it (reused, never allocated).
                let matched_blocks =
                    self.prefix_cache
                        .peek_matched_tokens(s.prompt_tokens, bs, s.seq.adapter_id)
                        / bs;
                let have = s.seq.block_table.len() + matched_blocks;
                needed += through.saturating_sub(have);
            }
            // Evicting cached blocks is always safe (the cache is an
            // optimization) and mutates no sequence, so it is fair game inside a
            // "pure" pre-flight. Loop because one eviction can free zero blocks
            // when a live sequence still holds the evicted node's block.
            while kv_cache.num_free_blocks() < needed {
                let short = needed - kv_cache.num_free_blocks();
                let evicted = self.prefix_cache.evict(short);
                if evicted.is_empty() {
                    break;
                }
                super::super::super::block_mgmt::apply_evicted_blocks(evicted, &mut kv_cache);
            }
            let free = kv_cache.num_free_blocks();
            if free < needed {
                tracing::debug!(
                    target: "atlas::q12",
                    n = streams.len(),
                    needed,
                    free,
                    "Q12 kernel-batched declined: cohort needs more KV than is \
                     reclaimable — falling back to per-stream so one exhaustion \
                     cannot fail the whole batch"
                );
                return Ok(KernelBatchResult::NotAdmitted);
            }
        }

        // Cache admission happens while every sequence is pristine. A rejected
        // plan has released its exact radix references and may safely take the
        // established per-stream path. Once admitted, Phase A consumes these
        // reservations instead of walking the cache again.
        let reserved_prefix_matches =
            match self.prefill_b_reserve_batched_prefix_matches(streams, kv_cache.block_size()) {
                Some(matches) => matches,
                None => return Ok(KernelBatchResult::NotAdmitted),
            };

        // Zero shared buffers once (instead of N times in per-stream).
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if streams[0].chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }

        let hidden_base = self.buffers.hidden_states();
        let _residual_base = self.buffers.residual();

        // ── PHASE A: per-stream Phase 1-3 setup at stacked offsets ──
        //
        // Each stream's per-stream meta uses a distinct scratch slice so
        // staged metadata doesn't clobber another stream's. Final stacked
        // BatchedAttnMetadata is staged AFTER all per-stream metas.

        // Per-stream metadata collected across the setup loop.
        struct PerStreamMeta {
            chunk_start: usize,
            proc_start: usize,
            proc_count: usize,
            effective_seq_len_start: usize,
            kv_write_start_eff: usize,
            block_table_dev: DevicePtr,
            seq_len_dev: DevicePtr,
            num_blocks: usize,
            // Σ proc_count of prior streams — this stream's hidden packing offset
            // (== cu_seqlens layout). Used by PHASE-C finalize.
            proc_off: usize,
        }
        let mut per_stream: Vec<PerStreamMeta> = Vec::with_capacity(n);
        // VARLEN admits chunk-0 batches on its own (`allow_chunk_zero` in
        // `check_kernel_batched_eligible` is codispatch OR varlen), so the
        // paged upload must fire on the same predicate: without it a chunk-0
        // stream's `block_table_dev` stays NULL, and the batched paged
        // attention kernel dereferences `block_table_ptrs[b]` unless the
        // opt-in FlashInfer ragged arm happens to be enabled.
        let force_paged_first_chunk = streams[0].chunk_start == 0
            && (crate::layers::ops::prefill_batched_first_chunk_enabled() || varlen);

        // Tracks MRoPE / paged-flag agreement across streams.
        let mut use_mrope: Option<bool> = None;
        let mut needs_paged: Option<bool> = None;

        // Per-stream scratch slot size: positions + MRoPE H/W (optional) +
        // slot table. Conservative estimate: 12 bytes per token + small
        // header. Reserved 4 KB per stream is plenty for chunk_len ≤ 256.
        // For larger chunk_len the scratch budget scales with arena_cap.
        let per_stream_meta_bytes = ((max_chunk_len * 16) + 64).max(4096);
        // Cumulative scratch offset cursor — starts after MoE topk
        // staging area (per single-stream upload_meta convention). Sized by the
        // largest per-stream chunk so a long VARLEN stream's topk staging fits.
        // VARLEN packs routed-MoE metadata by `cu_seqlens`, just like hidden
        // states and BatchedAttnMetadata. Charging every stream at the longest
        // request length both disagrees with the admission SSOT and can push
        // the first per-stream metadata upload past scratch capacity.
        let moe_scratch_tokens = if varlen {
            streams.iter().map(|slice| slice.chunk_len).sum()
        } else {
            max_chunk_len * n
        };
        let moe_scratch_bytes = moe_scratch_tokens * self.config.num_experts_per_tok * 4 * 2;
        let mut scratch_cursor = (moe_scratch_bytes + 63) & !63;

        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            // Per-stream chunk_len (VARLEN: differs per stream; legacy: == chunk_len).
            let cl = slice.chunk_len;
            let total = tokens.len();
            let seq = &mut *slice.seq;

            // Embed the FULL `cl` tokens at this stream's proc_off slot (==
            // Σ proc_count of prior streams, the cu_seqlens layout). proc_range
            // below re-embeds the processed suffix at the SAME slot on a
            // partial cache hit. A full-cl embed may write a stale tail past
            // proc_count, but that tail is in stream b+1's region which b+1's
            // own embed (next iteration) overwrites — every region
            // [proc_off[j], proc_off[j]+proc_count[j]) is LAST-written by
            // stream j's own embed, so correctness holds without reordering.
            let proc_off_b = running_proc_off;
            let hidden_b = hidden_base.offset(proc_off_b * h * dtype_bytes);
            // Skip the cached prefix when the arena is charged by effective
            // tokens: `proc_range` re-embeds exactly the uncached suffix at this
            // same slot moments later, so embedding the cached head is pure
            // redundant work — and, more importantly, writing `cl` tokens from
            // `proc_off_b` is what forces the arena to hold Σ chunk_len. With
            // the suffix-only embed the footprint is Σ proc_count, which is what
            // the packed cu_seqlens layout actually consumes.
            let embed_skip = if super::batch_kernel::eligible::effective_arena_charge_enabled() {
                let bs_probe = self.kv_cache.lock().block_size();
                self.prefix_cache
                    .peek_matched_tokens(tokens, bs_probe, seq.adapter_id)
                    .saturating_sub(chunk_start)
                    .min(cl)
            } else {
                0
            };
            if embed_skip < cl {
                self.prefill_b_embed_chunk_at(
                    tokens,
                    chunk_start + embed_skip,
                    cl - embed_skip,
                    hidden_b,
                    stream,
                )?;
            }

            // Prefix-cache lookup, EP-sync, Marconi restore. Cache-admitted
            // batches consume their preflight reservation exactly once.
            let reserved_match = if self.prefix_cache.is_active() && chunk_start == 0 {
                Some(reserved_prefix_matches[b].clone())
            } else {
                None
            };
            let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
                tokens,
                seq,
                chunk_start,
                total,
                &mut kv_cache,
                stream,
                reserved_match,
            )?;

            // Block allocation through end of chunk.
            let bs = kv_cache.block_size();
            let end_pos = chunk_start + cl;
            let blocks_needed = (end_pos - 1) / bs + 1;
            super::super::super::block_mgmt::ensure_blocks_through_prefill(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;

            // Effective processing range. DEFECT 1 fix: pass this stream's
            // proc_off hidden slot so any cache-hit re-embed lands in THIS
            // stream's region, not the offset-0 base buffer.
            let (proc_start, proc_count, effective_seq_len_start) = match self
                .prefill_b_proc_range(
                    tokens,
                    seq,
                    chunk_start,
                    cl,
                    is_last_chunk,
                    kv_write_start,
                    marconi_skip,
                    hidden_b,
                    stream,
                )? {
                ProcRange::Compute {
                    proc_start,
                    proc_count,
                    effective_seq_len_start,
                } => (proc_start, proc_count, effective_seq_len_start),
                ProcRange::EarlyReturn(_) => anyhow::bail!(
                    "kernel-batched: stream {b} early-returned during proc_range \
                         — eligibility check missed this. Caller should fall back."
                ),
            };

            // Cross-stream consistency: all streams must share proc_count
            // and effective_seq_len_start (q_offset) for the batched
            // attention kernel.
            if b > 0 {
                // VARLEN allows differing proc_count (cu_seqlens geometry); the
                // legacy batched-attention/GDN kernels require uniform proc_count.
                if !varlen && per_stream[0].proc_count != proc_count {
                    anyhow::bail!(
                        "kernel-batched: stream {b} proc_count={} differs from \
                         stream 0 proc_count={}. Caller should fall back.",
                        proc_count,
                        per_stream[0].proc_count
                    );
                }
                if per_stream[0].effective_seq_len_start != effective_seq_len_start {
                    anyhow::bail!(
                        "kernel-batched: stream {b} effective_seq_len_start={} \
                         differs from stream 0={}. Caller should fall back.",
                        effective_seq_len_start,
                        per_stream[0].effective_seq_len_start
                    );
                }
            }

            // Per-stream meta upload to distinct scratch slice.
            let meta_base = self.buffers.scratch().offset(scratch_cursor);
            // This stream's slice runs from `scratch_cursor` to the end of the
            // arena; the per-stream stride advance below keeps successive
            // blocks from overlapping.
            let meta_region_bytes = self.buffers.scratch_bytes().saturating_sub(scratch_cursor);
            let layout = self.prefill_b_upload_meta_at(
                tokens,
                seq,
                chunk_start,
                cl,
                proc_start,
                proc_count,
                effective_seq_len_start,
                &kv_cache,
                meta_base,
                meta_region_bytes,
                stream,
            )?;
            if layout.needs_paged || force_paged_first_chunk {
                self.prefill_b_upload_paged(
                    seq,
                    total,
                    proc_start,
                    proc_count,
                    meta_base,
                    layout.slot_offset,
                    &kv_cache,
                    stream,
                )?;
            }
            scratch_cursor += per_stream_meta_bytes;

            // First-stream sets the MRoPE / paged flags; subsequent streams
            // must match.
            match (use_mrope, layout.use_mrope) {
                (None, m) => use_mrope = Some(m),
                (Some(prev), m) if prev != m => {
                    anyhow::bail!("kernel-batched: stream {b} use_mrope={m} mismatch with stream 0")
                }
                _ => {}
            }
            match (needs_paged, layout.needs_paged) {
                (None, p) => needs_paged = Some(p),
                (Some(prev), p) if prev != p => anyhow::bail!(
                    "kernel-batched: stream {b} needs_paged={p} mismatch with stream 0"
                ),
                _ => {}
            }

            let kv_write_start_eff = if marconi_skip { 0 } else { kv_write_start };
            let (block_table_dev, seq_len_dev) = if layout.needs_paged || force_paged_first_chunk {
                let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
                (page_meta.block_table, page_meta.seq_len)
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };
            let num_blocks = seq.block_table.len();

            per_stream.push(PerStreamMeta {
                chunk_start,
                proc_start,
                proc_count,
                effective_seq_len_start,
                kv_write_start_eff,
                block_table_dev,
                seq_len_dev,
                num_blocks,
                proc_off: proc_off_b,
            });
            // Advance the running prefix-sum AFTER proc_count is known so the
            // next stream packs at Σ proc_count (cu_seqlens SSOT).
            // The arena bound was pre-flighted from a PROBE of each stream's
            // cached prefix. An eviction triggered by an earlier stream in this
            // same batch can shrink a later match, making its real proc_count
            // larger than predicted — so re-check against the true arena before
            // trusting the packed layout. Bailing here routes the batch to the
            // per-stream path rather than writing past the buffer (CUDA 700).
            if running_proc_off + proc_count > arena_cap_tokens {
                anyhow::bail!(
                    "Q12 batched staging overran the arena: stream {b} needs                      {proc_count} tokens at offset {running_proc_off} > cap                      {arena_cap_tokens} (prefix match shrank after pre-flight)"
                );
            }
            running_proc_off += proc_count;
        }

        // H2D barrier before kernel compute (GB10 DMA quirk).
        self.gpu.synchronize(stream)?;

        // ── PHASE B: stage BatchedAttnMetadata + outer layer loop ──
        let use_mrope = use_mrope.unwrap();
        let proc_count = per_stream[0].proc_count;
        let seq_lens_start = per_stream[0].effective_seq_len_start;

        // Build per-stream stage info (re-borrows from streams since
        // PerStreamStageInfo holds &seq).
        let streams_info: Vec<PerStreamStageInfo<'_>> = streams
            .iter()
            .zip(per_stream.iter())
            .map(|(slice, m)| PerStreamStageInfo {
                proc_start: m.proc_start,
                proc_count: m.proc_count,
                block_table_dev: m.block_table_dev,
                seq_len_dev: m.seq_len_dev,
                num_blocks: m.num_blocks,
                seq: &*slice.seq,
            })
            .collect();

        let meta = self.stage_batched_attn_metadata(
            &streams_info,
            &kv_cache,
            use_mrope,
            scratch_cursor,
            stream,
        )?;
        // Advance cursor by the EXACT staged footprint (#110): the prior
        // heuristic under-estimated it, placing h_state_ptrs_off inside the
        // live slot_stacked array → corrupted KV slots → CUDA-700.
        // `staged_bytes` is the SSOT matching `q12_batched_scratch_bytes`.
        let stage_size = meta.staged_bytes;
        scratch_cursor += stage_size;

        // Q12 safety: bail if the h_state_ptrs JIT slot (N*8 B) would exceed
        // scratch rather than overrun into another buffer.
        let scratch_bytes = self.buffers.sizes().scratch;
        let projected_usage = scratch_cursor + (n * std::mem::size_of::<u64>());
        if projected_usage > scratch_bytes {
            anyhow::bail!(
                "kernel-batched prefill scratch overflow: projected {} bytes \
                 > scratch capacity {} bytes (n={n}, chunk_len={chunk_len}, \
                 proc_count={proc_count}). Falling back to per-stream.",
                projected_usage,
                scratch_bytes
            );
        }

        // GDN buffers (for SSM layers).
        let gdn_bufs = GdnPrefillBuffers {
            qkv: self.gdn_buf_qkv,
            gate_beta: self.gdn_buf_gate_beta,
            output: self.gdn_buf_out,
            z: self.gdn_buf_z,
            // VARLEN: packed total = Σ proc_count (running_proc_off after the
            // PHASE-A loop) — the cu_seqlens SSOT. Was Σ chunk_len (total_tokens),
            // which over-counts on a partial cache hit and makes the GDN scan
            // walk phantom tokens past the packed data. Legacy uniform: proc_count*n.
            total_len: if varlen {
                running_proc_off
            } else {
                proc_count * n
            },
        };

        // ForwardContext for batched layer calls. attn_metadata is
        // intentionally None — layers read BatchedAttnMetadata directly
        // through the model-level dispatcher arguments.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            // Always COLD (warm hybrid-SSM matches fall back to the per-stream
            // path), but a cold pass is the reference a later warm Marconi
            // replay must reproduce — same kernel rule applies.
            gdn_exact_replay: self.gdn_exact_replay_for_prefill(false),
            token_ids: None,
            host_token_ids: None,
            // #30: batched multi-seq prefill legitimately mixes adapters and keeps
            // the bgmv (via multi_seq/qkv.rs); its attn_metadata is None so it never
            // reaches paged_qkv's routed path anyway. Must stay None.
            routed_lora_layers: None,
            midchunk_capture: None,
            // Codispatch packs multiple requests into one forward: per-row MoE
            // adapter identity needs the device-side fold (follow-up), so a MoE
            // adapter here REFUSES loudly rather than fold one adapter onto every
            // packed row. Inert when no MoE adapter is installed (hook no-ops).
            moe_lora_route: crate::layer::MoeLoraRoute::Refuse,
        };

        // h_state_ptrs scratch slot offset (used JIT per SSM layer).
        let h_state_ptrs_off = scratch_cursor;

        // Per-stream kv_write_starts vector for attention dispatcher.
        let kv_write_starts: Vec<usize> = per_stream.iter().map(|m| m.kv_write_start_eff).collect();

        // Outer layer loop with mixed dispatch.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Gather per-stream seq refs for this layer.
            let mut seqs_vec: Vec<&mut SequenceState> =
                streams.iter_mut().map(|s| &mut *s.seq).collect();

            if layer.is_ssm_layer() {
                let proc_starts: Vec<usize> = per_stream.iter().map(|m| m.proc_start).collect();
                self.prefill_ssm_batched_layer(
                    layer.as_ref(),
                    layer_idx,
                    hidden_base,
                    _residual_base,
                    &mut seqs_vec,
                    &mut kv_cache,
                    &proc_starts,
                    &meta,
                    &gdn_bufs,
                    h_state_ptrs_off,
                    &ctx,
                    stream,
                )?;
            } else {
                self.prefill_attn_batched_layer(
                    layer.as_ref(),
                    layer_idx,
                    hidden_base,
                    _residual_base,
                    &mut seqs_vec,
                    &mut kv_cache,
                    &kv_write_starts,
                    seq_lens_start,
                    &meta,
                    &ctx,
                    stream,
                )?;
            }
        }

        // DIAG: detect cross-stream physical-block sharing (co-dispatch KV
        // double-issue hypothesis for the n>=5 decode-bleed bug). Gated.
        self.codispatch_btcheck(streams, n);

        // ── PHASE C: per-stream finalize ──
        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);
        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            let cl = slice.chunk_len;
            let seq = &mut *slice.seq;
            let m = &per_stream[b];

            // Phase 5: sequence-state update.
            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + cl]);
            seq.seq_len = chunk_start + cl;

            let logits = if is_last_chunk {
                self.prefill_b_finalize_last_at(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    cl,
                    m.proc_count,
                    // hidden_stream_offset_tokens = proc_off[b] (Σ proc_count of
                    // prior streams, the cu_seqlens layout) — NOT cu_off[b].
                    // finalize reads last_token = proc_off + proc_count - 1.
                    m.proc_off,
                    // Shifted clear of the decode lanes in a mixed step; `b`
                    // alone would land on decode lane `b`'s logits row.
                    row_base + b,
                    stream,
                )?
            } else {
                self.prefill_b_save_checkpoint(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    cl,
                    stream,
                )?;
                DevicePtr::NULL
            };
            logits_out.push(logits);
        }

        Ok(KernelBatchResult::Completed(logits_out))
    }
}

// Unit tests for `check_kernel_batched_eligible` live in a sibling
// file `batch_kernel_tests.rs` (mounted by `prefill_b.rs`) to keep
// this file under the 500-LoC file-size-cap.
