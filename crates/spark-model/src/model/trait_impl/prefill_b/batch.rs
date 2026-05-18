// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_batch_chunk_dispatch` — batched prefill orchestrator (Q12).
//!
//! Mirrors `prefill_chunk_dispatch` (the single-stream path in `prefill_b.rs`)
//! but processes N concurrent streams in one model-level call. The current
//! implementation runs the streams **sequentially** through the phase
//! helpers while holding the KV-cache mutex once for the whole batch.
//!
//! ## What this commit *does* deliver
//!
//! - **Single mutex acquire**: the default trait impl in `traits/model.rs`
//!   re-locks `self.kv_cache` per stream. This dispatch locks once and
//!   threads `&mut kv_cache` through each stream's phase calls. On a
//!   single-threaded scheduler the lock contention is zero, so the win is
//!   purely "fewer atomic ops per dispatch" — not measurable but cheap.
//! - **Single orchestration source**: all prefill orchestration code lives
//!   in one file (this one + `prefill_b.rs` for the N=1 entry). Future
//!   Phase 2b (SSM kernel batching) and Phase 3 (attention kernel rewrite)
//!   patch sites are scoped to *layer overrides* — `Qwen3SsmLayer::prefill_batched`
//!   and `Qwen3AttentionLayer::prefill_batched` — rather than scattered
//!   across the trait's default loop.
//!
//! ## What this commit deliberately does *not* deliver
//!
//! Real kernel-level batching (one kernel launch processing N streams' QKV
//! through shared SMEM/L2) is **not** part of this commit. The motivating
//! win in the Q12 plan — per-layer L2-amortised weight load — does **not**
//! materialise from per-stream-sequential calls because layer weights are
//! orders of magnitude larger than the GB10 L2 cache (gigabytes vs ~24 MB)
//! so the second stream's call re-streams every byte of weight regardless.
//!
//! The actual win paths require:
//!
//! 1. **Phase 2b — SSM/GDN kernel batching.** The `gated_delta_rule_*`
//!    kernel family indexes `h_state + (b * num_v_heads + vh) * K_DIM *
//!    V_DIM`, which assumes one contiguous `h_state` buffer. Each
//!    `SsmLayerState` currently owns its own GPU allocation, so batching
//!    requires either (a) staging per-stream h_states into one contiguous
//!    buffer pre-launch and copying back post-launch (~10 ms / dispatch for
//!    Qwen3.6-27B SSM stack), or (b) patching every GDN variant to take
//!    `float* const* h_state_ptrs` and dereference per batch index (kernel
//!    change across ~7 variants + Rust ops bindings + model-specific
//!    overrides). The latter is the right long-term shape; the former is
//!    a 1-day intermediate.
//!
//! 2. **Phase 3 — Attention prefill kernel rewrite.** Today's grid in
//!    `inferspark_prefill_paged_fp8.cu` is `(num_q_heads, q_chunks, 1)`
//!    with no batch axis. Adding `cu_seqlens` + `block_table_offsets` and
//!    a per-block stream identifier is the actual TTFT win for Q12 — but
//!    it's multi-day CUDA work that's deliberately out of scope here.
//!
//! Once those land, this dispatch's *body* changes minimally: each layer's
//! `prefill_batched` override starts returning Ok with a one-shot
//! kernel-batched path, and the per-stream phase calls below stay correct
//! because each phase helper already takes a `&mut SequenceState`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::proc_range::ProcRange;
use super::upload_meta::MetaLayout;
use crate::traits::{Model, PrefillSlice, SequenceState};

impl TransformerModel {
    /// Batched-prefill dispatch for N concurrent streams. See module docs.
    ///
    /// Returns `Vec<DevicePtr>` parallel to `streams`: each entry is the
    /// last-token logits pointer for that stream when its chunk is the last,
    /// or `DevicePtr::NULL` otherwise. Order matches `streams`.
    pub(in crate::model) fn prefill_batch_chunk_dispatch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let n = streams.len();
        // Q12 diagnostic: dispatch entry. Useful to confirm scheduler is
        // funneling concurrent prefills here. Debug-level by default;
        // promote with `RUST_LOG=atlas::q12=debug`.
        tracing::debug!(
            target: "atlas::q12",
            n = n,
            "prefill_batch_chunk_dispatch entry"
        );
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            // Fast path: N=1 has no batching to do. Delegate to the
            // single-stream dispatch and skip the per-stream-loop bookkeeping.
            let s = &mut streams[0];
            let logits = self.prefill_chunk_dispatch(
                s.prompt_tokens,
                s.seq,
                s.chunk_start,
                s.chunk_len,
                s.is_last_chunk,
                stream,
            )?;
            return Ok(vec![logits]);
        }

        // Buffer-arena fit check (per-stream sequential layout still respects
        // arena cap on each call). Bail loud if any stream's chunk exceeds
        // the arena — CUDA 700 territory.
        let arena_cap = self.buffers.max_batch_tokens();
        for (i, s) in streams.iter().enumerate() {
            if s.chunk_len > arena_cap {
                anyhow::bail!(
                    "Batched prefill stream {i} chunk_len={} exceeds arena \
                     capacity {arena_cap}. Reduce --max-prefill-tokens.",
                    s.chunk_len
                );
            }
        }

        // Q12 Path B: kernel-batched fast path. Check eligibility upfront
        // (cheap) and, when viable, dispatch to the outer-layer-loop
        // implementation that uses BatchedAttnMetadata + per-layer batched
        // dispatchers. On Err from the kernel path, fall through to the
        // per-stream body below. The kernel path bails BEFORE any state
        // mutation on the structural eligibility check; mid-Phase-A bails
        // (e.g. proc_count mismatch from differing prefix-cache hits) leave
        // the streams in a partially-mutated state — we propagate that Err
        // so the caller can retry single-stream or surface to the user.
        //
        // Runtime kill switch: set `ATLAS_Q12_BATCHED=0` to force-disable
        // the kernel-batched path without rebuilding. Default is enabled
        // (any unset / non-"0" value). Useful for the kernel-validation
        // session when isolating a regression to the batched path.
        let q12_batched_enabled = std::env::var("ATLAS_Q12_BATCHED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);
        if q12_batched_enabled && self.kernel_batched_eligible(streams) {
            tracing::debug!(
                target: "atlas::q12",
                n = n,
                chunk_len = streams[0].chunk_len,
                is_last_chunk = streams[0].is_last_chunk,
                "Q12 kernel-batched dispatch attempt"
            );
            match self.prefill_batch_chunk_kernel_batched(streams, stream) {
                Ok(v) => {
                    tracing::debug!(target: "atlas::q12", "Q12 kernel-batched succeeded");
                    return Ok(v);
                }
                Err(e) => {
                    // Structural bails (proc_count/seq_lens_start mismatch,
                    // unsupported layer feature) are logged at info so the
                    // first occurrence is visible in production logs without
                    // requiring debug-level tracing. Subsequent bails are
                    // still logged but with reduced verbosity in tight loops.
                    tracing::info!(
                        target: "atlas::q12",
                        "Q12 kernel-batched bailed → falling back to per-stream: {e}"
                    );
                }
            }
        } else if !q12_batched_enabled {
            tracing::trace!(
                target: "atlas::q12",
                "Q12 kernel-batched disabled via ATLAS_Q12_BATCHED=0"
            );
        } else {
            // Observability: eligibility failed. Surface why so operators
            // can diagnose silent fallback. Logged at debug to avoid log
            // floods on hot paths.
            let chunk_lens: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
            let chunk_starts: Vec<usize> = streams.iter().map(|s| s.chunk_start).collect();
            let total: usize = chunk_lens.iter().sum();
            tracing::debug!(
                target: "atlas::q12",
                n = n,
                chunk_lens = ?chunk_lens,
                chunk_starts = ?chunk_starts,
                total = total,
                arena_cap = self.buffers.max_batch_tokens(),
                head_dim = self.config.head_dim,
                model_type = self.config.model_type.as_str(),
                "Q12 kernel-batched ineligible — falling back to per-stream"
            );
        }

        // EP active → NCCL needs the default stream.
        let stream = if self.comm.is_some() && self.config.ep_world_size > 1 {
            self.gpu.default_stream()
        } else {
            stream
        };

        // Lock KV cache once for the whole batched dispatch.
        let mut kv_cache = self.kv_cache.lock();

        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);

        for slice in streams.iter_mut() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            let chunk_len = slice.chunk_len;
            let is_last_chunk = slice.is_last_chunk;
            let total = tokens.len();
            let seq = &mut *slice.seq;

            // EP=2 zeroes ALL buffers per chunk for NCCL defence-in-depth.
            // EP=1 zeroes essentials only at chunk_start==0 (stale data).
            if self.comm.is_some() {
                self.buffers.zero_all(self.gpu.as_ref(), stream)?;
            } else if chunk_start == 0 {
                self.buffers.zero_all(self.gpu.as_ref(), stream)?;
            }

            // Phase 1+1b: embed at the shared hidden-buffer offset 0.
            // (Per-stream offsets in the buffer are deferred to Phase 2b/3
            // when the kernel-batched path actually reads N streams' worth
            // of hidden at once; today each stream's layer-loop consumes
            // offset 0 before the next stream overwrites it.)
            self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

            // Phase 2: prefix-cache + EP-sync + Marconi.
            let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
                tokens,
                seq,
                chunk_start,
                total,
                &mut kv_cache,
                stream,
            )?;

            // Block allocation through end of chunk.
            let bs = kv_cache.block_size();
            let end_pos = chunk_start + chunk_len;
            let blocks_needed = (end_pos - 1) / bs + 1;
            super::super::super::block_mgmt::ensure_blocks_through_prefill(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
            )?;

            // Phase 2b: proc range (may early-return on full prefix hit
            // of an intermediate chunk).
            let (proc_start, proc_count, effective_seq_len_start) = match self
                .prefill_b_proc_range(
                    tokens,
                    seq,
                    chunk_start,
                    chunk_len,
                    is_last_chunk,
                    kv_write_start,
                    marconi_skip,
                    stream,
                )? {
                ProcRange::Compute {
                    proc_start,
                    proc_count,
                    effective_seq_len_start,
                } => (proc_start, proc_count, effective_seq_len_start),
                ProcRange::EarlyReturn(ptr) => {
                    logits_out.push(ptr);
                    continue;
                }
            };

            // Phase 3+3b: positions / MRoPE / paged metadata.
            let MetaLayout {
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

            // Synchronise H2D before layer compute (GB10 DMA quirk —
            // see prefill_chunk_dispatch comment).
            self.gpu.synchronize(stream)?;

            // Phase 4: forward through all layers (per-stream — Phase 2b/3
            // will hoist this out of the loop with `layer.prefill_batched`).
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
                stream,
            )?;

            // Phase 5: update sequence state.
            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
            seq.seq_len = chunk_start + chunk_len;

            let logits = if is_last_chunk {
                self.prefill_b_finalize_last(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    chunk_len,
                    proc_count,
                    stream,
                )?
            } else {
                self.prefill_b_save_checkpoint(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    chunk_len,
                    stream,
                )?;
                DevicePtr::NULL
            };
            logits_out.push(logits);
        }

        Ok(logits_out)
    }
}
