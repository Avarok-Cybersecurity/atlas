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
//!   - All streams share `chunk_len`, `seq_len_start` (q_offset), and
//!     `is_last_chunk` flag
//!   - Total stacked tokens fits in buffer arena
//!   - No MLA / HDIM=512 / HSS-engaged layer in the model
//!   - All batched kernel handles loaded
//!
//! Validation status: compile-only. Hardware validation pending.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use super::proc_range::ProcRange;
use super::stage_batched::PerStreamStageInfo;
use super::upload_meta::MetaLayout;
use crate::layer::{
    BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState, TransformerLayer,
};
use crate::traits::{Model, PrefillSlice, SequenceState};

impl TransformerModel {
    /// Returns true when the batched-kernel path is viable for these
    /// streams. Cheap upfront check — caller (dispatch) falls back to
    /// per-stream when false.
    pub(in crate::model) fn kernel_batched_eligible(&self, streams: &[PrefillSlice<'_>]) -> bool {
        check_kernel_batched_eligible(
            streams
                .iter()
                .map(|s| (s.chunk_len, s.chunk_start, s.is_last_chunk)),
            streams.len(),
            self.buffers.max_batch_tokens(),
            &self.config.model_type,
            self.config.head_dim,
        )
    }
}

/// Pure-data predicate extracted from [`TransformerModel::kernel_batched_eligible`]
/// so the gating rules are unit-testable without a real `TransformerModel`.
/// Caller materialises per-stream tuples `(chunk_len, chunk_start, is_last_chunk)`.
pub(in crate::model) fn check_kernel_batched_eligible<I>(
    streams: I,
    n: usize,
    arena_cap: usize,
    model_type: &str,
    head_dim: usize,
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
    for (chunk_len, chunk_start, is_last) in streams {
        // `chunk_len`, `chunk_start`, and `is_last_chunk` must all
        // match across streams. Different `chunk_start` produces
        // different `effective_seq_len_start` post-Marconi (which the
        // batched attention kernel cannot handle); mixing
        // `is_last_chunk` can't dispatch one finalize_last and one
        // save_checkpoint in a single batched call.
        match first {
            None => first = Some((chunk_len, chunk_start, is_last)),
            Some((cl, cs, il)) => {
                if chunk_len != cl || chunk_start != cs || is_last != il {
                    return false;
                }
            }
        }
        total += chunk_len;
    }
    // Total stacked tokens fit in arena.
    total <= arena_cap
}

impl TransformerModel {
    /// Q12 Path B: full kernel-batched prefill orchestration.
    ///
    /// Caller (prefill_batch_chunk_dispatch) MUST have verified
    /// `kernel_batched_eligible` before calling this; if a per-stream
    /// constraint is later detected here (e.g. proc_count mismatch from
    /// differing prefix-cache hits), this function bails Err.
    pub(in crate::model) fn prefill_batch_chunk_kernel_batched(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        let n = streams.len();
        let chunk_len = streams[0].chunk_len;
        let is_last_chunk = streams[0].is_last_chunk;
        let h = self.config.hidden_size;
        let dtype_bytes = if self.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        // EP active → NCCL needs the default stream.
        let stream = if self.comm.is_some() && self.config.ep_world_size > 1 {
            self.gpu.default_stream()
        } else {
            stream
        };

        // Lock KV cache once.
        let mut kv_cache = self.kv_cache.lock();

        // Zero shared buffers once (instead of N times in per-stream).
        if self.comm.is_some() || streams[0].chunk_start == 0 {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
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
        }
        let mut per_stream: Vec<PerStreamMeta> = Vec::with_capacity(n);

        // Tracks MRoPE / paged-flag agreement across streams.
        let mut use_mrope: Option<bool> = None;
        let mut needs_paged: Option<bool> = None;

        // Per-stream scratch slot size: positions + MRoPE H/W (optional) +
        // slot table. Conservative estimate: 12 bytes per token + small
        // header. Reserved 4 KB per stream is plenty for chunk_len ≤ 256.
        // For larger chunk_len the scratch budget scales with arena_cap.
        let per_stream_meta_bytes = ((chunk_len * 16) + 64).max(4096);
        // Cumulative scratch offset cursor — starts after MoE topk
        // staging area (per single-stream upload_meta convention).
        let moe_scratch_bytes = chunk_len * self.config.num_experts_per_tok * 4 * 2 * n;
        let mut scratch_cursor = (moe_scratch_bytes + 63) & !63;

        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            let total = tokens.len();
            let seq = &mut *slice.seq;

            // Embed at b*chunk_len*H offset into shared hidden buffer.
            let hidden_b = hidden_base.offset(b * chunk_len * h * dtype_bytes);
            self.prefill_b_embed_chunk_at(tokens, chunk_start, chunk_len, hidden_b, stream)?;

            // Prefix-cache lookup, EP-sync, Marconi restore.
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

            // Effective processing range.
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
                ProcRange::EarlyReturn(_) => anyhow::bail!(
                    "kernel-batched: stream {b} early-returned during proc_range \
                         — eligibility check missed this. Caller should fall back."
                ),
            };

            // Cross-stream consistency: all streams must share proc_count
            // and effective_seq_len_start (q_offset) for the batched
            // attention kernel.
            if b > 0 {
                if per_stream[0].proc_count != proc_count {
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
            let layout = self.prefill_b_upload_meta_at(
                tokens,
                seq,
                chunk_start,
                chunk_len,
                proc_start,
                proc_count,
                effective_seq_len_start,
                &kv_cache,
                meta_base,
                stream,
            )?;
            if layout.needs_paged {
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
            let (block_table_dev, seq_len_dev) = if layout.needs_paged {
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
            });
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
        // Advance cursor past the staged BatchedAttnMetadata. Generous
        // upper bound for the staging size: positions(MRoPE ×3) + slots
        // + pointer arrays.
        let stage_size = (proc_count * n * 16) + (n * 32) + 256;
        scratch_cursor += stage_size;

        // Q12 safety: assert scratch headroom before consuming the
        // h_state_ptrs JIT slot per SSM layer. h_state_ptrs is N*8 bytes.
        // The scratch budget for batched dispatch is dominated by the
        // per-stream meta blocks + stacked BatchedAttnMetadata; bail if
        // we'd exceed scratch capacity rather than overrun into another
        // buffer.
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
            total_len: proc_count * n,
        };

        // ForwardContext for batched layer calls. attn_metadata is
        // intentionally None — layers read BatchedAttnMetadata directly
        // through the model-level dispatcher arguments.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            attn_metadata: None,
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
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

        // ── PHASE C: per-stream finalize ──
        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(n);
        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            let seq = &mut *slice.seq;
            let m = &per_stream[b];

            // Phase 5: sequence-state update.
            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
            seq.seq_len = chunk_start + chunk_len;

            let logits = if is_last_chunk {
                self.prefill_b_finalize_last_at(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    chunk_len,
                    m.proc_count,
                    b * chunk_len,
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

// Unit tests for `check_kernel_batched_eligible` live in a sibling
// file `batch_kernel_tests.rs` (mounted by `prefill_b.rs`) to keep
// this file under the 500-LoC file-size-cap.
