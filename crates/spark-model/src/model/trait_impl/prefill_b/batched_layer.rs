// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Path B model-level batched per-layer dispatchers.
//!
//! Three methods on `TransformerModel`:
//!   - `prefill_attn_batched_layer` — runs one attention layer over N
//!     stacked-input streams, using the batched paged-prefill kernel for
//!     the attention compute step.
//!   - `prefill_ssm_batched_layer` — runs one SSM layer over N streams,
//!     using the batched GDN kernel for the recurrent step.
//!   - `prefill_dense_batched_layer` — runs one dense (FFN-only or
//!     attention-only) layer that has no SSM state. Falls back to
//!     stacked-input single kernel call (per-token kernels naturally
//!     parallelise across the stacked layout).
//!
//! All three are called from `prefill_batch_chunk_dispatch`'s outer
//! layer loop after `stage_batched_attn_metadata` has built the
//! per-call metadata.
//!
//! ## Status (commit on 2026-05-10): scaffolded.
//!
//! Each method below currently delegates to N per-stream `layer.prefill(...)`
//! calls — same behaviour as the trait default impl — but owns the
//! routing decision per layer type. Replacing the body with the actual
//! batched kernel calls is bounded:
//!
//! **Attention (~150 LoC body replacement)**:
//!   1. ONE rms_norm + residual on stacked hidden [N*chunk_len, H].
//!   2. ONE q_proj/k_proj/v_proj GEMM on stacked input (token-parallel
//!      kernels naturally handle stacked layout).
//!   3. ONE RoPE using `meta.positions_stacked`.
//!   4. ONE reshape_and_cache using `meta.slot_stacked` for KV writes.
//!   5. ONE batched paged-prefill via `prefill_attention_paged_*_batched`
//!      using `meta.block_table_ptrs`. Grid `(num_q_heads, q_chunks,
//!      batch_size)`.
//!   6. ONE o_proj + residual on stacked output.
//!
//! **SSM (~200 LoC body replacement)**:
//!   1-6. Per-stream phase1 with `token_offset = b * chunk_len` writing
//!        into stacked GdnPrefillBuffers (model-owned, sized for
//!        max_batch_tokens).
//!   7. Build `h_state_ptrs[N]` device array from each stream's
//!      `SsmLayerState::h_state` (JIT per-layer-call, ~5μs H2D).
//!   8. ONE batched GDN via `gdn_prefill_persistent_smem_batched` (or
//!      sibling) with `batch_size = N`, `seq_len = chunk_len`.
//!   9-12. Per-stream phase3 with `token_offset = b * chunk_len`.
//!
//! Hardware validation pending — golden trace comparison vs N per-stream
//! single-stream runs, then Q12 repro for end-to-end TTFT win.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layer::{
    AttnMetadataDev, BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState,
    TransformerLayer,
};
use crate::traits::SequenceState;

impl TransformerModel {
    /// Run one attention layer over N stacked-input streams.
    ///
    /// `hidden_stacked` and `residual_stacked` are at the arena's
    /// `hidden_states()` / `residual()` pointers respectively, and
    /// contain N streams' tokens at offsets `b * chunk_len * H * dtype`.
    /// `seqs` provides per-stream `SequenceState` for KV-write routing
    /// and per-stream layer state (which is `EmptyLayerState` for
    /// attention but kept in the slice for symmetry with SSM).
    /// `meta` is the per-call `BatchedAttnMetadata` from
    /// `stage_batched_attn_metadata`.
    pub(in crate::model) fn prefill_attn_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        kv_write_starts: &[usize],
        seq_lens_start: usize,
        meta: &BatchedAttnMetadata,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // TODO(kernel-session): batched attention layer body.
        //
        // The body replacement is documented in the module-level docs above.
        // Until it lands, this stub returns Err so callers can detect the
        // unimplemented state and fall back to per-stream
        // `prefill_chunk_dispatch`.
        let _ = (layer, layer_idx, hidden_stacked, residual_stacked, seqs,
                 kv_cache, kv_write_starts, seq_lens_start, meta, ctx, stream);
        anyhow::bail!(
            "prefill_attn_batched_layer: stub body — caller should fall back \
             to per-stream prefill_chunk_dispatch until kernel-session wiring \
             lands. See module docstring for the body replacement plan."
        )
    }

    /// Run one SSM layer over N stacked-input streams.
    ///
    /// Same args as `prefill_attn_batched_layer` plus access to the
    /// model's SSM layer state pool via `seqs[b].layer_states[layer_idx]`.
    pub(in crate::model) fn prefill_ssm_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        meta: &BatchedAttnMetadata,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // TODO(kernel-session): batched SSM layer body.
        //
        // Plan:
        //   for b in 0..n {
        //       layer.prefill_phase1(... hidden_b, residual_b, chunk_len,
        //           seqs[b].layer_states[layer_idx], kv_cache,
        //           seqs[b].seq_len_start, ..., gdn_bufs,
        //           token_offset = b * chunk_len, ctx, stream)?;
        //   }
        //   // Build h_state_ptrs[N] device array
        //   let h_state_ptrs_dev = stage_h_state_ptrs(layer_idx, seqs, ctx)?;
        //   // ONE batched GDN
        //   layer.prefill_gdn_full_batched(h_state_ptrs_dev, gdn_bufs,
        //       batch_size=n, seq_len=chunk_len, ctx, stream)?;
        //   for b in 0..n {
        //       layer.prefill_phase3(... hidden_b, residual_b, chunk_len,
        //           gdn_bufs, token_offset = b * chunk_len, ctx, stream)?;
        //   }
        //
        // Stub: bail and let caller fall back.
        let _ = (layer, layer_idx, hidden_stacked, residual_stacked, seqs,
                 kv_cache, meta, gdn_bufs, ctx, stream);
        anyhow::bail!(
            "prefill_ssm_batched_layer: stub body — caller should fall back \
             to per-stream prefill_chunk_dispatch until kernel-session wiring \
             lands. See module docstring for the body replacement plan."
        )
    }

    /// Run one dense (non-SSM, non-attention-stateful) layer over N stacked-
    /// input streams. Per-token kernels (rms_norm, GEMM, MoE) handle the
    /// stacked layout naturally without per-stream metadata.
    pub(in crate::model) fn prefill_dense_batched_layer(
        &self,
        layer: &dyn TransformerLayer,
        layer_idx: usize,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        seqs: &mut [&mut SequenceState],
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // For dense layers (no per-stream state), call layer.prefill once
        // with total_tokens. Per-token kernels (rms_norm + GEMM + MoE) all
        // parallelise across the stacked input naturally.
        // CAVEAT: today's `layer.prefill` reads `ctx.attn_metadata` for
        // positions / RoPE — for batched, ctx must carry the stacked
        // positions. The caller is responsible for setting this up before
        // entering the layer loop.
        if seqs.is_empty() {
            return Ok(());
        }
        let first_seq = &mut **seqs.first_mut().unwrap();
        // Use the first stream's state placeholder (dense layers don't
        // mutate per-stream state). Block tables: all streams share the
        // same paged cache view — kernel reads via stacked slot indices.
        layer.prefill(
            hidden_stacked,
            residual_stacked,
            total_tokens,
            first_seq.layer_states[layer_idx].as_mut(),
            kv_cache,
            0, // seq_len_start unused for dense layers
            &mut first_seq.block_table,
            &mut first_seq.disk_block_ids,
            &mut first_seq.disk_last_offloaded_per_layer,
            0,
            ctx,
            stream,
        )
    }
}
