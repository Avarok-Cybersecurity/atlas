// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_batch_chunk_dispatch` — batched prefill orchestrator (Q12).
//!
//! Mirrors `prefill_chunk_dispatch` (the single-stream path in `prefill_b.rs`)
//! but processes N concurrent streams in one model-level call. The motivating
//! win is **per-layer L2-amortised weight load**: today's per-stream loop
//! (default trait impl) re-streams the full layer-0 weights for stream 1 even
//! though they were just loaded for stream 0.
//!
//! ## Layout
//!
//! - `hidden`/`residual` use the shared buffer arena, sized for
//!   `max_batch_tokens`. Each stream's chunk lands at offset
//!   `cu_seqlens[i] * hidden_size * residual_dtype` in the buffer.
//! - Per-stream metadata (positions, slots, paged block tables) lives in the
//!   scratch buffer at distinct offsets — built once per dispatch by reusing
//!   the existing `prefill_b_upload_meta` and `prefill_b_upload_paged` phase
//!   helpers, called sequentially per stream.
//! - The layer loop is shared: one `for (i, layer) in self.layers` iteration
//!   calls `layer.prefill_batched(hidden, residual, cu_seqlens, &mut states,
//!   &mut block_tables, ...)`. Each layer override decides whether to issue
//!   one kernel for all N streams (Phase 2b SSM, Phase 3 attention) or to
//!   loop per-stream internally (default trait impl).
//!
//! ## Status
//!
//! **Phase 4b stub (this commit).** The function is the entry point that the
//! `Model::prefill_batch_chunk` override eventually delegates to, but the
//! actual per-layer-batched implementation is staged in pieces:
//!
//! 1. Buffer-arena fit check + bail to default when `total_tokens` would
//!    overflow `max_batch_tokens`. Implemented.
//! 2. Per-stream embed/prefix/blocks/metadata setup. **TODO** — needs
//!    refactoring `prefill_b_embed_chunk` / `prefill_b_prefix_lookup` /
//!    `prefill_b_proc_range` / `prefill_b_upload_meta` /
//!    `prefill_b_upload_paged` to write into per-stream offsets of the
//!    shared metadata buffer instead of a single fixed offset.
//! 3. Shared per-layer loop calling `layer.prefill_batched`. **TODO** —
//!    requires building the stacked `cu_seqlens`, the N-vector of
//!    `&mut LayerState`, the N-vector of `&mut Vec<u32>` block tables, and
//!    the per-stream `seq_lens_start` slice. Falls back gracefully to the
//!    layer's default per-stream-loop impl.
//! 4. Per-stream finalize (last chunk → sample first token; intermediate
//!    chunk → save Marconi snapshot). **TODO**.
//!
//! Until pieces 2-4 land, this dispatch returns `Err(NotImplemented)` and
//! the trait's default impl handles batched prefill by looping over
//! `prefill_chunk` per stream — same behaviour as before this commit.
//! The override exists so future commits can fill it in incrementally
//! without changing the trait or the scheduler.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, anyhow};
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
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
        if n == 0 {
            return Ok(Vec::new());
        }

        // Total tokens to batch through the layer loop.
        let total_tokens: usize = streams.iter().map(|s| s.chunk_len).sum();
        let arena_cap = self.buffers.max_batch_tokens();

        // Phase-4b-stub fit check: when the stacked layout doesn't fit in
        // the buffer arena, abort and let the trait default impl handle it
        // via per-stream `prefill_chunk` calls (no L2 amortisation).
        // Bumping `max_batch_tokens` to host stacked-stream layouts is a
        // recipe-side knob that should land alongside the per-layer
        // batched dispatch.
        if total_tokens > arena_cap {
            anyhow::bail!(
                "Batched prefill total_tokens={total_tokens} exceeds arena \
                 capacity {arena_cap} (n={n} streams). Bump \
                 --max-prefill-tokens or run --max-batch-size lower so the \
                 stacked layout fits, or fall through to single-stream \
                 prefill_chunk per stream (the trait default)."
            );
        }

        // Phase 4b TODO: per-stream embed → per-stream prefix lookup →
        // shared layer loop → per-stream finalize.
        //
        // Until that ships, signal to the caller that this dispatch isn't
        // ready by returning a dedicated error. The trait default impl
        // (which loops over `prefill_chunk` per stream) is still wired
        // through at the trait level, so callers that need batched prefill
        // for correctness keep working — the only thing this stub gives up
        // is the L2-amortised path. Dropping into the default impl is one
        // catch-and-fallback away in the `prefill_batch_chunk` override
        // (see TransformerModel impl in `mod.rs`).
        //
        // Mark unused to suppress dead-code warnings on n+total_tokens until
        // the stub is fleshed out.
        let _ = stream;
        let _ = (n, total_tokens);
        Err(anyhow!(
            "TransformerModel::prefill_batch_chunk_dispatch — Phase 4b \
             body not yet implemented; caller should fall back to default \
             trait impl (per-stream prefill_chunk loop)."
        ))
    }
}
