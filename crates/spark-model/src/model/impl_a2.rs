// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn comm_ref(&self) -> Option<&dyn spark_comm::CommBackend> {
        self.comm.as_deref()
    }

    /// Per-vision-pad-token-id helper. Vision prompts splice ViT
    /// embeddings into placeholder `<|image_pad|>` token positions; the
    /// hashed token-ID stream therefore looks identical for two
    /// distinct images of the same prompt, and naive prefix-cache reuse
    /// would resurrect the FIRST image's KV/SSM blocks for the SECOND
    /// image. Skip cache lookup AND insert whenever any `image_pad`
    /// token is present in the prefill window.
    pub(super) fn tokens_have_vision_pad(&self, tokens: &[u32]) -> bool {
        let pad_id = match self.config.vision.as_ref().map(|v| v.image_pad_token_id) {
            Some(id) if id != 0 => id,
            _ => crate::layers::vision_encoder::IMAGE_PAD_TOKEN_ID,
        };
        tokens.contains(&pad_id)
    }

    /// Free pinned host memory on model destruction.
    pub(super) fn drop_pinned_staging(&self) {
        // SAFETY: Called from Drop, which runs on the owning thread.
        let staging = unsafe { &*self.pinned_staging.get() };
        if !staging.ptr.is_null()
            && let Err(e) = self.gpu.free_host_pinned(staging.ptr, staging.bytes)
        {
            tracing::warn!("Failed to free pinned staging: {e}");
        }
    }

    pub(super) fn ensure_chunked_prefill_meta<'a>(
        &self,
        seq: &'a mut SequenceState,
        total_tokens: usize,
        block_size: usize,
    ) -> Result<&'a mut ChunkedPrefillPageMetadata> {
        let required_blocks = total_tokens.saturating_sub(1) / block_size + 1;
        if seq.chunked_prefill_meta.is_none() {
            seq.chunked_prefill_meta = Some(ChunkedPrefillPageMetadata {
                block_table: self.gpu.alloc(required_blocks.max(1) * 4)?,
                seq_len: self.gpu.alloc(std::mem::size_of::<u32>())?,
                block_capacity: required_blocks,
                uploaded_blocks: 0,
            });
        }

        let meta = seq.chunked_prefill_meta.as_mut().unwrap();
        if meta.block_capacity < required_blocks {
            bail!(
                "chunked prefill metadata capacity {} < required {} blocks",
                meta.block_capacity,
                required_blocks,
            );
        }
        Ok(meta)
    }

    pub(super) fn free_chunked_prefill_meta(&self, seq: &mut SequenceState) -> Result<()> {
        if let Some(meta) = seq.chunked_prefill_meta.take() {
            if !meta.block_table.is_null() {
                self.gpu.free(meta.block_table)?;
            }
            if !meta.seq_len.is_null() {
                self.gpu.free(meta.seq_len)?;
            }
        }
        Ok(())
    }

    /// Bulk broadcast: send an array of u32 tokens from rank 0 to all ranks.
    ///
    /// Uses a single NCCL broadcast instead of per-token broadcasts.
    /// Per-token broadcasting causes NCCL deadlocks on prompts >4K tokens.
    pub(super) fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        let n = tokens.len();
        if self.comm.is_none() {
            return Ok(tokens.to_vec());
        }
        let comm = self.comm.as_ref().unwrap();
        let byte_len = n * 4;
        let stream = self.gpu.default_stream();

        // Use scratch buffer as device staging. This is safe because
        // ep_broadcast_tokens is called BEFORE prefill_chunk, which overwrites
        // scratch with its own metadata. Scratch is sized from the prefill
        // CHUNK size, not the full prompt length, so a long prompt's token
        // payload (n*4 bytes) can exceed it — bound-check before the H2D copy
        // and NCCL broadcast rather than overrun into adjacent device buffers
        // (which raises CUDA error 700 and wedges the GPU).
        let scratch_bytes = self.buffers.sizes().scratch;
        if byte_len > scratch_bytes {
            anyhow::bail!(
                "ep_broadcast_tokens: token payload {byte_len} bytes (n={n}) \
                 exceeds scratch capacity {scratch_bytes} bytes",
            );
        }
        let dev_buf = self.buffers.scratch();

        if comm.rank() == 0 {
            // H2D: copy token bytes to device scratch (synchronous, blocks until done)
            let token_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(tokens.as_ptr() as *const u8, byte_len) };
            self.gpu.copy_h2d(token_bytes, dev_buf)?;
        }

        // Single NCCL broadcast of all tokens at once (root=0)
        comm.broadcast(dev_buf.0, byte_len, 0)?;

        if comm.rank() != 0 {
            // D2H: read received tokens from device
            self.gpu.synchronize(stream)?;
            let mut result = vec![0u32; n];
            let result_bytes =
                unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, byte_len) };
            self.gpu.copy_d2h(dev_buf, result_bytes)?;
            Ok(result)
        } else {
            Ok(tokens.to_vec())
        }
    }

    /// F83 (2026-04-30): all-reduce-min on a single u32 across all
    /// EP ranks. Used by the prefix-cache cache-hit handshake so head
    /// and worker agree on the same `matched_tokens` count even when
    /// their independent local prefix caches disagree. Implemented via
    /// `world_size` rooted broadcasts (one rooted at each rank), each
    /// rank min-reducing the values it observes. NCCL has a native
    /// allreduce-MIN but Atlas's spark-comm trait only exposes SUM
    /// allreduce; the rooted-broadcast loop is portable and adds at
    /// most 2 NCCL ops per chunk-0 cache hit (negligible vs the prefill
    /// compute it unblocks).
    pub(super) fn ep_min_u32(&self, val: u32) -> Result<u32> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(val);
        };
        let stream = self.gpu.default_stream();
        // Loop over the ranks of the ACTUAL communicator: under pure TP
        // (`--tp-size 2 --ep-size 1`) `ep_world_size` is 1 but the comm
        // spans `tp_world_size` ranks — looping only `0..1` would leave
        // the head min-reducing over its own value alone (asymmetric
        // agreement → proc_count mismatch → collective deadlock on a warm
        // cache-hit divergence). For EP-only and overlapping TP==EP
        // topologies `max()` is identical to the previous value.
        let world = self.config.ep_world_size.max(self.config.tp_world_size);
        let mut min_val = val;
        for root in 0..world {
            let v = if comm.rank() == root {
                self.gpu.copy_h2d(&val.to_le_bytes(), self.ep_cmd_buf)?;
                comm.broadcast(self.ep_cmd_buf.0, 4, root)?;
                val
            } else {
                comm.broadcast(self.ep_cmd_buf.0, 4, root)?;
                self.gpu.synchronize(stream)?;
                let mut buf = [0u8; 4];
                self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
                u32::from_le_bytes(buf)
            };
            min_val = min_val.min(v);
        }
        Ok(min_val)
    }

    /// Broadcast a `(seq_id, cmd)` pair from rank 0 to all ranks.
    ///
    /// When `v2` is true, this fires a `seq_id` broadcast immediately before
    /// the existing `cmd` broadcast. Workers reading the stream pick up the
    /// preamble via [`Self::ep_recv_seq_and_cmd`] and route the command to
    /// the matching `SequenceState` slot.
    ///
    /// When `v2` is false, the preamble is skipped and the wire shape is
    /// byte-identical to the legacy single-sequence protocol — head and
    /// worker built before this change continue to interoperate.
    ///
    /// Both ranks must agree on `v2` at startup (e.g. via the same env
    /// var). Disagreement causes the worker to misread the next u32 as a
    /// command code and is the kind of misconfiguration we want to fail
    /// loudly in development — there's no graceful fallback.
    /// True when the head↔worker command protocol must be live: a
    /// multi-rank NCCL world exists — EP **or** pure TP. Under
    /// `--tp-size 2 --ep-size 1` (GDN HeadParallel 2-node) rank>0 still
    /// runs the command-driven worker loop, so the head MUST emit the
    /// same wire protocol as EP mode.
    ///
    /// Root cause of the 2026-07-03 2-node GDN deadlock: every broadcast
    /// helper gated on `ep_world_size > 1` alone, so under pure TP the
    /// head silently dropped ALL commands (prefill cmd + args + decode
    /// cmds) while `ep_broadcast_tokens` (gated only on `comm`) still
    /// fired — the rank-1 worker's 4-byte cmd recv paired with the head's
    /// token-bulk broadcast, read `prompt_tokens[0]` as a decode command,
    /// decoded (and CUDA-graph-captured) while the head was mid-prefill,
    /// and both ranks wedged in shape-mismatched collectives (head stuck
    /// in `sample_first_token` D2H, worker in the next cmd recv, both
    /// GPUs spinning in NCCL kernels).
    pub(crate) fn multi_rank_protocol_active(&self) -> bool {
        self.comm.is_some() && (self.config.ep_world_size > 1 || self.config.tp_world_size > 1)
    }

    pub(super) fn ep_broadcast_seq_and_cmd(&self, seq_id: u32, cmd: u32, v2: bool) -> Result<()> {
        // No-op unless a multi-rank worker protocol is active (EP or pure
        // TP — see `multi_rank_protocol_active`). The per-seq broadcast
        // helpers (`ep_broadcast_cmd_for_seq`) are called unconditionally
        // from the head's prefill / decode / mtp / lifecycle paths,
        // exactly like the original `ep_broadcast_cmd` which no-ops here
        // via `ep_broadcast_cmd_dispatch`. Without this guard
        // `ep_broadcast_u32` panics ("ep_broadcast_u32 without comm") on
        // every single-GPU generation, since `self.comm` is `None`.
        // (Regression from the EP=2 slot-mux work, which only exercised
        // the 2-rank path.)
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        if v2 {
            self.ep_broadcast_u32(seq_id)?;
        }
        self.ep_broadcast_u32(cmd)?;
        Ok(())
    }

    /// Wire-protocol shape for v2 batched decode (`0xFFFFFFE0`):
    ///
    /// ```text
    /// preamble seq_id = 0  (ignored — cmd routes the whole batch)
    /// cmd = 0xFFFFFFE0
    /// N (u32)
    /// seq_ids[N]  (one bulk broadcast)
    /// tokens[N]   (one bulk broadcast)
    /// ```
    ///
    /// The matched receive on the worker is `ep_worker_decode_batch` in
    /// `ep_worker_step_impl`'s dispatch. Both ranks then call the
    /// `decode_batch_compute_main` path which runs the existing batched
    /// `decode_multi_seq` per-layer with N tokens — same per-layer NCCL
    /// allreduce sequence on both ranks, comm-stream order matches.
    ///
    /// Caller must hold `self.comm.is_some()` (no-op on world_size=1) and
    /// `self.ep_protocol_v2 == true` (without the preamble, the worker
    /// would mis-parse the seq_id u32 as a cmd code). Both conditions are
    /// guaranteed at the only caller — `decode_batch_dispatch`'s EP
    /// branch — but asserted defensively here.
    pub(super) fn ep_broadcast_decode_batch_dispatch(
        &self,
        seq_ids: &[u32],
        tokens: &[u32],
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        debug_assert!(
            self.ep_protocol_v2,
            "ep_broadcast_decode_batch_dispatch called without ATLAS_EP_PROTOCOL=v2"
        );
        debug_assert_eq!(
            seq_ids.len(),
            tokens.len(),
            "seq_ids and tokens length mismatch"
        );
        self.ep_broadcast_seq_and_cmd(0, 0xFFFFFFE0, true)?;
        self.ep_broadcast_u32(seq_ids.len() as u32)?;
        self.ep_broadcast_tokens(seq_ids)?;
        self.ep_broadcast_tokens(tokens)?;
        Ok(())
    }

    /// Receive a `(seq_id, cmd)` pair from rank 0. Worker-side counterpart
    /// of [`Self::ep_broadcast_seq_and_cmd`].
    ///
    /// With `v2` enabled the returned `seq_id` is the slot the head wants
    /// the worker to dispatch the command into; with `v2` disabled the
    /// returned `seq_id` is always 0 (the legacy singleton slot).
    pub(super) fn ep_recv_seq_and_cmd(&self, v2: bool) -> Result<(u32, u32)> {
        let seq_id = if v2 { self.ep_broadcast_u32(0)? } else { 0 };
        let cmd = self.ep_broadcast_u32(0)?;
        Ok((seq_id, cmd))
    }

    /// Broadcast a u32 command from rank 0 to all ranks.
    /// Rank 0 writes `val` to GPU buffer and broadcasts.
    /// Other ranks receive the value and return it.
    pub(super) fn ep_broadcast_u32(&self, val: u32) -> Result<u32> {
        let comm = self.comm.as_ref().expect("ep_broadcast_u32 without comm");
        let stream = self.gpu.default_stream();
        if comm.rank() == 0 {
            // Sender: H2D + broadcast. Stream ordering ensures completion
            // before next GPU operation on the same stream. No sync needed.
            self.gpu.copy_h2d(&val.to_le_bytes(), self.ep_cmd_buf)?;
            comm.broadcast(self.ep_cmd_buf.0, 4, 0)?;
            Ok(val)
        } else {
            // Receiver: broadcast + sync + D2H to read the received value.
            comm.broadcast(self.ep_cmd_buf.0, 4, 0)?;
            self.gpu.synchronize(stream)?;
            let mut buf = [0u8; 4];
            self.gpu.copy_d2h(self.ep_cmd_buf, &mut buf)?;
            Ok(u32::from_le_bytes(buf))
        }
    }

    /// EP worker step: receive a (seq_id, cmd) preamble from rank 0 and
    /// execute the command in the addressed slot.
    ///
    /// Returns false when the worker should shut down.
    ///
    /// Protocol (`ATLAS_EP_PROTOCOL=v2`): rank 0 broadcasts the slot
    /// identifier first (worker uses it to pick the right `SequenceState`
    /// from `slots`), then the command code, then any per-command follow-on
    /// data. With v1 (the default) the preamble is skipped and every
    /// command targets slot 0 — equivalent to the singleton path this
    /// function originally implemented.
    ///
    /// Command codes:
    /// - 0..0xFFFFFFEF: token ID → decode in the addressed slot
    /// - 0xFFFFFFF0: prefill start → chunk_len, chunk_start, full_len, then full_len tokens
    /// - 0xFFFFFFF1: alloc slot (frees any prior occupant first, then re-allocates)
    /// - 0xFFFFFFF2/3/4: verify K=2/3/4 → K tokens, then accept/num_accepted
    /// - 0xFFFFFFFF: shutdown (seq_id is ignored; applies to the whole worker)
    pub(super) fn ep_worker_step_impl(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        let (seq_id, cmd) = self.ep_recv_seq_and_cmd(self.ep_protocol_v2)?;

        // Shutdown applies to the whole worker — seq_id is ignored.
        if cmd == 0xFFFFFFFF {
            return Ok(false);
        }

        // Batched-decode (`0xFFFFFFE0`): the preamble seq_id is sentinel-0;
        // the real per-token routing lives in the seq_ids[N] payload that
        // follows. Hand off to the batched handler which reads N + seq_ids
        // + tokens off the wire and dispatches the matched compute.
        if cmd == 0xFFFFFFE0 {
            return self.ep_worker_decode_batch(slots);
        }

        let slot_idx = seq_id as usize;
        if slot_idx >= slots.len() {
            anyhow::bail!(
                "ep_worker_step: seq_id {} exceeds slot capacity {} \
                 (head and worker likely disagree on max_batch_size)",
                seq_id,
                slots.len(),
            );
        }

        // `alloc-slot` (0xFFFFFFF1): replace the slot's sequence wholesale.
        // Frees the prior occupant if any, then allocates a fresh one. The
        // SSM-pool slot the new sequence claims may or may not equal
        // slot_idx — head and worker stay aligned because both ranks call
        // `claim_slot()` from a free-list pop in matched order. Defensive
        // bail if they ever diverge so we fail fast rather than corrupt KV.
        if cmd == 0xFFFFFFF1 {
            if let Some(mut old) = slots[slot_idx].take() {
                self.free_sequence(&mut old)?;
            }
            let new_seq = self.alloc_sequence()?;
            if self.ep_protocol_v2 && new_seq.slot_idx != slot_idx {
                anyhow::bail!(
                    "ep_worker_step: SSM-pool slot {} doesn't match head's seq_id {} \
                     after alloc — claim_slot ordering invariant violated",
                    new_seq.slot_idx,
                    slot_idx,
                );
            }
            slots[slot_idx] = Some(new_seq);
            return Ok(true);
        }

        // All other commands operate on an already-allocated slot.
        let seq = slots[slot_idx].as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "ep_worker_step: cmd {:#x} arrived for unallocated slot {} \
                 — head dispatched without a prior alloc",
                cmd,
                slot_idx,
            )
        })?;

        self.ep_worker_dispatch_cmd(cmd, seq)
    }

    /// Per-command dispatch for [`Self::ep_worker_step_impl`]. The
    /// (seq_id, cmd) preamble + slot lookup + shutdown + alloc are already
    /// handled by the caller; this routine assumes `seq` is the right
    /// slot's allocated `SequenceState`.
    fn ep_worker_dispatch_cmd(&self, cmd: u32, seq: &mut SequenceState) -> Result<bool> {
        let stream = self.gpu.default_stream();

        match cmd {
            0xFFFFFFF0 => {
                // Prefill chunk: receive chunk_len, chunk_start, full prompt length,
                // then ALL prompt tokens via bulk broadcast (single NCCL op).
                let chunk_len = self.ep_broadcast_u32(0)? as usize;
                let chunk_start = self.ep_broadcast_u32(0)? as usize;
                let full_len = self.ep_broadcast_u32(0)? as usize;
                let full_tokens = self.ep_broadcast_tokens(&vec![0u32; full_len])?;
                // Compute is_last from chunk bounds — must match rank 0's
                // value so Marconi skip branches are identical (bug #33).
                let is_last = chunk_start + chunk_len >= full_len;
                let _ =
                    self.prefill_chunk(&full_tokens, seq, chunk_start, chunk_len, is_last, stream)?;
                // Normalize SSM states after every chunk — must mirror the head's
                // normalize_ssm_states call (scheduler.rs line 584). Without this,
                // SSM states diverge between ranks causing MoE all-reduce corruption
                // and gibberish output after the first token (bug #41).
                if let Err(e) = self.normalize_ssm_states(seq, stream) {
                    tracing::warn!("Worker SSM state normalization failed: {e:#}");
                }
            }
            0xFFFFFFF2 => {
                // Verify K=2: receive 2 tokens, run verify, receive accept/reject
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                self.sync_secondary()?;
                self.decode_verify_graphed(&[t0, t1], seq, stream)?;
                let accepted = self.ep_broadcast_u32(0)?;
                if accepted == 1 {
                    self.start_checkpoint_async(seq)?;
                    self.trim_proposer_state(seq, 1, 0)?;
                } else {
                    seq.seq_len -= 1;
                    seq.tokens.pop();
                    self.trim_proposer_state(seq, 0, 0)?;
                    self.start_rollback_and_checkpoint_async(seq, 1)?;
                }
            }
            0xFFFFFFF3 => {
                // Verify K=3: receive 3 tokens, run verify, receive num_accepted (0/1/2)
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let t2 = self.ep_broadcast_u32(0)?;
                self.sync_secondary()?;
                self.decode_verify_graphed_k3(&[t0, t1, t2], seq, stream)?;
                let num_accepted = self.ep_broadcast_u32(0)?;
                self.trim_proposer_state(seq, num_accepted as usize, 0)?;
                match num_accepted {
                    2 => {
                        self.start_checkpoint_async(seq)?;
                    }
                    1 => {
                        seq.seq_len -= 1;
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 2)?;
                    }
                    _ => {
                        seq.seq_len -= 2;
                        seq.tokens.pop();
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 1)?;
                    }
                }
            }
            0xFFFFFFF4 => {
                // Verify K=4: receive 4 tokens, run verify, receive num_accepted (0/1/2/3)
                let t0 = self.ep_broadcast_u32(0)?;
                let t1 = self.ep_broadcast_u32(0)?;
                let t2 = self.ep_broadcast_u32(0)?;
                let t3 = self.ep_broadcast_u32(0)?;
                self.sync_secondary()?;
                self.decode_verify_graphed_k4(&[t0, t1, t2, t3], seq, stream)?;
                let num_accepted = self.ep_broadcast_u32(0)?;
                self.trim_proposer_state(seq, num_accepted as usize, 0)?;
                match num_accepted {
                    3 => {
                        self.start_checkpoint_async(seq)?;
                    }
                    2 => {
                        seq.seq_len -= 1;
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 3)?;
                    }
                    1 => {
                        seq.seq_len -= 2;
                        seq.tokens.pop();
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 2)?;
                    }
                    _ => {
                        seq.seq_len -= 3;
                        seq.tokens.pop();
                        seq.tokens.pop();
                        seq.tokens.pop();
                        self.start_rollback_and_checkpoint_async(seq, 1)?;
                    }
                }
            }
            token => {
                // Regular decode
                self.decode(token, seq, stream)?;
            }
        }

        Ok(true)
    }

    /// Worker-side handler for the batched-decode protocol (`0xFFFFFFE0`).
    ///
    /// Reads `N` (u32), `seq_ids[N]` (bulk broadcast), and `tokens[N]`
    /// (bulk broadcast) off the wire — matching what the head wrote in
    /// `ep_broadcast_decode_batch_dispatch`. Then builds an in-order
    /// `Vec<&mut SequenceState>` from the addressed slots and hands off
    /// to the shared compute path. The compute does the same per-layer
    /// `decode_multi_seq` the non-EP main batched path runs, with the
    /// NCCL allreduces inside each layer matching the head's submission
    /// order on the comm.
    ///
    /// Validates seq_ids up-front (bounds + duplicates) so a malformed
    /// payload from a buggy head fails before touching slot state.
    fn ep_worker_decode_batch(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        let seq_ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let tokens = self.ep_broadcast_tokens(&vec![0u32; n])?;

        // Validate up front so we fail before touching slot state.
        let mut seen = std::collections::HashSet::new();
        for &id in &seq_ids {
            let idx = id as usize;
            if idx >= slots.len() {
                anyhow::bail!(
                    "ep_worker_decode_batch: seq_id {} exceeds slot capacity {}",
                    id,
                    slots.len(),
                );
            }
            if !seen.insert(id) {
                anyhow::bail!("ep_worker_decode_batch: duplicate seq_id {} in batch", id);
            }
        }

        // Drain populated slots into a (idx, ref) Vec we can index by
        // position with `swap_remove`. The borrow checker won't let us
        // index `slots[seq_ids[i]]` in a loop because each `&mut` is
        // distinct but the indexer can't prove non-overlap.
        let mut slot_refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_mut().map(|s| (i, s)))
            .collect();

        // Order the refs to match the head's seq_ids order so the
        // compute path processes tokens in the same batch index as the
        // head — critical for KV-cache row alignment per slot.
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(n);
        for &id in &seq_ids {
            let idx = id as usize;
            let pos = slot_refs
                .iter()
                .position(|(i, _)| *i == idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("ep_worker_decode_batch: slot {} not allocated", idx)
                })?;
            let (_, seq) = slot_refs.swap_remove(pos);
            refs.push(seq);
        }

        let stream = self.gpu.default_stream();
        self.decode_batch_compute_main(&tokens, &mut refs, stream)?;
        Ok(true)
    }
}
