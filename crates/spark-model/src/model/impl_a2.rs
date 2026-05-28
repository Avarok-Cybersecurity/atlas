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

        // Use scratch buffer (guaranteed large enough for metadata) as device staging.
        // This is safe because ep_broadcast_tokens is called BEFORE prefill_chunk,
        // which overwrites scratch with its own metadata.
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
        let world = self.config.ep_world_size;
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
    pub(super) fn ep_broadcast_seq_and_cmd(&self, seq_id: u32, cmd: u32, v2: bool) -> Result<()> {
        if v2 {
            self.ep_broadcast_u32(seq_id)?;
        }
        self.ep_broadcast_u32(cmd)?;
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
}
