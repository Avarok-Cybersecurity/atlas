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

    /// EP worker step: receive a command from rank 0 and execute it.
    ///
    /// Returns false when the worker should shut down.
    /// Protocol: rank 0 broadcasts u32 commands before each model operation:
    /// - 0..0xFFFFFFF0: token ID → decode
    /// - 0xFFFFFFF0: prefill start → next broadcast = length, then length tokens
    /// - 0xFFFFFFF1: free+realloc sequence
    /// - 0xFFFFFFF2: verify K=2 → next 2 broadcasts = tokens, then accept/reject
    /// - 0xFFFFFFF3: verify K=3 → next 3 broadcasts = tokens, then num_accepted
    /// - 0xFFFFFFF4: verify K=4 → next 4 broadcasts = tokens, then num_accepted
    /// - 0xFFFFFFFF: shutdown
    pub(super) fn ep_worker_step_impl(&self, seq: &mut SequenceState) -> Result<bool> {
        let cmd = self.ep_broadcast_u32(0)?;
        let stream = self.gpu.default_stream();

        match cmd {
            0xFFFFFFFF => return Ok(false), // shutdown
            0xFFFFFFF1 => {
                // Free and realloc sequence
                self.free_sequence(seq)?;
                *seq = self.alloc_sequence()?;
            }
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
