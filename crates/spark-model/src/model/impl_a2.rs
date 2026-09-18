// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use avarok_core::config::{LayerType, ModelConfig};
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

/// Sequence retirement: run `cache_sequence` (finish-leaf SSM snapshot + radix
/// insert) on the worker too.
///
/// Without it only the head took finish-leaf snapshots, so the ranks' snapshot
/// pools held different entries, evicted differently, and eventually proposed
/// different Marconi anchors — different SSM replay lengths, mismatched
/// collectives, NCCL spin. This branch's command space uses 0xF0..0xF4 plus
/// 0xE0; 0xFFFF_FFF6 is free here and matches the code on `research/glm-exl3`,
/// so the two trees stay wire-compatible.
pub(crate) const EP_CMD_CACHE_SEQ: u32 = 0xFFFF_FFF6;

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
    /// The two placeholder token ids vision input can occupy:
    /// `(<|image_pad|>, <|video_pad|>)`, each falling back to the family
    /// default when the checkpoint declares none.
    pub(super) fn vision_pad_ids(&self) -> (u32, u32) {
        let v = self.config.vision.as_ref();
        let image = v
            .map(|v| v.image_pad_token_id)
            .filter(|id| *id != 0)
            .unwrap_or(crate::layers::vision_encoder::IMAGE_PAD_TOKEN_ID);
        let video = v
            .map(|v| v.video_pad_token_id)
            .filter(|id| *id != 0)
            .unwrap_or(crate::layers::vision_encoder::VIDEO_PAD_TOKEN_ID);
        (image, video)
    }

    /// Whether `tok` is a vision placeholder of EITHER modality.
    ///
    /// Every splice and every cache decision must use this rather than
    /// comparing against the image token alone. A video's pad tokens are a
    /// different id, and treating them as ordinary text is silent: the
    /// prompt still tokenises, the counts still add up, and the model simply
    /// never receives the pixels.
    pub(super) fn is_vision_pad(&self, tok: u32) -> bool {
        let (image, video) = self.vision_pad_ids();
        tok == image || tok == video
    }

    pub(super) fn tokens_have_vision_pad(&self, tokens: &[u32]) -> bool {
        self.first_vision_pad_index(tokens).is_some()
    }

    /// Index of the first vision pad in `tokens`, if any.
    ///
    /// The prefix cache needs the POSITION, not just the presence. Every image
    /// expands to the SAME pad token id, so a token-keyed radix tree cannot
    /// tell two images apart and must never match ACROSS a pad — but that
    /// ambiguity begins AT the first pad. Everything before it is ordinary
    /// text with plain sequential MRoPE positions (the image has not happened
    /// yet), so that head is what this chat's earlier text-only turns already
    /// inserted, and is safe to reuse.
    pub(super) fn first_vision_pad_index(&self, tokens: &[u32]) -> Option<usize> {
        let (image, video) = self.vision_pad_ids();
        tokens.iter().position(|&t| t == image || t == video)
    }

    /// The slice of `tokens` the prefix cache may key on: everything before the
    /// first vision pad.
    ///
    /// ONE function owns this rule so the lookup and the probes that PREDICT
    /// the lookup cannot drift — they already had, before this: the three
    /// full-prefill paths each re-derived "has a pad => match nothing" while
    /// the `peek_matched_tokens` probes that size the batched arena derived
    /// nothing and happily reported a match the real lookup then refused.
    ///
    /// A pad at index 0 yields an empty slice, which the radix walk answers
    /// with zero matched blocks, so the all-or-nothing case needs no special
    /// handling. The insert side stays vision-gated, so no pad-bearing sequence
    /// ever enters the tree; this only reads back what pad-free turns put there.
    pub(super) fn prefix_lookup_tokens<'a>(&self, tokens: &'a [u32]) -> &'a [u32] {
        match self.first_vision_pad_index(tokens) {
            Some(cut) => &tokens[..cut],
            None => tokens,
        }
    }

    /// Whether `--high-speed-swap` has slid this sequence's rolling window, so
    /// `block_table` no longer parallels the token stream from position 0.
    ///
    /// Every prefix-cache insert assumes it does: the radix tree is keyed on the
    /// token stream from position 0 and files `block_table[i]` on the node for
    /// token chunk `i`. Once HSS slides (`hss_window_start() > 0`),
    /// `block_table[0]` no longer holds position 0 — the front of the table holds
    /// the most RECENT positions — so inserting files a block under a token chunk
    /// whose KV it does not contain, and a later warm hit reuses the wrong KV as
    /// if it were a valid prefix.
    ///
    /// There is no correct partial insert to fall back on: the tree indexes
    /// prefixes from the root and what survives in a slid window is a
    /// mid-sequence suffix, so a slid sequence has nothing cacheable. Skip.
    ///
    /// `cache_sequence` and `save_checkpoint`'s boundary insert already guarded
    /// on this; the prefill-time inserts in `prefill_d`/`finalize_last` did not.
    pub(super) fn hss_window_slid(&self, seq: &SequenceState) -> bool {
        seq.hss_window_start() > 0
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
            // SAFETY: `byte_len = n * 4` and `n = tokens.len()` (both bound at the
            // top of this fn), so the reinterpreted span is exactly
            // `tokens.len() * size_of::<u32>()` bytes — the whole of `tokens` and
            // not one byte more. `tokens: &[u32]` is a live shared borrow, so every
            // byte is initialised and no `&mut` to it can exist. u8 has alignment 1,
            // so the cast cannot under-align.
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
            // SAFETY: `result` was just built by `vec![0u32; n]`, so its length is
            // exactly `n` and every element is initialised; `byte_len = n * 4 =
            // result.len() * size_of::<u32>()`, so the span is exactly the Vec's
            // buffer. `result_bytes` is the only reference derived from `result`
            // while it is live (the next use of `result` is the `Ok(result)` move,
            // after `result_bytes` is dead), so the `&mut` is unaliased.
            let result_bytes =
                unsafe { std::slice::from_raw_parts_mut(result.as_mut_ptr() as *mut u8, byte_len) };
            self.gpu.copy_d2h(dev_buf, result_bytes)?;
            Ok(result)
        } else {
            Ok(tokens.to_vec())
        }
    }

    /// Share rank 0's vision embeddings — and the grids that place them — with
    /// every other rank, immediately before a prefill chunk.
    ///
    /// Only rank 0 receives the image bytes, so only rank 0 ran the ViT. Every
    /// other rank embedded the prompt itself and found `<|image_pad|>` where
    /// the picture should be: it kept the raw placeholder row, and with no
    /// grids it also built a LINEAR position stream where rank 0 built
    /// (T, H, W). Under TP the ranks all-reduce every layer, so half of every
    /// contribution at every image position came from a hidden state with no
    /// image in it, at the wrong positions. The model still answered
    /// fluently — it saw a blurred impression of the picture and described it
    /// with confidence, which is why this survived so long behind checks that
    /// only ever looked at rank 0.
    ///
    /// Wire order, always the same number of collectives on every rank so the
    /// stream cannot desynchronise: item count, then (when non-zero) the
    /// flattened `(t_len, gh, gw)` grids, the row count, the slice bases, and
    /// finally the packed BF16 rows broadcast straight out of one rank's
    /// `buf_out` into the others' — same buffer, same role, no staging.
    pub(super) fn ep_exchange_vision(&self, tokens: &[u32]) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        // Gate on THIS prompt's tokens, which every rank already holds
        // identically, so the ranks agree without a handshake. The gate is not
        // an optimisation detail: the head never clears its vision state
        // between requests, so an unconditional exchange would re-broadcast the
        // last image — or the last VIDEO — on every text-only prefill that
        // followed it.
        if !self.tokens_have_vision_pad(tokens) {
            return Ok(());
        }
        let comm = self.comm.as_ref().expect("ep_exchange_vision without comm");
        let is_head = comm.rank() == 0;

        let grids = if is_head {
            self.vision_image_grids.lock().clone()
        } else {
            Vec::new()
        };
        let n_items = self.ep_broadcast_u32(grids.len() as u32)? as usize;
        if n_items == 0 {
            if !is_head {
                self.vision_image_grids.lock().clear();
                *self.vision_embed_patches.lock() = 0;
            }
            return Ok(());
        }

        let flat: Vec<u32> = if is_head {
            grids
                .iter()
                .flat_map(|&(t, h, w)| [t as u32, h as u32, w as u32])
                .collect()
        } else {
            vec![0u32; n_items * 3]
        };
        let flat = self.ep_broadcast_tokens(&flat)?;

        let n_rows = self.ep_broadcast_u32(if is_head {
            *self.vision_embed_patches.lock() as u32
        } else {
            0
        })? as usize;
        // Co-dispatch bases travel too: the splice and the MRoPE walk both index
        // the shared packed buffer through them, and a worker that defaulted to
        // zero would read another request's rows.
        let row_base = self.ep_broadcast_u32(if is_head {
            *self.vision_row_base.lock() as u32
        } else {
            0
        })? as usize;
        let grid_base = self.ep_broadcast_u32(if is_head {
            *self.vision_grid_base.lock() as u32
        } else {
            0
        })? as usize;
        let owned = self.ep_broadcast_u32(if is_head {
            *self.vision_owned_images.lock() as u32
        } else {
            0
        })? as usize;

        let ve = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("ep_exchange_vision: rank has no vision encoder"))?;
        // The worker never saw an image, so its ViT scratch is still unallocated
        // (it is deferred to the first image precisely to keep text-only serving
        // cheap). Allocate before using `buf_out` as the broadcast destination.
        ve.scratch_init(self.gpu.as_ref())?;
        let byte_len = n_rows
            .checked_mul(ve.out_hidden_size)
            .and_then(|e| e.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("ep_exchange_vision: row payload overflows"))?;
        // Bound against the ROW budget: this writes into buf_out, whose size is
        // `out_rows`, not `p_max`. 🪤 Head and worker must derive out_rows from
        // the SAME config or the broadcast length disagrees across ranks — the
        // rank-1-blind failure class.
        anyhow::ensure!(
            n_rows <= ve.out_rows,
            "ep_exchange_vision: {n_rows} rows exceed the encoder row budget {} — \
             the broadcast would write past buf_out",
            ve.out_rows
        );
        comm.broadcast(ve.scratch().buf_out.0, byte_len, 0)?;

        if !is_head {
            *self.vision_image_grids.lock() = flat
                .chunks_exact(3)
                .map(|c| (c[0] as usize, c[1] as usize, c[2] as usize))
                .collect();
            *self.vision_embed_patches.lock() = n_rows;
            *self.vision_row_base.lock() = row_base;
            *self.vision_grid_base.lock() = grid_base;
            *self.vision_owned_images.lock() = owned;
            self.gpu.synchronize(self.gpu.default_stream())?;
        }
        Ok(())
    }

    /// F83 (2026-04-30): all-reduce-min on a single u32 across all
    /// EP ranks. Used by the prefix-cache cache-hit handshake so head
    /// and worker agree on the same `matched_tokens` count even when
    /// their independent local prefix caches disagree. Implemented via
    /// `world_size` rooted broadcasts (one rooted at each rank), each
    /// rank min-reducing the values it observes. NCCL has a native
    /// allreduce-MIN but Avarok's spark-comm trait only exposes SUM
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

    /// Both bounds of `val` across the ranks, in the SAME rooted-broadcast pass
    /// [`Self::ep_min_u32`] already pays for — so agreement costs no extra
    /// collective over the min it replaces.
    ///
    /// `min == max` is the only proof that every rank proposed the same value.
    /// A min alone is not enough when the value SELECTS A RESOURCE: the rank
    /// holding the minimum would use it while a rank that proposed something
    /// larger has no such resource to fall back to, and the two then diverge
    /// anyway. See [`Self::ep_all_agree_u32`].
    pub(super) fn ep_minmax_u32(&self, val: u32) -> Result<(u32, u32)> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok((val, val));
        };
        let stream = self.gpu.default_stream();
        // Same `max()` reasoning as ep_min_u32: under pure TP `ep_world_size`
        // is 1 while the comm spans `tp_world_size` ranks.
        let world = self.config.ep_world_size.max(self.config.tp_world_size);
        let (mut min_val, mut max_val) = (val, val);
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
            max_val = max_val.max(v);
        }
        Ok((min_val, max_val))
    }

    /// Do ALL ranks propose `val`?
    ///
    /// 🪤 A COLLECTIVE. Every rank must call it the same number of times, at
    /// the same point — call it unconditionally on the multi-rank path, never
    /// behind a rank-local `if`, exactly as F83 learned for
    /// [`Self::ep_min_u32`] (gating that on `matched_tokens > 0` deadlocked
    /// when one rank matched and the other did not).
    ///
    /// Returns `true` on a single-rank world, so callers need no guard.
    pub(crate) fn ep_all_agree_u32(&self, val: u32) -> Result<bool> {
        if !self.multi_rank_protocol_active() {
            return Ok(true);
        }
        let (mn, mx) = self.ep_minmax_u32(val)?;
        Ok(mn == mx)
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
            "ep_broadcast_decode_batch_dispatch called without AVAROK_EP_PROTOCOL=v2"
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

    /// Announce a batched multi-sequence verify to the workers.
    ///
    /// Mirrors [`Self::ep_broadcast_decode_batch_dispatch`] and carries one
    /// extra list: `ks`, because a verify's rows-per-sequence is not 1 and can
    /// differ between sequences on the MTP ladder.
    ///
    /// 🪤 `seq_ids` must be in the HEAD'S dispatch order, not slot order. The
    /// batched forward issues ONE all-reduce over the contiguous row span, so a
    /// worker that rebuilt its refs in a different order would sum partials of
    /// different sequences — silently, and only under TP>1.
    pub(super) fn ep_broadcast_verify_batch_dispatch(
        &self,
        seq_ids: &[u32],
        ks: &[u32],
        tokens: &[u32],
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        debug_assert!(
            self.ep_protocol_v2,
            "ep_broadcast_verify_batch_dispatch called without AVAROK_EP_PROTOCOL=v2"
        );
        debug_assert_eq!(seq_ids.len(), ks.len(), "seq_ids and ks length mismatch");
        debug_assert_eq!(
            tokens.len(),
            ks.iter().map(|&k| k as usize).sum::<usize>(),
            "tokens must be the seq-major concatenation of every sequence's rows"
        );
        self.ep_broadcast_seq_and_cmd(0, crate::speculative::EP_CMD_VERIFY_BATCH, true)?;
        self.ep_broadcast_u32(seq_ids.len() as u32)?;
        self.ep_broadcast_tokens(seq_ids)?;
        self.ep_broadcast_tokens(ks)?;
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
    /// Protocol (`AVAROK_EP_PROTOCOL=v2`): rank 0 broadcasts the slot
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
    /// - 0xFFFFFFF7: verify K=N → k, then k tokens, then num_accepted
    /// - 0xFFFFFFFF: shutdown (seq_id is ignored; applies to the whole worker)
    pub(super) fn ep_worker_step_impl(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        // 🔴 The RECEIVE is the only fatal half. If it fails the link to the head is gone
        // and the worker must exit; everything after it is a per-request fault that the head
        // raises identically and answers the client with, so it is tagged `EpCommandFailed`
        // and the worker survives it. Breaking on both is what silently killed rank 1 and
        // left rank 0 spinning in a collective against a dead peer — ANOMALIES A60/A62.
        let (seq_id, cmd) = self.ep_recv_seq_and_cmd(self.ep_protocol_v2)?;
        self.ep_worker_execute(seq_id, cmd, slots)
            .map_err(|e| anyhow::Error::new(crate::traits::EpCommandFailed(e)))
    }

    /// Execute one already-received worker command. Every error out of here is
    /// request-scoped by construction — see the caller.
    fn ep_worker_execute(
        &self,
        seq_id: u32,
        cmd: u32,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
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

        // Batched VERIFY (`0xFFFF_FFE1`): same list shape as batched decode —
        // sentinel preamble seq_id, real routing in the seq_ids[N] payload.
        if cmd == crate::speculative::EP_CMD_VERIFY_BATCH {
            return self.ep_worker_verify_batch(slots);
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
                // Per-request control-vector selection, sent because it cannot
                // be re-derived from the tokens the way seq_len, the block
                // table and the SSM state are. A worker that stayed at 0 here
                // would steer nothing while the head steers, and the two ranks
                // would diverge on a highway that is supposed to be replicated.
                let cvec_lo = self.ep_broadcast_u32(0)? as u64;
                let cvec_hi = self.ep_broadcast_u32(0)? as u64;
                let cvec_id = (cvec_hi << 32) | cvec_lo;
                // A selection this rank cannot resolve means the ranks booted
                // with different --control-vector files. Fail loudly: the
                // alternative is one-sided steering, which produces plausible
                // text and no counter that would ever show it.
                anyhow::ensure!(
                    cvec_id == 0 || self.control_vectors.by_id(cvec_id).is_some(),
                    "EP worker: head selected control vector {cvec_id:#018x}, which \
                     this rank has not registered (has: {:?}). The ranks were \
                     started with different --control-vector flags.",
                    self.control_vectors.names().collect::<Vec<_>>()
                );
                seq.cvec_id = cvec_id;
                let full_tokens = self.ep_broadcast_tokens(&vec![0u32; full_len])?;
                // Receive rank 0's ViT output before embedding: the splice and
                // the MRoPE walk both read this state, and without it this rank
                // embeds the raw placeholder token at every image position.
                self.ep_exchange_vision(&full_tokens)?;
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
                // Land the PLE/QSA carries at the committed row count. The head
                // does this inside `commit_accepted_prefix`; the worker rewinds
                // its own SSM state and token buffers and never called it, so
                // QSA's `ingested` mark stayed `k - committed` rows ahead of
                // `seq_len` and the NEXT decode died on
                //   QSA: decode at pos N but N+2 tokens ingested
                // (2-node EP=2 agentic run, 2026-09-07 — the gap was exactly
                // the draft width). AFTER the rewind, not before:
                // `commit_verify_aux_rows` asserts `seq_len == base +
                // num_accepted`, so calling it first fails with
                //   batched mHC commit: 1/3 rows from 3513, but seq_len=3516
                // No-op unless the K-row batched mHC verify recorded a span.
                self.commit_verify_aux_rows(seq, accepted as usize + 1, stream)?;
            }
            crate::speculative::EP_CMD_MTP_PROPOSE => {
                // Run the SAME drafter forward rank 0 is running, so its collectives have a
                // partner. The drafts themselves are discarded — rank 0 broadcasts the tokens
                // it actually verifies — but the drafter KV this writes must stay in lockstep,
                // which it does because both ranks consume identical `(last_token, position)`
                // and identical target hiddens (the target forward is already collective-correct).
                let last_token = self.ep_broadcast_u32(0)?;
                let position = self.ep_broadcast_u32(0)? as usize;
                let num_drafts = self.ep_broadcast_u32(0)? as usize;
                let hidden_idx = self.ep_broadcast_u32(0)? as usize;
                // Mirror the head's `save_hidden_for_mtp`: the drafter's input vector must be
                // the SAME on both ranks or the all-reduce sums partials of different inputs.
                // No worker command arm writes `mtp_hidden_save`, so it has to happen here.
                if let Err(e) = self.save_hidden_for_mtp(hidden_idx, stream) {
                    tracing::warn!("EP worker save_hidden_for_mtp({hidden_idx}) failed: {e:#}");
                }
                if let Err(e) =
                    self.run_mtp_propose_inner(last_token, position, num_drafts, seq, None)
                {
                    // Never fail the worker on a drafter error: rank 0 decides what is
                    // verified, so a degraded worker draft costs acceptance, not correctness.
                    // Bailing here would desynchronise the command stream instead.
                    tracing::warn!("EP worker MTP propose failed (continuing): {e:#}");
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
                // Aux carries AFTER the rewind — see the K=2 arm.
                self.commit_verify_aux_rows(seq, num_accepted as usize + 1, stream)?;
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
                // Aux carries AFTER the rewind — see the K=2 arm.
                self.commit_verify_aux_rows(seq, num_accepted as usize + 1, stream)?;
            }
            crate::speculative::EP_CMD_VERIFY_KN => {
                // Width-generic verify. `k` arrives FIRST so this loop and the
                // head's send loop are driven by the same word — neither rank
                // can guess the count wrong, which is the only way a new EP
                // path deadlocks. Everything after is the K=4 arm with the
                // width lifted out.
                let k = self.ep_broadcast_u32(0)? as usize;
                anyhow::ensure!(
                    (2..=64).contains(&k),
                    "EP verify K=N: implausible width {k} off the wire"
                );
                let mut toks = Vec::with_capacity(k);
                for _ in 0..k {
                    toks.push(self.ep_broadcast_u32(0)?);
                }
                self.sync_secondary()?;
                self.decode_verify_graphed_kn(&toks, seq, stream)?;
                let num_accepted = self.ep_broadcast_u32(0)? as usize;
                self.trim_proposer_state(seq, num_accepted, 0)?;
                // K=4 rewinds by (K-1) - num_accepted and checkpoints at full
                // accept; that is this, with K no longer a literal. The pool
                // must be sized for k-1 intermediates
                // (`ssm_reserve::mtp_pool_draft_width`), and both ranks size
                // from the same inputs, so a width the head can send is a
                // width this rank can verify.
                let rewind = (k - 1).saturating_sub(num_accepted);
                if rewind == 0 {
                    self.start_checkpoint_async(seq)?;
                } else {
                    seq.seq_len -= rewind;
                    for _ in 0..rewind {
                        seq.tokens.pop();
                    }
                    self.start_rollback_and_checkpoint_async(seq, num_accepted + 1)?;
                }
                // Aux carries AFTER the rewind — see the K=2 arm.
                self.commit_verify_aux_rows(seq, num_accepted + 1, stream)?;
            }
            EP_CMD_CACHE_SEQ => {
                // Sequence retirement. Run the SAME bookkeeping the head runs
                // so both ranks' snapshot pools hold the same entries — that
                // symmetry is what keeps the Marconi anchor decision equal on
                // both ranks. No collective inside, so no ordering guard.
                self.cache_sequence_dispatch(seq);
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

    /// Worker side of [`crate::speculative::EP_CMD_VERIFY_BATCH`].
    ///
    /// Runs the SAME batched forward the head runs, then applies the same
    /// per-sequence rewind the head applies. Structured exactly like
    /// [`Self::ep_worker_decode_batch`] — receive the lists, order the refs by
    /// the head's `seq_ids`, call the shared compute — with the verify's extra
    /// step of reading one verdict word per sequence AFTER the forward.
    ///
    /// 🔴 This arm is what lets the `comm.is_none()` conjunct come off
    /// `can_batch_verify_dispatch`. Without it the head would issue a batched
    /// forward whose all-reduces no worker is answering, and both ranks spin at
    /// ~96% util on an NCCL wait — the documented multi-rank speculation hang.
    fn ep_worker_verify_batch(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        let seq_ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let ks_u32 = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let ks: Vec<usize> = ks_u32.iter().map(|&k| k as usize).collect();
        let r_total: usize = ks.iter().sum();
        let tokens = self.ep_broadcast_tokens(&vec![0u32; r_total])?;

        // Validate before touching slot state, and bail rather than serve a
        // partial batch: a mismatch means head and worker disagree about the
        // batch, and verifying the wrong sequence's rows corrupts its KV and
        // its recurrent state at once.
        let mut seen = std::collections::HashSet::new();
        for (i, &id) in seq_ids.iter().enumerate() {
            let idx = id as usize;
            if idx >= slots.len() {
                anyhow::bail!(
                    "ep_worker_verify_batch: seq_id {id} exceeds slot capacity {}",
                    slots.len(),
                );
            }
            if !seen.insert(id) {
                anyhow::bail!("ep_worker_verify_batch: duplicate seq_id {id} in batch");
            }
            if ks[i] == 0 {
                anyhow::bail!("ep_worker_verify_batch: seq_id {id} has k = 0");
            }
        }

        let mut slot_refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, opt)| opt.as_mut().map(|s| (i, s)))
            .collect();

        // The head's order, not slot order — see the broadcast helper.
        let mut refs: Vec<&mut SequenceState> = Vec::with_capacity(n);
        for &id in &seq_ids {
            let idx = id as usize;
            let pos = slot_refs
                .iter()
                .position(|(i, _)| *i == idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("ep_worker_verify_batch: slot {idx} not allocated")
                })?;
            let (_, seq) = slot_refs.swap_remove(pos);
            refs.push(seq);
        }

        let stream = self.gpu.default_stream();
        self.sync_secondary()?;
        self.decode_verify_batched_dispatch(&tokens, &ks, &mut refs, stream)?;

        // Verdicts arrive after the head's accept walk, one word per sequence,
        // in the same order. Read them ALL before acting: an early bail with
        // words still on the wire desynchronises every later command.
        let mut accepted: Vec<usize> = Vec::with_capacity(n);
        for _ in 0..n {
            accepted.push(self.ep_broadcast_u32(0)? as usize);
        }

        // 🔴 Mirror `k4_apply_verdict` — the BATCHED head step — not the
        // single-sequence K=3/K=4 arms above. Both restore `intermediate[na]`,
        // but they are different entry points with different width arguments,
        // and the head this worker is paired with is the batched one. Mirroring
        // the wrong head leaves the two ranks' recurrent state disagreeing,
        // which does not fault: it degrades a later sequence's logits.
        //
        // The ORDER is part of the contract too — pops before `trim`, then the
        // commit — because `trim_proposer_state` reads the sequence length it
        // is trimming against.
        for (i, seq) in refs.iter_mut().enumerate() {
            let k_rows = ks[i];
            let nd = k_rows.saturating_sub(1);
            let na = accepted[i].min(nd);
            if na == nd {
                // Full accept: the verify kernel already wrote the canonical
                // h_state, so this commit is the no-op the head takes.
                self.commit_accepted_prefix(seq, k_rows, k_rows)?;
                // ...and the head ALSO trims the proposer here
                // (`k4_apply_verdict`, "Full-accept branch trims AFTER the
                // hidden save"). This arm did not, so on every fully-accepted
                // step the head advanced its drafter's row accounting and the
                // worker left stale rows behind. Nothing faults: the next
                // drafter forward is a collective, so the two ranks then
                // all-reduce partials computed over different drafter state,
                // and the pair's drafts stop matching what either rank would
                // have produced alone.
                //
                // Only the BATCHED arm was missing it — both per-sequence arms
                // above trim on both branches — which is why the symptom was
                // "output depends on batch width". At C>=2 with the batched
                // verify on, a ~1K-token prompt answered at temperature 0 gave
                // different text from the same prompt run alone, with the
                // concurrent replies disagreeing among themselves; with
                // AVAROK_MTP_EP_BATCH_VERIFY=0 it was byte-identical at every
                // width.
                self.trim_proposer_state(seq, na, 0)?;
            } else {
                seq.seq_len -= nd - na;
                for _ in 0..(nd - na) {
                    seq.tokens.pop();
                }
                self.trim_proposer_state(seq, na, 0)?;
                self.commit_accepted_prefix(seq, na + 1, k_rows)?;
            }
        }
        Ok(true)
    }
}
