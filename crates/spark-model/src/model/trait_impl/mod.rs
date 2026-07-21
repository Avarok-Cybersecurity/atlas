// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Model for TransformerModel` — thin trait impl that delegates to
//! `<method>_dispatch` helpers split across sibling files for the ≤500
//! LoC cap. Each sibling adds methods to the `TransformerModel`
//! inherent impl. The trait impl below is purely one-line delegators.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{AttnMetadataDev, LayerState};
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, PrefillSlice, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights};

mod async_chkpt;
mod decode_a;
mod decode_a2;
mod decode_b;
mod decode_b2;
mod decode_checkpoint;
mod ep_misc;
mod meta;
mod prefill_a;
mod prefill_b;
mod prefill_c;
mod prefill_d;
mod sequence;
mod speculative;
mod ssm_fault_in;
mod verify_a;
mod verify_b;
mod verify_c;
mod verify_c2;
mod verify_d;
mod verify_fused;

impl Model for TransformerModel {
    fn prepare_vision_embed(&self, images: &[(Vec<f32>, usize, usize)]) -> Result<()> {
        self.prepare_vision_embed_dispatch(images)
    }
    fn prepare_vision_embed_batched(
        &self,
        per_request: &[Vec<(Vec<f32>, usize, usize)>],
    ) -> Result<Vec<(usize, usize, usize, usize)>> {
        self.prepare_vision_embed_batched_dispatch(per_request)
    }
    fn set_vision_slice_base(&self, row_base: usize, grid_base: usize, owned_images: usize) {
        *self.vision_row_base.lock() = row_base;
        *self.vision_grid_base.lock() = grid_base;
        *self.vision_owned_images.lock() = owned_images;
    }
    fn prefill(&self, tokens: &[u32], seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.prefill_dispatch(tokens, seq, stream)
    }
    fn prefill_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch(tokens, seq, chunk_start, chunk_len, is_last_chunk, stream)
    }
    fn prefill_twophase(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_size: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_twophase_dispatch(tokens, seq, chunk_size, stream)
    }
    fn decode(&self, token: u32, seq: &mut SequenceState, _stream: u64) -> Result<DevicePtr> {
        self.decode_dispatch(token, seq, _stream)
    }
    fn decode_batch(
        &self,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.decode_batch_dispatch(tokens, seqs, stream)
    }
    fn mixed_forward(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        prefill_tokens: &[u32],
        prefill_seq: &mut SequenceState,
        prefill_chunk_start: usize,
        prefill_chunk_len: usize,
        prefill_is_last: bool,
        stream: u64,
    ) -> Result<crate::traits::MixedForwardResult> {
        self.mixed_forward_dispatch(
            decode_tokens,
            decode_seqs,
            prefill_tokens,
            prefill_seq,
            prefill_chunk_start,
            prefill_chunk_len,
            prefill_is_last,
            stream,
        )
    }

    /// Q12 Phase 4b override: try the model-level batched dispatch
    /// (`prefill_batch_chunk_dispatch`) first; on the not-yet-implemented
    /// stub failure, fall back to the trait's default per-stream loop.
    /// This keeps callers correct while the per-layer-batched body is
    /// staged in subsequent commits.
    fn prefill_batch_chunk(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
    ) -> Result<Vec<DevicePtr>> {
        // Try the concrete dispatch. The Phase 4b stub returns Err for the
        // "not-yet-implemented" path so we transparently downgrade to the
        // single-stream-loop default impl. Once Phase 2b/3 land, the
        // dispatch returns Ok with logits and this fallback becomes dead
        // code that we can drop.
        match self.prefill_batch_chunk_dispatch(streams, stream) {
            Ok(v) => Ok(v),
            Err(e) => {
                // Log at debug — under expected for this stub. Promotes to
                // info if a real error is encountered (future Phase 4b body).
                tracing::debug!(
                    "prefill_batch_chunk_dispatch unavailable, falling back to \
                     per-stream loop: {e}"
                );
                let mut out = Vec::with_capacity(streams.len());
                for slice in streams.iter_mut() {
                    let logits = self.prefill_chunk(
                        slice.prompt_tokens,
                        slice.seq,
                        slice.chunk_start,
                        slice.chunk_len,
                        slice.is_last_chunk,
                        stream,
                    )?;
                    out.push(logits);
                }
                Ok(out)
            }
        }
    }
    fn vocab_size(&self) -> usize {
        self.vocab_size_dispatch()
    }
    fn set_active_lora(&mut self, name: &str) -> Result<()> {
        self.rotate_lora_to(name)
    }
    fn adapter_id_for(&self, slot: i32) -> u64 {
        self.adapter_id_for_slot(slot)
    }
    fn acquire_adapter_slot(&self, slot: i32) -> i32 {
        TransformerModel::acquire_adapter_slot(self, slot)
    }
    fn release_adapter_slot(&self, resolved: i32) {
        TransformerModel::release_adapter_slot(self, resolved)
    }
    fn swap_lora_from_disk(
        &mut self,
        dir: &std::path::Path,
        name: &str,
        slot: usize,
    ) -> Result<()> {
        // Disk staging is plain file I/O and is portable; only the PEER path
        // needs RDMA. Still cuda-gated, since it lands into a device pool.
        #[cfg(feature = "cuda")]
        {
            self.swap_lora_slot_from_disk(dir, name, slot)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dir, name, slot);
            anyhow::bail!("LoRA disk swap requires the cuda feature")
        }
    }
    fn promote_lora_from_peer(
        &mut self,
        peer_addr: &str,
        adapter_id: &str,
        name: &str,
        peft: atlas_core::config::PeftAdapterConfig,
    ) -> Result<(usize, Option<String>)> {
        #[cfg(all(feature = "cuda", unix))]
        {
            self.promote_lora_slot_from_peer(peer_addr, adapter_id, name, peft)
        }
        #[cfg(not(all(feature = "cuda", unix)))]
        {
            let _ = (peer_addr, adapter_id, name, peft);
            anyhow::bail!("LoRA peer promotion stages over RDMA (rdma-core); unix-only")
        }
    }
    fn promote_lora_from_disk(
        &mut self,
        dir: &std::path::Path,
        name: &str,
    ) -> Result<(usize, Option<String>)> {
        #[cfg(feature = "cuda")]
        {
            self.promote_lora_slot_from_disk(dir, name)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (dir, name);
            anyhow::bail!("LoRA disk promotion requires the cuda feature")
        }
    }
    fn high_speed_swap_dims(&self) -> Option<spark_storage::ModelDims> {
        self.high_speed_swap_dims_dispatch()
    }
    fn normalize_ssm_states(&self, seq: &SequenceState, stream: u64) -> Result<()> {
        self.normalize_ssm_states_dispatch(seq, stream)
    }
    fn bind_gpu_to_thread(&self) -> Result<()> {
        self.bind_gpu_to_thread_dispatch()
    }
    fn alloc_sequence(&self) -> Result<SequenceState> {
        self.alloc_sequence_dispatch()
    }
    fn copy_logits_to_host(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.copy_logits_to_host_dispatch(logits_ptr, dst)
    }
    fn logits_ptr_is_fp32(&self, logits_ptr: DevicePtr) -> bool {
        self.logits_ptr_is_fp32_dispatch(logits_ptr)
    }
    fn logits_buffer_ptr(&self) -> DevicePtr {
        self.logits_buffer_ptr_dispatch()
    }
    fn argmax_on_device(&self, logits_ptr: DevicePtr, _stream: u64) -> Result<u32> {
        self.argmax_on_device_dispatch(logits_ptr, _stream)
    }
    fn argmax_batch(&self, logits_ptr: DevicePtr, n: usize, _stream: u64) -> Result<Vec<u32>> {
        self.argmax_batch_dispatch(logits_ptr, n, _stream)
    }
    fn hidden_after_norm(&self) -> DevicePtr {
        self.hidden_after_norm_dispatch()
    }
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_dispatch(tokens, seq, stream)
    }
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()> {
        self.checkpoint_ssm_states_dispatch(seq)
    }
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()> {
        self.rollback_ssm_states_dispatch(seq, num_accepted)
    }
    fn has_ssm_layers(&self) -> bool {
        self.ssm_pool.num_ssm_layers > 0
    }
    fn decode_rollback_ring_slots(&self) -> usize {
        if self.ssm_snapshots.decode_rollback_enabled() {
            self.ssm_snapshots.decode_ring_slots
        } else {
            0
        }
    }
    fn save_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.save_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }
    fn restore_decode_ssm_snapshot(&self, seq: &SequenceState, ring_slot: usize) -> Result<()> {
        self.restore_decode_ssm_snapshot_dispatch(seq, ring_slot)
    }
    fn generate_speculative(
        &self,
        prompt_tokens: &[u32],
        params: &spark_runtime::sampler::SamplingParams,
        num_drafts: usize,
    ) -> Result<crate::engine::GenerateResult> {
        self.generate_speculative_dispatch(prompt_tokens, params, num_drafts)
    }
    fn has_proposer(&self) -> bool {
        self.has_proposer_dispatch()
    }
    fn has_self_speculative(&self) -> bool {
        self.has_self_speculative_dispatch()
    }
    fn decode_draft(&self, token: u32, seq: &mut SequenceState, stream: u64) -> Result<DevicePtr> {
        self.decode_draft_dispatch(token, seq, stream)
    }
    fn cache_sequence(&self, seq: &SequenceState) {
        self.cache_sequence_dispatch(seq)
    }
    fn decode_marconi_checkpoint(&self, seq: &mut SequenceState) {
        self.decode_marconi_checkpoint_dispatch(seq)
    }
    fn free_sequence(&self, seq: &mut SequenceState) -> Result<()> {
        self.free_sequence_dispatch(seq)
    }
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 2]> {
        self.decode_verify_graphed_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 3]> {
        self.decode_verify_graphed_k3_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<[u32; 4]> {
        self.decode_verify_graphed_k4_dispatch(tokens, seq, _stream)
    }
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_graphed_kgamma_dispatch(tokens, seq, _stream)
    }
    fn decode_and_verify_fused(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_and_verify_fused_dispatch(tokens, seq, _stream)
    }
    fn save_hidden_for_catchup(&self, token_idx: usize, pos: usize) -> Result<()> {
        self.save_hidden_for_catchup_dispatch(token_idx, pos)
    }

    fn save_hidden_for_mtp(&self, token_idx: usize, _stream: u64) -> Result<()> {
        self.save_hidden_for_mtp_dispatch(token_idx, _stream)
    }
    fn save_dflash_hidden_for_propose(&self, token_idx: usize, _stream: u64) -> Result<()> {
        self.save_dflash_hidden_dispatch(token_idx, _stream)
    }

    fn dflash_accept_append(&self, seq: &mut SequenceState) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = prop
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not DFlash proposer state"))?;
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let save_1 = base.offset(ctx_slot_bytes);
        let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(save_1, dst, ctx_slot_bytes, self.gpu.default_stream())?;
        d.ctx_positions.push((seq.seq_len as i32).saturating_sub(1));
        d.ctx_len += 1;
        Ok(())
    }

    fn dflash_eagle_accept_append(&self, seq: &mut SequenceState) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = prop
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not DFlash proposer state"))?;
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();
        let pos_row0 = (seq.seq_len as i32).saturating_sub(2);
        let pos_row1 = (seq.seq_len as i32).saturating_sub(1);
        // Row 0 @ N
        let save_0 = base;
        let dst_0 = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(save_0, dst_0, ctx_slot_bytes, stream)?;
        d.ctx_positions.push(pos_row0);
        d.ctx_len += 1;
        // Row 1 @ N+1
        let save_1 = base.offset(ctx_slot_bytes);
        let dst_1 = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu
            .copy_d2d_async(save_1, dst_1, ctx_slot_bytes, stream)?;
        d.ctx_positions.push(pos_row1);
        d.ctx_len += 1;
        d.skip_next_decode_append = true;
        Ok(())
    }

    fn dflash_eagle_kgamma_append(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        base_pos: usize,
    ) -> Result<()> {
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        let d = prop
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
            .ok_or_else(|| anyhow::anyhow!("not DFlash proposer state"))?;
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();
        for t in 0..=num_accepted {
            let row = base.offset(t * ctx_slot_bytes);
            let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
            self.gpu.copy_d2d_async(row, dst, ctx_slot_bytes, stream)?;
            let pos = (base_pos + t) as i32;
            d.ctx_positions.push(pos);
            d.ctx_len += 1;
        }
        d.skip_next_decode_append = true;
        Ok(())
    }

    fn commit_ctx(
        &self,
        seq: &mut SequenceState,
        num_committed: usize,
        base_pos: usize,
    ) -> Result<()> {
        if num_committed == 0 {
            return Ok(());
        }
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        // Graceful no-op for non-DFlash proposers (shared bootstrap path).
        let d = match prop
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        {
            Some(d) => d,
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();

        // Watermark slide FIRST, on the ctx_len (row-index) axis. If the
        // incoming rows would exceed capacity, keep the NEWEST rows and drop
        // the oldest (mirrors dflash_serial_ctx_append). keep is clamped so
        // drop_n >= keep — the single D2D copy's src/dst can never overlap.
        // ctx_committed resets to 0 (next propose re-precomputes the slid
        // rows chunk-wise); ctx_positions values (absolute RoPE positions)
        // are preserved by the drain, so stamps stay exact across the slide.
        if d.ctx_len + num_committed > d.max_ctx_len {
            let keep = (d.max_ctx_len / 2).min(d.max_ctx_len.saturating_sub(num_committed));
            let drop_n = d.ctx_len.saturating_sub(keep);
            if drop_n > 0 {
                let src = d.ctx_hidden_acc.offset(drop_n * ctx_slot_bytes);
                let dst0 = d.ctx_hidden_acc.offset(0);
                self.gpu
                    .copy_d2d_async(src, dst0, keep * ctx_slot_bytes, stream)?;
                d.ctx_positions.drain(..drop_n);
                d.ctx_len = keep;
                d.ctx_committed = 0;
                tracing::info!(
                    "DFlash UNIFIED_CTX watermark: slid ctx window (dropped {} oldest, keep {})",
                    drop_n,
                    keep,
                );
            }
        }

        // Append num_committed rows at the TAIL (ctx_len axis). dst uses
        // ctx_len (acc row index); base_pos stamps ctx_positions (RoPE axis).
        // Conflating the two axes is the DDD §4.1 landmine: they coincide
        // only until the first slide — and the sliding prompts ARE the reds.
        debug_assert_eq!(d.ctx_positions.len(), d.ctx_len);
        for t in 0..num_committed {
            let row = base.offset(t * ctx_slot_bytes);
            let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
            self.gpu.copy_d2d_async(row, dst, ctx_slot_bytes, stream)?;
            d.ctx_positions.push((base_pos + t) as i32);
            d.ctx_len += 1;
        }
        // Freshest ctx slot = row (num_committed-1) = the bonus generator
        // (EAGLE order, matches kgamma_append). Block the next propose()'s
        // internal decode-append so this capture is never double-appended.
        d.skip_next_decode_append = true;

        // One-shot activation log so A/B runs can confirm the path is live.
        static UNIFIED_CTX_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !UNIFIED_CTX_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "DFlash UNIFIED_CTX ACTIVE: first commit_ctx rows={} base_pos={} ctx_len={}",
                num_committed,
                base_pos,
                d.ctx_len,
            );
        }
        Ok(())
    }

    fn dflash_serial_ctx_append(&self, seq: &mut SequenceState) -> Result<()> {
        // Ctx-holes fix: append the serial-decoded token's captured hidden.
        // The decode layer loop (decode_a.rs try_dflash_capture) already
        // filled `dflash_hidden_save` row 0 with this token's per-layer
        // hiddens — the same [slot0|..|slot4] layout as one accumulator row.
        let base = match self.dflash_hidden_save {
            Some(p) => p,
            None => return Ok(()),
        };
        let prop = match seq.proposer_state.as_mut() {
            Some(p) => p.as_mut(),
            None => return Ok(()),
        };
        // Graceful no-op for non-DFlash proposers (this bootstrap path is
        // shared with EAGLE/MTP, unlike the DFlash-only eagle append above).
        let d = match prop
            .as_any_mut()
            .downcast_mut::<crate::layers::DflashProposerState>()
        {
            Some(d) => d,
            None => return Ok(()),
        };
        let n_layers = self.dflash_capture_layers.len();
        if n_layers == 0 {
            return Ok(());
        }
        let ctx_slot_bytes = n_layers * self.config.hidden_size * 2;
        let stream = self.gpu.default_stream();
        // Bounded watermark: accumulator full → slide the window. Keep the
        // NEWEST keep = max/2 rows, drop the oldest (dropping the newest
        // would starve the drafter of exactly the tokens that drive
        // acceptance — the 846-token think overrun). drop_n >= keep holds
        // whenever ctx_len >= max_ctx_len, so src/dst regions of the single
        // D2D copy can never overlap — no ring arithmetic, no status-1.
        // ctx_committed resets to 0: the next propose re-precomputes the
        // slid rows chunk-wise (ctx_window rows/pass) and rewrites their
        // paged K/V at the new slot indices; ctx_positions values (absolute
        // positions) are preserved by the drain, so RoPE stamps stay exact.
        if d.ctx_len >= d.max_ctx_len {
            let keep = d.max_ctx_len / 2;
            let drop_n = d.ctx_len - keep;
            let src = d.ctx_hidden_acc.offset(drop_n * ctx_slot_bytes);
            let dst0 = d.ctx_hidden_acc.offset(0);
            self.gpu
                .copy_d2d_async(src, dst0, keep * ctx_slot_bytes, stream)?;
            d.ctx_positions.drain(..drop_n);
            d.ctx_len = keep;
            d.ctx_committed = 0;
            tracing::info!(
                "DFlash SERIAL_APPEND watermark: slid ctx window (dropped {} oldest, keep {})",
                drop_n,
                keep,
            );
        }
        let dst = d.ctx_hidden_acc.offset(d.ctx_len * ctx_slot_bytes);
        self.gpu.copy_d2d_async(base, dst, ctx_slot_bytes, stream)?;
        // One-shot activation log so A/B runs can confirm the fix is live.
        static SERIAL_APPEND_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !SERIAL_APPEND_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "DFlash SERIAL_APPEND ACTIVE: first serial ctx append at ctx_len={} pos={}",
                d.ctx_len,
                seq.seq_len.saturating_sub(1),
            );
        }
        // Position convention: decode() advanced seq_len past the token we
        // just processed, so its true absolute position is seq_len - 1 —
        // identical to propose.rs's `position.saturating_sub(1)` stamp.
        debug_assert_eq!(d.ctx_positions.len(), d.ctx_len);
        d.ctx_positions.push(seq.seq_len.saturating_sub(1) as i32);
        d.ctx_len += 1;
        // The latest capture is now in ctx; a propose() firing later (e.g.
        // adaptive re-probe) must not decode-append it again.
        d.skip_next_decode_append = true;
        Ok(())
    }
    fn run_mtp_propose(
        &self,
        token: u32,
        position: usize,
        seq: &mut SequenceState,
        _stream: u64,
    ) -> Result<Option<u32>> {
        self.run_mtp_propose_dispatch(token, position, seq, _stream)
    }
    fn run_mtp_propose_multi(
        &self,
        token: u32,
        position: usize,
        num_drafts: usize,
        seq: &mut SequenceState,
        _stream: u64,
        grammar_bitmask: Option<&[i32]>,
    ) -> Result<Vec<u32>> {
        self.run_mtp_propose_multi_dispatch(
            token,
            position,
            num_drafts,
            seq,
            _stream,
            grammar_bitmask,
        )
    }
    fn read_deferred_draft_token(&self) -> Result<u32> {
        self.read_deferred_draft_token_dispatch()
    }
    fn trim_proposer_state(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        _stream: u64,
    ) -> Result<()> {
        self.trim_proposer_state_dispatch(seq, num_accepted, _stream)
    }
    fn compact_sequence(&self, seq: &mut SequenceState, new_slot: usize) -> Result<()> {
        self.compact_sequence_dispatch(seq, new_slot)
    }
    fn detach_slot_for_reuse(&self, seq: &mut SequenceState) {
        self.detach_slot_for_reuse_dispatch(seq)
    }
    fn save_sequence_state(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        self.save_sequence_state_dispatch(seq, writer)
    }
    fn restore_sequence_state(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        self.restore_sequence_state_dispatch(seq, num_blocks, reader)
    }
    fn num_free_blocks(&self) -> usize {
        self.num_free_blocks_dispatch()
    }
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.start_checkpoint_async_dispatch(seq)
    }
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        self.start_rollback_and_checkpoint_async_dispatch(seq, num_accepted)
    }
    fn sync_secondary(&self) -> Result<()> {
        self.sync_secondary_dispatch()
    }
    fn pre_verify_copy_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.pre_verify_copy_async_dispatch(seq)
    }
    fn commit_verify_state_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        self.commit_verify_state_async_dispatch(seq, num_accepted, k)
    }
    fn commit_accepted_prefix(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
        k: usize,
    ) -> Result<()> {
        self.commit_accepted_prefix_dispatch(seq, num_accepted, k)
    }
    fn ep_worker_step(&self, slots: &mut [Option<SequenceState>]) -> Result<bool> {
        self.ep_worker_step_dispatch(slots)
    }
    fn is_ep(&self) -> bool {
        self.is_ep_dispatch()
    }
    fn is_mla(&self) -> bool {
        self.is_mla_dispatch()
    }

    fn kv_block_size(&self) -> Option<usize> {
        Some(self.kv_cache.lock().block_size())
    }
    fn decode_logits_fp32(&self) -> bool {
        self.decode_logits_fp32_dispatch()
    }
    fn decode_logits_ptr(&self) -> DevicePtr {
        self.decode_logits_ptr_dispatch()
    }
    fn ep_broadcast_cmd(&self, cmd: u32) -> Result<()> {
        self.ep_broadcast_cmd_dispatch(cmd)
    }
    fn ep_broadcast_cmd_for_seq(&self, seq_id: u32, cmd: u32) -> Result<()> {
        // Routes to the helper added in 21e2130. Behaviour depends on the
        // ep_protocol_v2 field set at construction from ATLAS_EP_PROTOCOL.
        self.ep_broadcast_seq_and_cmd(seq_id, cmd, self.ep_protocol_v2)
    }
    fn ep_protocol_v2(&self) -> bool {
        self.ep_protocol_v2
    }
    fn ep_broadcast_tokens(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        self.ep_broadcast_tokens_dispatch(tokens)
    }
    fn default_stream(&self) -> u64 {
        self.default_stream_dispatch()
    }
    fn create_stream(&self) -> Result<u64> {
        self.create_stream_dispatch()
    }
    fn create_event(&self) -> Result<u64> {
        self.create_event_dispatch()
    }
    fn record_event(&self, event: u64, stream: u64) -> Result<()> {
        self.record_event_dispatch(event, stream)
    }
    fn stream_wait_event(&self, stream: u64, event: u64) -> Result<()> {
        self.stream_wait_event_dispatch(stream, event)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.synchronize_dispatch(stream)
    }
}
