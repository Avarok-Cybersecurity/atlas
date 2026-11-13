// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: `impl TransformerLayer for Lfm3ConvLayer`: the trait
//! entry points (decode, batched and multi-sequence decode, prefill), the
//! state lifecycle (alloc, release, snapshot, restore) and the capability
//! answers the scheduler and the model read (pool, graphs, speculation).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::Lfm3ConvLayer;
use super::state::{self, Lfm3ConvState};
use crate::layer::{ForwardContext, LayerState, TransformerLayer};

impl Lfm3ConvLayer {
    /// The sequence's own window (strict: a single-sequence entry point never
    /// receives a padding row).
    fn own_state(&self, state: &dyn LayerState) -> Result<DevicePtr> {
        let s = state
            .as_any()
            .downcast_ref::<Lfm3ConvState>()
            .with_context(|| {
                format!(
                    "LFM3 conv layer {}: expected Lfm3ConvState (the layer opts out of the SSM \
                     pool; a pool-shaped state reached it)",
                    self.cfg.layer_idx
                )
            })?;
        anyhow::ensure!(
            !s.conv.is_null(),
            "LFM3 conv layer {}: the sequence's window was already released",
            self.cfg.layer_idx
        );
        Ok(s.conv)
    }
}

impl TransformerLayer for Lfm3ConvLayer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let s = self.own_state(state)?;
        self.decode_rows(hidden, residual, &[s], ctx, stream)
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let s = self.own_state(state)?;
        self.forward_tokens(hidden, residual, num_tokens, s, ctx, stream)
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let s = self.own_state(state)?;
        self.forward_tokens(hidden, residual, num_tokens, s, ctx, stream)
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            states.len() >= num_seqs,
            "LFM3 conv layer {}: decode_multi_seq got {} states for {num_seqs} rows",
            self.cfg.layer_idx,
            states.len()
        );
        let ptrs = states[..num_seqs]
            .iter()
            .map(|s| self.row_state(&**s))
            .collect::<Result<Vec<_>>>()?;
        self.decode_rows(hidden, residual, &ptrs, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let bytes = self.cfg.state_bytes();
        let conv = gpu.alloc(bytes)?;
        // A fresh window is zero: the reference's left zero padding.
        gpu.memset(conv, 0, bytes)?;
        Ok(Box::new(Lfm3ConvState { conv, bytes }))
    }

    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        // Refuse by type: only a window this layer allocated is freed; the
        // pool-dummy states of padding rows are never ours.
        if let Some(s) = state.as_any_mut().downcast_mut::<Lfm3ConvState>()
            && !s.conv.is_null()
        {
            let ptr = std::mem::replace(&mut s.conv, DevicePtr::NULL);
            gpu.free(ptr)?;
        }
        Ok(())
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let s = self.own_state(state)?;
        let mut payload = vec![0u8; self.cfg.state_bytes()];
        // Stream-ordered: the window reflects every kernel already queued.
        gpu.copy_d2h_on_stream(s, &mut payload, stream)?;
        Ok(Some(state::encode_blob(self.cfg.hidden, &payload)?))
    }

    fn has_aux_state(&self) -> bool {
        true
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let s = self.own_state(state)?;
        let payload = state::decode_blob(self.cfg.hidden, blob)
            .with_context(|| format!("LFM3 conv layer {}: restore_aux", self.cfg.layer_idx))?;
        gpu.copy_h2d_async(payload, s, stream)
    }

    /// The window is per-sequence memory, so a slot's captured graphs bake
    /// one sequence's addresses and must be dropped when it is freed.
    fn graph_stale_on_new_sequence(&self) -> bool {
        true
    }

    /// No checkpoint ring: lowering the KV cursor does not rewind the window.
    fn decode_rollback_unsupported(&self) -> bool {
        true
    }

    /// No per-row intermediate windows for a batched K-row verify.
    fn decode_verify_multi_unsupported(&self) -> bool {
        true
    }

    /// The window is not GDN-shaped and is allocated by `alloc_state`.
    fn uses_ssm_pool(&self) -> bool {
        false
    }
}
