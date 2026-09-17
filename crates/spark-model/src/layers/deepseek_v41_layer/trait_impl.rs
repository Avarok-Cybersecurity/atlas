// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `impl TransformerLayer for DeepSeekV41Layer`: decode and prefill both go
//! through `step`; multi-sequence decode and graph capture are declined.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DeepSeekV41Layer, V41LayerState};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::attn_v41::AttnV41LayerState;

impl crate::layer::TransformerLayer for DeepSeekV41Layer {
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.step(hidden, 1, seq_len, state, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.step(hidden, num_tokens, seq_len_start, state, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(V41LayerState {
            attn: AttnV41LayerState::new(gpu, &self.rt.attn_cfg, self.role)?,
        }))
    }

    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }

    /// Lowering `seq_len` rewinds none of this layer's state: the shared
    /// compressed-row count only grows (`SharedV41::compress_len`), the
    /// ratio-2 compressor holds a partial group of the dropped tokens, and
    /// the engram hasher's n-gram history ends in them. A watchdog rollback
    /// then re-steers from a corrupted context (09-16: the second MinHeap
    /// request spliced `_wHere` at the boundary and degenerated from
    /// there). Declined until a rewind exists; the loop watchdog still
    /// ends the response.
    fn decode_rollback_unsupported(&self) -> bool {
        true
    }
}
