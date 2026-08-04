// SPDX-License-Identifier: AGPL-3.0-only

//! Default per-token fallback loop bodies for `TransformerLayer`, split out
//! of `transformer_layer.rs` for the ≤500 LoC file-size cap. Each function
//! is the verbatim body of the corresponding default trait method, taking
//! the layer as an explicit `&dyn TransformerLayer` argument.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::TransformerLayer;
use crate::layer::{ForwardContext, LayerState};

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_default(
    layer: &(impl TransformerLayer + ?Sized),
    hidden: DevicePtr,
    residual: DevicePtr,
    num_tokens: usize,
    state: &mut dyn LayerState,
    kv_cache: &mut PagedKvCache,
    seq_len_start: usize,
    block_table: &mut Vec<u32>,
    disk_block_ids: &mut Vec<u32>,
    disk_last_offloaded_per_layer: &mut Vec<u32>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let h = ctx.config.hidden_size;
    for t in 0..num_tokens {
        let offset = t * h * 2; // BF16 = 2 bytes per element
        let h_t = hidden.offset(offset);
        let r_t = residual.offset(offset);
        layer.decode(
            h_t,
            r_t,
            state,
            kv_cache,
            seq_len_start + t,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_batched_default(
    layer: &(impl TransformerLayer + ?Sized),
    hidden: DevicePtr,
    residual: DevicePtr,
    num_tokens: usize,
    state: &mut dyn LayerState,
    kv_cache: &mut PagedKvCache,
    seq_len: usize,
    block_table: &mut Vec<u32>,
    disk_block_ids: &mut Vec<u32>,
    disk_last_offloaded_per_layer: &mut Vec<u32>,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let h = ctx.config.hidden_size;
    for t in 0..num_tokens {
        let offset = (t * h * 2) as u64; // BF16 = 2 bytes per element
        let h_t = hidden.offset(offset as usize);
        let r_t = residual.offset(offset as usize);
        layer.decode(
            h_t,
            r_t,
            state,
            kv_cache,
            seq_len + t,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn decode_multi_seq_default<'a, 'b: 'a>(
    layer: &(impl TransformerLayer + ?Sized),
    hidden: DevicePtr,
    residual: DevicePtr,
    num_seqs: usize,
    states: &'a mut [&'b mut (dyn LayerState + 'static)],
    kv_cache: &mut PagedKvCache,
    seq_lens: &[usize],
    block_tables: &[Vec<u32>],
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let h = ctx.config.hidden_size;
    for i in 0..num_seqs {
        let offset = i * h * 2;
        let h_i = hidden.offset(offset);
        let r_i = residual.offset(offset);
        let mut bt = block_tables[i].clone();
        // Phase 6.1: per-seq disk_block_ids aren't threaded through this
        // default impl yet (chunked-prefill / batched-decode are Phase 6.2
        // scope). Pass empty stubs so the trait sig is satisfied; layers
        // that need disk IDs (attention) override decode_multi_seq.
        let mut stub_disk = Vec::<u32>::new();
        let mut stub_last_offloaded = Vec::<u32>::new();
        layer.decode(
            h_i,
            r_i,
            states[i],
            kv_cache,
            seq_lens[i],
            &mut bt,
            &mut stub_disk,
            &mut stub_last_offloaded,
            ctx,
            stream,
        )?;
    }
    Ok(())
}
