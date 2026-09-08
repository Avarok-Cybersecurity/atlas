// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence save/restore state I/O for `TransformerModel` (split from sequence.rs for the 500-LoC cap).
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::model::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use crate::model::ssm_pool::SsmStatePool;
use crate::model::ssm_snapshot::SsmSnapshotPool;
use crate::model::types::{PinnedMetaStaging, TransformerModel};
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// KV blocks staged on the host per spill/restore window.
///
/// Bounds host residency during a swap: at 48 layers and ~16 KB per block-layer
/// side this is ~24 MB, against the ~3.1 GB a 32K-token sequence staged when the
/// gather was unbounded — more than the whole `--swap-space-gb` budget (default
/// 3) the spill exists to respect. Both directions use it, and both preserve the
/// on-disk order (block-outer, layer-inner, K then V) because the window only
/// chunks the outer loop.
const SPILL_WINDOW_BLOCKS: usize = 32;

impl TransformerModel {
    pub(crate) fn save_sequence_state_dispatch(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // Phases 1+2, WINDOWED: gather a bounded run of blocks under the lock,
        // release it, write that run to disk, repeat.
        //
        // This used to be one unbounded phase 1 that pulled the ENTIRE KV image
        // into host `Vec`s before writing a single byte. At 48 layers, block
        // size 16 and ~16 KB per block-layer side, a 32K-token sequence stages
        // ~3.1 GB of host memory — more than the whole `--swap-space-gb` budget
        // (default 3) this spill exists to respect — through ~196,608 fresh
        // pageable allocations. Windowing bounds the residency to
        // `SPILL_WINDOW_BLOCKS` blocks (~24 MB at these dimensions) and leaves
        // everything else identical.
        //
        // The window is a residency bound, NOT a copy optimisation. The drain
        // per `read_block` is real but was measured cheap on this box: the
        // 2026-09-07 aux-collect A/B removed 13 drains per snapshot for ~1.2%,
        // because GB10 is unified memory and a D2H is not a bus transfer. So the
        // per-copy machinery is deliberately left alone here — the defect worth
        // fixing was the 3.1 GB, not the syscall count.
        //
        // Byte order on disk is unchanged and load-bearing (`restore` reads it
        // back sequentially): block-outer, layer-inner, K then V. Chunking
        // preserves it because `chunks()` keeps `block_table` order and the
        // inner loop is untouched. Disk I/O still happens with the lock
        // RELEASED, as before.
        for window in seq.block_table.chunks(SPILL_WINDOW_BLOCKS) {
            let kv_buffers = {
                let kv = self.kv_cache.lock();
                let mut bufs = Vec::with_capacity(window.len() * kv.num_layers());
                for &block_idx in window {
                    for layer_idx in 0..kv.num_layers() {
                        bufs.push(kv.read_block(layer_idx, block_idx, gpu)?);
                    }
                }
                bufs
            }; // Lock released here.

            for (k_data, v_data) in &kv_buffers {
                writer.write_all(k_data)?;
                writer.write_all(v_data)?;
            }
        }

        // Phase 3: Copy SSM states from GPU to host, then write to disk.
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // Pool h STORAGE width: the swap record holds exactly what the
                // slot holds (FP32 today; f16 under the stage-3 sized pool).
                let mut h_buf = vec![0u8; self.ssm_pool.h_stored_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                gpu.copy_d2h(ssm.h_state, &mut h_buf)?;
                gpu.copy_d2h(ssm.conv_state, &mut c_buf)?;
                writer.write_all(&h_buf)?;
                writer.write_all(&c_buf)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub(crate) fn restore_sequence_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // Phase 1: Read all KV block data from disk into host buffers.
        let (num_layers, layer_strides) = {
            let kv = self.kv_cache.lock();
            let n = kv.num_layers();
            let strides: Vec<usize> = (0..n).map(|i| kv.block_stride_bytes_for_layer(i)).collect();
            (n, strides)
        };

        // Phases 1+2, WINDOWED — the mirror of the save side, and for the same
        // reason: the unbounded form staged the ENTIRE KV image on the host
        // (~3.1 GB at 32K, larger than the whole `--swap-space-gb` budget)
        // before a single byte reached the GPU. Read a bounded run from disk,
        // upload it under the lock, repeat. Disk reads stay OUTSIDE the lock.
        //
        // Byte order is the save side's, consumed sequentially: block-outer,
        // layer-inner, K then V.
        //
        // `seq.block_table` is extended AS BLOCKS ARE ALLOCATED rather than
        // assigned once at the end. That is what makes a mid-restore failure
        // recoverable: `restore_swapped_image` calls `free_sequence` on error,
        // which frees whatever `block_table` holds. Assigning at the end — as
        // this did — leaked every block already allocated if any `alloc_block`
        // or `write_block` failed part-way, and windowing would have widened
        // that window from "phase 2" to "the whole restore".
        seq.block_table.clear();
        seq.block_table.reserve(num_blocks);
        let mut remaining = num_blocks;
        while remaining > 0 {
            let this_window = remaining.min(SPILL_WINDOW_BLOCKS);

            let mut kv_buffers = Vec::with_capacity(this_window * num_layers);
            for _ in 0..this_window {
                for layer_idx in 0..num_layers {
                    let stride = layer_strides[layer_idx];
                    let mut k_data = vec![0u8; stride];
                    let mut v_data = vec![0u8; stride];
                    reader.read_exact(&mut k_data)?;
                    reader.read_exact(&mut v_data)?;
                    kv_buffers.push((k_data, v_data));
                }
            }

            {
                let mut kv = self.kv_cache.lock();
                let mut buf_idx = 0;
                for _ in 0..this_window {
                    let block_idx = kv.alloc_block()?;
                    seq.block_table.push(block_idx);
                    for layer_idx in 0..num_layers {
                        let (ref k_data, ref v_data) = kv_buffers[buf_idx];
                        kv.write_block(layer_idx, block_idx, k_data, v_data, gpu)?;
                        buf_idx += 1;
                    }
                }
            } // Lock released here.

            remaining -= this_window;
        }

        // Phase 3: Read SSM state data from disk and upload to GPU.
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                // Pool h STORAGE width: the swap record holds exactly what the
                // slot holds (FP32 today; f16 under the stage-3 sized pool).
                let mut h_buf = vec![0u8; self.ssm_pool.h_stored_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                reader.read_exact(&mut h_buf)?;
                reader.read_exact(&mut c_buf)?;
                gpu.copy_h2d(&h_buf, ssm.h_state)?;
                gpu.copy_h2d(&c_buf, ssm.conv_state)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SPILL_WINDOW_BLOCKS;

    /// Emulate the SAVE traversal: `block_table.chunks(W)`, then layer-inner.
    fn save_order(num_blocks: usize, num_layers: usize) -> Vec<(u32, usize)> {
        let blocks: Vec<u32> = (0..num_blocks as u32).collect();
        let mut out = Vec::new();
        for window in blocks.chunks(SPILL_WINDOW_BLOCKS) {
            for &b in window {
                for l in 0..num_layers {
                    out.push((b, l));
                }
            }
        }
        out
    }

    /// Emulate the RESTORE traversal: the `remaining` / `this_window` loop,
    /// with blocks allocated in order.
    fn restore_order(num_blocks: usize, num_layers: usize) -> Vec<(u32, usize)> {
        let mut out = Vec::new();
        let mut remaining = num_blocks;
        let mut next_block = 0u32;
        while remaining > 0 {
            let this_window = remaining.min(SPILL_WINDOW_BLOCKS);
            for _ in 0..this_window {
                for l in 0..num_layers {
                    out.push((next_block, l));
                }
                next_block += 1;
            }
            remaining -= this_window;
        }
        out
    }

    /// The unwindowed traversal this replaced: the on-disk format itself.
    fn flat_order(num_blocks: usize, num_layers: usize) -> Vec<(u32, usize)> {
        (0..num_blocks as u32)
            .flat_map(|b| (0..num_layers).map(move |l| (b, l)))
            .collect()
    }

    /// Windowing must not perturb the (block, layer) visit order in EITHER
    /// direction: that order is the file format, so an off-by-one in the window
    /// loop would not fail loudly — it would write a KV image that restores as
    /// shuffled attention state. Sizes deliberately straddle the window bound.
    #[test]
    fn windowing_preserves_the_on_disk_order() {
        for &num_blocks in &[0usize, 1, 31, 32, 33, 63, 64, 65, 100] {
            for &num_layers in &[1usize, 48] {
                let flat = flat_order(num_blocks, num_layers);
                assert_eq!(
                    save_order(num_blocks, num_layers),
                    flat,
                    "save order diverged at {num_blocks} blocks x {num_layers} layers"
                );
                assert_eq!(
                    restore_order(num_blocks, num_layers),
                    flat,
                    "restore order diverged at {num_blocks} blocks x {num_layers} layers"
                );
                // Save and restore must agree with each other, which is the
                // property that actually round-trips.
                assert_eq!(
                    save_order(num_blocks, num_layers),
                    restore_order(num_blocks, num_layers),
                    "save/restore disagree at {num_blocks} blocks x {num_layers} layers"
                );
                assert_eq!(flat.len(), num_blocks * num_layers);
            }
        }
    }

    /// Every block is visited exactly once — no window drops or repeats one.
    #[test]
    fn every_block_is_visited_exactly_once() {
        for &num_blocks in &[0usize, 33, 64, 65] {
            let seen: Vec<u32> = restore_order(num_blocks, 1).into_iter().map(|(b, _)| b).collect();
            let expected: Vec<u32> = (0..num_blocks as u32).collect();
            assert_eq!(seen, expected, "{num_blocks} blocks");
        }
    }
}
