// SPDX-License-Identifier: AGPL-3.0-only
// Batched arm: design after LMCache csrc/cuda/mp_mem_kernels.cu:358-849 (one batched copy
// submission per object instead of one blocking copy per region), commit 3aefae1, Apache-2.0; no
// code copied. Changes: the swap image streamed through one reusable page-locked staging buffer
// (`spark_runtime::kv_cache::block_io::swap`), by Atlas contributors.

//! Responsibility: sequence swap save/restore for `TransformerModel`: the image a swapped-out
//! sequence leaves on disk and the restore that reads it back.
//!
//! The image is, for each block of the table, for each attention layer, the K block then the V
//! block; then for each GDN layer, the `h` state (pool storage width) then the conv state. Two
//! writers produce it and two readers consume it, byte for byte the same file:
//!
//! * the per-region path (row `kv_batched_swap` off, the default everywhere): blocking copies
//!   into fresh `Vec`s, unchanged from before the port sprint except for an optional
//!   `AVAROK_KV_SWAP_TIMING=1` log line (which also feeds `atlas_kv_swap_*`);
//! * the batched path (row on): every region of the image is gathered with batched async
//!   copies into a reused page-locked staging buffer and written in large chunks, the copy of
//!   chunk `i` overlapping the write of chunk `i - 1`; restore mirrors it.
//!
//! Restore through the batched path reads the K and V block of each layer at their own strides,
//! so it also reads files of asymmetric K/V dtypes, which the per-region reader reads with the K
//! stride for both sides (kept as it was: the row-off path does not change).

use std::io::{Read, Write};
use std::time::Instant;

use anyhow::{Result, anyhow};
use avarok_core::config::LayerType;
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::kv_cache::block_io::swap::{self, SwapDir, SwapReport, SwapStaging};

use crate::layer::{LayerState, SsmLayerState};
use crate::layers::ops::port_rows::{kv_batched_swap, ssm_store_lru};
use crate::model::types::TransformerModel;
use crate::traits::SequenceState;

/// One GDN layer's part of the image: the `h` state then the conv state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SsmRegion {
    pub h: DevicePtr,
    pub h_bytes: usize,
    pub conv: DevicePtr,
    pub conv_bytes: usize,
}

impl TransformerModel {
    /// The SSM regions of `states` in layer order, one per GDN layer. Lazy, so the per-region
    /// path meets a bad layer state at the same point of the file as before.
    fn ssm_regions<'a>(
        &'a self,
        states: &'a [Box<dyn LayerState>],
    ) -> impl Iterator<Item = Result<SsmRegion>> + 'a {
        let (h_bytes, conv_bytes) = (self.ssm_pool.h_stored_bytes, self.ssm_pool.conv_bytes);
        states.iter().enumerate().filter_map(move |(i, state)| {
            // Port L38: LFM3's conv window is layer-owned device memory, not a pool region; it
            // stays with the sequence across a swap, so it has no part in the image. The skip is
            // keyed on the LFM3 state type alone, so every model that existed before the sprint
            // is unchanged: pool-backed layers (Qwen GDN, GLM KDA) write their regions, and any
            // other own-state linear-attention layer (Kimi K3's KDA) still fails the downcast
            // below, at the same point of the file as before.
            if self.config.layer_type(i) != LayerType::LinearAttention
                || state.as_any().is::<crate::port::lfm3_conv::Lfm3ConvState>()
            {
                return None;
            }
            // Pool h STORAGE width: the swap record holds exactly what the slot holds (FP32
            // today; f16 under the stage-3 sized pool).
            Some(
                state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .map(|ssm| SsmRegion {
                        h: ssm.h_state,
                        h_bytes,
                        conv: ssm.conv_state,
                        conv_bytes,
                    })
                    .ok_or_else(|| anyhow!("Expected SsmLayerState at layer {i}")),
            )
        })
    }

    pub(crate) fn save_sequence_state_dispatch(
        &self,
        seq: &SequenceState,
        writer: &mut dyn Write,
    ) -> Result<()> {
        // Port L36 (row `kv_window_desk`): a block table with pages in host RAM has no complete
        // image; refuse (the scheduler takes its existing preemption-failure path).
        crate::port::kv_window::swap_out_guard(&seq.block_table)?;
        let gpu = self.gpu.as_ref();
        let ssm = self.ssm_regions(&seq.layer_states);
        if kv_batched_swap::row() {
            save_batched(
                gpu,
                &self.kv_cache,
                &seq.block_table,
                ssm,
                &self.port.kv_swap,
                writer,
            )
            .map(|_| ())
        } else {
            save_legacy(gpu, &self.kv_cache, &seq.block_table, ssm, writer)
        }
    }

    pub(crate) fn restore_sequence_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn Read,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();
        let ssm = self.ssm_regions(&seq.layer_states);
        let table = &mut seq.block_table;
        if kv_batched_swap::row() {
            restore_batched(
                gpu,
                &self.kv_cache,
                num_blocks,
                table,
                ssm,
                &self.port.kv_swap,
                reader,
            )
            .map(|_| ())
        } else {
            restore_legacy(gpu, &self.kv_cache, num_blocks, table, ssm, reader)
        }
    }

    /// Drop the SSM tier blobs the snapshot index orphaned by re-homing a spilled prefix to HBM
    /// (`radix_tree/snapshot_insert.rs`). Row `ssm_store_lru` only; a model without a tier store
    /// or a prefix cache without a snapshot index has nothing to reap.
    pub(in crate::model::trait_impl) fn reap_orphaned_ssm_blobs(&self) {
        if !ssm_store_lru::row() {
            return;
        }
        let Some(store) = self.ssm_tier_store.as_ref() else {
            return;
        };
        for key in self.prefix_cache.take_orphaned_snapshot_tier_keys() {
            store.remove(key);
        }
    }
}

/// Count and log one per-region image when `AVAROK_KV_SWAP_TIMING` is on. With it off the
/// per-region path touches no counter, so a row-off serve's `/metrics` stays zero.
fn legacy_timing(dir: SwapDir, t0: Instant, kv_bytes: usize, bytes: usize, regions: usize) {
    if !swap::timing_enabled() {
        return;
    }
    let elapsed = t0.elapsed();
    let rep = SwapReport {
        bytes,
        regions,
        total_us: u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        staging: "none",
        ..SwapReport::default()
    };
    swap::record(dir, bytes, elapsed);
    swap::log_report(dir, "legacy", kv_bytes, &rep);
}

/// The per-region writer (row off): the pre-sprint body, plus the optional timing line.
pub(super) fn save_legacy(
    gpu: &dyn GpuBackend,
    kv_cache: &Mutex<PagedKvCache>,
    block_table: &[u32],
    ssm: impl Iterator<Item = Result<SsmRegion>>,
    writer: &mut dyn Write,
) -> Result<()> {
    let t0 = Instant::now();
    // Phase 1: Copy all KV block data from GPU to host buffers under the lock.
    let kv_buffers = {
        let kv = kv_cache.lock();
        let mut bufs = Vec::with_capacity(block_table.len() * kv.num_layers());
        for &block_idx in block_table {
            for layer_idx in 0..kv.num_layers() {
                bufs.push(kv.read_block(layer_idx, block_idx, gpu)?);
            }
        }
        bufs
    }; // Lock released here.

    // Phase 2: Write KV data to disk (no lock held).
    let mut kv_bytes = 0usize;
    for (k_data, v_data) in &kv_buffers {
        writer.write_all(k_data)?;
        writer.write_all(v_data)?;
        kv_bytes += k_data.len() + v_data.len();
    }

    // Phase 3: Copy SSM states from GPU to host, then write to disk.
    let (mut bytes, mut regions) = (kv_bytes, 2 * kv_buffers.len());
    for r in ssm {
        let r = r?;
        let mut h_buf = vec![0u8; r.h_bytes];
        let mut c_buf = vec![0u8; r.conv_bytes];
        gpu.copy_d2h(r.h, &mut h_buf)?;
        gpu.copy_d2h(r.conv, &mut c_buf)?;
        writer.write_all(&h_buf)?;
        writer.write_all(&c_buf)?;
        bytes += r.h_bytes + r.conv_bytes;
        regions += 2;
    }

    writer.flush()?;
    legacy_timing(SwapDir::Out, t0, kv_bytes, bytes, regions);
    Ok(())
}

/// The per-region reader (row off): the pre-sprint body, plus the optional timing line.
pub(super) fn restore_legacy(
    gpu: &dyn GpuBackend,
    kv_cache: &Mutex<PagedKvCache>,
    num_blocks: usize,
    block_table: &mut Vec<u32>,
    ssm: impl Iterator<Item = Result<SsmRegion>>,
    reader: &mut dyn Read,
) -> Result<()> {
    let t0 = Instant::now();
    // Phase 1: Read all KV block data from disk into host buffers.
    let (num_layers, layer_strides) = {
        let kv = kv_cache.lock();
        let n = kv.num_layers();
        let strides: Vec<usize> = (0..n).map(|i| kv.block_stride_bytes_for_layer(i)).collect();
        (n, strides)
    };

    let mut kv_buffers = Vec::with_capacity(num_blocks * num_layers);
    for _ in 0..num_blocks {
        for &stride in layer_strides.iter().take(num_layers) {
            let mut k_data = vec![0u8; stride];
            let mut v_data = vec![0u8; stride];
            reader.read_exact(&mut k_data)?;
            reader.read_exact(&mut v_data)?;
            kv_buffers.push((k_data, v_data));
        }
    }

    // Phase 2: Allocate blocks and write data under the lock.
    let mut kv_bytes = 0usize;
    {
        let mut kv = kv_cache.lock();
        let mut new_block_table = Vec::with_capacity(num_blocks);
        let mut buf_idx = 0;
        for _ in 0..num_blocks {
            let block_idx = kv.alloc_block()?;
            for layer_idx in 0..num_layers {
                let (ref k_data, ref v_data) = kv_buffers[buf_idx];
                kv.write_block(layer_idx, block_idx, k_data, v_data, gpu)?;
                kv_bytes += k_data.len() + v_data.len();
                buf_idx += 1;
            }
            new_block_table.push(block_idx);
        }
        *block_table = new_block_table;
    } // Lock released here.

    // Phase 3: Read SSM state data from disk and upload to GPU.
    let (mut bytes, mut regions) = (kv_bytes, 2 * kv_buffers.len());
    for r in ssm {
        let r = r?;
        let mut h_buf = vec![0u8; r.h_bytes];
        let mut c_buf = vec![0u8; r.conv_bytes];
        reader.read_exact(&mut h_buf)?;
        reader.read_exact(&mut c_buf)?;
        gpu.copy_h2d(&h_buf, r.h)?;
        gpu.copy_h2d(&c_buf, r.conv)?;
        bytes += r.h_bytes + r.conv_bytes;
        regions += 2;
    }

    legacy_timing(SwapDir::In, t0, kv_bytes, bytes, regions);
    Ok(())
}

/// Append the SSM regions to `regions` (h then conv per GDN layer); returns the KV byte count
/// that precedes them.
fn push_ssm(
    regions: &mut Vec<(DevicePtr, usize)>,
    ssm: impl Iterator<Item = Result<SsmRegion>>,
) -> Result<usize> {
    let kv_bytes = regions.iter().map(|&(_, n)| n).sum();
    for r in ssm {
        let r = r?;
        regions.push((r.h, r.h_bytes));
        regions.push((r.conv, r.conv_bytes));
    }
    Ok(kv_bytes)
}

/// The batched writer (row on). Copies run on the backend's default stream, so they are ordered
/// after every kernel that wrote the sequence's KV and SSM state. The pointers are taken under
/// the KV lock; the blocks stay this sequence's (it holds their references) for the whole save.
pub(super) fn save_batched(
    gpu: &dyn GpuBackend,
    kv_cache: &Mutex<PagedKvCache>,
    block_table: &[u32],
    ssm: impl Iterator<Item = Result<SsmRegion>>,
    staging: &SwapStaging,
    writer: &mut dyn Write,
) -> Result<SwapReport> {
    let mut regions: Vec<(DevicePtr, usize)> = Vec::new();
    for b in kv_cache.lock().table_regions(block_table)? {
        regions.extend(b.regions);
    }
    let kv_bytes = push_ssm(&mut regions, ssm)?;
    let rep = swap::write_image(gpu, &regions, staging, writer, gpu.default_stream())?;
    swap::log_report(SwapDir::Out, "batched", kv_bytes, &rep);
    Ok(rep)
}

/// The batched reader (row on): allocate the blocks, stream the file into them and into the SSM
/// state, and only then publish the new table. On any failure the blocks it allocated are freed
/// and `block_table` is left as it was.
pub(super) fn restore_batched(
    gpu: &dyn GpuBackend,
    kv_cache: &Mutex<PagedKvCache>,
    num_blocks: usize,
    block_table: &mut Vec<u32>,
    ssm: impl Iterator<Item = Result<SsmRegion>>,
    staging: &SwapStaging,
    reader: &mut dyn Read,
) -> Result<SwapReport> {
    let (table, mut regions) = {
        let mut kv = kv_cache.lock();
        let mut table = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            match kv.alloc_block() {
                Ok(b) => table.push(b),
                Err(e) => {
                    kv.free_blocks(&table);
                    return Err(e);
                }
            }
        }
        let regions: Vec<(DevicePtr, usize)> = match kv.table_regions(&table) {
            Ok(blocks) => blocks.into_iter().flat_map(|b| b.regions).collect(),
            Err(e) => {
                kv.free_blocks(&table);
                return Err(e);
            }
        };
        (table, regions)
    };
    let run = push_ssm(&mut regions, ssm).and_then(|kv_bytes| {
        let rep = swap::read_image(gpu, &regions, staging, reader, gpu.default_stream())?;
        Ok((kv_bytes, rep))
    });
    match run {
        Ok((kv_bytes, rep)) => {
            *block_table = table;
            swap::log_report(SwapDir::In, "batched", kv_bytes, &rep);
            Ok(rep)
        }
        Err(e) => {
            kv_cache.lock().free_blocks(&table);
            Err(e)
        }
    }
}

#[cfg(test)]
#[path = "state_io_tests.rs"]
mod tests;
