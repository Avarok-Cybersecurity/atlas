// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache csrc/cuda/mp_mem_kernels.cu:358-849 (commit 3aefae1, Apache-2.0): one batched
// submission of many (device, host, bytes) entries instead of one blocking copy per region; no code
// copied. Atlas needs no layout transform (every layer's K or V block is one contiguous region), so
// the per-format offset math upstream carries has no counterpart here.

//! Responsibility: the KV block transfer primitives the sequence swap and the host KV tier share.
//!
//! * [`BlockRegions`] lists the device regions of ONE physical block across every attention layer,
//!   K then V per layer, in layer order. That is the byte order the swap file has always used
//!   (`sequence/state_io.rs`: for each block, for each layer, K then V), so a gather of a block
//!   table into one host buffer IS the KV half of the swap image.
//! * [`gather_to_host`] / [`scatter_from_host`] enqueue one batched copy per call
//!   (`GpuBackend::copy_batch_*`, `cuMemcpyBatchAsync` where it resolved) between those regions and
//!   one contiguous host buffer; the `*_hosts` variants take one host address per block (the host
//!   tier's page runs). None of them synchronizes: the caller records an event or synchronizes the
//!   stream before it reads (D2H) or reuses (H2D) the host bytes.
//! * [`plan_chunks`] cuts an ordered region list into host chunks of at most `cap` bytes (splitting
//!   a region across two chunks when it straddles the boundary), so an image of any size streams
//!   through a fixed staging buffer; concatenating the chunks in order reproduces the image.
//!
//! Nothing here reads a row. The swap path (`swap`, a child module) is selected by spark-model's
//! `kv_batched_swap` row; with it off nothing in this file runs.

use anyhow::{Result, bail};

use super::PagedKvCache;
use crate::gpu::{CopyEntry, DevicePtr, GpuBackend};

/// The batched swap image path: staging, chunked write and read, `/metrics` series.
#[path = "block_io_swap.rs"]
pub mod swap;

/// Entries per `copy_batch_*` call. `cuMemcpyBatchAsync` takes host arrays of descriptors; a
/// 32K-token BF16 sequence on the 4B is 2,048 blocks x 16 regions = 32,768 entries, so the submit
/// is split to keep each descriptor array small. Bytes moved do not depend on the split.
pub const MAX_BATCH_ENTRIES: usize = 4096;

/// The device regions of one physical KV block across every attention layer: for each layer in
/// order, `(K pointer, K bytes)` then `(V pointer, V bytes)`. Sizes come from the per-side block
/// strides, so asymmetric K/V dtypes are exact.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockRegions {
    pub regions: Vec<(DevicePtr, usize)>,
}

impl BlockRegions {
    /// Total bytes of the block across all layers, K and V.
    pub fn bytes(&self) -> usize {
        self.regions.iter().map(|&(_, n)| n).sum()
    }

    /// Number of regions (two per attention layer).
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    /// True when the cache has no attention layers (or was released).
    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

impl PagedKvCache {
    /// The regions of physical block `block` (all layers, K then V: the swap byte order).
    ///
    /// Pure pointer arithmetic over the pools, like `k_cache_ptr`: `block` must be a block of this
    /// cache. Use [`PagedKvCache::checked_block_regions`] when it comes from outside.
    pub fn block_regions(&self, block: u32) -> BlockRegions {
        let mut regions = Vec::with_capacity(self.layers.len() * 2);
        for layer in &self.layers {
            let k = layer.k_pool.offset(block as usize * layer.k_block_stride);
            let v = layer.v_pool.offset(block as usize * layer.v_block_stride);
            regions.push((k, layer.k_block_stride));
            regions.push((v, layer.v_block_stride));
        }
        BlockRegions { regions }
    }

    /// [`PagedKvCache::block_regions`] with the block index checked against the pool.
    pub fn checked_block_regions(&self, block: u32) -> Result<BlockRegions> {
        if block as usize >= self.num_blocks {
            bail!(
                "block_regions: block {block} out of range (cache has {} blocks)",
                self.num_blocks
            );
        }
        Ok(self.block_regions(block))
    }

    /// The regions of every block of `table`, in table order.
    pub fn table_regions(&self, table: &[u32]) -> Result<Vec<BlockRegions>> {
        table
            .iter()
            .map(|&b| self.checked_block_regions(b))
            .collect()
    }
}

/// Total bytes of `blocks`.
pub fn total_bytes(blocks: &[BlockRegions]) -> usize {
    blocks.iter().map(BlockRegions::bytes).sum()
}

/// Copy entries for `blocks` laid out back to back from host address `base`.
fn contiguous_entries(blocks: &[BlockRegions], base: usize) -> Vec<CopyEntry> {
    let mut out = Vec::with_capacity(blocks.iter().map(BlockRegions::len).sum());
    let mut off = 0usize;
    for b in blocks {
        for &(dev, bytes) in &b.regions {
            out.push(CopyEntry {
                dev,
                host: base + off,
                bytes,
            });
            off += bytes;
        }
    }
    out
}

/// Copy entries for `blocks`, block `i` laid out from host address `hosts[i]`.
fn per_block_entries(blocks: &[BlockRegions], hosts: &[usize]) -> Result<Vec<CopyEntry>> {
    if blocks.len() != hosts.len() {
        bail!(
            "block_io: {} blocks but {} host addresses",
            blocks.len(),
            hosts.len()
        );
    }
    let mut out = Vec::with_capacity(blocks.iter().map(BlockRegions::len).sum());
    for (b, &base) in blocks.iter().zip(hosts) {
        let mut off = 0usize;
        for &(dev, bytes) in &b.regions {
            out.push(CopyEntry {
                dev,
                host: base + off,
                bytes,
            });
            off += bytes;
        }
    }
    Ok(out)
}

/// Enqueue `entries` D2H on `stream`, at most [`MAX_BATCH_ENTRIES`] per backend call.
pub(crate) fn submit_d2h(gpu: &dyn GpuBackend, entries: &[CopyEntry], stream: u64) -> Result<()> {
    for part in entries.chunks(MAX_BATCH_ENTRIES) {
        gpu.copy_batch_d2h_async(part, stream)?;
    }
    Ok(())
}

/// Enqueue `entries` H2D on `stream`, at most [`MAX_BATCH_ENTRIES`] per backend call.
pub(crate) fn submit_h2d(
    gpu: &dyn GpuBackend,
    entries: &[CopyEntry],
    stream: u64,
    src_any_order: bool,
) -> Result<()> {
    for part in entries.chunks(MAX_BATCH_ENTRIES) {
        gpu.copy_batch_h2d_async(part, stream, src_any_order)?;
    }
    Ok(())
}

/// Enqueue the D2H gather of `blocks` into `host` (back to back, in block then region order).
/// Returns the bytes enqueued.
///
/// Does NOT synchronize. `host` must stay valid and must not be read until the next sync point on
/// `stream` (an event recorded after this call, or `synchronize`): the contract of
/// `GpuBackend::copy_d2h_async`. A page-locked `host` (pinned pool, pinned staging) is what makes
/// the copies truly asynchronous; a pageable one is correct but bounces through the driver.
pub fn gather_to_host(
    gpu: &dyn GpuBackend,
    blocks: &[BlockRegions],
    host: &mut [u8],
    stream: u64,
) -> Result<usize> {
    let total = total_bytes(blocks);
    if host.len() < total {
        bail!(
            "gather_to_host: host buffer {} B < {total} B of blocks",
            host.len()
        );
    }
    submit_d2h(
        gpu,
        &contiguous_entries(blocks, host.as_mut_ptr() as usize),
        stream,
    )?;
    Ok(total)
}

/// Enqueue the H2D scatter of `host` (laid out as [`gather_to_host`] writes it) into `blocks`.
/// Returns the bytes enqueued.
///
/// Does NOT synchronize, and never adds the implicit sync of `copy_h2d_async`: `host` must stay
/// valid and unchanged until the next sync point on `stream`. `src_any_order` lets the copy engine
/// read the sources out of stream order (tier-owned memory that nothing rewrites in flight).
pub fn scatter_from_host(
    gpu: &dyn GpuBackend,
    blocks: &[BlockRegions],
    host: &[u8],
    stream: u64,
    src_any_order: bool,
) -> Result<usize> {
    let total = total_bytes(blocks);
    if host.len() < total {
        bail!(
            "scatter_from_host: host buffer {} B < {total} B of blocks",
            host.len()
        );
    }
    let entries = contiguous_entries(blocks, host.as_ptr() as usize);
    submit_h2d(gpu, &entries, stream, src_any_order)?;
    Ok(total)
}

/// [`gather_to_host`] with one host destination per block: block `i` lands at `hosts[i]`, its
/// regions back to back. Returns the bytes enqueued.
///
/// # Safety
///
/// For every `i`, `[hosts[i], hosts[i] + blocks[i].bytes())` must be valid writable host memory
/// that nothing else reads or writes until the next sync point on `stream`, and no two of those
/// ranges may overlap.
pub unsafe fn gather_to_hosts(
    gpu: &dyn GpuBackend,
    blocks: &[BlockRegions],
    hosts: &[usize],
    stream: u64,
) -> Result<usize> {
    let entries = per_block_entries(blocks, hosts)?;
    submit_d2h(gpu, &entries, stream)?;
    Ok(total_bytes(blocks))
}

/// [`scatter_from_host`] with one host source per block: block `i` is read from `hosts[i]`.
/// Returns the bytes enqueued.
///
/// # Safety
///
/// For every `i`, `[hosts[i], hosts[i] + blocks[i].bytes())` must be valid readable host memory
/// that stays unchanged until the next sync point on `stream`.
pub unsafe fn scatter_from_hosts(
    gpu: &dyn GpuBackend,
    blocks: &[BlockRegions],
    hosts: &[usize],
    stream: u64,
    src_any_order: bool,
) -> Result<usize> {
    let entries = per_block_entries(blocks, hosts)?;
    submit_h2d(gpu, &entries, stream, src_any_order)?;
    Ok(total_bytes(blocks))
}

/// One host chunk of an image: copy entries `(device pointer, offset in the chunk, bytes)` that
/// fill `[0, bytes)` of the chunk exactly, in image order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkPlan {
    pub entries: Vec<(DevicePtr, usize, usize)>,
    pub bytes: usize,
}

/// Cut the ordered `regions` of an image into chunks of at most `cap` bytes. Every chunk but the
/// last is exactly `cap` bytes; a region that straddles a chunk boundary is split into two entries
/// (its device pointer advanced by the bytes already taken). Zero-byte regions are skipped. The
/// chunks, concatenated in order, are the regions' bytes in order.
pub fn plan_chunks(regions: &[(DevicePtr, usize)], cap: usize) -> Result<Vec<ChunkPlan>> {
    if cap == 0 {
        bail!("plan_chunks: chunk capacity is zero");
    }
    let mut out = Vec::new();
    let mut cur = ChunkPlan::default();
    for &(dev, len) in regions {
        let mut done = 0usize;
        while done < len {
            let take = (cap - cur.bytes).min(len - done);
            cur.entries.push((dev.offset(done), cur.bytes, take));
            cur.bytes += take;
            done += take;
            if cur.bytes == cap {
                out.push(std::mem::take(&mut cur));
            }
        }
    }
    if cur.bytes > 0 {
        out.push(cur);
    }
    Ok(out)
}

fn chunk_entries(plan: &ChunkPlan, base: usize) -> Vec<CopyEntry> {
    plan.entries
        .iter()
        .map(|&(dev, off, bytes)| CopyEntry {
            dev,
            host: base + off,
            bytes,
        })
        .collect()
}

/// Enqueue the D2H copies of one chunk into `host[..plan.bytes]`. Same contract as
/// [`gather_to_host`] (no sync; `host` untouched until the next sync point).
pub fn gather_chunk(
    gpu: &dyn GpuBackend,
    plan: &ChunkPlan,
    host: &mut [u8],
    stream: u64,
) -> Result<()> {
    if host.len() < plan.bytes {
        bail!(
            "gather_chunk: host buffer {} B < chunk {} B",
            host.len(),
            plan.bytes
        );
    }
    submit_d2h(
        gpu,
        &chunk_entries(plan, host.as_mut_ptr() as usize),
        stream,
    )
}

/// Enqueue the H2D copies of one chunk from `host[..plan.bytes]`. Same contract as
/// [`scatter_from_host`].
pub fn scatter_chunk(
    gpu: &dyn GpuBackend,
    plan: &ChunkPlan,
    host: &[u8],
    stream: u64,
    src_any_order: bool,
) -> Result<()> {
    if host.len() < plan.bytes {
        bail!(
            "scatter_chunk: host buffer {} B < chunk {} B",
            host.len(),
            plan.bytes
        );
    }
    let entries = chunk_entries(plan, host.as_ptr() as usize);
    submit_h2d(gpu, &entries, stream, src_any_order)
}

#[cfg(test)]
#[path = "block_io_tests.rs"]
mod tests;
