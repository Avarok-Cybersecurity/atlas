// SPDX-License-Identifier: AGPL-3.0-only
//
// Storage backend trait + impls for the high-speed-swap path.
//
// SBIO contract: tiled-attention / scratch-pool code never opens a file or
// issues a syscall. Every NVMe-touching operation flows through a
// `StorageBackend` impl, so the predictor / scratch / kernel layers can be
// tested with the deterministic POSIX backend and swap in the io_uring
// production backend transparently.

use anyhow::Result;

use crate::group::{GroupKey, GroupLayout, KvKind};

pub mod io_uring;
pub mod posix;

pub use self::io_uring::IoUringBackend;
pub use posix::PosixBackend;

/// One read request: pull `group` from disk, land it at `dst_dev_ptr`.
#[derive(Clone, Copy, Debug)]
pub struct ReadRequest {
    pub group: GroupKey,
    pub dst_dev_ptr: u64,
}

/// One block-granular read: land the whole block (all kv-heads' K then V,
/// `block_bytes`) into the device slot based at `dst_dev_ptr`
/// (== `ScratchPool::slot_dev_ptr(slot)`). `base_key` carries `kv_head = 0,
/// kind = K` by convention; only its `layer` and `block` are load-bearing.
#[derive(Clone, Copy, Debug)]
pub struct BlockReadRequest {
    pub base_key: GroupKey,
    pub dst_dev_ptr: u64,
}

/// Expand each `BlockReadRequest` into the exact `2·nkv` per-head `ReadRequest`s
/// the un-coalesced path issues, in the SAME order the caller loops emit
/// (interleaved `K(kh), V(kh)` for `kh` in `0..nkv`) with device destinations
/// at `dst + kh·gs` (K) and `dst + (nkv+kh)·gs` (V).
///
/// This is the SINGLE source of the per-head fan-out: the default `read_blocks`
/// / `write_block_from_host` trait impls AND the unit tests consume it, so the
/// RDMA/Cascade backends (which inherit the default) can never drift from the
/// caller-side per-head layout, and byte-identity is pinned host-side.
pub fn expand_blocks_to_groups(spec: &GroupLayout, reqs: &[BlockReadRequest]) -> Vec<ReadRequest> {
    let nkv = spec.num_kv_heads;
    let gs = spec.group_stride;
    let mut out = Vec::with_capacity(reqs.len() * 2 * nkv as usize);
    for r in reqs {
        let layer = r.base_key.layer;
        let block = r.base_key.block;
        for kh in 0..nkv {
            out.push(ReadRequest {
                group: GroupKey::new(layer, block, kh, KvKind::K),
                dst_dev_ptr: r.dst_dev_ptr + (kh as u64) * gs,
            });
            out.push(ReadRequest {
                group: GroupKey::new(layer, block, kh, KvKind::V),
                dst_dev_ptr: r.dst_dev_ptr + (nkv as u64 + kh as u64) * gs,
            });
        }
    }
    out
}

pub trait StorageBackend: Send + Sync {
    /// Synchronously fulfil all `requests`, returning when the corresponding
    /// HBM destinations are populated and visible on `stream`. The backend
    /// chooses how to schedule (blocking POSIX `pread`, batched `io_uring`,
    /// etc.). At return, the `stream` has been synchronised so the caller
    /// can issue subsequent kernels that depend on the data.
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()>;

    /// Async variant of `read`: enqueue the tier read + H2D on `stream` and
    /// return WITHOUT a terminal host `stream_sync`. Default = the synchronous
    /// `read`, so file backends need no change and the on-demand path stays
    /// byte-identical.
    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read(requests, stream)
    }

    /// One-shot sequential write — used at offload time to populate disk
    /// from a host-side K/V buffer.
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()>;

    /// Immutable disk/device geometry. The default block methods below use it to
    /// fan a block op back out to the per-head path; io_uring/posix return their
    /// layout spec, Cascade delegates to its backing, RDMA returns its layout.
    fn group_layout(&self) -> GroupLayout;

    /// Block-granular read: fulfil each request with ONE contiguous
    /// `block_bytes` op instead of `2·nkv` per-head reads. Same stream contract
    /// as `read`. The DEFAULT fans out to `read` via `expand_blocks_to_groups`,
    /// so posix/RDMA/Cascade stay correct (just un-coalesced) with no change.
    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let groups = expand_blocks_to_groups(&self.group_layout(), requests);
        self.read(&groups, stream)
    }

    /// Async block-granular read — the coalesced twin of `read_async` for the
    /// prefetch path. DEFAULT fans out to `read_async`.
    fn read_blocks_async(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let groups = expand_blocks_to_groups(&self.group_layout(), requests);
        self.read_async(&groups, stream)
    }

    /// Block-granular write: ONE contiguous `block_bytes` op. `src` is exactly
    /// `block_bytes` laid out `[K0,K1,…,K(nkv-1),V0,…,V(nkv-1)]` at `group_stride`
    /// pitch. `base_key` carries the block identity (kv_head/kind ignored).
    /// DEFAULT splits `src` back into the `2·nkv` per-head `group_stride` stripes
    /// and calls `write_from_host` per head — byte-identical on-disk image.
    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let spec = self.group_layout();
        let nkv = spec.num_kv_heads as usize;
        let gs = spec.group_stride as usize;
        let expect = 2 * nkv * gs;
        if src.len() != expect {
            anyhow::bail!(
                "write_block_from_host: src len {} != block bytes {expect}",
                src.len()
            );
        }
        let layer = base_key.layer;
        let block = base_key.block;
        for kh in 0..nkv {
            let k_off = kh * gs;
            let v_off = (nkv + kh) * gs;
            self.write_from_host(
                GroupKey::new(layer, block, kh as u16, KvKind::K),
                &src[k_off..k_off + gs],
            )?;
            self.write_from_host(
                GroupKey::new(layer, block, kh as u16, KvKind::V),
                &src[v_off..v_off + gs],
            )?;
        }
        Ok(())
    }

    /// Write a run of `run_len` strictly-consecutive same-layer blocks in ONE
    /// contiguous op. `base_key` carries the run's FIRST block; `src` is exactly
    /// `run_len · block_bytes`. DEFAULT fans out to `run_len`
    /// `write_block_from_host` calls — byte- AND op-identical to the
    /// un-coalesced path (and `run_len == 1` is exactly one call).
    fn write_blocks_run(&mut self, base_key: GroupKey, run_len: usize, src: &[u8]) -> Result<()> {
        let spec = self.group_layout();
        let block_bytes = spec.block_bytes() as usize;
        let expect = run_len * block_bytes;
        if src.len() != expect {
            anyhow::bail!(
                "write_blocks_run: src len {} != run bytes {expect} ({run_len} × {block_bytes})",
                src.len()
            );
        }
        for i in 0..run_len {
            let off = i * block_bytes;
            self.write_block_from_host(
                GroupKey::new(base_key.layer, base_key.block + i as u32, 0, KvKind::K),
                &src[off..off + block_bytes],
            )?;
        }
        Ok(())
    }

    /// Whether this backend can service `write_blocks_run` as a single wide op.
    /// DEFAULT `false`: RDMA/Cascade keep the per-block fan-out, and the caller
    /// stays on the per-block write path.
    fn supports_write_run_coalescing(&self) -> bool {
        false
    }

    /// Optionally pre-register `[base, base+len)` as the read-landing region.
    /// The RDMA backend registers it as ONE MR (per rail) so zero-copy restore
    /// reuses that lkey for every slot within it. No-op for the file backends.
    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        let _ = (base, len);
        Ok(())
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod coalesce_tests;
