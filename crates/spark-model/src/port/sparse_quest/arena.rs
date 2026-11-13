// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the Quest selector's device memory (port lane L30): the
//! page-summary arena (sized and charged by `budget.rs`), the fixed View
//! arena (compacted block table, lengths, page scores and the two device
//! counters), the per-stream upload buffers for host block tables (chunk-0
//! prefill only), and which K pool each attention layer was first seen with.
//!
//! Graph rule (master plan 3.7.6): every pointer handed to a captured kernel
//! is fixed for the arena's lifetime. The View arena's regions sit at constant
//! offsets, independent of the step's shape, so two captures of the same shape
//! always see the same table and lengths. Allocation happens only on an eager
//! hook (never while a stream is capturing).
//!
//! An arena belongs to one backend and one summary generation. A model swap
//! builds a new backend (whose sweep frees the old allocations) and a new
//! generation; the next eager hook allocates afresh.

use std::collections::HashMap;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::page_summary::PageSummaryBuffers;

use super::budget::Geometry;
use super::launch::Kernels;

/// Rows a View can carry (lengths region).
pub const MAX_ROWS: usize = 4096;
const LENS_OFF: usize = 0;
const STATS_OFF: usize = MAX_ROWS * 4;
const SCORES_OFF: usize = 32 << 10;
/// Bytes of the page-score region (`[groups, score_stride]` f32).
pub const SCORES_BYTES: usize = 4 << 20;
const TABLE_OFF: usize = SCORES_OFF + SCORES_BYTES;
/// Bytes of the fixed View arena the selector reserves through the seam.
pub const VIEW_ARENA_BYTES: usize = 32 << 20;
/// Bytes of the compacted block-table region.
pub const TABLE_BYTES: usize = VIEW_ARENA_BYTES - TABLE_OFF;

/// Whether a step's shape fits the View arena.
pub fn view_fits(rows: usize, row_stride: usize, groups: usize, score_stride: usize) -> bool {
    rows <= MAX_ROWS
        && rows.saturating_mul(row_stride).saturating_mul(4) <= TABLE_BYTES
        && groups.saturating_mul(score_stride).saturating_mul(4) <= SCORES_BYTES
}

/// Address of the backend an arena was allocated on.
pub fn backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu).cast::<()>() as usize
}

/// The fixed regions of the View arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViewRegions {
    pub seq_lens: DevicePtr,
    pub stats: DevicePtr,
    pub scores: DevicePtr,
    pub block_table: DevicePtr,
}

impl ViewRegions {
    pub fn at(base: DevicePtr) -> Self {
        Self {
            seq_lens: base.offset(LENS_OFF),
            stats: base.offset(STATS_OFF),
            scores: base.offset(SCORES_OFF),
            block_table: base.offset(TABLE_OFF),
        }
    }
}

/// One backend's allocations.
pub struct Arena {
    pub owner: usize,
    pub generation: u64,
    pub summary: PageSummaryBuffers,
    pub views_base: DevicePtr,
    pub views: ViewRegions,
    pub kernels: Option<Kernels>,
    /// Per attention layer, the K pool first seen (a later call with another
    /// pool, such as a drafter's own cache, is left dense and untouched).
    pools: Vec<Option<DevicePtr>>,
    uploads: HashMap<u64, (DevicePtr, usize)>,
}

impl Arena {
    /// Allocate the summary and View arenas on `stream` (eager only).
    pub fn new(gpu: &dyn GpuBackend, geometry: &Geometry, stream: u64) -> Result<Self> {
        let summary = PageSummaryBuffers::alloc(gpu, geometry.layout.clone(), stream)?;
        let views_base = match gpu.alloc(VIEW_ARENA_BYTES) {
            Ok(p) => p,
            Err(e) => {
                let _ = summary.free(gpu);
                return Err(e);
            }
        };
        let views = ViewRegions::at(views_base);
        gpu.memset_async(views.stats, 0, 16, stream)?;
        let layers = geometry.layout.layers.len();
        Ok(Self {
            owner: backend_key(gpu),
            generation: geometry.generation,
            summary,
            views_base,
            views,
            kernels: Kernels::resolve(gpu),
            pools: vec![None; layers],
            uploads: HashMap::new(),
        })
    }

    /// True when this arena serves `gpu` at `generation`.
    pub fn serves(&self, gpu: &dyn GpuBackend, generation: u64) -> bool {
        self.owner == backend_key(gpu) && self.generation == generation
    }

    /// Record or check the K pool of an attention layer. False when the layer
    /// is outside the layout or was first seen with a different pool.
    pub fn claim_pool(&mut self, layer: usize, k_pool: DevicePtr) -> bool {
        let Some(slot) = self.pools.get_mut(layer) else {
            return false;
        };
        match *slot {
            None => {
                *slot = Some(k_pool);
                true
            }
            Some(p) => p == k_pool,
        }
    }

    /// Upload a host block table for an eager launch on `stream` and return its
    /// device copy. The buffer is per stream; a launch reading it is ordered
    /// before the next upload on the same stream. Growth waits for the stream.
    pub fn upload_table(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
        table: &[u32],
    ) -> Result<DevicePtr> {
        let bytes = (table.len() * 4).max(4);
        let entry = self.uploads.get(&stream).copied();
        let buf = match entry {
            Some((p, cap)) if cap >= bytes => p,
            other => {
                if let Some((old, _)) = other {
                    gpu.synchronize(stream)?;
                    gpu.free(old)?;
                }
                let cap = bytes.next_power_of_two().max(64 << 10);
                let p = gpu.alloc(cap)?;
                self.uploads.insert(stream, (p, cap));
                p
            }
        };
        let mut raw = Vec::with_capacity(bytes);
        for &b in table {
            raw.extend_from_slice(&(b as i32).to_le_bytes());
        }
        if raw.is_empty() {
            raw.extend_from_slice(&0i32.to_le_bytes());
        }
        gpu.copy_h2d_async(&raw, buf, stream)?;
        Ok(buf)
    }

    /// Free everything through the backend that allocated it, after `stream`
    /// drained (same backend, new generation).
    pub fn release(self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        gpu.synchronize(stream)?;
        for (_, (p, _)) in self.uploads {
            gpu.free(p)?;
        }
        gpu.free(self.views_base)?;
        self.summary.free(gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_are_disjoint_and_fixed() {
        let r = ViewRegions::at(DevicePtr(0x1000_0000));
        assert_eq!(r.seq_lens.0, 0x1000_0000);
        assert!(r.stats.0 >= r.seq_lens.0 + (MAX_ROWS * 4) as u64);
        assert!(r.scores.0 >= r.stats.0 + 16);
        assert_eq!(r.block_table.0 - r.scores.0, SCORES_BYTES as u64);
        assert_eq!(
            r.block_table.0 + TABLE_BYTES as u64,
            0x1000_0000 + VIEW_ARENA_BYTES as u64
        );
        assert_eq!(r.block_table.0 % 256, 0);
        assert_eq!(r.scores.0 % 256, 0);
    }

    #[test]
    fn shapes_up_to_a_262k_batch_of_eight_fit() {
        // 262,144 tokens = 16,384 blocks per row; C=8 x verify K=4 rows = 32 rows.
        assert!(view_fits(32, 16_384, 8, 4_096));
        assert!(view_fits(1, 65_536, 1, 16_384));
        assert!(!view_fits(MAX_ROWS + 1, 1, 1, 1));
        assert!(!view_fits(512, 16_384, 1, 1));
        assert!(!view_fits(1, 1, 2, SCORES_BYTES / 4));
    }
}
