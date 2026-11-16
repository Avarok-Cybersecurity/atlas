// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: the bounded view's fixed device buffers (port lane L32) and
//! the launch of `kv_view_build_tables` (`kernels/ada/common/kv_view_table_ada.cu`).
//!
//! One allocation, made outside any graph capture at the first prefill call
//! after the model build, holds:
//! - the view table arena and the view lengths every decode and verify plan
//!   returns (the same pointers on every step: the graph rule).
//!   The kernel rebuilds them from the real tables on every launch,
//!   so a replayed graph always reads this step's view;
//! - the descriptor table and its page pool, written by the host only between
//!   steps, on the decode stream, so a replay never reads a torn entry;
//! - the prefill view table (host-composed, uploaded per chunk on the prefill
//!   stream) and the score scratch.
//!
//! The kernel is Ada-only (an `[kernels] overrides` file). Where the module
//! is absent the view turns itself off with one warning and every plan is
//! `Dense`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::core_api::Geom;
use super::view::{DESC_WORDS, Desc};

/// The kernel's module (file stem) and entry point.
pub const KERNEL_MODULE: &str = "kv_view_table_ada";
pub const KERNEL_FN: &str = "kv_view_build_tables";

/// Views alive at once (one per sequence with a selection).
pub const DESC_ENTRIES: usize = 64;
/// View table arena, in i32 entries (16 MiB): rows x stride of the largest
/// batched step, or the whole row of a single-sequence step.
pub const OUT_TABLE_ENTRIES: usize = 4 << 20;
/// View length rows (decode rows times verify rows per sequence).
pub const OUT_ROWS: usize = 4096;
/// Threads per row of the kernel.
const THREADS: u32 = 128;
/// Sub-buffer alignment.
const ALIGN: usize = 256;
/// Prefill table and score scratch sizes the reservation assumes (the actual
/// sizes follow the model's longest sequence and are logged).
const PREFILL_ENTRIES_ESTIMATE: usize = 1 << 18;

fn align(v: usize) -> usize {
    v.div_ceil(ALIGN) * ALIGN
}

/// Byte layout of the one allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    out_table: usize,
    out_lens: usize,
    desc: usize,
    sel_pool: usize,
    prefill: usize,
    scores: usize,
    total: usize,
}

fn layout(sel_cap: u32, prefill_entries: usize, score_cap: usize) -> Layout {
    let mut at = 0usize;
    let mut take = |bytes: usize| {
        let o = at;
        at += align(bytes);
        o
    };
    let out_table = take(OUT_TABLE_ENTRIES * 4);
    let out_lens = take(OUT_ROWS * 4);
    let desc = take(DESC_ENTRIES * DESC_WORDS * 4);
    let sel_pool = take(DESC_ENTRIES * sel_cap as usize * 4);
    let prefill = take(prefill_entries * 4);
    let scores = take(score_cap * 4);
    Layout {
        out_table,
        out_lens,
        desc,
        sel_pool,
        prefill,
        scores,
        total: at,
    }
}

/// Bytes of the fixed buffers for a page pool of `sel_cap` pages per view,
/// at the sizes the reservation assumes.
pub fn fixed_bytes(sel_cap: u32) -> u64 {
    layout(sel_cap, PREFILL_ENTRIES_ESTIMATE, PREFILL_ENTRIES_ESTIMATE).total as u64
}

fn i32_bytes(v: impl IntoIterator<Item = i32>) -> Vec<u8> {
    v.into_iter().flat_map(i32::to_le_bytes).collect()
}

/// The view's device buffers on one backend.
pub struct DeviceState {
    /// Address of the backend the buffers live on.
    pub owner: usize,
    kernel: KernelHandle,
    base: DevicePtr,
    l: Layout,
    sel_cap: u32,
    prefill_entries: usize,
    score_cap: usize,
}

/// The identity of a backend (buffers are never reused across backends).
pub fn backend_key(gpu: &dyn GpuBackend) -> usize {
    std::ptr::from_ref(gpu).cast::<()>() as usize
}

impl DeviceState {
    /// Allocate on `gpu`. `Ok(None)` when the kernel module is not in this
    /// build (every target but Ada).
    pub fn new(gpu: &dyn GpuBackend, geom: &Geom, sel_cap: u32) -> Result<Option<Self>> {
        let kernel = crate::layers::try_target_kernel(gpu, KERNEL_MODULE, KERNEL_FN);
        if kernel.0 == 0 {
            return Ok(None);
        }
        let prefill_entries = geom.max_blocks_per_seq.max(1) as usize;
        let score_cap = prefill_entries.div_ceil(geom.block_pages() as usize).max(1);
        let l = layout(sel_cap, prefill_entries, score_cap);
        let base = gpu.alloc(l.total)?;
        // Every descriptor starts inactive.
        gpu.memset(base.offset(l.desc), 0, DESC_ENTRIES * DESC_WORDS * 4)?;
        Ok(Some(Self {
            owner: backend_key(gpu),
            kernel,
            base,
            l,
            sel_cap,
            prefill_entries,
            score_cap,
        }))
    }

    /// Bytes allocated.
    pub fn bytes(&self) -> u64 {
        self.l.total as u64
    }

    /// Free the buffers (only on the backend that allocated them).
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if backend_key(gpu) == self.owner {
            gpu.free(self.base)?;
        }
        Ok(())
    }

    /// Pages one view may select before its tail.
    pub fn sel_cap(&self) -> u32 {
        self.sel_cap
    }

    /// The score scratch (`score_cap` f32).
    pub fn scores(&self) -> (DevicePtr, usize) {
        (self.base.offset(self.l.scores), self.score_cap)
    }

    /// Write descriptor `e` (or clear it with `None`) on `stream`. The entry
    /// is marked inactive first and activated last, so a launch between the
    /// copies (same stream: none can run) would see a whole entry or none.
    pub fn write_entry(
        &self,
        gpu: &dyn GpuBackend,
        e: usize,
        desc: Option<&Desc>,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(e < DESC_ENTRIES, "kv view: descriptor {e} out of range");
        let header = self.base.offset(self.l.desc + e * DESC_WORDS * 4);
        gpu.copy_h2d_async(&i32_bytes([0i32; DESC_WORDS]), header, stream)?;
        let Some(d) = desc else {
            return Ok(());
        };
        let pages = &d.shape.sel_pages;
        anyhow::ensure!(
            pages.len() <= self.sel_cap as usize,
            "kv view: {} selected pages exceed the {}-page pool",
            pages.len(),
            self.sel_cap
        );
        if !pages.is_empty() {
            let pool = self
                .base
                .offset(self.l.sel_pool + e * self.sel_cap as usize * 4);
            gpu.copy_h2d_async(&i32_bytes(pages.iter().map(|&p| p as i32)), pool, stream)?;
        }
        gpu.copy_h2d_async(&i32_bytes(d.words()), header, stream)
    }

    /// Build this step's view tables for the rows of a decode or verify call
    /// (`real_table` `[n * stride]`, `real_lens` `[n]`). `None` when the
    /// shape does not fit the arena (the caller plans `Dense` for it).
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        real_table: DevicePtr,
        real_lens: DevicePtr,
        n: usize,
        stride: u32,
        block_size: u32,
        stream: u64,
    ) -> Result<Option<(DevicePtr, DevicePtr)>> {
        let Some(row_cap) = row_capacity(n, stride) else {
            return Ok(None);
        };
        if real_table.is_null() || real_lens.is_null() {
            return Ok(None);
        }
        let out_table = self.base.offset(self.l.out_table);
        let out_lens = self.base.offset(self.l.out_lens);
        KernelLaunch::new(gpu, self.kernel)
            .grid([n as u32, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(real_table)
            .arg_ptr(real_lens)
            .arg_u32(n as u32)
            .arg_u32(stride)
            .arg_u32(row_cap)
            .arg_u32(block_size)
            .arg_ptr(self.base.offset(self.l.desc))
            .arg_u32(DESC_ENTRIES as u32)
            .arg_ptr(self.base.offset(self.l.sel_pool))
            .arg_u32(self.sel_cap)
            .arg_ptr(out_table)
            .arg_ptr(out_lens)
            .launch(stream)?;
        Ok(Some((out_table, out_lens)))
    }

    /// Upload a host-composed prefill view table on `stream`. `None` when it
    /// is longer than the buffer.
    pub fn upload_prefill(
        &self,
        gpu: &dyn GpuBackend,
        table: &[u32],
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        if table.is_empty() || table.len() > self.prefill_entries {
            return Ok(None);
        }
        let dst = self.base.offset(self.l.prefill);
        gpu.copy_h2d_async(&i32_bytes(table.iter().map(|&p| p as i32)), dst, stream)?;
        Ok(Some(dst))
    }
}

/// Entries a row may hold in the arena: the whole arena for one row (the
/// single-sequence decode replays one graph for any length, so its stride is
/// not a bound), the stride for a batched step. `None` when the step does not
/// fit.
pub fn row_capacity(n: usize, stride: u32) -> Option<u32> {
    if n == 0 || stride == 0 || n > OUT_ROWS {
        return None;
    }
    if n == 1 {
        return Some(OUT_TABLE_ENTRIES as u32);
    }
    (n * stride as usize <= OUT_TABLE_ENTRIES).then_some(stride)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_aligned_and_disjoint() {
        let l = layout(1032, 16385, 2049);
        let starts = [
            l.out_table,
            l.out_lens,
            l.desc,
            l.sel_pool,
            l.prefill,
            l.scores,
        ];
        assert!(starts.iter().all(|s| s % ALIGN == 0));
        assert!(starts.windows(2).all(|w| w[0] < w[1]));
        assert!(l.scores + 2049 * 4 <= l.total);
        assert!(fixed_bytes(1032) >= (OUT_TABLE_ENTRIES * 4) as u64);
    }

    #[test]
    fn row_capacity_bounds_each_shape() {
        assert_eq!(row_capacity(0, 10), None);
        assert_eq!(row_capacity(1, 3), Some(OUT_TABLE_ENTRIES as u32));
        assert_eq!(row_capacity(8, 16385), Some(16385));
        assert_eq!(row_capacity(OUT_ROWS + 1, 1), None);
        assert_eq!(row_capacity(512, 16385), None);
    }
}
