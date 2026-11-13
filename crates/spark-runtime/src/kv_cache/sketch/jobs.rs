// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

//! Responsibility: what the sketch kernels are asked to recompute: the host-planned page
//! jobs of an eager (prefill) update ([`plan_jobs`]) and the shape of a graph-safe
//! decode/verify update ([`TailBatch`], [`tail_pages_span`]). Pure host values; the
//! launches are in the parent module.

use crate::gpu::DevicePtr;

/// One page to recompute: its physical id (or -1 to resolve it on the device from the
/// stream's block table), the stream it belongs to, its index in the stream's page list
/// and how many of its tokens are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SketchJob {
    pub block: i32,
    pub stream: u32,
    pub logical_page: u32,
    pub fill: u32,
}

/// Rows one stream just wrote: positions `first_pos .. first_pos + n_new` are now stored,
/// and nothing past them. `host_table` is the stream's page list when the caller has it.
#[derive(Clone, Copy, Debug)]
pub struct WrittenRange<'a> {
    pub first_pos: u32,
    pub n_new: u32,
    pub host_table: Option<&'a [u32]>,
}

/// The page jobs that bring the sketch up to date after `ranges` were written, and the
/// number of written rows no job can cover (no host table and `device_tables` false, or a
/// page past the end of the host table). Every page any written row lands in is
/// recomputed with `fill = min(block_size, end - page_start)`.
pub fn plan_jobs(
    ranges: &[WrittenRange],
    block_size: usize,
    device_tables: bool,
) -> (Vec<SketchJob>, usize) {
    let bs = block_size.max(1) as u64;
    let mut jobs = Vec::new();
    let mut uncovered = 0usize;
    for (stream, r) in ranges.iter().enumerate() {
        if r.n_new == 0 {
            continue;
        }
        let end = u64::from(r.first_pos) + u64::from(r.n_new);
        for lp in u64::from(r.first_pos) / bs..=(end - 1) / bs {
            let fill = (end - lp * bs).min(bs) as u32;
            let rows_here = {
                let lo = (lp * bs).max(u64::from(r.first_pos));
                let hi = ((lp + 1) * bs).min(end);
                (hi - lo) as usize
            };
            let block = match r.host_table {
                Some(t) => match t.get(lp as usize).and_then(|&b| i32::try_from(b).ok()) {
                    Some(b) => b,
                    None => {
                        uncovered += rows_here;
                        continue;
                    }
                },
                None if device_tables => -1,
                None => {
                    uncovered += rows_here;
                    continue;
                }
            };
            jobs.push(SketchJob {
                block,
                stream: stream as u32,
                logical_page: lp as u32,
                fill,
            });
        }
    }
    (jobs, uncovered)
}

/// The graph-safe decode/verify update: `n_rows` attention rows, `rows_per_seq` per
/// sequence (consecutive, the last one carrying the sequence's full extent in `seq_lens`),
/// each sequence's last `n_new` tokens just written.
#[derive(Clone, Copy, Debug)]
pub struct TailBatch {
    /// `[n_rows * max_blocks_per_seq]` i32 (the step's real table, not a view).
    pub block_table: DevicePtr,
    /// `[n_rows]` i32 extents, this step's tokens included (0 = padding row).
    pub seq_lens: DevicePtr,
    pub max_blocks_per_seq: u32,
    pub n_rows: u32,
    pub rows_per_seq: u32,
    pub n_new: u32,
}

/// Pages `n_new` consecutive tokens can touch: `ceil((n_new - 1) / block_size) + 1`.
pub fn tail_pages_span(n_new: u32, block_size: u32) -> u32 {
    if n_new == 0 {
        return 0;
    }
    (n_new - 1).div_ceil(block_size.max(1)) + 1
}
