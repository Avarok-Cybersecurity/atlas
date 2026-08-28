// SPDX-License-Identifier: AGPL-3.0-only

//! Batched NVMe fault-in for [`super::ngram_cache::NgramRowCache`] — split
//! out of `ngram_cache.rs` (500-LoC cap).
//!
//! `resolve` used to issue one blocking `read_exact_at` PER MISS while
//! holding the table mutex (QD=1), so a diverse prefill paid
//! `misses x NVMe latency` serially — the stall the per-gather stats were
//! logged to catch. The two-phase resolve assigns every miss its slot first
//! (pure bookkeeping), then this module faults all of them with a bounded
//! thread pool of positional reads: `read_at(&File)` is thread-safe, each
//! job's destination slot region is disjoint, and each worker carries its
//! own 4 KiB-aligned bounce, so the only shared state is the atomic work
//! index. Decode-scale batches (a few misses) keep a serial arm — thread
//! spawn would cost more than it saves.

use std::fs::File;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};

use super::ngram_cache::{AlignedBlock, BLOCK};

/// Fan out to threads only past this many misses: below it, spawn cost
/// rivals the reads themselves (decode gathers are 16 ids, 0-2 misses).
const PARALLEL_MIN: usize = 4;

/// Cap on fault workers. NVMe queue depth benefits flatten out well below
/// this; the reads are 4-8 KiB each.
const MAX_WORKERS: usize = 16;

/// One miss, fully resolved to byte offsets — no `&self` reaches the
/// workers.
pub(super) struct FaultJob {
    pub(super) row_id: u64,
    pub(super) slot: u32,
    /// 4 KiB-aligned file offset of the row's containing block(s).
    pub(super) block_off: u64,
    /// Row start within the block window.
    pub(super) within: usize,
    /// 1, or 2 when the row straddles a block boundary.
    pub(super) nblocks: usize,
    /// Destination (pinned arena slot bytes) as an address; the region
    /// `[dst, dst + row_stride)` is disjoint per job.
    pub(super) dst: usize,
    /// FP8 tables: `(scale_block_off, scale_within, scale_dst)`.
    pub(super) scale: Option<(u64, usize, usize)>,
}

// SAFETY: `dst`/`scale.2` are raw addresses into the pinned arena; each
// job's region is disjoint and the arena outlives the scoped threads.
unsafe impl Send for FaultJob {}
unsafe impl Sync for FaultJob {}

fn run_one(
    job: &FaultJob,
    file: &File,
    scale_file: Option<&File>,
    row_stride: usize,
    bounce: &mut AlignedBlock,
) -> Result<()> {
    atlas_tier::pio::read_exact_at(file, bounce.blocks(job.nblocks), job.block_off)
        .with_context(|| format!("NgramRowCache: read row {}", job.row_id))?;
    // SAFETY: disjoint per-job region inside the live pinned arena.
    let dst = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, row_stride) };
    dst.copy_from_slice(&bounce.blocks(job.nblocks)[job.within..job.within + row_stride]);

    if let Some((sblock, swithin, sdst)) = job.scale {
        let sf = scale_file.expect("scale job without scale file");
        atlas_tier::pio::read_exact_at(sf, bounce.blocks(1), sblock)
            .with_context(|| format!("NgramRowCache: read scale {}", job.row_id))?;
        // SAFETY: disjoint 4-byte per-job region in the scale arena.
        let sdst = unsafe { std::slice::from_raw_parts_mut(sdst as *mut u8, 4) };
        sdst.copy_from_slice(&bounce.blocks(1)[swithin..swithin + 4]);
    }
    Ok(())
}

/// `ATLAS_PLE_SERIAL_FAULT=1`: keep the pre-parallel QD=1 arm, so the
/// speedup can be measured against the thing it replaced rather than
/// asserted (same A/B convention as `ATLAS_HC_DECODE_SPLIT`).
fn serial_forced() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("ATLAS_PLE_SERIAL_FAULT").as_deref() == Ok("1"))
}

/// Fault every job in, serial below [`PARALLEL_MIN`], scoped threads above.
pub(super) fn fault_all(
    jobs: &[FaultJob],
    file: &File,
    scale_file: Option<&File>,
    row_stride: usize,
    bounce: &mut AlignedBlock,
) -> Result<()> {
    if jobs.len() < PARALLEL_MIN || serial_forced() {
        for job in jobs {
            run_one(job, file, scale_file, row_stride, bounce)?;
        }
        return Ok(());
    }
    let next = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let workers = jobs.len().min(MAX_WORKERS);
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                let mut bounce = AlignedBlock::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    if let Err(e) = run_one(job, file, scale_file, row_stride, &mut bounce) {
                        *first_err.lock().unwrap() = Some(e);
                        break;
                    }
                    if first_err.lock().unwrap().is_some() {
                        break;
                    }
                }
            });
        }
    });
    match first_err.into_inner().unwrap() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Sanity used by the job builder: a row never needs more than two blocks.
pub(super) fn nblocks_for(within: usize, row_stride: usize) -> usize {
    if within + row_stride > BLOCK { 2 } else { 1 }
}
