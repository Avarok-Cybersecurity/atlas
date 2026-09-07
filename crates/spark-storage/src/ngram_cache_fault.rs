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

/// Cap on fault workers.
///
/// MEASURED 2026-09-06 on this box's NVMe against the real 36.4 GiB n-gram file
/// (`.research/exl3_decode_perf/ple_fault_bench.c` — 32768 random 4 KiB
/// O_DIRECT preads, the shape `run_one` issues; 3 repeats, wall ms):
///
/// ```text
///   workers    16     32     64     96    128
///   wall ms   332    222    147    138    138
///             334    222    147    138    139
///             335    224    148    137    137
/// ```
///
/// The knee is 64 and the curve is flat past 96 — the previous cap of 16 left
/// 2.3x on the table, and the comment it carried ("benefits flatten out well
/// below this") was an assumption this drive does not support. A real gather of
/// 31745 misses resolved in 288 ms at 16 workers.
const MAX_WORKERS: usize = 64;

/// Jobs a worker should have to justify its own spawn (~20 us) — below this,
/// threads would outnumber the reads they issue. 31745 misses -> 64 workers,
/// 100 -> 13, 32 -> 4.
const JOBS_PER_WORKER: usize = 8;

/// `ATLAS_PLE_FAULT_WORKERS=<n>` overrides the cap (clamped to 1..=256) so the
/// curve above can be re-measured in situ without a rebuild. Unparsable values
/// fall back to [`MAX_WORKERS`] with one warning.
fn max_workers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("ATLAS_PLE_FAULT_WORKERS") {
        Err(_) => MAX_WORKERS,
        Ok(v) => match v.trim().parse::<usize>() {
            Ok(n) if n >= 1 => n.min(256),
            _ => {
                tracing::warn!(
                    "ATLAS_PLE_FAULT_WORKERS={v:?} is not a worker count — using {MAX_WORKERS}"
                );
                MAX_WORKERS
            }
        },
    })
}

/// Workers for `jobs` misses: one per [`JOBS_PER_WORKER`], capped, never more
/// than there are jobs. Pure so the policy is testable without a drive.
pub(super) fn worker_count(jobs: usize, cap: usize) -> usize {
    jobs.div_ceil(JOBS_PER_WORKER)
        .clamp(1, cap)
        .min(jobs.max(1))
}

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
    /// Which backing file `block_off` is an offset INTO. A segmented table's
    /// shards can live in different safetensors files, so the offset alone
    /// named a byte in whichever file happened to be open.
    pub(super) file_idx: usize,
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
    // `read_at_least_at`, not `read_exact_at`: the block covering a row near the
    // tail of a shard runs past EOF (no safetensors file is block-aligned -- all
    // 21 shards of the shipped checkpoint end mid-block), and demanding the whole
    // block failed the request over padding that was never part of a row. The row
    // itself must arrive in full, which is what `within + row_stride` asserts.
    atlas_tier::pio::read_at_least_at(
        file,
        bounce.blocks(job.nblocks),
        job.block_off,
        job.within + row_stride,
    )
    .with_context(|| format!("NgramRowCache: read row {}", job.row_id))?;
    // SAFETY: disjoint per-job region inside the live pinned arena.
    let dst = unsafe { std::slice::from_raw_parts_mut(job.dst as *mut u8, row_stride) };
    dst.copy_from_slice(&bounce.blocks(job.nblocks)[job.within..job.within + row_stride]);

    if let Some((sblock, swithin, sdst)) = job.scale {
        let sf = scale_file.expect("scale job without scale file");
        // Same tail: a per-row scale near the end of its file sits in a block
        // that runs past EOF.
        atlas_tier::pio::read_at_least_at(sf, bounce.blocks(1), sblock, swithin + 4)
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
    files: &[&File],
    scale_file: Option<&File>,
    row_stride: usize,
    bounce: &mut AlignedBlock,
) -> Result<()> {
    // Indexed per job, not per call: two jobs in one batch can name rows in
    // different shards, and therefore in different files.
    let pick = |job: &FaultJob| -> Result<&File> {
        files.get(job.file_idx).copied().ok_or_else(|| {
            anyhow::anyhow!(
                "NgramRowCache: row {} names backing file {} of {}",
                job.row_id,
                job.file_idx,
                files.len()
            )
        })
    };
    if jobs.len() < PARALLEL_MIN || serial_forced() {
        for job in jobs {
            run_one(job, pick(job)?, scale_file, row_stride, bounce)?;
        }
        return Ok(());
    }
    let next = AtomicUsize::new(0);
    let first_err: Mutex<Option<anyhow::Error>> = Mutex::new(None);
    let workers = worker_count(jobs.len(), max_workers());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| {
                let mut bounce = AlignedBlock::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(i) else { break };
                    let f = match pick(job) {
                        Ok(f) => f,
                        Err(e) => {
                            *first_err.lock().unwrap() = Some(e);
                            break;
                        }
                    };
                    if let Err(e) = run_one(job, f, scale_file, row_stride, &mut bounce) {
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

#[cfg(test)]
mod tests {
    use super::{JOBS_PER_WORKER, MAX_WORKERS, worker_count};

    #[test]
    fn worker_count_scales_with_jobs_and_respects_the_cap() {
        // Never more workers than jobs — a worker with nothing to steal is
        // pure spawn cost.
        for jobs in 0..=JOBS_PER_WORKER {
            assert!(
                worker_count(jobs, MAX_WORKERS) <= jobs.max(1),
                "jobs={jobs}"
            );
        }
        // One worker per JOBS_PER_WORKER, rounded up.
        assert_eq!(worker_count(1, MAX_WORKERS), 1);
        assert_eq!(worker_count(JOBS_PER_WORKER, MAX_WORKERS), 1);
        assert_eq!(worker_count(JOBS_PER_WORKER + 1, MAX_WORKERS), 2);
        assert_eq!(
            worker_count(100, MAX_WORKERS),
            100_usize.div_ceil(JOBS_PER_WORKER)
        );
        // A real 8K-prefill gather saturates the cap (measured: 31745 misses).
        assert_eq!(worker_count(31745, MAX_WORKERS), MAX_WORKERS);
        // The cap is the measured knee, not the old assumption.
        assert_eq!(MAX_WORKERS, 64);
        // An operator override is honoured in both directions.
        assert_eq!(worker_count(31745, 16), 16);
        assert_eq!(worker_count(31745, 128), 128);
        // Zero jobs never reaches the pool (fault_all returns early), but the
        // policy must still be total.
        assert_eq!(worker_count(0, MAX_WORKERS), 1);
    }
}
