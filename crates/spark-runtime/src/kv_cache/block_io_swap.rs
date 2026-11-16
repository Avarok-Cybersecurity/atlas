// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache csrc/cuda/mp_mem_kernels.cu:251-345 (staged gather, then DMA in plan steps)
// and csrc/cuda/phase_timing_recorder.cu (per-phase GB/s), commit 3aefae1, Apache-2.0; no code
// copied. Changes: a two-buffer host staging that overlaps the copy of chunk i with the file write
// (or read) of chunk i-1, behind spark-model's `kv_batched_swap` row, by Atlas contributors.

//! Responsibility: stream a sequence's swap image (KV blocks, then SSM state) between the device
//! and a `Write`/`Read` through ONE reusable page-locked staging buffer, with batched copies.
//!
//! The image is an ordered list of device regions; the file is their bytes back to back, exactly
//! the bytes the per-region path (`sequence/state_io.rs`, row off) writes, so either reader reads
//! either writer's file. The staging is two halves: chunk `i` is gathered into half `i % 2` and
//! an event is recorded; the file write of chunk `i - 1` waits only for ITS event, so the copy
//! engine and the disk run side by side. Restore mirrors it: half `i % 2` is reused only after the
//! scatter that last read it completed. Without events (a backend that cannot create them) every
//! wait is a full `synchronize`, still correct, just not overlapped.
//!
//! Also owns the `atlas_kv_swap_*` series and the `AVAROK_KV_SWAP_TIMING=1` log line, which the
//! per-region path reports into as well, so the two are compared on the same scale.

use std::io::{Read, Write};
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use parking_lot::{Mutex, MutexGuard};

use super::{ChunkPlan, gather_chunk, plan_chunks, scatter_chunk};
use crate::gpu::{DevicePtr, GpuBackend};

/// The `atlas_kv_swap_*` series and the timing log line, split out to keep this file under the cap.
#[path = "block_io_swap_metrics.rs"]
mod metrics;
pub use metrics::{
    SwapDir, format_report, log_report, record, render_prometheus, timing_enabled, totals,
};

/// Smallest staging half: below this the per-chunk event and write overhead dominates.
const MIN_HALF_BYTES: usize = 1 << 20;
const ALIGN: usize = 4096;

/// The staging half for an image of `image_bytes` when the pool chunk is `chunk_bytes`: the image
/// rounded up to 1 MiB, clamped to `[1 MiB, chunk_bytes / 2]`. The whole staging (two halves) is
/// therefore at most one pool chunk (64 MiB by default, `--pinned-pool-chunk-mb`).
pub fn staging_half_bytes(image_bytes: usize, chunk_bytes: usize) -> usize {
    let cap = (chunk_bytes / 2).max(MIN_HALF_BYTES);
    image_bytes
        .div_ceil(MIN_HALF_BYTES)
        .saturating_mul(MIN_HALF_BYTES)
        .clamp(MIN_HALF_BYTES, cap)
}

/// The staging allocation: `2 * half` bytes, page-locked when the backend allows.
struct Blob {
    ptr: *mut u8,
    half: usize,
    pinned: bool,
    /// One event per half, `None` when the backend could not create them.
    events: Option<[u64; 2]>,
}

// SAFETY: `ptr` is `alloc_host_pinned` memory (process-global, usable from any thread) or a heap
// allocation; it is reachable only through the `Mutex` in `SwapStaging`, so access is exclusive.
unsafe impl Send for Blob {}

/// The reusable swap staging buffer, allocated on first use and grown (never shrunk) to the
/// largest half requested. Held per model (`ModelPortState::kv_swap`) and released by
/// [`SwapStaging::free`] from the model's `Drop`; a model whose `kv_batched_swap` row is off never
/// allocates it.
#[derive(Default)]
pub struct SwapStaging {
    slot: Mutex<Option<Blob>>,
}

impl SwapStaging {
    /// Borrow the staging with halves of at least `half` bytes, allocating or growing it.
    pub fn acquire<'a>(&'a self, gpu: &dyn GpuBackend, half: usize) -> Result<StagingGuard<'a>> {
        if half == 0 {
            bail!("swap staging: zero-byte half requested");
        }
        let mut slot = self.slot.lock();
        if slot.as_ref().is_none_or(|b| b.half < half) {
            if let Some(old) = slot.take() {
                free_blob(gpu, old);
            }
            *slot = Some(alloc_blob(gpu, half.div_ceil(ALIGN) * ALIGN)?);
        }
        Ok(StagingGuard { slot })
    }

    /// Release the buffer and its events. No-op when it was never allocated.
    pub fn free(&self, gpu: &dyn GpuBackend) {
        if let Some(b) = self.slot.lock().take() {
            free_blob(gpu, b);
        }
    }

    /// Bytes currently held (both halves), 0 when not allocated.
    pub fn allocated_bytes(&self) -> usize {
        self.slot.lock().as_ref().map_or(0, |b| 2 * b.half)
    }
}

fn alloc_blob(gpu: &dyn GpuBackend, half: usize) -> Result<Blob> {
    let bytes = 2 * half;
    let (ptr, pinned) = match gpu.alloc_host_pinned(bytes) {
        Ok(p) if !p.is_null() => (p, true),
        other => {
            if let Err(e) = other {
                tracing::warn!(
                    "KV swap: could not page-lock a {bytes} B staging buffer ({e:#}); using heap \
                     memory (the swap stays correct but the copies bounce through the driver)"
                );
            }
            let layout = std::alloc::Layout::from_size_align(bytes, ALIGN)?;
            // SAFETY: `bytes > 0` (half > 0); freed by `free_blob` with the same layout. Zeroed so
            // a slice over it never covers uninitialized bytes.
            let p = unsafe { std::alloc::alloc_zeroed(layout) };
            if p.is_null() {
                bail!("KV swap: heap staging allocation of {bytes} B failed");
            }
            (p, false)
        }
    };
    let events = match (gpu.create_event(), gpu.create_event()) {
        (Ok(a), Ok(b)) => Some([a, b]),
        (a, b) => {
            for e in [a, b].into_iter().flatten() {
                let _ = gpu.destroy_event(e);
            }
            None
        }
    };
    Ok(Blob {
        ptr,
        half,
        pinned,
        events,
    })
}

fn free_blob(gpu: &dyn GpuBackend, b: Blob) {
    if let Some(ev) = b.events {
        for e in ev {
            let _ = gpu.destroy_event(e);
        }
    }
    if b.pinned {
        if let Err(e) = gpu.free_host_pinned(b.ptr, 2 * b.half) {
            tracing::warn!("KV swap: failed to free the pinned staging buffer: {e:#}");
        }
        return;
    }
    if let Ok(layout) = std::alloc::Layout::from_size_align(2 * b.half, ALIGN) {
        // SAFETY: allocated by `alloc_blob`'s heap arm with exactly this layout.
        unsafe { std::alloc::dealloc(b.ptr, layout) };
    }
}

/// Exclusive borrow of the staging. Dropping it releases the lock; every user synchronizes the
/// stream (or waits for both events) first, since in-flight copies reference these bytes.
pub struct StagingGuard<'a> {
    slot: MutexGuard<'a, Option<Blob>>,
}

impl StagingGuard<'_> {
    /// The two halves, the per-half events and `"pinned"` / `"heap"`.
    fn parts(&mut self) -> (&mut [u8], &mut [u8], Option<[u64; 2]>, &'static str) {
        let b = self
            .slot
            .as_mut()
            .expect("acquire installs the blob before returning the guard");
        // SAFETY: `b.ptr` owns `2 * b.half` initialized bytes and this guard is the only reference.
        let all = unsafe { std::slice::from_raw_parts_mut(b.ptr, 2 * b.half) };
        let (lo, hi) = all.split_at_mut(b.half);
        (lo, hi, b.events, if b.pinned { "pinned" } else { "heap" })
    }
}

/// What one image save or restore did, for the timing log and the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SwapReport {
    pub bytes: usize,
    pub regions: usize,
    pub chunks: usize,
    pub chunk_bytes: usize,
    /// Host time blocked on copy completion (events or stream syncs).
    pub wait_us: u64,
    /// Host time in file writes (save) or reads (restore).
    pub io_us: u64,
    pub total_us: u64,
    pub staging: &'static str,
}

impl SwapReport {
    /// Image bytes over the whole call's wall time, GB/s (1e9).
    pub fn gb_per_s(&self) -> f64 {
        if self.total_us == 0 {
            return 0.0;
        }
        self.bytes as f64 / (self.total_us as f64 * 1e3)
    }
}

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// Wait until the copies before `event` completed, or the whole stream without events.
fn wait(gpu: &dyn GpuBackend, event: Option<u64>, stream: u64) -> Result<()> {
    match event {
        Some(e) => gpu.event_synchronize(e),
        None => gpu.synchronize(stream),
    }
}

/// Save: gather `regions` (in order) and write their bytes to `writer`, then flush.
pub fn write_image(
    gpu: &dyn GpuBackend,
    regions: &[(DevicePtr, usize)],
    staging: &SwapStaging,
    writer: &mut dyn Write,
    stream: u64,
) -> Result<SwapReport> {
    let bytes: usize = regions.iter().map(|&(_, n)| n).sum();
    let half = staging_half_bytes(bytes, crate::pinned_pool::default_chunk_bytes());
    write_image_with(gpu, regions, staging, writer, stream, half)
}

/// [`write_image`] with an explicit staging half (tests force many chunks with a small one).
pub fn write_image_with(
    gpu: &dyn GpuBackend,
    regions: &[(DevicePtr, usize)],
    staging: &SwapStaging,
    writer: &mut dyn Write,
    stream: u64,
    half: usize,
) -> Result<SwapReport> {
    let t0 = Instant::now();
    let bytes: usize = regions.iter().map(|&(_, n)| n).sum();
    let mut rep = SwapReport {
        bytes,
        regions: regions.len(),
        staging: "none",
        ..SwapReport::default()
    };
    if bytes > 0 {
        let plans = plan_chunks(regions, half)?;
        let mut guard = staging.acquire(gpu, half)?;
        let (lo, hi, events, kind) = guard.parts();
        rep.staging = kind;
        rep.chunks = plans.len();
        rep.chunk_bytes = half;
        let mut bufs: [&mut [u8]; 2] = [lo, hi];
        let res = write_chunks(gpu, &plans, &mut bufs, events, writer, stream, &mut rep);
        if res.is_err() {
            // Nothing may still be copying into the staging once the guard is released.
            let _ = gpu.synchronize(stream);
        }
        res?;
    }
    let tf = Instant::now();
    writer.flush()?;
    rep.io_us += micros(tf.elapsed());
    rep.total_us = micros(t0.elapsed());
    record(SwapDir::Out, bytes, t0.elapsed());
    Ok(rep)
}

fn write_chunks(
    gpu: &dyn GpuBackend,
    plans: &[ChunkPlan],
    bufs: &mut [&mut [u8]; 2],
    events: Option<[u64; 2]>,
    writer: &mut dyn Write,
    stream: u64,
    rep: &mut SwapReport,
) -> Result<()> {
    let mut pending: Option<usize> = None;
    for (i, plan) in plans.iter().enumerate() {
        let slot = i % 2;
        // Half `slot` last held chunk i - 2, written out in the previous iteration.
        gather_chunk(gpu, plan, &mut *bufs[slot], stream)?;
        if let Some(ev) = events {
            gpu.record_event(ev[slot], stream)?;
        }
        if let Some(p) = pending {
            flush_chunk(
                gpu,
                &plans[p],
                &*bufs[p % 2],
                events.map(|e| e[p % 2]),
                writer,
                stream,
                rep,
            )?;
        }
        pending = Some(i);
    }
    if let Some(p) = pending {
        flush_chunk(
            gpu,
            &plans[p],
            &*bufs[p % 2],
            events.map(|e| e[p % 2]),
            writer,
            stream,
            rep,
        )?;
    }
    Ok(())
}

fn flush_chunk(
    gpu: &dyn GpuBackend,
    plan: &ChunkPlan,
    buf: &[u8],
    event: Option<u64>,
    writer: &mut dyn Write,
    stream: u64,
    rep: &mut SwapReport,
) -> Result<()> {
    let tw = Instant::now();
    wait(gpu, event, stream)?;
    rep.wait_us += micros(tw.elapsed());
    let tio = Instant::now();
    writer.write_all(&buf[..plan.bytes])?;
    rep.io_us += micros(tio.elapsed());
    Ok(())
}

/// Restore: read the bytes of `regions` (in order) from `reader` and scatter them to the device.
/// Returns once every copy completed (the stream is synchronized).
pub fn read_image(
    gpu: &dyn GpuBackend,
    regions: &[(DevicePtr, usize)],
    staging: &SwapStaging,
    reader: &mut dyn Read,
    stream: u64,
) -> Result<SwapReport> {
    let bytes: usize = regions.iter().map(|&(_, n)| n).sum();
    let half = staging_half_bytes(bytes, crate::pinned_pool::default_chunk_bytes());
    read_image_with(gpu, regions, staging, reader, stream, half)
}

/// [`read_image`] with an explicit staging half.
pub fn read_image_with(
    gpu: &dyn GpuBackend,
    regions: &[(DevicePtr, usize)],
    staging: &SwapStaging,
    reader: &mut dyn Read,
    stream: u64,
    half: usize,
) -> Result<SwapReport> {
    let t0 = Instant::now();
    let bytes: usize = regions.iter().map(|&(_, n)| n).sum();
    let mut rep = SwapReport {
        bytes,
        regions: regions.len(),
        staging: "none",
        ..SwapReport::default()
    };
    if bytes > 0 {
        let plans = plan_chunks(regions, half)?;
        let mut guard = staging.acquire(gpu, half)?;
        let (lo, hi, events, kind) = guard.parts();
        rep.staging = kind;
        rep.chunks = plans.len();
        rep.chunk_bytes = half;
        let mut bufs: [&mut [u8]; 2] = [lo, hi];
        let res = read_chunks(gpu, &plans, &mut bufs, events, reader, stream, &mut rep);
        let ts = Instant::now();
        let synced = gpu.synchronize(stream);
        rep.wait_us += micros(ts.elapsed());
        res?;
        synced?;
    }
    rep.total_us = micros(t0.elapsed());
    record(SwapDir::In, bytes, t0.elapsed());
    Ok(rep)
}

fn read_chunks(
    gpu: &dyn GpuBackend,
    plans: &[ChunkPlan],
    bufs: &mut [&mut [u8]; 2],
    events: Option<[u64; 2]>,
    reader: &mut dyn Read,
    stream: u64,
    rep: &mut SwapReport,
) -> Result<()> {
    for (i, plan) in plans.iter().enumerate() {
        let slot = i % 2;
        if i >= 2 {
            // The scatter of chunk i - 2 still reads this half until its event completes.
            let tw = Instant::now();
            wait(gpu, events.map(|e| e[slot]), stream)?;
            rep.wait_us += micros(tw.elapsed());
        }
        let tio = Instant::now();
        reader.read_exact(&mut bufs[slot][..plan.bytes])?;
        rep.io_us += micros(tio.elapsed());
        scatter_chunk(gpu, plan, &*bufs[slot], stream, false)?;
        if let Some(ev) = events {
            gpu.record_event(ev[slot], stream)?;
        }
    }
    Ok(())
}
