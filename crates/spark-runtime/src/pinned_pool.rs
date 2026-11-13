// SPDX-License-Identifier: AGPL-3.0-only
// Design after LMCache lmcache/v1/memory_allocators/lazy_memory_allocator.py:80-200 (reserve the
// whole range once, pin it in 64 MB chunks, grow in a background thread, serve from the first
// chunks at once), paged_tensor_memory_allocator.py (page-granular runs) and csrc/cuda/mem_alloc.cpp
// (page-aligned mmap plus host registration), commit 3aefae1, Apache-2.0; no code copied.
// Changes: Rust, `GpuBackend::host_register` (tier-owned in `pinned_hosts`), best-fit runs that
// never cross a pinning chunk, no hugepage path, by Atlas contributors.

//! Responsibility: a pool of page-locked host memory for the host KV tier and the SSM blob store
//! that starts serving at once and pins the rest of its budget in the background.
//!
//! * The whole budget is reserved once as one anonymous mapping (page-aligned, so 4096-aligned for
//!   a later O_DIRECT disk tier) and pinned in chunks (`--pinned-pool-chunk-mb`, 64 MiB default)
//!   through `GpuBackend::host_register`, which records each chunk in `pinned_hosts` as TIER-OWNED
//!   (tier DMA from it never takes the implicit sync of `copy_h2d_async`).
//! * `init` bytes are pinned before [`PinnedPool::new`] returns; a background thread, which first
//!   binds the backend's context (`cuMemHostRegister` needs a current context), pins the rest one
//!   chunk at a time. A refused pin (a WSL2 or `ulimit -l` cap) stops the growth with one log line;
//!   the pool keeps serving what it pinned.
//! * Allocation hands out page runs (multiples of 4096 bytes, at most one chunk) best-fit from a
//!   coalescing free list; only pinned bytes are ever handed out. The only state the pin thread
//!   shares is an atomic high-water mark and the mutex around the free list.
//! * `Drop` stops and joins the thread, unregisters every pinned chunk, then unmaps. The owner
//!   must have completed every DMA into or out of the pool first.
//!
//! The pool is inert until a lane builds one; with every port row off nothing constructs it.

use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use parking_lot::Mutex;

use crate::gpu::GpuBackend;

#[path = "pinned_pool_free.rs"]
mod free_list;
use free_list::FreeList;

/// Allocation granularity and alignment of every run.
pub const PAGE_BYTES: usize = 4096;
/// Default pinning chunk, MiB (`--pinned-pool-chunk-mb`).
pub const DEFAULT_CHUNK_MB: u32 = 64;
/// Largest accepted pinning chunk, MiB.
pub const MAX_CHUNK_MB: u32 = 4096;
const MIB: usize = 1 << 20;

static CHUNK_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_CHUNK_MB as usize * MIB);

/// Publish `--pinned-pool-chunk-mb` (from `port_apply`). `None` or an out-of-range value keeps
/// the current chunk (validation reports the bad value first, see [`validate_chunk_mb`]).
pub fn set_default_chunk_mb(mb: Option<u32>) {
    if let Some(mb) = mb
        && (1..=MAX_CHUNK_MB).contains(&mb)
    {
        CHUNK_BYTES.store(mb as usize * MIB, Ordering::Relaxed);
    }
}

/// The pinning chunk new pools use by default, bytes. Also bounds the swap staging.
pub fn default_chunk_bytes() -> usize {
    CHUNK_BYTES.load(Ordering::Relaxed)
}

/// Startup check for `--pinned-pool-chunk-mb`.
pub fn validate_chunk_mb(mb: Option<u32>) -> Result<()> {
    match mb {
        Some(v) if !(1..=MAX_CHUNK_MB).contains(&v) => {
            bail!("--pinned-pool-chunk-mb {v} is out of range (1..={MAX_CHUNK_MB})")
        }
        _ => Ok(()),
    }
}

// Process-wide gauges over every live pool (`atlas_pinned_pool_bytes{state}`).
static G_TOTAL: AtomicUsize = AtomicUsize::new(0);
static G_PINNED: AtomicUsize = AtomicUsize::new(0);
static G_USED: AtomicUsize = AtomicUsize::new(0);

/// Append `atlas_pinned_pool_bytes{state="total|pinned|used"}`.
pub fn render_prometheus(out: &mut String) {
    let (t, p, u) = (
        G_TOTAL.load(Ordering::Relaxed),
        G_PINNED.load(Ordering::Relaxed),
        G_USED.load(Ordering::Relaxed),
    );
    let _ = writeln!(
        out,
        "# HELP atlas_pinned_pool_bytes Page-locked host pool bytes over every live pool, by \
         state: total (reserved range), pinned (page-locked so far), used (handed out as runs).\n\
         # TYPE atlas_pinned_pool_bytes gauge\n\
         atlas_pinned_pool_bytes{{state=\"total\"}} {t}\n\
         atlas_pinned_pool_bytes{{state=\"pinned\"}} {p}\n\
         atlas_pinned_pool_bytes{{state=\"used\"}} {u}"
    );
}

/// One allocation: `bytes` (as requested) at host address `ptr`, page-aligned. Not `Clone`: the
/// holder owns the run until it hands it back with [`PinnedPool::free`]. The fields are public
/// for the tier's DMA descriptors; never build or duplicate a run by hand.
#[derive(Debug, PartialEq, Eq)]
pub struct PageRun {
    pub ptr: usize,
    pub bytes: usize,
    pub id: u64,
}

/// A snapshot of one pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    /// Reserved bytes (the budget, rounded up to whole chunks).
    pub total: usize,
    /// Page-locked bytes (the high-water mark).
    pub pinned: usize,
    /// Bytes handed out (runs rounded up to pages).
    pub used: usize,
    /// Pinned bytes not handed out.
    pub free: usize,
    pub runs: usize,
    pub largest_free: usize,
    pub chunk: usize,
    /// The pin thread is still running.
    pub growing: bool,
    /// A pin was refused; the pool stopped growing below `total`.
    pub pin_failed: bool,
    /// `alloc_run` calls that found no room.
    pub refusals: u64,
}

struct Inner {
    gpu: Arc<dyn GpuBackend>,
    base: usize,
    total: usize,
    chunk: usize,
    /// Pinned prefix `[0, pinned)` of the range. Written by one thread at a time.
    pinned: AtomicUsize,
    free: Mutex<FreeList>,
    stop: AtomicBool,
    growing: AtomicBool,
    pin_failed: AtomicBool,
}

impl Inner {
    /// Pin the next chunk. `Ok(false)` once the whole range is pinned.
    fn pin_next(&self) -> Result<bool> {
        let off = self.pinned.load(Ordering::Acquire);
        if off >= self.total {
            return Ok(false);
        }
        self.gpu.host_register(self.base + off, self.chunk)?;
        self.free.lock().add_chunk(off);
        self.pinned.store(off + self.chunk, Ordering::Release);
        G_PINNED.fetch_add(self.chunk, Ordering::Relaxed);
        Ok(true)
    }
}

/// The background half of [`PinnedPool::new`].
fn grow(inner: Arc<Inner>) {
    if let Err(e) = inner.gpu.bind_to_thread() {
        tracing::warn!("pinned pool: pin thread could not bind the device context ({e:#})");
        inner.pin_failed.store(true, Ordering::Relaxed);
        inner.growing.store(false, Ordering::Release);
        return;
    }
    while !inner.stop.load(Ordering::Acquire) {
        match inner.pin_next() {
            Ok(true) => {}
            Ok(false) => break,
            Err(e) => {
                tracing::warn!(
                    "pinned pool: stopped growing at {} of {} B ({e:#}); serving what is pinned",
                    inner.pinned.load(Ordering::Acquire),
                    inner.total
                );
                inner.pin_failed.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
    inner.growing.store(false, Ordering::Release);
}

/// A lazily pinned host memory pool (see the module doc).
pub struct PinnedPool {
    inner: Arc<Inner>,
    next_id: AtomicU64,
    used: AtomicUsize,
    refusals: AtomicU64,
    worker: Mutex<Option<JoinHandle<()>>>,
    // Dropped after `Drop::drop` unregistered every chunk.
    _map: memmap2::MmapMut,
}

impl PinnedPool {
    /// Reserve `total` bytes, pin `init` of them now and the rest in `chunk`-sized steps on a
    /// background thread. `chunk` is rounded up to a page, `total` up to whole chunks and `init`
    /// up to whole chunks (capped at `total`). The calling thread must have the backend's context
    /// current, as for any other backend call.
    pub fn new(
        gpu: Arc<dyn GpuBackend>,
        total: usize,
        init: usize,
        chunk: usize,
    ) -> Result<Arc<Self>> {
        if total == 0 {
            bail!("pinned pool: zero-byte budget");
        }
        let chunk = chunk.max(PAGE_BYTES).div_ceil(PAGE_BYTES) * PAGE_BYTES;
        let total = total
            .div_ceil(chunk)
            .checked_mul(chunk)
            .context("pinned pool: budget overflows")?;
        let init = init.div_ceil(chunk).saturating_mul(chunk).min(total);
        let mut map = memmap2::MmapMut::map_anon(total)
            .with_context(|| format!("pinned pool: reserving {total} B of address space"))?;
        let base = map.as_mut_ptr() as usize;
        let inner = Arc::new(Inner {
            gpu,
            base,
            total,
            chunk,
            pinned: AtomicUsize::new(0),
            free: Mutex::new(FreeList::new(chunk)),
            stop: AtomicBool::new(false),
            growing: AtomicBool::new(false),
            pin_failed: AtomicBool::new(false),
        });
        G_TOTAL.fetch_add(total, Ordering::Relaxed);
        // From here on `Drop` undoes everything, including on the error returns below.
        let pool = PinnedPool {
            inner,
            next_id: AtomicU64::new(1),
            used: AtomicUsize::new(0),
            refusals: AtomicU64::new(0),
            worker: Mutex::new(None),
            _map: map,
        };
        while pool.inner.pinned.load(Ordering::Acquire) < init {
            pool.inner
                .pin_next()
                .with_context(|| format!("pinned pool: pinning the first {init} B"))?;
        }
        if pool.inner.pinned.load(Ordering::Acquire) < total {
            pool.inner.growing.store(true, Ordering::Release);
            let shared = Arc::clone(&pool.inner);
            let handle = std::thread::Builder::new()
                .name("pinned-pool".to_string())
                .spawn(move || grow(shared));
            match handle {
                Ok(h) => *pool.worker.lock() = Some(h),
                Err(e) => {
                    pool.inner.growing.store(false, Ordering::Release);
                    tracing::warn!("pinned pool: no pin thread ({e}); serving {init} B");
                }
            }
        }
        Ok(Arc::new(pool))
    }

    /// A run of at least `bytes` (rounded up to whole pages), or `None` when no pinned extent
    /// holds it (pool full or fragmented, still pinning, or `bytes` larger than one chunk).
    /// Run memory is NOT cleared on reuse: the holder overwrites it before reading.
    pub fn alloc_run(&self, bytes: usize) -> Option<PageRun> {
        if bytes == 0 {
            return None;
        }
        let need = bytes.div_ceil(PAGE_BYTES).checked_mul(PAGE_BYTES)?;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let Some(off) = self.inner.free.lock().alloc(need, id) else {
            self.refusals.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        self.used.fetch_add(need, Ordering::Relaxed);
        G_USED.fetch_add(need, Ordering::Relaxed);
        Some(PageRun {
            ptr: self.inner.base + off,
            bytes,
            id,
        })
    }

    /// Hand a run back. A run this pool does not hold (freed twice, from another pool) is
    /// reported and ignored.
    pub fn free(&self, run: PageRun) {
        let freed = self
            .offset_of(run.ptr)
            .and_then(|off| self.inner.free.lock().free(off, run.id));
        match freed {
            Some(len) => {
                self.used.fetch_sub(len, Ordering::Relaxed);
                G_USED.fetch_sub(len, Ordering::Relaxed);
            }
            None => tracing::warn!(
                "pinned pool: ignored free of a run it does not hold (ptr {:#x}, id {})",
                run.ptr,
                run.id
            ),
        }
    }

    fn offset_of(&self, ptr: usize) -> Option<usize> {
        ptr.checked_sub(self.inner.base)
            .filter(|&off| off < self.inner.total)
    }

    /// Run `f` over the live run's bytes with the free list locked, so the run cannot be freed
    /// (and its pages reused) while `f` reads or writes them.
    fn with_run<R>(&self, run: &PageRun, f: impl FnOnce(*mut u8) -> R) -> Result<R> {
        let off = self
            .offset_of(run.ptr)
            .context("pinned pool: run outside this pool")?;
        let free = self.inner.free.lock();
        match free.live_len(off, run.id) {
            Some(len) if run.bytes <= len => Ok(f(run.ptr as *mut u8)),
            _ => bail!("pinned pool: run {} at {:#x} is not live", run.id, run.ptr),
        }
    }

    /// Copy `src` into `run` at byte `offset`.
    pub fn copy_in(&self, run: &PageRun, offset: usize, src: &[u8]) -> Result<()> {
        if offset
            .checked_add(src.len())
            .is_none_or(|end| end > run.bytes)
        {
            bail!(
                "pinned pool: copy_in of {} B at {offset} overruns a {} B run",
                src.len(),
                run.bytes
            );
        }
        self.with_run(run, |p| {
            // SAFETY: the run is live (checked under the lock, held for the copy) and spans
            // `run.bytes >= offset + src.len()` bytes of mapped memory; `src` is a distinct
            // caller buffer.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p.add(offset), src.len()) }
        })
    }

    /// Copy `dst.len()` bytes of `run` from byte `offset` into `dst`.
    pub fn copy_out(&self, run: &PageRun, offset: usize, dst: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(dst.len())
            .is_none_or(|end| end > run.bytes)
        {
            bail!(
                "pinned pool: copy_out of {} B at {offset} overruns a {} B run",
                dst.len(),
                run.bytes
            );
        }
        self.with_run(run, |p| {
            // SAFETY: as in `copy_in`, reading instead of writing.
            unsafe { std::ptr::copy_nonoverlapping(p.add(offset), dst.as_mut_ptr(), dst.len()) }
        })
    }

    /// Whether `[ptr, ptr + bytes)` lies in this pool's PINNED prefix.
    pub fn contains(&self, ptr: usize, bytes: usize) -> bool {
        let pinned = self.inner.pinned.load(Ordering::Acquire);
        self.offset_of(ptr)
            .and_then(|off| off.checked_add(bytes))
            .is_some_and(|end| end <= pinned)
    }

    /// Block until at least `bytes` are pinned, the thread stopped, or `timeout` passed. Returns
    /// whether `bytes` are pinned.
    pub fn wait_until_pinned(&self, bytes: usize, timeout: Duration) -> bool {
        let want = bytes.min(self.inner.total);
        let t0 = Instant::now();
        loop {
            if self.inner.pinned.load(Ordering::Acquire) >= want {
                return true;
            }
            if !self.inner.growing.load(Ordering::Acquire) || t0.elapsed() >= timeout {
                return self.inner.pinned.load(Ordering::Acquire) >= want;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// The pinning chunk, bytes (also the largest run).
    pub fn chunk_bytes(&self) -> usize {
        self.inner.chunk
    }

    /// The reserved range, bytes.
    pub fn capacity(&self) -> usize {
        self.inner.total
    }

    pub fn stats(&self) -> PoolStats {
        let free = self.inner.free.lock();
        PoolStats {
            total: self.inner.total,
            pinned: self.inner.pinned.load(Ordering::Acquire),
            used: self.used.load(Ordering::Relaxed),
            free: free.free_bytes(),
            runs: free.live_runs(),
            largest_free: free.largest_free(),
            chunk: self.inner.chunk,
            growing: self.inner.growing.load(Ordering::Acquire),
            pin_failed: self.inner.pin_failed.load(Ordering::Relaxed),
            refusals: self.refusals.load(Ordering::Relaxed),
        }
    }
}

impl Drop for PinnedPool {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::Release);
        if let Some(h) = self.worker.lock().take()
            && h.join().is_err()
        {
            tracing::warn!("pinned pool: the pin thread panicked");
        }
        let pinned = self.inner.pinned.load(Ordering::Acquire);
        let mut off = 0usize;
        while off < pinned {
            if let Err(e) = self.inner.gpu.host_unregister(self.inner.base + off) {
                tracing::warn!("pinned pool: unregistering the chunk at offset {off}: {e:#}");
            }
            off += self.inner.chunk;
        }
        let live = self.inner.free.lock().live_runs();
        if live > 0 {
            tracing::debug!("pinned pool: dropped with {live} runs still handed out");
        }
        G_PINNED.fetch_sub(pinned, Ordering::Relaxed);
        G_TOTAL.fetch_sub(self.inner.total, Ordering::Relaxed);
        G_USED.fetch_sub(self.used.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

#[cfg(test)]
#[path = "pinned_pool_tests.rs"]
mod tests;
