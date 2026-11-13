// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the pinned pool and its free list on the mock backend: the
//! eager and background pinning, run allocation, coalescing, chunk-boundary rule, growth under
//! concurrent load, refused pins, `pinned_hosts` agreement, the byte helpers and teardown.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Result, bail};

use super::free_list::FreeList;
use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend, KernelHandle};

const CHUNK: usize = 64 * 1024;

fn mock() -> Arc<dyn GpuBackend> {
    Arc::new(MockGpuBackend::new())
}

/// Delegates to the mock, but refuses `host_register` after `ok` successful pins (a WSL2 or
/// `ulimit -l` cap) and counts context binds.
struct Refusing {
    inner: MockGpuBackend,
    ok: usize,
    pins: AtomicUsize,
    binds: AtomicUsize,
}

impl Refusing {
    fn new(ok: usize) -> Self {
        Self {
            inner: MockGpuBackend::new(),
            ok,
            pins: AtomicUsize::new(0),
            binds: AtomicUsize::new(0),
        }
    }
}

impl GpuBackend for Refusing {
    fn alloc(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc(bytes)
    }
    fn alloc_managed(&self, bytes: usize) -> Result<DevicePtr> {
        self.inner.alloc_managed(bytes)
    }
    fn free(&self, ptr: DevicePtr) -> Result<()> {
        self.inner.free(ptr)
    }
    fn copy_h2d(&self, src: &[u8], dst: DevicePtr) -> Result<()> {
        self.inner.copy_h2d(src, dst)
    }
    fn copy_d2h(&self, src: DevicePtr, dst: &mut [u8]) -> Result<()> {
        self.inner.copy_d2h(src, dst)
    }
    fn copy_d2d(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        self.inner.copy_d2d(src, dst, bytes)
    }
    fn launch(
        &self,
        func: KernelHandle,
        grid: [u32; 3],
        block: [u32; 3],
        shared_mem: u32,
        stream: u64,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        self.inner
            .launch(func, grid, block, shared_mem, stream, params)
    }
    fn synchronize(&self, stream: u64) -> Result<()> {
        self.inner.synchronize(stream)
    }
    fn default_stream(&self) -> u64 {
        0
    }
    fn kernel(&self, module: &str, func_name: &str) -> Result<KernelHandle> {
        self.inner.kernel(module, func_name)
    }
    fn has_module(&self, module: &str) -> bool {
        self.inner.has_module(module)
    }
    fn op_cache(&self) -> &crate::op_cache::OpCache {
        self.inner.op_cache()
    }
    fn memset(&self, ptr: DevicePtr, value: u8, bytes: usize) -> Result<()> {
        self.inner.memset(ptr, value, bytes)
    }
    fn memset_async(&self, ptr: DevicePtr, value: u8, bytes: usize, stream: u64) -> Result<()> {
        self.inner.memset_async(ptr, value, bytes, stream)
    }
    fn total_memory(&self) -> Result<usize> {
        self.inner.total_memory()
    }
    fn free_memory(&self) -> Result<usize> {
        self.inner.free_memory()
    }
    fn sm_count(&self) -> Result<u32> {
        self.inner.sm_count()
    }
    fn bind_to_thread(&self) -> Result<()> {
        self.binds.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn host_register(&self, ptr: usize, bytes: usize) -> Result<()> {
        if self.pins.fetch_add(1, Ordering::Relaxed) >= self.ok {
            bail!("refused: page-lock limit");
        }
        self.inner.host_register(ptr, bytes)
    }
    fn host_unregister(&self, ptr: usize) -> Result<()> {
        self.inner.host_unregister(ptr)
    }
}

fn full_pool(chunks: usize) -> Arc<PinnedPool> {
    PinnedPool::new(mock(), chunks * CHUNK, chunks * CHUNK, CHUNK).unwrap()
}

#[test]
fn new_pins_init_before_returning_and_rounds_to_chunks() {
    let pool = PinnedPool::new(mock(), 3 * CHUNK + 1, CHUNK + 1, CHUNK).unwrap();
    let s = pool.stats();
    assert_eq!(s.total, 4 * CHUNK, "budget rounded up to whole chunks");
    assert!(
        s.pinned >= 2 * CHUNK,
        "init rounded up to whole chunks and pinned eagerly"
    );
    assert_eq!(s.chunk, CHUNK);
    assert!(pool.wait_until_pinned(4 * CHUNK, Duration::from_secs(10)));
    let s = pool.stats();
    assert_eq!(s.pinned, 4 * CHUNK);
    assert!(!s.growing);
    assert!(!s.pin_failed);
    assert_eq!(s.free, 4 * CHUNK);
    assert_eq!(pool.capacity(), 4 * CHUNK);
    assert_eq!(pool.chunk_bytes(), CHUNK);
}

#[test]
fn zero_budget_is_refused_and_chunk_is_page_rounded() {
    assert!(PinnedPool::new(mock(), 0, 0, CHUNK).is_err());
    let pool = PinnedPool::new(mock(), 10_000, 10_000, 5000).unwrap();
    assert_eq!(pool.chunk_bytes(), 8192);
    assert_eq!(pool.capacity(), 16_384);
}

#[test]
fn runs_are_page_aligned_and_agree_with_pinned_hosts() {
    let pool = full_pool(2);
    let run = pool.alloc_run(5000).unwrap();
    assert_eq!(run.ptr % PAGE_BYTES, 0);
    assert_eq!(run.bytes, 5000);
    assert!(pool.contains(run.ptr, run.bytes));
    // SAFETY: the run is live and spans `run.bytes` bytes of mapped memory.
    let slice = unsafe { std::slice::from_raw_parts(run.ptr as *const u8, run.bytes) };
    assert!(crate::pinned_hosts::is_pinned(slice));
    assert!(
        crate::pinned_hosts::is_tier_owned(slice),
        "tier-owned: no implicit transient sync"
    );
    assert!(!crate::pinned_hosts::needs_transient_sync(slice));
    let s = pool.stats();
    assert_eq!(s.used, 8192, "rounded up to whole pages");
    assert_eq!(s.runs, 1);
    pool.free(run);
    assert_eq!(pool.stats().used, 0);
}

#[test]
fn zero_and_oversized_requests_are_refused() {
    let pool = full_pool(2);
    assert!(pool.alloc_run(0).is_none());
    assert!(
        pool.alloc_run(CHUNK + 1).is_none(),
        "a run never spans two chunks"
    );
    assert!(pool.alloc_run(CHUNK).is_some());
    assert!(pool.stats().refusals >= 1);
}

#[test]
fn freeing_coalesces_back_to_whole_chunks() {
    let pool = full_pool(2);
    let runs: Vec<PageRun> = (0..8)
        .map(|_| pool.alloc_run(PAGE_BYTES * 4).unwrap())
        .collect();
    assert_eq!(pool.stats().free, 2 * CHUNK - 8 * 4 * PAGE_BYTES);
    // Free in an interleaved order so both neighbors merge.
    let mut order = runs;
    let odd: Vec<PageRun> = (0..order.len())
        .rev()
        .filter(|i| i % 2 == 1)
        .map(|i| order.remove(i))
        .collect();
    for r in odd.into_iter().chain(order) {
        pool.free(r);
    }
    let s = pool.stats();
    assert_eq!(s.free, 2 * CHUNK);
    assert_eq!(
        s.largest_free, CHUNK,
        "coalescing stops at the chunk boundary"
    );
    assert_eq!(s.runs, 0);
    assert_eq!(
        pool.inner.free.lock().extents(),
        vec![(0, CHUNK), (CHUNK, CHUNK)]
    );
}

#[test]
fn best_fit_picks_the_smallest_hole() {
    let pool = full_pool(1);
    let a = pool.alloc_run(PAGE_BYTES).unwrap();
    let b = pool.alloc_run(3 * PAGE_BYTES).unwrap();
    let c = pool.alloc_run(PAGE_BYTES).unwrap();
    let d = pool.alloc_run(PAGE_BYTES).unwrap();
    let e = pool.alloc_run(PAGE_BYTES).unwrap();
    let (b_ptr, d_ptr) = (b.ptr, d.ptr);
    pool.free(b); // a 3-page hole
    pool.free(d); // a 1-page hole
    let one = pool.alloc_run(PAGE_BYTES).unwrap();
    assert_eq!(one.ptr, d_ptr, "the 1-page hole fits a 1-page run best");
    let three = pool.alloc_run(3 * PAGE_BYTES).unwrap();
    assert_eq!(three.ptr, b_ptr);
    for r in [a, c, e, one, three] {
        pool.free(r);
    }
    assert_eq!(pool.stats().free, CHUNK);
}

#[test]
fn a_double_or_foreign_free_is_ignored() {
    let pool = full_pool(1);
    let other = full_pool(1);
    let run = pool.alloc_run(PAGE_BYTES).unwrap();
    let forged = PageRun {
        ptr: run.ptr,
        bytes: run.bytes,
        id: run.id,
    };
    pool.free(run);
    pool.free(forged); // freed twice
    assert_eq!(pool.stats().used, 0);
    assert_eq!(pool.stats().free, CHUNK);
    let foreign = other.alloc_run(PAGE_BYTES).unwrap();
    let copy = PageRun {
        ptr: foreign.ptr,
        bytes: foreign.bytes,
        id: foreign.id,
    };
    pool.free(copy); // not this pool's
    assert_eq!(other.stats().runs, 1);
    other.free(foreign);
}

#[test]
fn a_stale_run_id_cannot_free_a_reused_extent() {
    let pool = full_pool(1);
    let first = pool.alloc_run(PAGE_BYTES).unwrap();
    let stale = PageRun {
        ptr: first.ptr,
        bytes: first.bytes,
        id: first.id,
    };
    pool.free(first);
    let second = pool.alloc_run(PAGE_BYTES).unwrap();
    assert_eq!(second.ptr, stale.ptr, "the same extent is handed out again");
    pool.free(stale);
    assert_eq!(pool.stats().runs, 1, "the live run survives a stale free");
    pool.free(second);
}

#[test]
fn copy_helpers_round_trip_and_refuse_overruns_and_dead_runs() {
    let pool = full_pool(1);
    let run = pool.alloc_run(100).unwrap();
    let src: Vec<u8> = (0..60u8).collect();
    pool.copy_in(&run, 40, &src).unwrap();
    let mut out = vec![0u8; 60];
    pool.copy_out(&run, 40, &mut out).unwrap();
    assert_eq!(out, src);
    assert!(
        pool.copy_in(&run, 41, &src).is_err(),
        "overruns the 100-byte run"
    );
    assert!(pool.copy_out(&run, usize::MAX, &mut out).is_err());
    let dead = PageRun {
        ptr: run.ptr,
        bytes: run.bytes,
        id: run.id,
    };
    pool.free(run);
    assert!(
        pool.copy_in(&dead, 0, &src).is_err(),
        "a freed run is not writable"
    );
    assert!(pool.copy_out(&dead, 0, &mut out).is_err());
}

#[test]
fn growth_continues_under_concurrent_allocation() {
    let chunks = 16;
    let pool = PinnedPool::new(mock(), chunks * CHUNK, CHUNK, CHUNK).unwrap();
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let p = Arc::clone(&pool);
            std::thread::spawn(move || {
                let mut held = Vec::new();
                for i in 0..400 {
                    let want = PAGE_BYTES * (1 + (i + t) % 5);
                    if let Some(r) = p.alloc_run(want) {
                        assert!(
                            p.contains(r.ptr, r.bytes),
                            "only pinned bytes are handed out"
                        );
                        p.copy_in(&r, 0, &[t as u8; 16]).unwrap();
                        held.push(r);
                    }
                    if held.len() > 6 {
                        let r = held.remove(0);
                        let mut b = [0u8; 16];
                        p.copy_out(&r, 0, &mut b).unwrap();
                        assert_eq!(b, [t as u8; 16], "no run is shared between holders");
                        p.free(r);
                    }
                }
                for r in held {
                    p.free(r);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert!(pool.wait_until_pinned(chunks * CHUNK, Duration::from_secs(10)));
    let s = pool.stats();
    assert_eq!(s.pinned, chunks * CHUNK);
    assert_eq!(s.used, 0);
    assert_eq!(s.free, chunks * CHUNK);
    assert_eq!(s.largest_free, CHUNK);
}

#[test]
fn a_refused_pin_stops_growth_and_the_pool_keeps_serving() {
    let gpu = Arc::new(Refusing::new(3));
    let dyn_gpu: Arc<dyn GpuBackend> = gpu.clone();
    let pool = PinnedPool::new(dyn_gpu, 8 * CHUNK, CHUNK, CHUNK).unwrap();
    assert!(!pool.wait_until_pinned(8 * CHUNK, Duration::from_secs(10)));
    let s = pool.stats();
    assert_eq!(s.pinned, 3 * CHUNK);
    assert!(s.pin_failed);
    assert!(!s.growing);
    assert_eq!(
        gpu.binds.load(Ordering::Relaxed),
        1,
        "the pin thread bound the context first"
    );
    let runs: Vec<PageRun> = (0..3).map(|_| pool.alloc_run(CHUNK).unwrap()).collect();
    assert!(
        pool.alloc_run(PAGE_BYTES).is_none(),
        "nothing beyond the pinned prefix"
    );
    for r in runs {
        pool.free(r);
    }
}

#[test]
fn a_refused_initial_pin_fails_construction_cleanly() {
    let gpu: Arc<dyn GpuBackend> = Arc::new(Refusing::new(1));
    let live = crate::pinned_hosts::tier_count();
    assert!(PinnedPool::new(gpu, 4 * CHUNK, 2 * CHUNK, CHUNK).is_err());
    // The one chunk pinned before the refusal was unregistered by Drop. Other tests register
    // concurrently, so compare against a bound rather than an exact count.
    assert!(crate::pinned_hosts::tier_count() <= live + 64);
}

#[test]
fn drop_unregisters_every_chunk() {
    let pool = full_pool(3);
    let base = pool.inner.base;
    let total = pool.capacity();
    assert!(crate::pinned_hosts::overlaps(base as *const u8, total));
    drop(pool);
    assert!(!crate::pinned_hosts::overlaps(base as *const u8, total));
}

#[test]
fn chunk_flag_validation_and_publication() {
    assert!(validate_chunk_mb(None).is_ok());
    assert!(validate_chunk_mb(Some(64)).is_ok());
    assert!(validate_chunk_mb(Some(0)).is_err());
    assert!(validate_chunk_mb(Some(MAX_CHUNK_MB + 1)).is_err());
    // Out-of-range and absent values never change the published chunk.
    let before = default_chunk_bytes();
    set_default_chunk_mb(Some(0));
    set_default_chunk_mb(None);
    assert_eq!(default_chunk_bytes(), before);
}

#[test]
fn gauges_render_all_three_states() {
    let pool = full_pool(1);
    let run = pool.alloc_run(PAGE_BYTES).unwrap();
    let mut out = String::new();
    render_prometheus(&mut out);
    for needle in [
        "# TYPE atlas_pinned_pool_bytes gauge",
        "atlas_pinned_pool_bytes{state=\"total\"}",
        "atlas_pinned_pool_bytes{state=\"pinned\"}",
        "atlas_pinned_pool_bytes{state=\"used\"}",
    ] {
        assert!(out.contains(needle), "missing {needle:?}");
    }
    pool.free(run);
}

#[test]
fn free_list_never_merges_across_chunks() {
    let mut fl = FreeList::new(8192);
    fl.add_chunk(0);
    fl.add_chunk(8192);
    let a = fl.alloc(4096, 1).unwrap();
    let b = fl.alloc(4096, 2).unwrap();
    let c = fl.alloc(4096, 3).unwrap();
    assert_eq!((a, b, c), (0, 4096, 8192));
    assert_eq!(fl.free(b, 2), Some(4096));
    assert_eq!(fl.free(c, 3), Some(4096));
    // b ends where c's chunk starts; they must stay two extents.
    assert_eq!(fl.extents(), vec![(4096, 4096), (8192, 8192)]);
    assert_eq!(fl.free(a, 1), Some(4096));
    assert_eq!(fl.extents(), vec![(0, 8192), (8192, 8192)]);
    assert_eq!(fl.free(a, 1), None, "double free changes nothing");
    assert_eq!(fl.free_bytes(), 16384);
    assert_eq!(fl.largest_free(), 8192);
    assert_eq!(fl.live_runs(), 0);
    assert!(fl.alloc(0, 9).is_none());
    assert!(fl.alloc(8193, 9).is_none());
}
