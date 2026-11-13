// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the page API (lane L36): demote then fetch restores
//! every byte, pages are visible only through completed copies, busy runs are never freed early,
//! a failed copy leaves the GPU blocks valid, and the slab arena keeps its budget.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::Mutex;

use super::{PageApi, PageOp, PageStats, SlabArena};
use crate::gpu::mock::MockGpuBackend;
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kv_cache::block_io::BlockRegions;
use crate::kv_host_tier::{HostArena, HostRun, MonotonicClock, PageDeps, RestoreState};

const LAYERS: usize = 3;
const REGION: usize = 64;
const BLOCK: usize = LAYERS * 2 * REGION;

/// Boxed buffers; `cap` limits how many runs may be out at once.
struct TestArena {
    runs: Mutex<HashMap<u64, Box<[u8]>>>,
    next: AtomicU64,
    cap: AtomicUsize,
}

impl TestArena {
    fn new(cap: usize) -> Self {
        Self {
            runs: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
            cap: AtomicUsize::new(cap),
        }
    }
    fn live(&self) -> usize {
        self.runs.lock().len()
    }
}

impl HostArena for TestArena {
    fn alloc(&self, bytes: usize) -> Option<HostRun> {
        let mut runs = self.runs.lock();
        if runs.len() >= self.cap.load(Ordering::Relaxed) {
            return None;
        }
        let mut buf = vec![0xEEu8; bytes].into_boxed_slice();
        let ptr = buf.as_mut_ptr() as usize;
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        runs.insert(id, buf);
        Some(HostRun { ptr, bytes, id })
    }
    fn free(&self, run: HostRun) {
        assert!(
            self.runs.lock().remove(&run.id).is_some(),
            "double free {run:?}"
        );
    }
}

struct Rig {
    gpu: Arc<MockGpuBackend>,
    arena: Arc<TestArena>,
    api: PageApi,
}

fn rig(cap: usize) -> Rig {
    let gpu = Arc::new(MockGpuBackend::new());
    gpu.set_unique_events(true);
    let arena = Arc::new(TestArena::new(cap));
    let copy_stream = gpu.create_copy_stream().expect("copy stream");
    let api = PageApi::with_parts(
        PageDeps {
            arena: arena.clone(),
            block_bytes: BLOCK,
            copy_stream,
        },
        Arc::new(MonotonicClock::new()),
        Arc::new(PageStats::default()),
    );
    Rig { gpu, arena, api }
}

fn block(gpu: &MockGpuBackend, seed: u8) -> BlockRegions {
    let regions = (0..LAYERS * 2)
        .map(|r| {
            let ptr = gpu.alloc(REGION).expect("alloc");
            let bytes: Vec<u8> = (0..REGION)
                .map(|i| seed.wrapping_mul(37).wrapping_add((r * REGION + i) as u8))
                .collect();
            gpu.copy_h2d(&bytes, ptr).expect("fill");
            (ptr, REGION)
        })
        .collect();
    BlockRegions { regions }
}

fn read(gpu: &MockGpuBackend, b: &BlockRegions) -> Vec<u8> {
    let mut out = Vec::new();
    for &(p, n) in &b.regions {
        let mut buf = vec![0u8; n];
        gpu.copy_d2h(p, &mut buf).expect("read");
        out.extend_from_slice(&buf);
    }
    out
}

fn clobber(gpu: &MockGpuBackend, b: &BlockRegions) {
    for &(p, n) in &b.regions {
        gpu.copy_h2d(&vec![0x5Au8; n], p).expect("clobber");
    }
}

#[test]
fn demote_then_fetch_restores_every_byte() {
    let r = rig(16);
    let src: Vec<BlockRegions> = (0..4).map(|i| block(&r.gpu, i as u8 + 1)).collect();
    let want: Vec<Vec<u8>> = src.iter().map(|b| read(&r.gpu, b)).collect();
    let rep = r
        .api
        .demote(r.gpu.as_ref(), 7, &[10, 11, 12, 13], &src, 0)
        .expect("demote");
    assert_eq!(rep.copied, vec![10, 11, 12, 13]);
    assert!(rep.resident.is_empty() && rep.refused.is_empty());
    let t = rep.ticket.expect("ticket");
    assert_eq!(t.op(), PageOp::Demote);
    assert_eq!(t.bytes(), (4 * BLOCK) as u64);
    assert_eq!(r.api.poll(r.gpu.as_ref(), &t).unwrap(), RestoreState::Ready);
    r.api.release(t);
    for (i, page) in [10u32, 11, 12, 13].iter().enumerate() {
        assert_eq!(
            r.api.read_page(7, *page).unwrap(),
            want[i],
            "host copy of page {page}"
        );
    }
    // The GPU blocks are reused by someone else; the pages come back into fresh blocks.
    for b in &src {
        clobber(&r.gpu, b);
    }
    let dst: Vec<BlockRegions> = (0..2).map(|_| block(&r.gpu, 0)).collect();
    let t = r
        .api
        .fetch(r.gpu.as_ref(), 7, &[12, 10], &dst, 0)
        .expect("fetch");
    assert_eq!(r.api.poll(r.gpu.as_ref(), &t).unwrap(), RestoreState::Ready);
    r.api.release(t);
    assert_eq!(read(&r.gpu, &dst[0]), want[2]);
    assert_eq!(read(&r.gpu, &dst[1]), want[0]);
    let s = r.api.stats().snapshot();
    assert_eq!(s.demote_pages, 4);
    assert_eq!(s.fetch_pages, 2);
    assert_eq!(s.fetch_bytes, (2 * BLOCK) as u64);
    assert_eq!(s.host_bytes, (4 * BLOCK) as u64);
    assert_eq!(r.api.seq_pages(7), 4);
    assert_eq!(r.api.release_seq(r.gpu.as_ref(), 7), 4);
    assert_eq!(r.arena.live(), 0);
    assert_eq!(r.api.stats().snapshot().host_bytes, 0);
}

#[test]
fn a_page_already_on_the_host_is_not_copied_again() {
    let r = rig(16);
    let b = block(&r.gpu, 3);
    let t = r
        .api
        .demote(r.gpu.as_ref(), 1, &[5], std::slice::from_ref(&b), 0)
        .unwrap()
        .ticket
        .unwrap();
    r.api.wait(r.gpu.as_ref(), &t).unwrap();
    r.api.release(t);
    let rep = r
        .api
        .demote(r.gpu.as_ref(), 1, &[5], std::slice::from_ref(&b), 0)
        .unwrap();
    assert!(rep.ticket.is_none());
    assert_eq!(rep.resident, vec![5]);
    assert_eq!(r.arena.live(), 1);
    // Another sequence's page 5 is a different page.
    let rep = r
        .api
        .demote(r.gpu.as_ref(), 2, &[5], std::slice::from_ref(&b), 0)
        .unwrap();
    assert_eq!(rep.copied, vec![5]);
}

#[test]
fn a_full_arena_refuses_pages_and_they_stay_on_the_gpu() {
    let r = rig(2);
    let src: Vec<BlockRegions> = (0..3).map(|i| block(&r.gpu, i)).collect();
    let rep = r
        .api
        .demote(r.gpu.as_ref(), 1, &[1, 2, 3], &src, 0)
        .unwrap();
    assert_eq!(rep.copied, vec![1, 2]);
    assert_eq!(rep.refused, vec![3]);
    assert!(!r.api.contains(1, 3));
    assert_eq!(r.api.stats().snapshot().refused_pages, 1);
}

#[test]
fn a_busy_page_is_freed_only_after_its_copy_completed() {
    let r = rig(16);
    r.gpu.hold_events(true);
    let b = block(&r.gpu, 9);
    let t = r
        .api
        .demote(r.gpu.as_ref(), 4, &[0], std::slice::from_ref(&b), 0)
        .unwrap()
        .ticket
        .unwrap();
    assert_eq!(
        r.api.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Pending
    );
    // Released while the D2H may still write the run: the run must survive.
    assert_eq!(r.api.release_seq(r.gpu.as_ref(), 4), 1);
    assert!(!r.api.contains(4, 0));
    assert_eq!(r.arena.live(), 1, "the run is a zombie, not freed");
    r.gpu.complete_event(t.event());
    assert_eq!(r.api.poll(r.gpu.as_ref(), &t).unwrap(), RestoreState::Ready);
    assert_eq!(r.arena.live(), 0);
    r.api.release(t);
    assert_eq!(r.api.running(), 0);
}

#[test]
fn a_fetch_keeps_its_source_page_alive() {
    let r = rig(16);
    let b = block(&r.gpu, 1);
    let t = r
        .api
        .demote(r.gpu.as_ref(), 3, &[2], std::slice::from_ref(&b), 0)
        .unwrap()
        .ticket
        .unwrap();
    r.api.wait(r.gpu.as_ref(), &t).unwrap();
    r.api.release(t);
    r.gpu.hold_events(true);
    let dst = block(&r.gpu, 0);
    let f = r
        .api
        .fetch(r.gpu.as_ref(), 3, &[2], std::slice::from_ref(&dst), 0)
        .unwrap();
    r.api.release_seq(r.gpu.as_ref(), 3);
    assert_eq!(r.arena.live(), 1, "the H2D still reads the run");
    r.gpu.complete_all_events();
    r.api.sweep(r.gpu.as_ref());
    assert_eq!(r.arena.live(), 0);
    assert_eq!(r.api.poll(r.gpu.as_ref(), &f).unwrap(), RestoreState::Ready);
    r.api.release(f);
}

#[test]
fn bad_requests_fail_without_side_effects() {
    let r = rig(16);
    let b = block(&r.gpu, 1);
    // Absent page.
    assert!(
        r.api
            .fetch(r.gpu.as_ref(), 1, &[9], std::slice::from_ref(&b), 0)
            .is_err()
    );
    // Duplicate page.
    assert!(
        r.api
            .demote(r.gpu.as_ref(), 1, &[1, 1], &[b.clone(), b.clone()], 0)
            .is_err()
    );
    // Length mismatch.
    assert!(
        r.api
            .demote(r.gpu.as_ref(), 1, &[1, 2], std::slice::from_ref(&b), 0)
            .is_err()
    );
    // Wrong block size.
    let short = BlockRegions {
        regions: vec![(b.regions[0].0, REGION)],
    };
    assert!(r.api.demote(r.gpu.as_ref(), 1, &[1], &[short], 0).is_err());
    assert_eq!(r.arena.live(), 0);
    // A copy that faults (unallocated device memory) frees its runs and leaves no page.
    let bogus = BlockRegions {
        regions: (0..LAYERS * 2)
            .map(|i| (DevicePtr(0x7E00_0000 + (i * REGION) as u64), REGION))
            .collect(),
    };
    assert!(r.api.demote(r.gpu.as_ref(), 1, &[3], &[bogus], 0).is_err());
    assert!(!r.api.contains(1, 3));
    assert_eq!(r.arena.live(), 0);
    assert_eq!(read(&r.gpu, &b).len(), BLOCK, "the real block is untouched");
}

#[test]
fn quiesce_empties_the_api() {
    let r = rig(16);
    r.gpu.hold_events(true);
    let src: Vec<BlockRegions> = (0..3).map(|i| block(&r.gpu, i)).collect();
    let rep = r
        .api
        .demote(r.gpu.as_ref(), 1, &[0, 1, 2], &src, 0)
        .unwrap();
    r.api.quiesce(r.gpu.as_ref());
    assert_eq!(r.arena.live(), 0);
    assert_eq!(r.api.seq_pages(1), 0);
    if let Some(t) = rep.ticket {
        r.api.release(t);
    }
}

#[test]
fn the_slab_arena_serves_fixed_slots_within_its_budget() {
    let gpu = MockGpuBackend::new();
    // 1000-byte pages round to 1024; 4 slots per slab; budget of 2 slabs.
    let a = SlabArena::new(1000, 4096, 8192);
    assert_eq!(a.slot_bytes(), 1024);
    assert!(a.alloc(1000).is_none(), "nothing reserved yet");
    assert_eq!(a.reserve(&gpu, 6), 8);
    assert_eq!(a.reserved_bytes(), 8192);
    assert_eq!(a.reserve(&gpu, 100), 8, "the budget stops growth");
    assert!(a.alloc(2000).is_none(), "larger than a slot");
    let runs: Vec<HostRun> = (0..8).map(|_| a.alloc(1000).expect("slot")).collect();
    assert!(a.alloc(1).is_none());
    let mut ptrs: Vec<usize> = runs.iter().map(|r| r.ptr).collect();
    ptrs.sort_unstable();
    ptrs.dedup();
    assert_eq!(ptrs.len(), 8, "every slot is distinct");
    assert!(ptrs.iter().all(|p| p.is_multiple_of(256)));
    // Slots never overlap.
    for w in ptrs.windows(2) {
        assert!(w[1] - w[0] >= 1024);
    }
    // SAFETY: the run is a live 1024-byte slot of a mapping the arena keeps.
    unsafe { std::ptr::write_bytes(runs[0].ptr as *mut u8, 7, 1024) };
    assert!(a.release_all(&gpu).is_err(), "runs are out");
    for r in runs {
        a.free(r);
    }
    assert_eq!(a.live_runs(), 0);
    a.release_all(&gpu).expect("release");
    assert_eq!(a.reserved_bytes(), 0);
}

#[test]
fn the_page_api_runs_over_the_slab_arena() {
    let gpu = Arc::new(MockGpuBackend::new());
    gpu.set_unique_events(true);
    let slab = Arc::new(SlabArena::new(BLOCK, 64 * BLOCK, (128 * BLOCK) as u64));
    assert!(slab.reserve(gpu.as_ref(), 2) >= 2);
    let api = PageApi::new(PageDeps {
        arena: slab.clone(),
        block_bytes: BLOCK,
        copy_stream: gpu.create_copy_stream().unwrap(),
    });
    let b = block(&gpu, 4);
    let want = read(&gpu, &b);
    let t = api
        .demote(gpu.as_ref(), 1, &[0], std::slice::from_ref(&b), 0)
        .unwrap()
        .ticket
        .unwrap();
    api.wait(gpu.as_ref(), &t).unwrap();
    api.release(t);
    assert_eq!(api.read_page(1, 0).unwrap(), want);
    api.quiesce(gpu.as_ref());
    assert_eq!(slab.live_runs(), 0);
    slab.release_all(gpu.as_ref()).unwrap();
}
