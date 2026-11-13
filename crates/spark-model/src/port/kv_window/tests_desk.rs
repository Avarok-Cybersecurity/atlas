// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: mock-backend tests of the window desk's state machine (lane L36): a short
//! sequence and a covering window are the identity; demotion hides the middle blocks and keeps
//! every position; demote then fetch restores every byte; references balance after the sequence
//! ends; a failed demote puts the blocks back; a reused first block drops the old state; an
//! inert device moves nothing.

use std::sync::Arc;

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use spark_runtime::kv_host_tier::PageDeps;
use spark_runtime::kv_host_tier::pages::{PageApi, PageStats, SlabArena};

use super::desk::{Desk, DeskConfig, DeviceHooks};
use super::fetch::{self, Pending, Settled};
use super::policy::Slot;
use super::selector::{PageScorer, ScoreRequest, compact_row};

const BS: usize = 16;
const BLOCKS: usize = 512;
const HEAD: usize = 32; // 2 blocks
const WINDOW: usize = 1024; // 64 blocks

struct TestDevice {
    ready: bool,
    holes: Mutex<Vec<Option<u32>>>,
}

impl TestDevice {
    fn new(ready: bool) -> Self {
        Self {
            ready,
            holes: Mutex::new(Vec::new()),
        }
    }
}

impl DeviceHooks for TestDevice {
    fn ready(&self, _gpu: &dyn GpuBackend, _stream: u64) -> Result<bool> {
        Ok(self.ready)
    }
    fn publish_hole(&self, _gpu: &dyn GpuBackend, hole: Option<u32>, _stream: u64) -> Result<()> {
        self.holes.lock().push(hole);
        Ok(())
    }
}

/// Ranks the candidates in the order of `prefer` (a list the test changes between rounds).
struct ListScorer {
    prefer: Mutex<Vec<u32>>,
    demoted: Mutex<Vec<(u32, u32)>>,
}

impl ListScorer {
    fn new(prefer: Vec<u32>) -> Self {
        Self {
            prefer: Mutex::new(prefer),
            demoted: Mutex::new(Vec::new()),
        }
    }
}

impl PageScorer for ListScorer {
    fn name(&self) -> &'static str {
        "list"
    }
    fn on_demote(
        &self,
        _gpu: &dyn GpuBackend,
        _stream: u64,
        _block0: u32,
        pages: &[(u32, u32)],
    ) -> Result<()> {
        self.demoted.lock().extend_from_slice(pages);
        Ok(())
    }
    fn rank(&self, _gpu: &dyn GpuBackend, req: &ScoreRequest, k: usize) -> Result<Vec<u32>> {
        Ok(self
            .prefer
            .lock()
            .iter()
            .copied()
            .filter(|p| req.candidates.contains(p))
            .take(k)
            .collect())
    }
}

fn cfg() -> DeskConfig {
    DeskConfig {
        head_tokens: HEAD,
        window_tokens: WINDOW,
        recall_tokens: 4 * BS,
        host_budget_bytes: 256 << 20,
    }
}

struct Rig {
    gpu: Arc<MockGpuBackend>,
    kv: PagedKvCache,
    desk: Desk,
}

fn rig() -> Rig {
    let gpu = Arc::new(MockGpuBackend::new());
    let kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: 8,
            num_layers: 2,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        BLOCKS,
        gpu.as_ref(),
    )
    .expect("kv cache");
    Rig {
        gpu,
        kv,
        desk: Desk::new(cfg(), Arc::new(PageStats::default())),
    }
}

fn fill(r: &Rig, block: u32) {
    for (j, &(p, n)) in r.kv.block_regions(block).regions.iter().enumerate() {
        let bytes: Vec<u8> = (0..n)
            .map(|i| (block as usize * 131 + j * 17 + i) as u8)
            .collect();
        r.gpu.copy_h2d(&bytes, p).expect("fill");
    }
}

fn bytes(r: &Rig, block: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for &(p, n) in &r.kv.block_regions(block).regions {
        let mut b = vec![0u8; n];
        r.gpu.copy_d2h(p, &mut b).expect("read");
        out.extend_from_slice(&b);
    }
    out
}

/// Grow `table` to `blocks` blocks, each filled with its own pattern.
fn grow(r: &mut Rig, table: &mut Vec<u32>, blocks: usize) {
    while table.len() < blocks {
        let b = r.kv.alloc_block().expect("alloc");
        fill(r, b);
        table.push(b);
    }
}

fn run(
    r: &mut Rig,
    table: &mut [u32],
    kv_tokens: usize,
    dev: &TestDevice,
    sc: Option<&dyn PageScorer>,
) -> super::desk::StepReport {
    r.desk
        .maintain(table, kv_tokens, &mut r.kv, r.gpu.as_ref(), 0, dev, sc)
        .expect("maintain")
}

#[test]
fn a_short_sequence_or_a_covering_window_is_the_identity() {
    let mut r = rig();
    let dev = TestDevice::new(true);
    let mut table = Vec::new();
    grow(&mut r, &mut table, 100);
    let before = table.clone();
    // 100 blocks: head 2 + window 64 leave 34 demotable, below one batch of 64.
    for kv in [0, 16, 800, 1600] {
        assert_eq!(run(&mut r, &mut table, kv, &dev, None), Default::default());
    }
    assert_eq!(table, before);
    assert!(r.desk.hole().is_none());
    assert!(r.desk.seq(table[0]).is_none());
    // A window longer than the context: never anything, at any length.
    let mut wide = Desk::new(
        DeskConfig {
            window_tokens: 1 << 20,
            ..cfg()
        },
        Arc::new(PageStats::default()),
    );
    grow(&mut r, &mut table, 400);
    let before = table.clone();
    for kv in (0..400 * BS).step_by(333) {
        let rep = wide
            .maintain(&mut table, kv, &mut r.kv, r.gpu.as_ref(), 0, &dev, None)
            .unwrap();
        assert_eq!(rep, Default::default());
    }
    assert_eq!(table, before);
    assert!(dev.holes.lock().is_empty());
}

#[test]
fn demotion_hides_the_middle_and_keeps_every_position() {
    let mut r = rig();
    let dev = TestDevice::new(true);
    let mut table = Vec::new();
    grow(&mut r, &mut table, 140);
    let orig = table.clone();
    let kv = 140 * BS;
    let free_before = r.kv.num_free_blocks();
    let rep = run(&mut r, &mut table, kv, &dev, None);
    // Demotable: blocks 2 .. (2240 - 1024) / 16 = 76.
    assert_eq!(rep.demoted, 74);
    let hole = r.desk.hole().expect("hole");
    assert_eq!(dev.holes.lock().as_slice(), &[Some(hole)]);
    assert_eq!(table.len(), 140, "logical length unchanged");
    assert_eq!(&table[..2], &orig[..2], "head sinks stay");
    assert_eq!(&table[76..], &orig[76..], "recent window stays");
    assert!(table[2..76].iter().all(|&b| b == hole));
    assert_eq!(r.kv.ref_count(hole), 1 + 74);
    // The originals are released once their copy completed (next call).
    assert_eq!(r.kv.num_free_blocks(), free_before - 1);
    let rep = run(&mut r, &mut table, kv, &dev, None);
    assert_eq!(rep.freed, 74);
    assert_eq!(r.kv.num_free_blocks(), free_before - 1 + 74);
    // The host holds every demoted block's bytes.
    let sd = r.desk.seq(table[0]).expect("state");
    let host = r.desk.host().expect("host side");
    assert_eq!(host.api.seq_pages(sd.id), 74);
    // The view the attention reads: sinks then the window, length less 74 blocks.
    let (view, len) = compact_row(&table, kv as u32, BS as u32, Some(hole));
    let mut want = orig[..2].to_vec();
    want.extend_from_slice(&orig[76..]);
    assert_eq!(view, want);
    assert_eq!(len as usize, kv - 74 * BS);
    assert_eq!(r.desk.resident_tokens(), kv - 74 * BS);
    assert!(r.desk.any_hidden());
}

#[test]
fn demote_then_fetch_restores_the_bytes_and_retires_by_delta() {
    let mut r = rig();
    let dev = TestDevice::new(true);
    let sc = ListScorer::new(vec![10, 20, 30]);
    let mut table = Vec::new();
    grow(&mut r, &mut table, 140);
    let orig = table.clone();
    let want: Vec<Vec<u8>> = orig.iter().map(|&b| bytes(&r, b)).collect();
    let mut kv = 140 * BS;
    run(&mut r, &mut table, kv, &dev, Some(&sc));
    assert_eq!(
        sc.demoted.lock().len(),
        74,
        "the scorer saw every demoted page"
    );
    run(&mut r, &mut table, kv, &dev, Some(&sc)); // settle the demote
    // The originals were freed and may be reused by anyone.
    for &b in &orig[2..76] {
        assert_eq!(r.kv.ref_count(b), 0);
    }
    // 1,024 positions later the recall round runs.
    kv += 1024;
    grow(&mut r, &mut table, kv / BS + 1);
    let rep = run(&mut r, &mut table, kv, &dev, Some(&sc));
    assert_eq!(rep.fetch_started, 3);
    let rep = run(&mut r, &mut table, kv, &dev, Some(&sc));
    assert_eq!(rep.installed, 3);
    let hole = r.desk.hole().unwrap();
    for i in [10usize, 20, 30] {
        assert_ne!(table[i], hole);
        assert_eq!(
            bytes(&r, table[i]),
            want[i],
            "page {i} restored byte for byte"
        );
    }
    let sd = r.desk.seq(table[0]).unwrap();
    assert_eq!(sd.slots[10], Slot::Recalled);
    // Next round prefers page 40: 10, 20, 30 go back on the shelf without a copy, 40 comes.
    *sc.prefer.lock() = vec![40];
    let recalled: Vec<u32> = [10, 20, 30].iter().map(|&i| table[i]).collect();
    kv += 1024;
    grow(&mut r, &mut table, kv / BS + 1);
    let rep = run(&mut r, &mut table, kv, &dev, Some(&sc));
    assert_eq!(rep.rehidden, 3);
    assert_eq!(rep.fetch_started, 1);
    for &i in &[10usize, 20, 30] {
        assert_eq!(table[i], hole);
    }
    run(&mut r, &mut table, kv, &dev, Some(&sc));
    assert_eq!(bytes(&r, table[40]), want[40]);
    // The recalled copies were freed (one may already serve as page 40's fresh block).
    for &b in &recalled {
        assert!(
            r.kv.ref_count(b) == 0 || b == table[40],
            "recalled block {b} still referenced"
        );
    }
    // A fetch only after a full step: the same position asks nothing more.
    let rep = run(&mut r, &mut table, kv, &dev, Some(&sc));
    assert_eq!(rep.fetch_started + rep.rehidden, 0);
}

#[test]
fn references_balance_after_the_sequence_ends() {
    let mut r = rig();
    let dev = TestDevice::new(true);
    let sc = ListScorer::new(vec![5, 6]);
    let free0 = r.kv.num_free_blocks();
    let mut table = Vec::new();
    grow(&mut r, &mut table, 140);
    let mut kv = 140 * BS;
    run(&mut r, &mut table, kv, &dev, Some(&sc));
    run(&mut r, &mut table, kv, &dev, Some(&sc));
    kv += 1024;
    grow(&mut r, &mut table, kv / BS + 1);
    run(&mut r, &mut table, kv, &dev, Some(&sc)); // demote batch 2 + fetch 5, 6
    run(&mut r, &mut table, kv, &dev, Some(&sc)); // settle both
    let hole = r.desk.hole().unwrap();
    // The sequence ends: the model frees every table entry (holes included).
    r.desk.on_free(&table, r.gpu.as_ref());
    r.kv.free_blocks(&table);
    // Another call settles what the ended sequence left running.
    let mut other = Vec::new();
    grow(&mut r, &mut other, 1);
    run(&mut r, &mut other, 16, &dev, None);
    r.kv.free_blocks(&other);
    assert_eq!(r.kv.ref_count(hole), 1, "only the desk's own reference");
    assert_eq!(
        r.kv.num_free_blocks(),
        free0 - 1,
        "every block but the hole is free again"
    );
    assert_eq!(r.desk.host().unwrap().api.seq_pages(1), 0);
    assert!(!r.desk.any_hidden());
}

#[test]
fn a_failed_demote_puts_the_blocks_back() {
    let mut r = rig();
    let gpu = r.gpu.clone();
    let mut table = Vec::new();
    grow(&mut r, &mut table, 4);
    let hole = r.kv.alloc_block().unwrap();
    let orig = table[1];
    // Table as a demote leaves it: index 1 points at the hole, the desk holds `orig`.
    table[1] = hole;
    r.kv.inc_ref(hole);
    let mut slots = vec![Slot::Resident, Slot::Hidden, Slot::Resident, Slot::Resident];
    // A ticket this API never issued polls as Failed (a stand-in for a faulted copy).
    let arena = Arc::new(SlabArena::new(1024, 1 << 20, 4 << 20));
    let deps = |arena: &Arc<SlabArena>| PageDeps {
        arena: arena.clone(),
        block_bytes: r.kv.block_regions(0).bytes(),
        copy_stream: gpu.create_copy_stream().unwrap(),
    };
    let other = PageApi::new(deps(&arena));
    arena.reserve(gpu.as_ref(), 1);
    let ticket = other
        .demote(gpu.as_ref(), 1, &[9], &[r.kv.block_regions(table[0])], 0)
        .unwrap()
        .ticket
        .unwrap();
    let api = PageApi::new(deps(&arena));
    let mut out = Settled::default();
    let left = fetch::settle(
        &api,
        gpu.as_ref(),
        vec![Pending::Demote {
            ticket,
            sources: vec![(1, orig)],
        }],
        &mut table,
        &mut slots,
        hole,
        &mut r.kv,
        &mut out,
    );
    assert!(left.is_empty());
    assert_eq!(out.reverted, 1);
    assert_eq!(table[1], orig);
    assert_eq!(slots[1], Slot::Resident);
    assert_eq!(r.kv.ref_count(hole), 1);
    assert_eq!(r.kv.ref_count(orig), 1, "the table owns the block again");
    other.quiesce(gpu.as_ref());
}

#[test]
fn a_reused_first_block_drops_the_old_state() {
    let mut r = rig();
    let dev = TestDevice::new(true);
    let mut table = Vec::new();
    grow(&mut r, &mut table, 140);
    let kv = 140 * BS;
    run(&mut r, &mut table, kv, &dev, None);
    run(&mut r, &mut table, kv, &dev, None);
    let block0 = table[0];
    let old_id = r.desk.seq(block0).unwrap().id;
    // The sequence ends without the free hook; a new one gets the same first block.
    r.kv.free_blocks(&table[1..]);
    let mut fresh = vec![block0];
    grow(&mut r, &mut fresh, 3);
    run(&mut r, &mut fresh, 3 * BS, &dev, None);
    assert!(
        r.desk.seq(block0).is_none(),
        "the old slots do not describe the new table"
    );
    assert_eq!(r.desk.host().unwrap().api.seq_pages(old_id), 0);
    assert!(!r.desk.any_hidden());
}

#[test]
fn an_inert_device_moves_nothing() {
    let mut r = rig();
    let dev = TestDevice::new(false);
    let mut table = Vec::new();
    grow(&mut r, &mut table, 300);
    let before = table.clone();
    let rep = run(&mut r, &mut table, 300 * BS, &dev, None);
    assert_eq!(rep, Default::default());
    assert_eq!(table, before);
    assert!(r.desk.hole().is_none());
}
