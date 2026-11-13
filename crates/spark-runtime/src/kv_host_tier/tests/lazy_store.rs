// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the lazy store mode: stores only buffer; a drain starts exactly the blocks
//! near the GPU eviction head (closest first, capped per drain), drops blocks whose GPU copy is
//! gone, keeps in-use and budget-limited blocks buffered, honors the deferral deadline, and the
//! restored bytes equal the bytes at store time.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

use super::*;
use crate::kv_host_tier::{DrainSignals, EvictionView, GpuBlockState, StoreMode, TierPolicy};

#[derive(Default)]
struct FakeView(RefCell<HashMap<BlockKey, GpuBlockState>>);

impl FakeView {
    fn set(&self, k: BlockKey, s: GpuBlockState) {
        self.0.borrow_mut().insert(k, s);
    }
}

impl EvictionView for FakeView {
    fn state(&self, key: &BlockKey) -> GpuBlockState {
        self.0
            .borrow()
            .get(key)
            .copied()
            .unwrap_or(GpuBlockState::Gone)
    }
}

fn lazy_rig(blocks: u64) -> Rig {
    let opts = TierOptions {
        store_mode: StoreMode::Lazy,
        ..test_opts()
    };
    rig_with(cfg_blocks(blocks, TierPolicy::Lru), opts)
}

fn demand(blocks_per_step: f64) -> DrainSignals {
    DrainSignals { blocks_per_step }
}

/// Buffer `n` blocks of one prompt; the view ranks block i at `rank(i)`.
fn buffer(
    r: &Rig,
    n: usize,
    rank: impl Fn(usize) -> usize,
) -> (Prompt, Vec<BlockRegions>, FakeView) {
    let p = Prompt::new(NS, 0, n);
    let blocks: Vec<BlockRegions> = (0..n).map(|i| device_block(&r.gpu, i as u8)).collect();
    let rep = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
        .unwrap();
    assert_eq!(rep.buffered.len(), n);
    assert!(
        rep.started.is_empty(),
        "lazy mode copies nothing at store time"
    );
    let view = FakeView::default();
    for (i, k) in p.keys.iter().enumerate() {
        view.set(*k, GpuBlockState::Queued { rank: rank(i) });
    }
    (p, blocks, view)
}

#[test]
fn a_drain_starts_only_the_blocks_near_the_eviction_head() {
    let r = lazy_rig(64);
    let (p, _blocks, view) = buffer(&r, 10, |i| i);
    assert_eq!(r.tier.lookup(&p.keys, &p.chunks()), 0);
    assert_eq!(r.tier.snapshot().lazy_pending_blocks, 10);
    // One block per step, horizon 2.5 steps: ranks 0, 1 and 2 are near the head.
    let started = r
        .tier
        .drain_lazy(r.gpu.as_ref(), demand(1.0), &view)
        .unwrap();
    assert_eq!(started, p.keys[..3].to_vec());
    assert_eq!(r.tier.poll_writebacks(r.gpu.as_ref()), p.keys[..3].to_vec());
    assert_eq!(r.tier.lookup(&p.keys, &p.chunks()), 3);
    // Nothing else is near the head: the next drain starts nothing.
    assert!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(1.0), &view)
            .unwrap()
            .is_empty()
    );
    assert_eq!(r.tier.snapshot().lazy_pending_blocks, 7);
    // No allocation pressure at all: nothing drains.
    assert!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(0.0), &view)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn closest_blocks_go_first_and_each_drain_is_capped() {
    let opts = TierOptions {
        store_mode: StoreMode::Lazy,
        max_drain_per_step: 2,
        ..test_opts()
    };
    let r = rig_with(cfg_blocks(64, TierPolicy::Lru), opts);
    // Ranks in reverse order of arrival.
    let (p, _blocks, view) = buffer(&r, 6, |i| 5 - i);
    let started = r
        .tier
        .drain_lazy(r.gpu.as_ref(), demand(10.0), &view)
        .unwrap();
    assert_eq!(
        started,
        vec![p.keys[5], p.keys[4]],
        "rank 0 then rank 1, two per drain"
    );
    assert_eq!(r.tier.snapshot().lazy_pending_blocks, 4);
}

#[test]
fn gone_blocks_are_dropped_and_in_use_blocks_wait() {
    let r = lazy_rig(64);
    let (p, _blocks, view) = buffer(&r, 4, |_| 0);
    view.set(p.keys[0], GpuBlockState::Gone);
    view.set(p.keys[1], GpuBlockState::InUse);
    let started = r
        .tier
        .drain_lazy(r.gpu.as_ref(), demand(4.0), &view)
        .unwrap();
    assert_eq!(started, vec![p.keys[2], p.keys[3]]);
    let s = r.tier.snapshot();
    assert_eq!(
        s.writeback_dropped, 1,
        "the evicted block can no longer be copied"
    );
    assert_eq!(s.lazy_pending_blocks, 1, "the in-use block waits");
}

#[test]
fn the_deferral_deadline_starts_old_blocks_anyway() {
    let opts = TierOptions {
        store_mode: StoreMode::Lazy,
        max_deferral: Some(Duration::from_secs(5)),
        ..test_opts()
    };
    let r = rig_with(cfg_blocks(64, TierPolicy::Lru), opts);
    let (p, _blocks, view) = buffer(&r, 2, |_| 1_000);
    view.set(p.keys[1], GpuBlockState::InUse);
    assert!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(1.0), &view)
            .unwrap()
            .is_empty()
    );
    r.clock.advance(Duration::from_secs(6));
    let started = r
        .tier
        .drain_lazy(r.gpu.as_ref(), demand(1.0), &view)
        .unwrap();
    assert_eq!(started.len(), 2);
}

#[test]
fn budget_limited_blocks_stay_buffered() {
    let mut cfg = cfg_blocks(64, TierPolicy::Lru);
    cfg.writeback_bytes_per_step = BLOCK_BYTES as u64;
    let opts = TierOptions {
        store_mode: StoreMode::Lazy,
        ..test_opts()
    };
    let r = rig_with(cfg, opts);
    let (p, _blocks, view) = buffer(&r, 3, |i| i);
    r.tier.begin_step();
    assert_eq!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(4.0), &view)
            .unwrap(),
        vec![p.keys[0]]
    );
    assert_eq!(r.tier.snapshot().lazy_pending_blocks, 2);
    assert_eq!(
        r.tier.snapshot().writeback_dropped,
        0,
        "not dropped, only deferred"
    );
    r.tier.begin_step();
    assert_eq!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(4.0), &view)
            .unwrap(),
        vec![p.keys[1]]
    );
}

#[test]
fn a_lazily_stored_block_restores_its_bytes() {
    let r = lazy_rig(64);
    let (p, blocks, view) = buffer(&r, 2, |i| i);
    let want: Vec<Vec<u8>> = blocks.iter().map(|b| read_block(&r.gpu, b)).collect();
    assert_eq!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(2.0), &view)
            .unwrap()
            .len(),
        2
    );
    r.tier.poll_writebacks(r.gpu.as_ref());
    for b in &blocks {
        clobber(&r.gpu, b);
    }
    let dst = [empty_block(&r.gpu), empty_block(&r.gpu)];
    let t = r.tier.begin_restore(r.gpu.as_ref(), &p.keys, &dst).unwrap();
    r.tier.wait(r.gpu.as_ref(), &t).unwrap();
    r.tier.release(t);
    for (d, w) in dst.iter().zip(&want) {
        assert_eq!(&read_block(&r.gpu, d), w);
    }
}

#[test]
fn a_repeated_lazy_store_is_present_not_buffered_twice() {
    let r = lazy_rig(64);
    let (p, blocks, _view) = buffer(&r, 2, |i| i);
    let rep = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
        .unwrap();
    assert_eq!(rep.present, vec![0, 1]);
    assert_eq!(r.tier.snapshot().lazy_pending_blocks, 2);
}

#[test]
fn eager_mode_has_nothing_to_drain() {
    let r = rig(16, TierPolicy::Lru);
    let view = FakeView::default();
    assert!(
        r.tier
            .drain_lazy(r.gpu.as_ref(), demand(8.0), &view)
            .unwrap()
            .is_empty()
    );
}
