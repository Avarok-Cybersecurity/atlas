// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU oracles of the lazy store mode (`--kv-tier-store lazy`) on the mock rig
//! of `tests.rs`: a buffered store copies nothing and keeps its block; a block the radix evicts
//! while buffered is copied at the next tick before it can be recycled and restores with
//! identical bytes; a block a live sequence still uses is not copied early; the eviction view
//! ranks what the reference counts say.

use spark_runtime::kv_host_tier::{EvictionView, GpuBlockState, StoreMode, TierOptions};
use spark_runtime::prefix_cache::PrefixCache;

use super::plan::PrefetchOutcome;
use super::store::HeldView;
use super::tests::{Rig, prompt};

fn lazy_rig() -> Rig {
    Rig::new(
        64,
        256,
        TierOptions {
            store_mode: StoreMode::Lazy,
            ..TierOptions::default()
        },
    )
}

fn lazy_held(rig: &Rig) -> usize {
    rig.live
        .books
        .lock()
        .held
        .values()
        .filter(|h| h.lazy)
        .count()
}

#[test]
fn a_buffered_store_copies_nothing_and_holds_its_block() {
    let mut rig = lazy_rig();
    let tokens = prompt(4, 100);
    let (table, _) = rig.prefill(&tokens, true);
    assert_eq!(
        lazy_held(&rig),
        4,
        "every block buffered, one reference each"
    );
    for &b in &table {
        assert_eq!(
            rig.kv.ref_count(b),
            2,
            "the cache's reference plus the glue's"
        );
    }
    assert_eq!(rig.live.tier.snapshot().lazy_pending_blocks, 4);
    // Nothing published: a restore finds nothing yet.
    rig.evict_all();
    assert_eq!(rig.prefetch(&tokens, false).outcome, PrefetchOutcome::Miss);
}

#[test]
fn an_evicted_buffered_block_is_copied_before_it_is_recycled() {
    let mut rig = lazy_rig();
    let tokens = prompt(5, 200);
    let (table, bytes) = rig.prefill(&tokens, true);
    rig.evict_all();
    for &b in &table {
        assert_eq!(
            rig.kv.ref_count(b),
            1,
            "evicted from the radix, held for the copy"
        );
    }
    let free = rig.kv.num_free_blocks();
    // The tick drains the urgent blocks (rank 0), and the copies complete at once on the mock.
    rig.live.tick(&rig.gpu, &mut rig.kv);
    assert_eq!(lazy_held(&rig), 0, "every evicted block started its copy");
    assert_eq!(rig.live.tier.snapshot().lazy_pending_blocks, 0);
    rig.live.tick(&rig.gpu, &mut rig.kv);
    assert_eq!(rig.held(), 0, "the next tick dropped the references");
    assert_eq!(rig.kv.num_free_blocks(), free + 5);
    rig.scribble_free();
    let p = rig.prefetch(&tokens, false);
    assert_eq!(p.outcome, PrefetchOutcome::Restored);
    assert_eq!(rig.gpu_hit(&tokens), bytes);
}

#[test]
fn a_block_in_use_is_not_copied_early() {
    let mut rig = lazy_rig();
    let tokens = prompt(3, 300);
    // The sequence stays live: its blocks carry its reference too.
    let (table, _) = rig.prefill(&tokens, false);
    for &b in &table {
        assert_eq!(rig.kv.ref_count(b), 3);
    }
    rig.live.tick(&rig.gpu, &mut rig.kv);
    assert_eq!(lazy_held(&rig), 3, "in use: still buffered");
    assert_eq!(rig.live.tier.snapshot().lazy_pending_blocks, 3);
}

#[test]
fn the_view_ranks_by_reference_count_then_age() {
    let mut rig = lazy_rig();
    let a = prompt(2, 400);
    let (ta, _) = rig.prefill(&a, false); // in use
    let b = prompt(2, 500);
    let (tb, _) = rig.prefill(&b, true); // cached, evictable
    let keys: Vec<_> = {
        let books = rig.live.books.lock();
        let mut by_block: Vec<_> = books.held.iter().map(|(k, h)| (h.block, *k)).collect();
        by_block.sort();
        by_block
    };
    let key_of = |blk: u32| keys.iter().find(|(b, _)| *b == blk).unwrap().1;
    // Evict b's tail block only: its node goes, a's are pinned by the live sequence.
    let ev = rig.cache.evict(1);
    assert_eq!(ev.physical, vec![tb[1]], "the LRU evictable leaf");
    for &blk in &ev.physical {
        rig.kv.return_evicted_block(blk);
    }
    let view = {
        let books = rig.live.books.lock();
        HeldView::new(&books, &rig.kv)
    };
    for &blk in &ta {
        assert_eq!(view.state(&key_of(blk)), GpuBlockState::InUse);
    }
    assert_eq!(
        view.state(&key_of(tb[1])),
        GpuBlockState::Queued { rank: 0 }
    );
    assert_eq!(
        view.state(&key_of(tb[0])),
        GpuBlockState::Queued { rank: 1 }
    );
    let unknown = spark_runtime::kv_host_tier::BlockKey([0xab; 16]);
    assert_eq!(view.state(&unknown), GpuBlockState::Gone);
    assert_eq!(view.into_asked().len(), ta.len() + tb.len() + 1);
}
