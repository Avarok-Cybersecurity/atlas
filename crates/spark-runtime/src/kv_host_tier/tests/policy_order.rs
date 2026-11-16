// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: eviction order through the tier and at the policy level: watermark batch
//! eviction, LRU tail-first ordering, restore touches, heat with halving, and victims never
//! chosen among entries the caller's predicate rejects.

use super::*;
use crate::kv_host_tier::TierPolicy;
use crate::kv_host_tier::policy::Policy;

fn key(i: u32) -> BlockKey {
    BlockKey::chain(NS, 0, None, &[i])
}

fn publish(r: &Rig, p: &Prompt, seed: u8) {
    let blocks: Vec<BlockRegions> = (0..p.keys.len())
        .map(|i| device_block(&r.gpu, seed + i as u8))
        .collect();
    assert_eq!(store_published(r, p, &blocks), p.keys.len());
}

#[test]
fn a_full_tier_evicts_a_batch_tail_first() {
    // Budget 10 blocks: past 8 blocks, evict down to 6.
    let r = rig(10, TierPolicy::Lru);
    let a = Prompt::new(NS, 0, 4);
    let b = Prompt::new(NS, 1000, 4);
    publish(&r, &a, 1);
    publish(&r, &b, 20);
    assert_eq!(r.tier.snapshot().evictions, 0);
    let c = Prompt::new(NS, 2000, 1);
    publish(&r, &c, 40);
    // 8 + 1 > 8: free 3 blocks, the LRU request's LAST blocks first.
    assert_eq!(r.tier.snapshot().evictions, 3);
    assert_eq!(
        r.tier.lookup(&a.keys, &a.chunks()),
        1,
        "a0 stays, a1..a3 left tail first"
    );
    assert_eq!(r.tier.lookup(&b.keys, &b.chunks()), 4);
    assert_eq!(r.tier.lookup(&c.keys, &c.chunks()), 1);
    assert_eq!(r.tier.snapshot().resident_bytes, 6 * BLOCK_BYTES as u64);
    assert_eq!(r.arena.live(), 6);
}

#[test]
fn a_restore_makes_its_blocks_most_recent() {
    let r = rig(10, TierPolicy::Lru);
    let a = Prompt::new(NS, 0, 4);
    let b = Prompt::new(NS, 1000, 4);
    publish(&r, &a, 1);
    publish(&r, &b, 20);
    let dst: Vec<BlockRegions> = (0..4).map(|_| empty_block(&r.gpu)).collect();
    let t = r.tier.begin_restore(r.gpu.as_ref(), &a.keys, &dst).unwrap();
    r.tier.poll(r.gpu.as_ref(), &t).unwrap();
    r.tier.release(t);
    publish(&r, &Prompt::new(NS, 2000, 1), 40);
    assert_eq!(r.tier.lookup(&a.keys, &a.chunks()), 4);
    assert_eq!(
        r.tier.lookup(&b.keys, &b.chunks()),
        1,
        "B is now least recent: b1..b3 left"
    );
}

#[test]
fn the_pool_refusing_a_run_evicts_one_more_block() {
    let r = rig(100, TierPolicy::Lru);
    let a = Prompt::new(NS, 0, 3);
    publish(&r, &a, 1);
    // Far under the watermark, but the arena (a pool still pinning) holds only 3 runs.
    r.arena.set_max_runs(3);
    let b = Prompt::new(NS, 1000, 1);
    publish(&r, &b, 9);
    assert_eq!(r.tier.snapshot().evictions, 1);
    assert_eq!(
        r.tier.lookup(&a.keys, &a.chunks()),
        2,
        "a2 (the tail) made room"
    );
}

#[test]
fn lru_policy_skips_rejected_keys() {
    let mut p = Policy::new(TierPolicy::Lru, 8, 1024);
    p.insert_batch(&[key(0), key(1), key(2)]);
    // Tail-first: key(2) is least recent, then key(1), then key(0).
    let v = p.victims(2, &|k: &BlockKey| *k != key(2));
    assert_eq!(v, vec![key(1), key(0)]);
    assert_eq!(p.len(), 1);
    assert_eq!(p.victims(5, &|_: &BlockKey| true), vec![key(2)]);
    assert_eq!(p.kind(), TierPolicy::Lru);
}

#[test]
fn heat_keeps_hot_blocks_and_halves_over_time() {
    let mut p = Policy::new(TierPolicy::Heat, 8, 4);
    p.insert_batch(&[key(0), key(1)]);
    p.touch_batch(&[key(0)]);
    {
        let h = p.as_heat().unwrap();
        // Three accesses so far (two inserts, one touch): no halving yet.
        assert_eq!(h.heat_of(&key(0)), Some(2));
        assert_eq!(h.heat_of(&key(1)), Some(1));
    }
    p.touch_batch(&[key(0)]);
    {
        let h = p.as_heat().unwrap();
        // The fourth access crossed the halving point: key(0) was 2, halved to 1, then +1.
        assert_eq!(h.heat_of(&key(0)), Some(2));
        assert_eq!(h.heat_of(&key(1)), Some(0), "halved away");
    }
    assert_eq!(p.victims(1, &|_: &BlockKey| true), vec![key(1)]);
    // Equal heat: the older stamp leaves first, tail-first within a batch.
    p.insert_batch(&[key(5), key(6)]);
    let v = p.victims(2, &|k: &BlockKey| *k != key(0));
    assert_eq!(v, vec![key(6), key(5)]);
}

#[test]
fn heat_through_the_tier_evicts_cold_requests_first() {
    let r = rig(10, TierPolicy::Heat);
    let a = Prompt::new(NS, 0, 4);
    let b = Prompt::new(NS, 1000, 4);
    publish(&r, &a, 1);
    publish(&r, &b, 20);
    for _ in 0..3 {
        let dst: Vec<BlockRegions> = (0..4).map(|_| empty_block(&r.gpu)).collect();
        let t = r.tier.begin_restore(r.gpu.as_ref(), &a.keys, &dst).unwrap();
        r.tier.poll(r.gpu.as_ref(), &t).unwrap();
        r.tier.release(t);
    }
    publish(&r, &Prompt::new(NS, 2000, 1), 40);
    assert_eq!(r.tier.lookup(&a.keys, &a.chunks()), 4, "A is hot");
    assert_eq!(
        r.tier.lookup(&b.keys, &b.chunks()),
        1,
        "B was cold: its tail left"
    );
}
