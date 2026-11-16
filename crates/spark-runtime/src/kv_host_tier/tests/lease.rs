// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: read leases and ticket lifetime: a leased entry survives eviction pressure,
//! a lease lasts exactly until the H2D event completed (never expires while the copy may run),
//! an abandoned ticket is reclaimed after the lease TTL, and the in-flight cap bounds leases.

use super::*;
use crate::kv_host_tier::{RestoreState, TierPolicy};

#[test]
fn a_leased_entry_is_never_evicted() {
    // Budget 5 blocks: eviction starts past 4 blocks and evicts down to 3.
    let r = rig(5, TierPolicy::Lru);
    let a = Prompt::new(NS, 0, 2);
    store_published(&r, &a, &[device_block(&r.gpu, 1), device_block(&r.gpu, 2)]);
    r.gpu.hold_events(true);
    let dst = [empty_block(&r.gpu), empty_block(&r.gpu)];
    let t = r.tier.begin_restore(r.gpu.as_ref(), &a.keys, &dst).unwrap();
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Pending
    );
    r.gpu.hold_events(false);
    // Three more blocks need eviction, but A is leased: only what fits without A is stored.
    let b = Prompt::new(NS, 1000, 3);
    let bb: Vec<BlockRegions> = (0..3).map(|i| device_block(&r.gpu, 10 + i)).collect();
    let rep = r
        .tier
        .store_detailed(r.gpu.as_ref(), &b.keys, &b.chunks(), &bb)
        .unwrap();
    assert_eq!(
        rep.started.len(),
        2,
        "the staging cap (half of 5 blocks) stops the third"
    );
    assert_eq!(rep.dropped, vec![2]);
    r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(
        r.tier.lookup(&a.keys, &a.chunks()),
        2,
        "leased blocks survived"
    );
    assert_eq!(r.tier.snapshot().evictions, 0);
    // The copy completes: leases drop and A becomes evictable again.
    r.gpu.complete_all_events();
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Ready
    );
    r.tier.release(t);
    assert_eq!(r.tier.snapshot().leased_bytes, 0);
    let c = Prompt::new(NS, 2000, 1);
    assert_eq!(
        r.tier
            .store(
                r.gpu.as_ref(),
                &c.keys,
                &c.chunks(),
                &[device_block(&r.gpu, 30)]
            )
            .unwrap(),
        1
    );
    assert!(r.tier.snapshot().evictions > 0);
}

#[test]
fn a_lease_lasts_until_the_copy_completed_even_past_the_ttl() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 0, 1);
    store_published(&r, &p, &[device_block(&r.gpu, 1)]);
    r.gpu.hold_events(true);
    let t = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys, &[empty_block(&r.gpu)])
        .unwrap();
    r.tier.release(t);
    r.clock.advance(std::time::Duration::from_secs(3600));
    r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(
        r.tier.snapshot().leased_bytes,
        BLOCK_BYTES as u64,
        "copy may still read the run"
    );
    r.gpu.complete_all_events();
    r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(r.tier.snapshot().leased_bytes, 0);
}

#[test]
fn an_abandoned_ticket_is_reclaimed_after_the_ttl() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 0, 1);
    store_published(&r, &p, &[device_block(&r.gpu, 1)]);
    let t = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys, &[empty_block(&r.gpu)])
        .unwrap();
    // The caller never polls or releases. A sweep finishes the copy and starts the TTL.
    r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(
        r.tier.snapshot().leased_bytes,
        0,
        "the finished copy dropped its lease"
    );
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Ready
    );
    r.clock
        .advance(r.tier.options().lease_ttl + std::time::Duration::from_secs(1));
    r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Failed,
        "the record was reclaimed"
    );
    r.tier.release(t);
}

#[test]
fn several_restores_may_lease_one_entry() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 0, 1);
    store_published(&r, &p, &[device_block(&r.gpu, 1)]);
    r.gpu.hold_events(true);
    let t1 = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys, &[empty_block(&r.gpu)])
        .unwrap();
    let t2 = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys, &[empty_block(&r.gpu)])
        .unwrap();
    r.gpu.complete_event(t1.event());
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t1).unwrap(),
        RestoreState::Ready
    );
    assert_eq!(
        r.tier.snapshot().leased_bytes,
        BLOCK_BYTES as u64,
        "t2 still leases it"
    );
    r.gpu.complete_event(t2.event());
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t2).unwrap(),
        RestoreState::Ready
    );
    assert_eq!(r.tier.snapshot().leased_bytes, 0);
    r.tier.release(t1);
    r.tier.release(t2);
}

#[test]
fn the_in_flight_cap_bounds_leased_bytes() {
    // Budget 8 blocks: at most 4 leased at once.
    let r = rig(8, TierPolicy::Lru);
    let p = Prompt::new(NS, 0, 6);
    let blocks: Vec<BlockRegions> = (0..6).map(|i| device_block(&r.gpu, i)).collect();
    // Two store calls: each stays under the staging cap.
    let chunks = p.chunks();
    for rg in [0..3, 3..6] {
        let n = r
            .tier
            .store(
                r.gpu.as_ref(),
                &p.keys[rg.clone()],
                &chunks[rg.clone()],
                &blocks[rg],
            )
            .unwrap();
        assert_eq!(n, 3);
        r.tier.poll_writebacks(r.gpu.as_ref());
    }
    assert_eq!(r.tier.max_restore_blocks(), 4);
    r.gpu.hold_events(true);
    let dst: Vec<BlockRegions> = (0..4).map(|_| empty_block(&r.gpu)).collect();
    let t = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys[..4], &dst)
        .unwrap();
    assert_eq!(r.tier.max_restore_blocks(), 0);
    assert!(
        r.tier
            .begin_restore(r.gpu.as_ref(), &p.keys[4..5], &[empty_block(&r.gpu)])
            .is_err()
    );
    r.gpu.complete_all_events();
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Ready
    );
    r.tier.release(t);
    assert_eq!(r.tier.max_restore_blocks(), 4);
}
