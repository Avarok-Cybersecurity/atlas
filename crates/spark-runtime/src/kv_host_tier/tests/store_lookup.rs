// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: store, publish, lookup and restore through the public tier API on the mock
//! backend: byte-identical restores, forged keys, staging invisibility, fingerprint misses,
//! failure cleanup, the per-step budget and the staging cap.

use super::*;
use crate::kv_host_tier::{RestoreState, TierPolicy};

#[test]
fn restore_replays_the_stored_bytes_after_the_gpu_copy_is_gone() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 100, 3);
    let blocks: Vec<BlockRegions> = (0..3).map(|i| device_block(&r.gpu, i as u8 + 1)).collect();
    let originals: Vec<Vec<u8>> = blocks.iter().map(|b| read_block(&r.gpu, b)).collect();
    assert_eq!(store_published(&r, &p, &blocks), 3);
    // The radix tree evicts the blocks and hands them to another sequence.
    for b in &blocks {
        clobber(&r.gpu, b);
    }
    assert_eq!(r.tier.lookup(&p.keys, &p.chunks()), 3);
    let dst: Vec<BlockRegions> = (0..3).map(|_| empty_block(&r.gpu)).collect();
    let t = r
        .tier
        .begin_restore(r.gpu.as_ref(), &p.keys, &dst)
        .expect("restore");
    assert_eq!(t.blocks(), 3);
    assert_eq!(t.tokens(), 3 * BS);
    assert_eq!(
        r.tier.poll(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Ready
    );
    for (d, want) in dst.iter().zip(&originals) {
        assert_eq!(&read_block(&r.gpu, d), want, "restored bytes differ");
    }
    r.tier.release(t);
    let s = r.tier.snapshot();
    assert_eq!(s.restore_blocks, 3);
    assert_eq!(s.restore_tokens, (3 * BS) as u64);
    assert_eq!(s.writeback_blocks, 3);
    assert_eq!(s.leased_bytes, 0);
    assert_eq!(s.hits, 1);
}

#[test]
fn a_forged_key_with_other_tokens_misses() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 7, 2);
    let blocks: Vec<BlockRegions> = (0..2).map(|i| device_block(&r.gpu, i)).collect();
    store_published(&r, &p, &blocks);
    // Same keys, other tokens: a hash-only index would serve the wrong prefix.
    let other: Vec<u32> = p.tokens.iter().map(|t| t + 1).collect();
    let chunks: Vec<&[u32]> = other.chunks_exact(BS).collect();
    assert_eq!(r.tier.lookup(&p.keys, &chunks), 0);
    // A store under an existing key with other tokens is dropped, never overwrites.
    let fresh = device_block(&r.gpu, 99);
    let rep = r
        .tier
        .store_detailed(
            r.gpu.as_ref(),
            &p.keys[..1],
            &chunks[..1],
            std::slice::from_ref(&fresh),
        )
        .unwrap();
    assert_eq!(rep.dropped, vec![0]);
    assert_eq!(r.tier.lookup(&p.keys, &p.chunks()), 2);
    assert!(r.tier.snapshot().misses >= 1);
}

#[test]
fn staging_entries_stay_invisible_until_their_copy_completed() {
    let r = rig(16, TierPolicy::Lru);
    r.gpu.hold_events(true);
    let p = Prompt::new(NS, 1, 2);
    let blocks: Vec<BlockRegions> = (0..2).map(|i| device_block(&r.gpu, i)).collect();
    let rep = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
        .unwrap();
    assert_eq!(rep.started, vec![0, 1]);
    assert_eq!(
        r.tier.lookup(&p.keys, &p.chunks()),
        0,
        "staging must not be visible"
    );
    assert!(
        r.tier.poll_writebacks(r.gpu.as_ref()).is_empty(),
        "nothing completed yet"
    );
    let dst = [empty_block(&r.gpu), empty_block(&r.gpu)];
    assert!(r.tier.begin_restore(r.gpu.as_ref(), &p.keys, &dst).is_err());
    assert_eq!(r.tier.snapshot().staging_bytes, 2 * BLOCK_BYTES as u64);
    // Storing again while staging is "present", not a second copy.
    let again = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
        .unwrap();
    assert_eq!(again.present, vec![0, 1]);
    r.gpu.complete_all_events();
    let done = r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(done, p.keys);
    assert_eq!(r.tier.lookup(&p.keys, &p.chunks()), 2);
    assert_eq!(r.tier.snapshot().staging_bytes, 0);
}

#[test]
fn a_stale_model_fingerprint_misses() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 40, 2);
    let blocks: Vec<BlockRegions> = (0..2).map(|i| device_block(&r.gpu, i)).collect();
    store_published(&r, &p, &blocks);
    let other_model = Prompt::new(NS + 1, 40, 2);
    assert_eq!(other_model.tokens, p.tokens);
    assert_eq!(r.tier.lookup(&other_model.keys, &other_model.chunks()), 0);
    let other_adapter = BlockKey::chain_blocks(NS, 3, &p.tokens, BS);
    assert_eq!(r.tier.lookup(&other_adapter, &p.chunks()), 0);
}

#[test]
fn lookup_counts_only_the_leading_run_of_hits() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 9, 4);
    let blocks: Vec<BlockRegions> = (0..4).map(|i| device_block(&r.gpu, i)).collect();
    let chunks = p.chunks();
    let keep = [0usize, 1, 3];
    let keys: Vec<BlockKey> = keep.iter().map(|&i| p.keys[i]).collect();
    let toks: Vec<&[u32]> = keep.iter().map(|&i| chunks[i]).collect();
    let regs: Vec<BlockRegions> = keep.iter().map(|&i| blocks[i].clone()).collect();
    assert_eq!(
        r.tier.store(r.gpu.as_ref(), &keys, &toks, &regs).unwrap(),
        3
    );
    r.tier.poll_writebacks(r.gpu.as_ref());
    assert_eq!(
        r.tier.lookup(&p.keys, &chunks),
        2,
        "block 2 is missing, so block 3 cannot count"
    );
}

#[test]
fn a_failed_submission_queues_nothing_and_returns_every_run() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 3, 2);
    let blocks = [device_block(&r.gpu, 1), bogus_block()];
    assert!(
        r.tier
            .store(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
            .is_err()
    );
    assert_eq!(r.arena.live(), 0, "every host run went back");
    let s = r.tier.snapshot();
    assert_eq!(s.entries, 0);
    assert_eq!(s.resident_bytes, 0);
    assert!(r.tier.poll_writebacks(r.gpu.as_ref()).is_empty());
    // The tier keeps working.
    let good = [device_block(&r.gpu, 1), device_block(&r.gpu, 2)];
    assert_eq!(store_published(&r, &p, &good), 2);
}

#[test]
fn a_failed_restore_leases_nothing() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 3, 1);
    store_published(&r, &p, &[device_block(&r.gpu, 1)]);
    assert!(
        r.tier
            .begin_restore(r.gpu.as_ref(), &p.keys, &[bogus_block()])
            .is_err()
    );
    assert_eq!(r.tier.snapshot().leased_bytes, 0);
    assert_eq!(r.tier.snapshot().restore_failed, 1);
    let dst = [empty_block(&r.gpu)];
    let t = r.tier.begin_restore(r.gpu.as_ref(), &p.keys, &dst).unwrap();
    assert_eq!(
        r.tier.wait(r.gpu.as_ref(), &t).unwrap(),
        RestoreState::Ready
    );
    r.tier.release(t);
}

#[test]
fn blocks_of_the_wrong_size_or_layout_are_refused() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 3, 1);
    let short = BlockRegions {
        regions: vec![(r.gpu.alloc(REGION).unwrap(), REGION)],
    };
    assert!(
        r.tier
            .store(r.gpu.as_ref(), &p.keys, &p.chunks(), &[short])
            .is_err()
    );
    store_published(&r, &p, &[device_block(&r.gpu, 1)]);
    // Same total bytes, other split: refused against the first store's layout.
    let half = REGION / 2;
    let odd = BlockRegions {
        regions: (0..LAYERS * 4)
            .map(|_| (r.gpu.alloc(half).unwrap(), half))
            .collect(),
    };
    let q = Prompt::new(NS, 50, 1);
    assert!(
        r.tier
            .store(
                r.gpu.as_ref(),
                &q.keys,
                &q.chunks(),
                std::slice::from_ref(&odd)
            )
            .is_err()
    );
    assert!(
        r.tier
            .begin_restore(r.gpu.as_ref(), &p.keys, &[odd])
            .is_err()
    );
}

#[test]
fn the_step_budget_caps_write_back_bytes() {
    let mut cfg = cfg_blocks(64, TierPolicy::Lru);
    cfg.writeback_bytes_per_step = 2 * BLOCK_BYTES as u64;
    let r = rig_with(cfg, test_opts());
    let p = Prompt::new(NS, 1, 5);
    let blocks: Vec<BlockRegions> = (0..5).map(|i| device_block(&r.gpu, i)).collect();
    r.tier.begin_step();
    let rep = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
        .unwrap();
    assert_eq!(rep.started, vec![0, 1]);
    assert_eq!(rep.dropped, vec![2, 3, 4]);
    // The same step has nothing left.
    let rep2 = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys[2..], &p.chunks()[2..], &blocks[2..])
        .unwrap();
    assert!(rep2.started.is_empty());
    r.tier.begin_step();
    let rep3 = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys[2..], &p.chunks()[2..], &blocks[2..])
        .unwrap();
    assert_eq!(rep3.started, vec![0, 1]);
    assert_eq!(r.tier.snapshot().writeback_dropped, 3 + 3 + 1);
}

#[test]
fn staging_bytes_stay_under_half_the_budget() {
    let r = rig(8, TierPolicy::Lru);
    r.gpu.hold_events(true);
    let p = Prompt::new(NS, 1, 6);
    let blocks: Vec<BlockRegions> = (0..6).map(|i| device_block(&r.gpu, i)).collect();
    let rep = r
        .tier
        .store_detailed(r.gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
        .unwrap();
    assert_eq!(rep.started.len(), 4, "in-flight cap is half of 8 blocks");
    assert_eq!(rep.dropped, vec![4, 5]);
    r.gpu.complete_all_events();
    assert_eq!(r.tier.poll_writebacks(r.gpu.as_ref()).len(), 4);
}

#[test]
fn shutdown_returns_every_run_and_every_pending_key() {
    let r = rig(16, TierPolicy::Lru);
    let p = Prompt::new(NS, 1, 2);
    store_published(&r, &p, &[device_block(&r.gpu, 1), device_block(&r.gpu, 2)]);
    r.gpu.hold_events(true);
    let q = Prompt::new(NS, 500, 1);
    r.tier
        .store(
            r.gpu.as_ref(),
            &q.keys,
            &q.chunks(),
            &[device_block(&r.gpu, 3)],
        )
        .unwrap();
    let done = r.tier.shutdown(r.gpu.as_ref());
    assert_eq!(done, q.keys);
    assert_eq!(r.arena.live(), 0);
    assert_eq!(r.tier.snapshot().resident_bytes, 0);
}

#[test]
fn the_contract_constructor_runs_on_the_pinned_pool() {
    let gpu = std::sync::Arc::new(MockGpuBackend::new());
    let dyn_gpu: std::sync::Arc<dyn GpuBackend> = gpu.clone();
    let pool =
        crate::pinned_pool::PinnedPool::new(dyn_gpu, 1 << 20, 1 << 20, 1 << 20).expect("pool");
    let cfg = TierConfig {
        budget_bytes: 1 << 20,
        ..TierConfig::default()
    };
    let stream = gpu.create_copy_stream().unwrap();
    let tier = KvHostTier::new(cfg, pool.clone(), BLOCK_BYTES, stream);
    let p = Prompt::new(tier.options().namespace, 1, 2);
    let blocks = [device_block(&gpu, 1), device_block(&gpu, 2)];
    let want: Vec<Vec<u8>> = blocks.iter().map(|b| read_block(&gpu, b)).collect();
    assert_eq!(
        tier.store(gpu.as_ref(), &p.keys, &p.chunks(), &blocks)
            .unwrap(),
        2
    );
    assert_eq!(tier.poll_writebacks(gpu.as_ref()).len(), 2);
    assert!(pool.stats().used >= 2 * BLOCK_BYTES);
    let dst = [empty_block(&gpu), empty_block(&gpu)];
    let t = tier.begin_restore(gpu.as_ref(), &p.keys, &dst).unwrap();
    assert_eq!(tier.poll(gpu.as_ref(), &t).unwrap(), RestoreState::Ready);
    tier.release(t);
    for (d, w) in dst.iter().zip(&want) {
        assert_eq!(&read_block(&gpu, d), w);
    }
    tier.shutdown(gpu.as_ref());
    assert_eq!(pool.stats().used, 0, "every page run went back to the pool");
}
