// SPDX-License-Identifier: AGPL-3.0-only

//! Hybrid prefix-cache invariant: lookup_paired is KV+SSM or it is a miss.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

#[test]
fn lookup_paired_kv_without_ssm_is_a_miss() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    let raw = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(
        raw.matched_tokens, 64,
        "raw lookup still reports the KV walk"
    );
    assert_eq!(raw.ssm_snapshot, None);
    tree.release(&tokens, 16, 0);

    let paired = tree.lookup_paired(&tokens, 16, 0, 0);
    assert!(
        paired.is_empty(),
        "hybrid hit requires a restorable SSM snapshot; got {paired:?}"
    );
    assert_eq!(paired.matched_tokens, 0);
    assert!(paired.ssm_snapshot.is_none());
}

#[test]
fn lookup_paired_restores_without_full_kv_recompute() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 42, 7, 0, 0);

    let m = tree.lookup_paired(&tokens, 16, 7, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(42));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    assert_eq!(
        m.paired_ssm_tokens(16),
        Some(64),
        "hit length equals the restored snapshot"
    );
    tree.release(&tokens, 16, 0);
}

#[test]
fn lookup_paired_block_alignment_edge_is_a_miss() {
    // Snapshot registered at 48 tokens; next-turn KV walk floors to 32.
    // That is the #353 one-block-short edge: not a hit.
    let tree = RadixTree::new();
    let full: Vec<u32> = (0..48).collect();
    tree.insert_with_snapshot(&full, &[10, 20, 30], &[], 16, 99, 7, 0, 0);

    let short: Vec<u32> = (0..32).collect();
    let raw = tree.lookup(&short, 16, 7, 0);
    assert_eq!(raw.matched_tokens, 32);
    assert_eq!(raw.ssm_snapshot, None);
    tree.release(&short, 16, 0);

    let paired = tree.lookup_paired(&short, 16, 7, 0);
    assert!(
        paired.is_empty(),
        "one-block-short of the snapshot is a miss, not a lying hit: {paired:?}"
    );
}

#[test]
fn lookup_paired_trims_kv_to_shallower_snapshot() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);
    let at_32: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&at_32, &[10, 20], &[], 16, 50, 7, 0, 0);

    let m = tree.lookup_paired(&tokens, 16, 7, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release_matched(&tokens, 16, 32, 0);
}

#[test]
fn lookup_paired_releases_unpaired_kv_refs() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    tree.insert(&tokens, &[10], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    let miss = tree.lookup_paired(&tokens, 16, 0, 0);
    assert!(miss.is_empty());

    // Unpaired lookup must not leave a walk-ref that pins the node.
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);
}

#[test]
fn lookup_paired_non_tail_is_content_keyed() {
    // #353: non-tail snapshots are content-addressed. A different
    // session_hash (the unstable first-1024 hash) must still hit.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20], &[], 16, 8, 0x1111, 0, 0);

    let m = tree.lookup_paired(&tokens, 16, 0x2222, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, Some(8));
    tree.release(&tokens, 16, 0);
}

#[test]
fn lookup_paired_tail_stays_session_gated() {
    // #345 / #353: tails bleed past the exact prefix and stay session-gated.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    let displaced = tree.insert_tail_snapshot(&tokens, 5, 0xaaaa, 0);
    assert!(displaced.is_empty());

    let foreign = tree.lookup_paired(&tokens, 16, 0xbbbb, 0);
    assert!(
        foreign.is_empty(),
        "a foreign session must not restore a tail: {foreign:?}"
    );

    let same = tree.lookup_paired(&tokens, 16, 0xaaaa, 0);
    assert_eq!(same.matched_tokens, 32);
    assert_eq!(same.ssm_snapshot, Some(5));
    assert!(same.ssm_snapshot_is_tail);
    tree.release(&tokens, 16, 0);
}
