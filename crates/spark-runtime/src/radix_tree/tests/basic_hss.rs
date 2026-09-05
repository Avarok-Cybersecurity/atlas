// SPDX-License-Identifier: AGPL-3.0-only

//! HSS (`--high-speed-swap`) disk-block-ref acquisition checks for
//! `RadixTree::insert`. Split out of `basic.rs` under the workspace 500-LoC
//! cap; nested via `#[path]` so they register as
//! `radix_tree::tests::basic::hss::*`.

use super::*;

/// Issue #17 regression: when HSS is active, the cache must report which
/// disk_block_ids it newly takes ownership of so the caller can balance
/// `inc_disk_ref`/`dec_disk_ref`. Without this, the second request panics
/// at `high_speed_swap.rs:167` because the cache holds a stale ID whose
/// refcount has already hit zero.
#[test]
fn test_hss_disk_ref_acquisition_cold_insert() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let block_table = vec![10, 20];
    let disk_ids = vec![100u32, 101u32];

    // Cold insert with HSS active: cache takes ownership of both disk_ids.
    let acquired = tree.insert(&tokens, &block_table, &disk_ids, 16, 0, 0);
    assert_eq!(acquired.disk_block_ids, vec![100, 101]);
    // Both nodes are new, so the cache also takes a KV ref on both blocks.
    assert_eq!(acquired.blocks, vec![10, 20]);
}

#[test]
fn test_hss_disk_ref_acquisition_re_insert_no_double() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    let block_table = vec![10, 20];
    let disk_ids = vec![100u32, 101u32];

    // First insert acquires both.
    let acquired1 = tree.insert(&tokens, &block_table, &disk_ids, 16, 0, 0);
    assert_eq!(acquired1.disk_block_ids, vec![100, 101]);

    // Second insert of the same prefix: nothing newly acquired (the cache
    // already owns these). Re-acquiring would over-inc and leak the disk
    // refcount.
    let acquired2 = tree.insert(&tokens, &block_table, &disk_ids, 16, 0, 0);
    assert!(
        acquired2.disk_block_ids.is_empty(),
        "re-insert should not re-acquire disk_ids; got {acquired2:?}"
    );
    // No node was created, so no KV ref is taken either — the cache already
    // holds exactly one per existing node.
    assert!(
        acquired2.blocks.is_empty(),
        "re-insert should not re-ref blocks; got {acquired2:?}"
    );
}

#[test]
fn test_hss_disk_ref_acquisition_extension() {
    let tree = RadixTree::new();
    let tokens_short: Vec<u32> = (0..32).collect();
    let tokens_long: Vec<u32> = (0..48).collect();

    // Insert 2 blocks first.
    let acquired1 = tree.insert(&tokens_short, &[10, 20], &[100u32, 101u32], 16, 0, 0);
    assert_eq!(acquired1.disk_block_ids, vec![100, 101]);

    // Extension: same first 2 blocks (already cached) + 1 new block.
    // Only the new block's disk_id should be reported as acquired.
    let acquired2 = tree.insert(
        &tokens_long,
        &[10, 20, 30],
        &[100u32, 101u32, 102u32],
        16,
        0,
        0,
    );
    assert_eq!(acquired2.disk_block_ids, vec![102]);
    // Only the newly created node's block is newly owned.
    assert_eq!(acquired2.blocks, vec![30]);
}

#[test]
fn test_hss_disk_ref_acquisition_no_op_when_hss_inactive() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();

    // Empty disk_ids slice ⇒ HSS not active ⇒ nothing acquired regardless
    // of whether nodes are new or pre-existing.
    let acquired = tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    assert!(acquired.disk_block_ids.is_empty());
}
