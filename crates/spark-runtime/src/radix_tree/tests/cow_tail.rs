// SPDX-License-Identifier: AGPL-3.0-only

//! Row `prefix_cow_tail` (issue #1193): a sub-block match through
//! `lookup_cow` never hands out a shared block as the requester's tail.
//!
//! The source of the tail comes back in `cow_tail`, outside
//! `matched_blocks`; the requester copies it into a private block (the
//! copy itself is `spark-model .../prefill_b/prefix_cow.rs`). These tests pin
//! the radix half: which source, how many rows, which references, and that
//! every other path (plain `lookup`, the row off, the legacy lever) is
//! unchanged.

use crate::prefix_cache::{PrefixCache, PrefixMatch};
use crate::radix_tree::RadixTree;

use super::{arm_cow_tail, arm_legacy_partial_tail};

const BS: usize = 16;

/// The shared part of a CoW match is exactly the full blocks, and the
/// source is not among them.
fn assert_tail_is_copied_not_shared(m: &PrefixMatch, src: u32, rows: usize) {
    assert_eq!(m.cow_tail, Some((src, rows)), "{m:?}");
    assert!(
        !m.matched_blocks.contains(&src),
        "the copy source must never be handed out as a shared block: {m:?}"
    );
    assert_eq!(m.matched_tokens, m.matched_blocks.len() * BS + rows);
}

/// Every node's reference count, in arena order (roots and free slots are 0).
fn refs(tree: &RadixTree) -> Vec<String> {
    tree.inner
        .lock()
        .test_dump()
        .into_iter()
        .filter(|l| l.contains(": blk="))
        .map(|l| {
            let blk = l.split("blk=").nth(1).and_then(|s| s.split(' ').next());
            let r = l.split("ref=").nth(1).and_then(|s| s.split(' ').next());
            format!("{blk:?}:{r:?}")
        })
        .collect()
}

/// The partial-suffix arm: the donor published its partially filled tail
/// block and still decodes into it. The requester gets a copy source, never
/// the block.
#[test]
fn a_partial_tail_becomes_a_copy_source() {
    let tree = RadixTree::with_levers(false, true);
    let prompt: Vec<u32> = (0..20).collect();
    tree.insert(&prompt, &[10, 11], &[], BS, 0, 0);

    let m = tree.lookup_cow(&prompt, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10]);
    assert_tail_is_copied_not_shared(&m, 11, 4);
    assert_eq!(m.matched_tokens, 20, "the whole prompt is reused");
    tree.release(&prompt, BS, 0);
}

/// The child-key arm: the requester's remainder is a prefix of a committed
/// full block other sequences read.
#[test]
fn a_committed_block_prefix_becomes_a_copy_source() {
    let tree = RadixTree::with_levers(false, true);
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);

    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10]);
    assert_tail_is_copied_not_shared(&m, 20, 6);
    tree.release(&requester, BS, 0);
}

/// A prompt shorter than one block can copy from a first-level block.
#[test]
fn a_prompt_shorter_than_a_block_copies_from_a_first_block() {
    let tree = RadixTree::with_levers(false, true);
    let donor: Vec<u32> = (0..32).collect();
    tree.insert(&donor, &[10, 20], &[], BS, 0, 0);
    let short: Vec<u32> = (0..9).collect();
    let m = tree.lookup_cow(&short, BS, 0, 0);
    assert!(m.matched_blocks.is_empty());
    assert_tail_is_copied_not_shared(&m, 10, 9);
    tree.release(&short, BS, 0);
}

/// Every other caller keeps block-aligned matches: plain `lookup` on a CoW
/// tree, `lookup_cow` on a tree with the row off, and both on a block-aligned
/// prompt.
#[test]
fn plain_lookup_and_the_row_off_stay_block_aligned() {
    let cow = RadixTree::with_levers(false, true);
    let off = RadixTree::with_levers(false, false);
    let donor: Vec<u32> = (0..35).collect();
    let requester: Vec<u32> = (0..22).collect();
    for tree in [&cow, &off] {
        tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);
    }
    let plain = cow.lookup(&requester, BS, 0, 0);
    assert_eq!((plain.matched_tokens, plain.cow_tail), (16, None));
    assert_eq!(plain.matched_blocks, vec![10]);
    cow.release(&requester, BS, 0);

    let m = off.lookup_cow(&requester, BS, 0, 0);
    assert_eq!((m.matched_tokens, m.cow_tail), (16, None));
    assert_eq!(m.matched_blocks, vec![10]);
    off.release(&requester, BS, 0);

    let aligned: Vec<u32> = (0..32).collect();
    let m = cow.lookup_cow(&aligned, BS, 0, 0);
    assert_eq!((m.matched_tokens, m.cow_tail), (32, None));
    cow.release(&aligned, BS, 0);
}

/// With the legacy lever armed (`AVAROK_PREFIX_SUBBLOCK=1`), `lookup_cow` is
/// the legacy lookup: the A/B instrument is untouched by the row.
#[test]
fn the_legacy_lever_wins_over_the_row() {
    let tree = RadixTree::with_levers(false, true);
    arm_legacy_partial_tail(&tree);
    let prompt: Vec<u32> = (0..20).collect();
    tree.insert(&prompt, &[10, 11], &[], BS, 0, 0);
    let m = tree.lookup_cow(&prompt, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10, 11], "legacy: the tail is shared");
    assert_eq!((m.matched_tokens, m.cow_tail), (20, None));
    tree.release(&prompt, BS, 0);
}

/// A CoW lookup takes exactly the references a plain lookup takes (full
/// blocks), so `release` and `release_matched` balance it and the tree drains.
#[test]
fn references_balance_and_the_tree_drains() {
    let tree = RadixTree::with_levers(true, true);
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);
    tree.release(&donor, BS, 0);
    let baseline = refs(&tree);

    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    assert_tail_is_copied_not_shared(&m, 20, 6);
    assert_ne!(refs(&tree), baseline, "the full block is referenced");
    tree.release(&requester, BS, 0);
    assert_eq!(refs(&tree), baseline, "release balances the CoW lookup");

    // Batched-admission rollback releases exactly `matched_tokens`.
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    tree.release_matched(&requester, BS, m.matched_tokens, 0);
    assert_eq!(refs(&tree), baseline, "release_matched balances it too");

    // The requester's own insert (its private copy at index 1) and release
    // leave every node evictable: the tree drains completely.
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    let acq = tree.insert(&requester, &[10, 99], &[], BS, m.matched_tokens, 0);
    assert_eq!(
        acq.blocks,
        vec![99],
        "the private copy is cached as a partial tail"
    );
    tree.release(&requester, BS, 0);
    let mut freed = tree.evict(usize::MAX).physical;
    freed.sort_unstable();
    assert_eq!(freed, vec![10, 20, 30, 99]);
    assert_eq!(tree.stats(), (0, 0));
}

/// The source the copy reads from is marked used, so the allocation for the
/// private block (which may evict) does not pick it first.
#[test]
fn the_copy_source_is_touched_so_it_is_not_the_next_victim() {
    let tree = RadixTree::with_levers(true, true);
    let old: Vec<u32> = (100..116).collect();
    tree.insert(&old, &[50], &[], BS, 0, 0);
    tree.release(&old, BS, 0);
    let donor: Vec<u32> = (0..32).collect();
    tree.insert(&donor, &[10, 20], &[], BS, 0, 0);
    tree.release(&donor, BS, 0);
    // Make the donor's leaf (block 20) the LRU candidate.
    {
        let mut inner = tree.inner.lock();
        let leaf = inner
            .test_live_nodes()
            .into_iter()
            .find(|&id| inner.test_dump()[id].contains("blk=20 "))
            .expect("block 20 is cached");
        inner.test_set_last_access(leaf, 0);
    }
    let requester: Vec<u32> = (0..21).collect();
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    assert_tail_is_copied_not_shared(&m, 20, 5);
    assert_eq!(tree.evict(1).physical, vec![50], "block 20 is being copied");
    tree.release(&requester, BS, 0);
}

/// Among several children whose key starts with the remainder, the lowest
/// block wins, so the source does not depend on hash-map order.
#[test]
fn the_source_choice_is_deterministic() {
    let tree = RadixTree::with_levers(false, true);
    let base: Vec<u32> = (0..16).collect();
    for (tail, blk) in [(7u32, 40u32), (8, 30), (9, 35)] {
        let mut t = base.clone();
        t.extend([1, 2, 3]);
        t.extend(std::iter::repeat_n(tail, 13));
        tree.insert(&t, &[10, blk], &[], BS, 16, 0);
    }
    let mut requester = base.clone();
    requester.extend([1, 2, 3]);
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    assert_tail_is_copied_not_shared(&m, 30, 3);
    tree.release(&requester, BS, 0);
}

/// `--high-speed-swap` blocks carry disk ids; a private copy would have no
/// disk image, so no CoW tail is offered there.
#[test]
fn no_copy_source_under_high_speed_swap() {
    let tree = RadixTree::with_levers(false, true);
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[5, 6, 7], BS, 0, 0);
    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    assert_eq!((m.matched_tokens, m.cow_tail), (16, None));
    tree.release(&requester, BS, 0);
}

/// Adapters stay disjoint: a CoW source is found only under the requester's
/// own adapter root.
#[test]
fn a_copy_source_never_crosses_adapters() {
    let tree = RadixTree::with_levers(false, true);
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 3);
    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup_cow(&requester, BS, 0, 0);
    assert!(m.is_empty() && m.cow_tail.is_none(), "{m:?}");
    let m = tree.lookup_cow(&requester, BS, 0, 3);
    assert_tail_is_copied_not_shared(&m, 20, 6);
    tree.release(&requester, BS, 3);
}

/// Hybrid models: an exact repeat of a non-block-aligned prompt reaches the
/// leaf snapshot at prompt end, because the snapshot lookup uses the
/// CoW-extended `matched_tokens`. Plain `lookup` stops at the aligned depth.
#[test]
fn an_exact_repeat_reaches_the_leaf_snapshot() {
    let tree = RadixTree::new();
    arm_cow_tail(&tree);
    let prompt: Vec<u32> = (0..20).collect();
    let (displaced, _) = tree.insert_with_snapshot(&prompt, &[10, 11], &[], BS, 7, 0, 0, 0);
    assert_eq!(displaced, None);

    let m = tree.lookup(&prompt, BS, 0, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.ssm_snapshot, None, "the leaf at 20 is deeper than 16");
    tree.release(&prompt, BS, 0);

    let m = tree.lookup_cow(&prompt, BS, 0, 0);
    assert_tail_is_copied_not_shared(&m, 11, 4);
    assert_eq!((m.ssm_snapshot, m.ssm_snapshot_tokens), (Some(7), 20));
    tree.release(&prompt, BS, 0);
}
