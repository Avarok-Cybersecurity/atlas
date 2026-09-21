// SPDX-License-Identifier: AGPL-3.0-only

//! Issue #1193: the radix walk must not end a match inside a KV block.
//!
//! Both sub-block arms hand the requester a physical block index that another
//! owner still holds, and the requester writes its own K/V into that block
//! from `matched_tokens` onward — there is no copy-on-write in the paged KV
//! path. These tests pin the SHIPPED contract (a match is block-aligned) and,
//! separately, that the legacy arms remain reachable for A/B.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::arm_legacy_partial_tail;

const BS: usize = 16;

/// Every block the walk hands out must be fully covered by `matched_tokens`,
/// so no handed-out block is also the requester's writable tail.
fn assert_no_writable_tail_handed_out(m: &crate::prefix_cache::PrefixMatch) {
    assert_eq!(
        m.matched_tokens % BS,
        0,
        "a match that ends inside a block makes that block the requester's \
         writable tail while its owner still writes it (#1193); matched={}",
        m.matched_tokens
    );
    assert_eq!(
        m.matched_blocks.len(),
        m.matched_tokens / BS,
        "block table must be exactly the fully-matched blocks; got {:?} for \
         matched={}",
        m.matched_blocks,
        m.matched_tokens
    );
}

/// The `partial_suffix` arm. The donor published its partially-filled tail
/// block at END OF PREFILL and is still decoding into it, so handing that
/// physical block to a second sequence puts two live writers on one set of
/// slots.
#[test]
fn a_partial_tail_block_is_never_handed_to_a_second_writer() {
    let tree = RadixTree::new();
    // Donor's 20-token prompt: full block 10, plus a 4-token tail in block 11
    // that the donor keeps writing as it decodes positions 20, 21, ...
    let prompt: Vec<u32> = (0..20).collect();
    tree.insert(&prompt, &[10, 11], &[], BS, 0, 0);

    // A second sequence arrives with the same prompt.
    let m = tree.lookup(&prompt, BS, 0, 0);
    assert_eq!(
        m.matched_tokens, 16,
        "only the donor's FULL blocks may be reused"
    );
    assert_eq!(m.matched_blocks, vec![10]);
    assert!(
        !m.matched_blocks.contains(&11),
        "block 11 is the donor's live tail; got {:?}",
        m.matched_blocks
    );
    assert_no_writable_tail_handed_out(&m);
    tree.release(&prompt, BS, 0);
}

/// The child-key arm, which #692 showed is unsound for the same reason against
/// a FULL donor block: the requester's tail writes land in offsets
/// `[remainder, block_size)` of a block other sequences map read-only as
/// committed prompt K/V. Flipping one default must close this arm too.
#[test]
fn a_committed_full_block_is_never_handed_out_as_a_second_writers_tail() {
    let tree = RadixTree::new();
    // Donor cached 35 tokens: full blocks 10 and 20, plus a 3-token tail (30).
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);

    // Requester's 22-token prompt: one full block, then a 6-token remainder
    // that is a strict PREFIX of block 20's key. The legacy arm served block
    // 20 here — and the requester would then write positions 22..32 into it.
    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&requester, BS, 0, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    assert!(
        !m.matched_blocks.contains(&20),
        "block 20 is a committed block other sequences read; got {:?}",
        m.matched_blocks
    );
    assert_no_writable_tail_handed_out(&m);
    tree.release(&requester, BS, 0);
}

/// The two tests above would also pass against a tree that matches NOTHING.
/// This is the other half: the same fixtures still match everything they
/// legitimately can, so what the fix removed is exactly the sub-block tail.
#[test]
fn full_block_reuse_is_untouched_by_the_default() {
    let tree = RadixTree::new();
    // 35 tokens, two full blocks + a tail. A block-aligned 32-token lookup
    // must still reuse both full blocks.
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);

    let aligned: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&aligned, BS, 0, 0);
    assert_eq!(m.matched_tokens, 32, "full-block reuse must still happen");
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&aligned, BS, 0);
}

/// The lever is an A/B instrument (#936 arm ladder), so the legacy arms must
/// stay operable. Armed on the same fixture as
/// `a_partial_tail_block_is_never_handed_to_a_second_writer`, with the default
/// as the only variable, the old non-aligned match comes back — which is also
/// what proves that test measures the default and not a broken walk.
#[test]
fn the_legacy_sub_block_arms_stay_reachable_for_ab() {
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);

    // partial-suffix arm.
    let prompt: Vec<u32> = (0..20).collect();
    tree.insert(&prompt, &[10, 11], &[], BS, 0, 0);
    let m = tree.lookup(&prompt, BS, 0, 0);
    assert_eq!(m.matched_tokens, 20, "armed: the partial tail is served");
    assert_eq!(m.matched_blocks, vec![10, 11]);
    tree.release(&prompt, BS, 0);

    // child-key arm, on a second tree so the fixtures do not interact.
    let tree = RadixTree::new();
    arm_legacy_partial_tail(&tree);
    let donor: Vec<u32> = (0..35).collect();
    tree.insert(&donor, &[10, 20, 30], &[], BS, 0, 0);
    let requester: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&requester, BS, 0, 0);
    assert_eq!(
        m.matched_tokens, 22,
        "armed: the child-key sub-block is served"
    );
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&requester, BS, 0);
}
