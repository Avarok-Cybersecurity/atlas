// SPDX-License-Identifier: AGPL-3.0-only

//! Basic radix-tree tests: insert, lookup, eviction, branching, partial
//! match, sub-block tokens, ref counting, and SSM-snapshot insert/lookup
//! through the [`super::super::RadixTree`] API.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

/// Issue #17 HSS disk-ref acquisition checks — split out to keep this file
/// under the workspace 500-LoC cap, using the same `#[path]` nesting the
/// snapshot index tests use.
#[path = "basic_hss.rs"]
mod hss;

#[test]
fn test_insert_and_lookup_exact() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect(); // 1 block of 16 tokens
    let block_table = vec![42];

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![42]);
}

#[test]
fn test_insert_and_lookup_multi_block() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..48).collect(); // 3 blocks of 16
    let block_table = vec![10, 20, 30];

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.matched_blocks, vec![10, 20, 30]);
}

#[test]
fn test_partial_match() {
    let tree = RadixTree::new();
    // Insert 32 tokens (2 blocks)
    let tokens_a: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    // Lookup 48 tokens (3 blocks) — only first 2 match
    let tokens_b: Vec<u32> = (0..48).collect();
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_b, 16, 0);

    // The match must stop at the DIVERGENCE, not merely somewhere before the
    // end of the request: the assertions above are equally satisfied by a walk
    // that stops one block short of the deepest cached node, because block 3
    // was never cached. Cache a third block under one continuation, then ask
    // for a different one.
    let mut tokens_c: Vec<u32> = (0..32).collect();
    tokens_c.extend(200..216);
    tree.insert(&tokens_c, &[10, 20, 30], &[], 16, 0, 0);
    tree.release(&tokens_c, 16, 0);

    let deep = tree.lookup(&tokens_c, 16, 0, 0);
    assert_eq!(
        deep.matched_tokens, 48,
        "the deepest cached block must match"
    );
    assert_eq!(deep.matched_blocks, vec![10, 20, 30]);
    tree.release(&tokens_c, 16, 0);

    let mut tokens_d: Vec<u32> = (0..32).collect();
    tokens_d.extend(700..716);
    let diverged = tree.lookup(&tokens_d, 16, 0, 0);
    assert_eq!(
        diverged.matched_tokens, 32,
        "the match stops at the divergent block, not before it"
    );
    assert_eq!(diverged.matched_blocks, vec![10, 20]);
    tree.release(&tokens_d, 16, 0);
}

#[test]
fn test_no_match() {
    let tree = RadixTree::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    tree.insert(&tokens_a, &[10], &[], 16, 0, 0);

    // Different tokens — no match
    let tokens_b: Vec<u32> = (100..116).collect();
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    assert!(m.is_empty());
    assert!(m.matched_blocks.is_empty(), "a miss hands back no blocks");

    // Positive control: the miss must be caused by the token mismatch, not by
    // a cache that misses everything. `is_empty()` alone is satisfied by any
    // lookup-wide defect (verified: a walk that stops one block short of the
    // deepest node leaves the assertion above green), which is exactly the
    // shape of failure a no-match check is least able to notice.
    let hit = tree.lookup(&tokens_a, 16, 0, 0);
    assert_eq!(
        hit.matched_tokens, 16,
        "the same cache still hits its own key"
    );
    assert_eq!(hit.matched_blocks, vec![10]);
    tree.release(&tokens_a, 16, 0);
}

#[test]
fn test_release_decrements_refcount() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    // Simulate the real cache-miss flow: insert from a seq that did NOT
    // match on walk (matched_tokens=0), then that seq exits (release).
    // After release the cache holds exactly one ref (its baseline).
    tree.insert(&tokens, &[42], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // A second seq comes in, walks (hits), then exits.
    let _ = tree.lookup(&tokens, 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // After both seqs exit, ref_count == cache baseline (1) → evictable.
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![42]);
}

#[test]
fn test_insert_release_lookup_survives() {
    // Regression: the original Marconi/prefix-cache regression (alpha-2.99)
    // was that the first seq to cache a prompt rendered its own cache
    // entry un-findable when it exited — `walk` required `ref_count > 0`
    // but `insert` created nodes at ref_count=1 and `release` decremented
    // them to 0. The fix threads matched_tokens through insert so the
    // inserting seq's pending release doesn't drop the cache's baseline.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0); // cache-miss insert
    tree.release(&tokens, 16, 0); // inserting seq exits

    // Second request with identical prompt — must find the cached entry.
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32, "stale cache entry after seq exit");
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_evict_lru_order() {
    let tree = RadixTree::new();

    // Insert block A (older) + release inserting seq
    let tokens_a: Vec<u32> = (0..16).collect();
    tree.insert(&tokens_a, &[10], &[], 16, 0, 0);
    tree.release(&tokens_a, 16, 0);

    // Insert block B (newer) + release inserting seq
    let tokens_b: Vec<u32> = (100..116).collect();
    tree.insert(&tokens_b, &[20], &[], 16, 0, 0);
    tree.release(&tokens_b, 16, 0);

    // A cache HIT on A makes it the most recently USED. Ordering the victims by
    // insertion order alone would still pick A here — this is the partition
    // that separates real LRU from FIFO, and the only one that observes
    // `inc_refs` refreshing `last_access` on a hit.
    let hit = tree.lookup(&tokens_a, 16, 0, 0);
    assert_eq!(hit.matched_blocks, vec![10]);
    tree.release(&tokens_a, 16, 0);

    // Evict 1 — must be B, the block that was NOT re-used.
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20], "the least recently USED block");

    // Evict 1 more — now A is the only entry left.
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);
    assert_eq!(tree.stats(), (0, 0), "and nothing else survived");
}

#[test]
fn test_evict_skips_referenced_nodes() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();
    // Cache-miss insert + release: node settles at ref_count=1 (cache baseline).
    tree.insert(&tokens, &[42], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // An active seq walks (matches) → ref_count bumps above the baseline.
    let _ = tree.lookup(&tokens, 16, 0, 0);

    // Evict should skip while the seq is active (ref_count > 1).
    let evicted = tree.evict(1);
    assert!(evicted.is_empty());

    // Seq exits → ref_count returns to the cache baseline → evictable.
    tree.release(&tokens, 16, 0);
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![42]);
}

#[test]
fn test_evict_chain_from_leaf() {
    let tree = RadixTree::new();
    // Insert 3-block chain + release inserting seq so the chain is evictable.
    let tokens: Vec<u32> = (0..48).collect();
    tree.insert(&tokens, &[10, 20, 30], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // Evict 1 — should remove leaf (block 30)
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![30]);

    // Now block 20 is a leaf, evict it
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);

    // Now block 10 is a leaf
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    // Tree is empty
    assert_eq!(tree.stats(), (0, 0));
}

#[test]
fn test_insert_idempotent() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    // Insert same sequence twice — should not create duplicate nodes
    tree.insert(&tokens, &[42], &[], 16, 0, 0);
    tree.insert(&tokens, &[42], &[], 16, 0, 0);

    assert_eq!(tree.stats(), (1, 1));

    // A node count cannot say WHICH block the surviving node holds. The second
    // sequence normally does not get its block from the cache (it never
    // matched, or it restored from a swap file), so it re-inserts the same
    // token chunk with a DIFFERENT physical block. The node must keep the block
    // the cache already holds a reference on: adopting the sequence's block
    // strands that reference, and evicting the node then returns a block a live
    // sequence still owns. See `InsertAcquired`.
    let second = tree.insert(&tokens, &[43], &[], 16, 0, 0);
    assert_eq!(tree.stats(), (1, 1), "still one node");
    assert!(
        second.blocks.is_empty(),
        "no node was created, so no KV ref is owed; got {second:?}"
    );
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(
        m.matched_blocks,
        vec![42],
        "the node keeps the block the cache references"
    );
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_branching_tree() {
    let tree = RadixTree::new();

    // Insert A: [0..32] → blocks [10, 20]
    let tokens_a: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    // Insert B: [0..16, 100..116] → blocks [10, 30]
    // Same first block, different second block
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..116);
    tree.insert(&tokens_b, &[10, 30], &[], 16, 0, 0);

    // Lookup A — full match
    let m = tree.lookup(&tokens_a, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_a, 16, 0);

    // Lookup B — full match
    let m = tree.lookup(&tokens_b, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&tokens_b, 16, 0);

    // 3 nodes total: block 10 (shared prefix), block 20 (A), block 30 (B)
    assert_eq!(tree.stats(), (3, 3));
}

#[test]
fn test_sub_block_tokens_ignored() {
    let tree = RadixTree::new();
    // Only 10 tokens — less than one block of 16, not inserted
    let tokens: Vec<u32> = (0..10).collect();
    tree.insert(&tokens, &[42], &[], 16, 0, 0);

    // No full blocks → nothing cached
    assert_eq!(tree.stats(), (0, 0));
}

#[test]
fn test_ssm_snapshot_insert_and_lookup() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();

    // Insert with snapshot on deepest node
    tree.insert_with_snapshot(&tokens, &[10, 20], &[], 16, 42, 0, 0, 0);

    // Lookup should return the snapshot at 32 tokens (leaf)
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, Some(42));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_ssm_snapshot_partial_match_returns_deepest() {
    let tree = RadixTree::new();

    // Insert 3-block sequence with snapshot on leaf
    let tokens: Vec<u32> = (0..48).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30], &[], 16, 99, 0, 0, 0);

    // Lookup only 2 blocks — snapshot is on block 3 (not matched)
    let tokens_short: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_short, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    tree.release(&tokens_short, 16, 0);
}

#[test]
fn test_ssm_snapshot_survives_tree_eviction() {
    // Snapshots are decoupled from tree nodes — evicting tree nodes
    // does NOT destroy the snapshot in the index.
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert_with_snapshot(&tokens, &[10], &[], 16, 7, 0, 0, 0);
    tree.release(&tokens, 16, 0); // inserting seq exits → node evictable

    // Evict the tree node
    let evicted_blocks = tree.evict(1);
    assert_eq!(evicted_blocks.physical, vec![10]);

    // Snapshot still exists in the index (decoupled from tree)
    assert_eq!(tree.snapshot_count(), 1);

    // Explicit LRU eviction of snapshot index
    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(7));
    assert_eq!(tree.snapshot_count(), 0);

    // Second LRU eviction returns None
    assert_eq!(tree.evict_snapshot_lru(), None);
}

#[test]
fn test_ssm_snapshot_overwrite_returns_displaced() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    // First insert — no displaced snapshot
    let (displaced, _acquired) = tree.insert_with_snapshot(&tokens, &[10], &[], 16, 5, 0, 0, 0);
    assert_eq!(displaced, None);

    // Re-insert same path with new snapshot — returns displaced ID 5
    let (displaced, _acquired) = tree.insert_with_snapshot(&tokens, &[10], &[], 16, 8, 0, 0, 0);
    assert_eq!(displaced, Some(5));

    // Only 1 entry in the index (overwrite, not append)
    assert_eq!(tree.snapshot_count(), 1);

    // Lookup should return new snapshot
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.ssm_snapshot, Some(8));
    assert_eq!(m.ssm_snapshot_tokens, 16);
    tree.release(&tokens, 16, 0);
}

/// Issue #58: the prefix-cache key is derived from token IDs only, so two
/// vision requests that differ ONLY in image pixel content but share the same
/// text prompt and image-pad placeholder run produce a byte-identical token
/// stream, and therefore collide on the same cache entry. This pins the
/// image-blind-key invariant the model-layer vision-pad gate depends on: if a
/// vision prefill were admitted here, the next page's identical token stream
/// would reuse the previous page's KV/snapshot and return the wrong page's
/// answer (the "returns the previous picture" report).
#[test]
fn test_vision_pad_tokens_are_image_blind_collision() {
    let tree = RadixTree::new();
    const IMAGE_PAD: u32 = 151655; // <|image_pad|>

    // Page A: 3 prompt tokens followed by a run of image-pad placeholders.
    let page_a: Vec<u32> = [1u32, 2, 3]
        .into_iter()
        .chain(std::iter::repeat_n(IMAGE_PAD, 29))
        .collect(); // 32 tokens == 2 blocks of 16
    // Page B is a DIFFERENT image but the same prompt and same pad count, so
    // pixels never enter the token IDs: the stream is identical to page A.
    let page_b = page_a.clone();

    // Admitting page A's blocks would let page B match them in full.
    tree.insert(&page_a, &[10, 20], &[], 16, 0, 0);
    let m = tree.lookup(&page_b, 16, 0, 0);

    assert_eq!(
        m.matched_tokens, 32,
        "distinct images with identical prompt+pad token streams collide on \
         the same prefix-cache key (issue #58); the model layer must not admit \
         vision prefills into the radix cache"
    );
    assert_eq!(m.matched_blocks, vec![10, 20]);
}

/// The cache must own a KV ref on a `partial_suffix` block, not just on full
/// blocks. It previously owned none, while `walk` still handed the block out
/// (no liveness check) and `evict` still handed it back — so once the inserting
/// sequence freed it, the block sat at 0 refs on the free list with a live node
/// pointing at it. Each later matching request then took it 0->1 while it was
/// still free and dropped it 1->0 again, pushing DUPLICATE free-list entries and
/// handing one block to several sequences. Seen on the agentic bench as a single
/// block underflowing 56 times.
#[test]
fn test_partial_suffix_block_is_owned_and_released() {
    let tree = RadixTree::new();
    // 20 tokens @ bs=16 => 1 full block (10) + a 4-token partial (11).
    let tokens: Vec<u32> = (0..20).collect();
    let acquired = tree.insert(&tokens, &[10, 11], &[], 16, 0, 0);
    assert!(
        acquired.blocks.contains(&11),
        "cache must take a ref on the partial-suffix block; got {:?}",
        acquired.blocks
    );
    assert!(acquired.blocks.contains(&10), "and on the full block");
    assert!(acquired.released_blocks.is_empty());

    // Re-inserting the SAME partial block must not re-acquire it.
    let again = tree.insert(&tokens, &[10, 11], &[], 16, 0, 0);
    assert!(
        !again.blocks.contains(&11),
        "re-insert must not double-ref the partial block; got {:?}",
        again.blocks
    );

    // Overwriting the slot with a DIFFERENT block acquires the new one and
    // releases the old, so the displaced block cannot be pinned forever.
    let other: Vec<u32> = (0..16).chain(90..94).collect();
    let swapped = tree.insert(&other, &[10, 12], &[], 16, 0, 0);
    assert!(swapped.blocks.contains(&12), "new partial block acquired");
    assert_eq!(
        swapped.released_blocks,
        vec![11],
        "displaced partial block released"
    );
}
