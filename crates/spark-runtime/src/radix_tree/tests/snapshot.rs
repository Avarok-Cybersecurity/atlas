// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot-side tests: intermediate snapshots, partial-suffix matching,
//! and the standalone snapshot index LRU/session/overwrite behaviours.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::super::hash_token_prefix;
use super::super::snapshot::SsmSnapshotIndex;

#[test]
fn test_insert_without_snapshot() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert(&tokens, &[10], &[], 16, 0);
    let m = tree.lookup(&tokens, 16, 0);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    tree.release(&tokens, 16);
}

#[test]
fn test_intermediate_snapshot_on_partial_match() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0);

    // Lookup all 4 blocks — should return intermediate snapshot at block 2
    let m = tree.lookup(&tokens, 16, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens, 16);
}

#[test]
fn test_intermediate_snapshot_deepest_wins() {
    let tree = RadixTree::new();

    // Insert 4-block sequence with leaf snapshot
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0);

    // Lookup all 4 blocks — leaf snapshot (deeper) wins
    let m = tree.lookup(&tokens, 16, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    tree.release(&tokens, 16);
}

#[test]
fn test_intermediate_snapshot_partial_prefix_hit() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0);

    // New request shares first 48 tokens, diverges at block 4
    let mut tokens_new: Vec<u32> = (0..48).collect();
    tokens_new.extend(200..216);
    let m = tree.lookup(&tokens_new, 16, 0);
    // Matches 3 blocks (48 tokens), intermediate snapshot at block 2
    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens_new, 16);
}

#[test]
fn test_intermediate_snapshot_survives_tree_eviction() {
    let tree = RadixTree::new();

    // Insert 2-block sequence with intermediate snapshot on block 1
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0);
    tree.release(&tokens, 16); // inserting seq exits → nodes evictable

    let tokens_at_1: Vec<u32> = (0..16).collect();
    tree.insert_intermediate_snapshot(&tokens_at_1, &[10], &[], 16, 50, 0, 0);

    // Evict both tree nodes — snapshot survives in index
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![20]);
    let evicted = tree.evict(1);
    assert_eq!(evicted.physical, vec![10]);

    // Snapshot still in index (decoupled from tree)
    assert_eq!(tree.snapshot_count(), 1);
    let snap = tree.evict_snapshot_lru();
    assert_eq!(snap, Some(50));
}

// ── Partial suffix tests ──

#[test]
fn test_partial_suffix_insert_and_lookup() {
    let tree = RadixTree::new();
    // 20 tokens = 1 full block (16) + 4 partial
    let tokens: Vec<u32> = (0..20).collect();
    let block_table = vec![10, 20]; // block for full + block for partial

    tree.insert(&tokens, &block_table, &[], 16, 0);
    let m = tree.lookup(&tokens, 16, 0);

    // Should match all 20 tokens (16 full + 4 partial)
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16);
}

#[test]
fn test_partial_suffix_no_match_different_suffix() {
    let tree = RadixTree::new();
    // Insert 20 tokens
    let tokens_a: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0);

    // Lookup 20 tokens with different suffix (same first 16, different last 4)
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..104);
    let m = tree.lookup(&tokens_b, 16, 0);

    // Should match only 16 full-block tokens (partial suffix doesn't match)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_b, 16);
}

#[test]
fn test_partial_suffix_not_matched_for_full_block_request() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 partial)
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0);

    // Lookup 32 tokens — 2 full blocks in request. Partial suffix is 4 tokens
    // but remainder is 0 (32 % 16 == 0), so partial check is skipped.
    let tokens_32: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_32, 16, 0);

    // Only first full block matches (second block [16..32] has no matching tree node)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_32, 16);
}

#[test]
fn test_partial_suffix_eviction_frees_both_blocks() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full block + 4 partial) + release inserting seq
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0);
    tree.release(&tokens, 16);

    // Evict 1 — should free block 10 (full) AND block 20 (partial suffix)
    let evicted = tree.evict(1);
    // Evicting the leaf node also frees its partial suffix block
    assert!(evicted.physical.contains(&10));
    assert!(evicted.physical.contains(&20));
}

#[test]
#[ignore = "tests removed behavior — partial-suffix clearing was replaced \
            with partial-block-matching during the radix-tree refactor; \
            assertions need rewriting against the new lookup semantics"]
fn test_partial_suffix_cleared_when_extended() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 partial)
    let tokens_20: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_20, &[10, 20], &[], 16, 0);

    // Insert 32 tokens (2 full blocks, extends past partial)
    let tokens_32: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_32, &[10, 30], &[], 16, 0);

    // Lookup 20 tokens — partial suffix was cleared by the 32-token insert
    let m = tree.lookup(&tokens_20, 16, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_20, 16);

    // Lookup 32 tokens — full match
    let m = tree.lookup(&tokens_32, 16, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&tokens_32, 16);
}

#[test]
fn test_partial_suffix_multi_block_prefix() {
    let tree = RadixTree::new();
    // 396 tokens = 24 full blocks + 12 partial
    let tokens: Vec<u32> = (0..396).collect();
    let block_table: Vec<u32> = (0..25).collect();
    // block_table[24] = partial block

    tree.insert(&tokens, &block_table, &[], 16, 0);
    let m = tree.lookup(&tokens, 16, 0);

    assert_eq!(m.matched_tokens, 396);
    assert_eq!(m.matched_blocks.len(), 25);
    tree.release(&tokens, 16);
}

#[test]
fn test_partial_suffix_prefix_match_shorter_lookup() {
    let tree = RadixTree::new();
    // Insert 31 tokens (1 full block + 15 partial) — simulates prompt+generation
    let tokens_31: Vec<u32> = (0..31).collect();
    tree.insert(&tokens_31, &[10, 20], &[], 16, 0);

    // Lookup 22 tokens (1 full block + 6 partial) — simulates repeat of prompt only
    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0);

    // Partial suffix [16..31] starts with [16..22], so prefix match succeeds
    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16);
}

#[test]
fn test_sub_block_match_via_child_key_prefix() {
    let tree = RadixTree::new();
    // Insert 35 tokens (2 full blocks + 3 partial) — prompt + generation
    let tokens_35: Vec<u32> = (0..35).collect();
    tree.insert(&tokens_35, &[10, 20, 30], &[], 16, 0);

    // Lookup 22 tokens (1 full block + 6 remaining) — same prompt
    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0);

    // Block 0 (0-15) matched as full block.
    // Remaining 6 tokens (16-21) are a prefix of block 1's key (16-31).
    // Sub-block matching should include block 1.
    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16);
}

#[test]
fn test_partial_suffix_sub_block_only() {
    let tree = RadixTree::new();
    // Only 10 tokens — no full blocks, partial suffix not stored (no parent)
    let tokens: Vec<u32> = (0..10).collect();
    tree.insert(&tokens, &[42], &[], 16, 0);

    // No full blocks → nothing cached or matched
    assert_eq!(tree.stats(), (0, 0));
    let m = tree.lookup(&tokens, 16, 0);
    assert_eq!(m.matched_tokens, 0);
}

// ── SsmSnapshotIndex tests ──

#[test]
fn test_snapshot_index_insert_lookup_roundtrip() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..32).collect();
    let prefix_hash = hash_token_prefix(&tokens, 32);

    assert!(idx.insert(prefix_hash, 42, 100, 32).is_none());
    let result = idx.lookup(&tokens, 32, 100);
    assert_eq!(result, Some((42, 32)));
}

#[test]
fn test_snapshot_index_lru_eviction() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    let tokens_b: Vec<u32> = (100..116).collect();
    let ha = hash_token_prefix(&tokens_a, 16);
    let hb = hash_token_prefix(&tokens_b, 16);

    idx.insert(ha, 1, 0, 16); // older
    idx.insert(hb, 2, 0, 16); // newer

    // LRU eviction should evict snapshot 1 (older)
    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(1));
    assert_eq!(idx.len(), 1);

    // Only snapshot 2 remains
    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(2));
    assert_eq!(idx.len(), 0);

    // Empty
    assert_eq!(idx.evict_lru(), None);
}

#[test]
fn test_snapshot_index_session_isolation() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = hash_token_prefix(&tokens, 16);

    // Insert snapshot for session 100
    idx.insert(prefix_hash, 42, 100, 16);

    // Lookup from session 200 — should NOT match (different session)
    let result = idx.lookup(&tokens, 16, 200);
    assert_eq!(result, None);

    // Lookup from session 100 — should match
    let result = idx.lookup(&tokens, 16, 100);
    assert_eq!(result, Some((42, 16)));

    // Lookup with session_hash=0 (legacy) — matches any session
    let result = idx.lookup(&tokens, 16, 0);
    assert_eq!(result, Some((42, 16)));
}

#[test]
fn test_snapshot_index_overwrite_existing() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = hash_token_prefix(&tokens, 16);

    // Insert first
    assert!(idx.insert(prefix_hash, 5, 0, 16).is_none());
    assert_eq!(idx.len(), 1);

    // Overwrite same prefix_hash — returns old snapshot_id
    let old = idx.insert(prefix_hash, 8, 0, 16);
    assert_eq!(old, Some(5));
    assert_eq!(idx.len(), 1); // still 1 entry, not 2

    // Lookup returns new value
    let result = idx.lookup(&tokens, 16, 0);
    assert_eq!(result, Some((8, 16)));
}

#[test]
fn test_snapshot_index_deepest_match() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..64).collect();

    // Snapshot at token 16
    let h16 = hash_token_prefix(&tokens, 16);
    idx.insert(h16, 10, 0, 16);

    // Snapshot at token 32
    let h32 = hash_token_prefix(&tokens, 32);
    idx.insert(h32, 20, 0, 32);

    // Lookup with 48 matched tokens — deepest snapshot at 32 wins
    let result = idx.lookup(&tokens, 48, 0);
    assert_eq!(result, Some((20, 32)));

    // Lookup with 20 matched tokens — only snapshot at 16 qualifies
    let result = idx.lookup(&tokens, 20, 0);
    assert_eq!(result, Some((10, 16)));
}
