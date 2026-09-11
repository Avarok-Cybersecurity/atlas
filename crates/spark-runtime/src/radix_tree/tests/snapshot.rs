// SPDX-License-Identifier: AGPL-3.0-only

//! Snapshot-side tests: intermediate snapshots and partial-suffix matching.
//! The standalone `SsmSnapshotIndex` LRU/session/overwrite behaviours live in
//! `snapshot_index.rs` (mounted as `radix_tree::snapshot`'s unit tests).

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::super::hash_token_prefix;
use super::super::snapshot::SsmSnapshotIndex;

#[test]
fn test_insert_without_snapshot() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..16).collect();

    tree.insert(&tokens, &[10], &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);
    // The KV walk must actually HIT first. `lookup` skips the snapshot index
    // entirely when `matched_tokens == 0`, so without this the snapshot
    // assertions below also hold for a tree that matches nothing at all.
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
    // The tier fields are the other half of "no snapshot": a spilled anchor
    // arrives there, not in `ssm_snapshot`.
    assert_eq!(m.ssm_snapshot_tier_key, None);
    assert_eq!(m.ssm_snapshot_tier_tokens, 0);
    assert!(!m.ssm_snapshot_is_tail);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_on_partial_match() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // Lookup all 4 blocks — should return intermediate snapshot at block 2
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_deepest_wins() {
    let tree = RadixTree::new();

    // Insert 4-block sequence with leaf snapshot
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 0, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // Lookup all 4 blocks — leaf snapshot (deeper) wins
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 64);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_intermediate_snapshot_partial_prefix_hit() {
    let tree = RadixTree::new();

    // Insert 4-block sequence
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert(&tokens, &[10, 20, 30, 40], &[], 16, 0, 0);

    // Attach intermediate snapshot at block 2 (token 32)
    let tokens_at_2: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_2, &[10, 20], &[], 16, 50, 0, 0, 0);

    // New request shares first 48 tokens, diverges at block 4
    let mut tokens_new: Vec<u32> = (0..48).collect();
    tokens_new.extend(200..216);
    let m = tree.lookup(&tokens_new, 16, 0, 0);
    // Matches 3 blocks (48 tokens), intermediate snapshot at block 2
    assert_eq!(m.matched_tokens, 48);
    assert_eq!(m.ssm_snapshot, Some(50));
    assert_eq!(m.ssm_snapshot_tokens, 32);
    tree.release(&tokens_new, 16, 0);
}

#[test]
fn test_intermediate_snapshot_survives_tree_eviction() {
    let tree = RadixTree::new();

    // Insert 2-block sequence with intermediate snapshot on block 1
    let tokens: Vec<u32> = (0..32).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0); // inserting seq exits → nodes evictable

    let tokens_at_1: Vec<u32> = (0..16).collect();
    tree.insert_intermediate_snapshot(&tokens_at_1, &[10], &[], 16, 50, 0, 0, 0);

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

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    // Should match all 20 tokens (16 full + 4 partial)
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_partial_suffix_no_match_different_suffix() {
    let tree = RadixTree::new();
    // Insert 20 tokens
    let tokens_a: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_a, &[10, 20], &[], 16, 0, 0);

    // Lookup 20 tokens with different suffix (same first 16, different last 4)
    let mut tokens_b: Vec<u32> = (0..16).collect();
    tokens_b.extend(100..104);
    let m = tree.lookup(&tokens_b, 16, 0, 0);

    // Should match only 16 full-block tokens (partial suffix doesn't match)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_b, 16, 0);
}

#[test]
fn test_partial_suffix_not_matched_for_full_block_request() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 partial)
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);

    // Lookup 32 tokens — 2 full blocks in request. Only the first matches, and
    // the unmatched remainder is a WHOLE block (16), so the sub-block arms are
    // out of range for it.
    let tokens_32: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_32, 16, 0, 0);

    // Only first full block matches (second block [16..32] has no matching tree node)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_32, 16, 0);

    // The case the name is actually about: a BLOCK-ALIGNED request. Its
    // remainder is zero, so the sub-block arms must not run at all — an empty
    // suffix is a prefix of every stored key, so a missing `remainder > 0`
    // guard appends the 4-token partial block to a 16-token match and hands
    // the caller a block table longer than `matched_tokens` describes.
    let tokens_16: Vec<u32> = (0..16).collect();
    let m16 = tree.lookup(&tokens_16, 16, 0, 0);
    assert_eq!(m16.matched_tokens, 16);
    assert_eq!(
        m16.matched_blocks,
        vec![10],
        "the partial slot must not be appended to a block-aligned match"
    );
    assert_eq!(
        m16.matched_blocks.len(),
        m16.matched_tokens / 16,
        "block table must stay aligned with matched_tokens"
    );
    tree.release(&tokens_16, 16, 0);
}

#[test]
fn test_partial_suffix_eviction_frees_both_blocks() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full block + 4 partial) + release inserting seq
    let tokens: Vec<u32> = (0..20).collect();
    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0);
    tree.release(&tokens, 16, 0);

    // Evict 1 — should free block 10 (full) AND block 20 (partial suffix)
    let evicted = tree.evict(1);
    // Evicting the leaf node also frees its partial suffix block
    assert!(evicted.physical.contains(&10));
    assert!(evicted.physical.contains(&20));
}

/// A real child node SUPERSEDES the partial-suffix slot it overlaps: the slot
/// is cleared and the KV block it held comes back in `released_blocks` so the
/// caller can drop the cache's reference on it.
///
/// Rewritten (was `#[ignore]`d as "tests removed behavior"): the OLD
/// assertions expected the 20-token prefix to become unmatchable, which the
/// sub-block child-key arm made false — the deeper node now serves those 20
/// tokens. The clearing itself is NOT removed behaviour; what it must produce
/// is the released-block accounting asserted below, and no serving of the
/// retired block. The partial→partial overwrite arm is covered by
/// `tests::basic::test_partial_suffix_block_is_owned_and_released`; this is
/// the distinct child-supersedes-partial arm.
#[test]
fn test_partial_suffix_cleared_when_extended() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + a 4-token partial held in block 20).
    let tokens_20: Vec<u32> = (0..20).collect();
    let first = tree.insert(&tokens_20, &[10, 20], &[], 16, 0, 0);
    assert!(
        first.blocks.contains(&20),
        "the partial slot takes a ref on block 20; got {:?}",
        first.blocks
    );

    // Insert 32 tokens: a real child now covers [16..32), superseding the slot.
    let tokens_32: Vec<u32> = (0..32).collect();
    let extended = tree.insert(&tokens_32, &[10, 30], &[], 16, 0, 0);
    assert_eq!(
        extended.released_blocks,
        vec![20],
        "the superseded partial block must be handed back, not leaked"
    );
    assert!(
        extended.blocks.contains(&30),
        "the superseding child block is acquired; got {:?}",
        extended.blocks
    );

    // Lookup 20 tokens — served by the DEEPER node (block 30) through the
    // sub-block child-key arm, never by the retired partial block 20.
    let m = tree.lookup(&tokens_20, 16, 0, 0);
    assert_eq!(m.matched_tokens, 20);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    assert!(
        !m.matched_blocks.contains(&20),
        "the retired partial block must never be served again"
    );
    tree.release(&tokens_20, 16, 0);

    // Lookup 32 tokens — full match
    let m = tree.lookup(&tokens_32, 16, 0, 0);
    assert_eq!(m.matched_tokens, 32);
    assert_eq!(m.matched_blocks, vec![10, 30]);
    tree.release(&tokens_32, 16, 0);
}

#[test]
fn test_partial_suffix_multi_block_prefix() {
    let tree = RadixTree::new();
    // 396 tokens = 24 full blocks + 12 partial
    let tokens: Vec<u32> = (0..396).collect();
    let block_table: Vec<u32> = (0..25).collect();
    // block_table[24] = partial block

    tree.insert(&tokens, &block_table, &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);

    assert_eq!(m.matched_tokens, 396);
    // Identity AND order, not just the count: a count-only oracle accepts a
    // walk that returns 25 wrong blocks (e.g. each node's parent block).
    assert_eq!(m.matched_blocks, block_table);
    tree.release(&tokens, 16, 0);
}

#[test]
fn test_partial_suffix_prefix_match_shorter_lookup() {
    let tree = RadixTree::new();
    // Insert 31 tokens (1 full block + 15 partial) — simulates prompt+generation
    let tokens_31: Vec<u32> = (0..31).collect();
    tree.insert(&tokens_31, &[10, 20], &[], 16, 0, 0);

    // Lookup 22 tokens (1 full block + 6 partial) — simulates repeat of prompt only
    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0, 0);

    // Partial suffix [16..31] starts with [16..22], so prefix match succeeds
    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16, 0);
}

#[test]
fn test_sub_block_match_via_child_key_prefix() {
    let tree = RadixTree::new();
    // Insert 35 tokens (2 full blocks + 3 partial) — prompt + generation
    let tokens_35: Vec<u32> = (0..35).collect();
    tree.insert(&tokens_35, &[10, 20, 30], &[], 16, 0, 0);

    // Lookup 22 tokens (1 full block + 6 remaining) — same prompt
    let tokens_22: Vec<u32> = (0..22).collect();
    let m = tree.lookup(&tokens_22, 16, 0, 0);

    // Block 0 (0-15) matched as full block.
    // Remaining 6 tokens (16-21) are a prefix of block 1's key (16-31).
    // Sub-block matching should include block 1.
    assert_eq!(m.matched_tokens, 22);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&tokens_22, 16, 0);
}

#[test]
fn test_partial_suffix_sub_block_only() {
    let tree = RadixTree::new();
    // Only 10 tokens — no full blocks, partial suffix not stored (no parent)
    let tokens: Vec<u32> = (0..10).collect();
    tree.insert(&tokens, &[42], &[], 16, 0, 0);

    // No full blocks → nothing cached or matched
    assert_eq!(tree.stats(), (0, 0));
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, 0);
}

// ── Task #24: adapter-correct SSM snapshots + base hash byte-identity ──

/// `hash_token_prefix(_, _, 0)` (base sentinel) must reduce EXACTLY to the
/// pre-#24 token-only FNV-1a value, so base prefix-cache/snapshot hit rates are
/// unchanged. A non-zero adapter_id must change the hash.
#[test]
fn test_hash_token_prefix_base_byte_identical() {
    let tokens: Vec<u32> = vec![7, 42, 1000, 65535, 3, 0, 128];
    // Recompute the exact pre-#24 formula inline.
    let mut expected: u64 = 0xcbf29ce484222325;
    for &t in &tokens {
        expected ^= t as u64;
        expected = expected.wrapping_mul(0x100000001b3);
    }
    assert_eq!(
        hash_token_prefix(&tokens, tokens.len(), 0),
        expected,
        "base (adapter_id=0) hash must be byte-identical to the pre-#24 value"
    );
    // Any non-zero adapter partitions the key.
    assert_ne!(
        hash_token_prefix(&tokens, tokens.len(), 0),
        hash_token_prefix(&tokens, tokens.len(), 99),
    );
    assert_ne!(
        hash_token_prefix(&tokens, tokens.len(), 7),
        hash_token_prefix(&tokens, tokens.len(), 9),
    );
}

/// The SSM snapshot index must isolate by adapter: a snapshot registered under
/// adapter A's prefix hash is not found by an adapter-B lookup, but is by an
/// adapter-A lookup.
#[test]
fn test_snapshot_index_adapter_isolation() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    const A: u64 = 0xAA;
    const B: u64 = 0xBB;

    // Register under adapter A (the tree computes prefix_hash with A folded in).
    let ph_a = hash_token_prefix(&tokens, 16, A);
    idx.insert(ph_a, 42, 0, 16);

    // Adapter B lookup recomputes with B → different hash → miss.
    assert_eq!(idx.lookup(&tokens, 16, 0, B), None);
    // Adapter A lookup → hit.
    assert_eq!(idx.lookup(&tokens, 16, 0, A), Some((42, 16)));
    // Base lookup → miss (base hash != A hash).
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), None);
}

/// End-to-end through the tree API: an SSM snapshot saved under adapter A is
/// not restored for an adapter-B request, but is for an adapter-A request.
#[test]
fn test_ssm_snapshot_adapter_isolation_via_tree() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0x55;
    const B: u64 = 0x66;

    tree.insert_with_snapshot(&tokens, &[10, 20], &[], 16, 42, 0, 0, A);
    tree.release(&tokens, 16, A);

    // Adapter B: KV misses AND no snapshot restore.
    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert!(m_b.is_empty());
    assert_eq!(m_b.ssm_snapshot, None);

    // Adapter A: KV hit + snapshot restored.
    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(m_a.matched_tokens, 32);
    assert_eq!(m_a.ssm_snapshot, Some(42));
    tree.release(&tokens, 16, A);

    // Give B its OWN KV for the same tokens (disjoint radix root) so B's walk
    // HITS. Without this, `lookup` short-circuits on `matched_tokens == 0` and
    // never reaches the snapshot index — so `m_b.ssm_snapshot == None` above is
    // proved by the tree's root isolation alone and says nothing about whether
    // the snapshot KEY carries the adapter.
    tree.insert(&tokens, &[30, 40], &[], 16, 0, B);
    tree.release(&tokens, 16, B);
    let m_b2 = tree.lookup(&tokens, 16, 0, B);
    assert_eq!(m_b2.matched_tokens, 32, "B now has its own cached KV");
    assert_eq!(m_b2.matched_blocks, vec![30, 40]);
    assert_eq!(
        m_b2.ssm_snapshot, None,
        "A's SSM snapshot must not restore for B even on a B-side KV hit"
    );
    assert_eq!(m_b2.ssm_snapshot_tier_key, None);
    tree.release(&tokens, 16, B);
}
