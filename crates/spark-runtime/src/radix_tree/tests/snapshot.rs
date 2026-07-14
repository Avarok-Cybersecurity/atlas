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

    tree.insert(&tokens, &[10], &[], 16, 0, 0);
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.ssm_snapshot, None);
    assert_eq!(m.ssm_snapshot_tokens, 0);
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

    // Lookup 32 tokens — 2 full blocks in request. Partial suffix is 4 tokens
    // but remainder is 0 (32 % 16 == 0), so partial check is skipped.
    let tokens_32: Vec<u32> = (0..32).collect();
    let m = tree.lookup(&tokens_32, 16, 0, 0);

    // Only first full block matches (second block [16..32] has no matching tree node)
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
    tree.release(&tokens_32, 16, 0);
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

#[test]
#[ignore = "tests removed behavior — partial-suffix clearing was replaced \
            with partial-block-matching during the radix-tree refactor; \
            assertions need rewriting against the new lookup semantics"]
fn test_partial_suffix_cleared_when_extended() {
    let tree = RadixTree::new();
    // Insert 20 tokens (1 full + 4 partial)
    let tokens_20: Vec<u32> = (0..20).collect();
    tree.insert(&tokens_20, &[10, 20], &[], 16, 0, 0);

    // Insert 32 tokens (2 full blocks, extends past partial)
    let tokens_32: Vec<u32> = (0..32).collect();
    tree.insert(&tokens_32, &[10, 30], &[], 16, 0, 0);

    // Lookup 20 tokens — partial suffix was cleared by the 32-token insert
    let m = tree.lookup(&tokens_20, 16, 0, 0);
    assert_eq!(m.matched_tokens, 16);
    assert_eq!(m.matched_blocks, vec![10]);
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
    assert_eq!(m.matched_blocks.len(), 25);
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

// ── SsmSnapshotIndex tests ──

#[test]
fn test_snapshot_index_insert_lookup_roundtrip() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..32).collect();
    let prefix_hash = hash_token_prefix(&tokens, 32, 0);

    assert!(idx.insert(prefix_hash, 42, 100, 32).is_none());
    let result = idx.lookup(&tokens, 32, 100, 0);
    assert_eq!(result, Some((42, 32)));
}

#[test]
fn test_snapshot_index_lru_eviction() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    let tokens_b: Vec<u32> = (100..116).collect();
    let ha = hash_token_prefix(&tokens_a, 16, 0);
    let hb = hash_token_prefix(&tokens_b, 16, 0);

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
    let prefix_hash = hash_token_prefix(&tokens, 16, 0);

    // Insert snapshot for session 100
    idx.insert(prefix_hash, 42, 100, 16);

    // Lookup from session 200 — should NOT match (different session)
    let result = idx.lookup(&tokens, 16, 200, 0);
    assert_eq!(result, None);

    // Lookup from session 100 — should match
    let result = idx.lookup(&tokens, 16, 100, 0);
    assert_eq!(result, Some((42, 16)));

    // Lookup with session_hash=0 (legacy) — matches any session
    let result = idx.lookup(&tokens, 16, 0, 0);
    assert_eq!(result, Some((42, 16)));
}

#[test]
fn test_snapshot_index_overwrite_existing() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = hash_token_prefix(&tokens, 16, 0);

    // Insert first
    assert!(idx.insert(prefix_hash, 5, 0, 16).is_none());
    assert_eq!(idx.len(), 1);

    // Overwrite same prefix_hash — returns old snapshot_id
    let old = idx.insert(prefix_hash, 8, 0, 16);
    assert_eq!(old, Some(5));
    assert_eq!(idx.len(), 1); // still 1 entry, not 2

    // Lookup returns new value
    let result = idx.lookup(&tokens, 16, 0, 0);
    assert_eq!(result, Some((8, 16)));
}

#[test]
fn test_snapshot_index_deepest_match() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..64).collect();

    // Snapshot at token 16
    let h16 = hash_token_prefix(&tokens, 16, 0);
    idx.insert(h16, 10, 0, 16);

    // Snapshot at token 32
    let h32 = hash_token_prefix(&tokens, 32, 0);
    idx.insert(h32, 20, 0, 32);

    // Lookup with 48 matched tokens — deepest snapshot at 32 wins
    let result = idx.lookup(&tokens, 48, 0, 0);
    assert_eq!(result, Some((20, 32)));

    // Lookup with 20 matched tokens — only snapshot at 16 qualifies
    let result = idx.lookup(&tokens, 20, 0, 0);
    assert_eq!(result, Some((10, 16)));
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
}

/// Phase 1b end-to-end trait surface: a snapshot's location transitions
/// resident → spilled → resident through the `PrefixCache` API, and
/// `lookup` reports each state correctly (the serving path's contract).
#[test]
fn test_spill_tier_lookup_transitions() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    // Register a leaf snapshot (slot 99) for this prefix, session 7.
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 7, 0, 0);

    // Resident: lookup returns the HBM slot, no tier key.
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    assert_eq!(m.ssm_snapshot_tier_key, None);

    // Spill: evict_to_tier KEEPS the entry and returns (freed_slot, key).
    let (freed, key) = tree.evict_snapshot_to_tier().expect("resident victim");
    assert_eq!(freed, 99, "the resident slot is freed for reuse");

    // Spilled: lookup now reports the anchor as tiered — ssm_snapshot is None
    // (no HBM slot) and the tier key is present, so the caller faults it in.
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, None, "no resident slot while spilled");
    assert_eq!(m.ssm_snapshot_tier_key, Some(key));
    assert_eq!(m.ssm_snapshot_tier_tokens, 64);

    // Fault-in completes into slot 123; promote re-homes the entry to HBM.
    assert!(tree.promote_snapshot(key, 123));

    // Resident again at the new slot.
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, Some(123));
    assert_eq!(m.ssm_snapshot_tier_key, None);
    tree.release(&tokens, 16, 0);
}

/// A caches-without-a-tier default: `evict_snapshot_to_tier` / `promote_snapshot`
/// no-op on the trait default (NoPrefixCaching), so a non-radix cache is safe.
#[test]
fn test_no_tier_default_impl() {
    use crate::prefix_cache::NoPrefixCaching;
    let c = NoPrefixCaching;
    assert_eq!(c.evict_snapshot_to_tier(), None);
    assert!(!c.promote_snapshot(123, 0));
}
