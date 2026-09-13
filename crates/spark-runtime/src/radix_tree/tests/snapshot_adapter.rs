// SPDX-License-Identifier: AGPL-3.0-only

//! The snapshot-adapter and SSM-anchor cases, split out of `snapshot.rs` to
//! keep it under the 500-line cap.

use super::*;

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

// ── Marconi re-anchor below the exact leaf (`lookup_ssm_anchor`) ──

/// The full-prompt-hit shape measured on qwen4exp (2026-08-30): a 3286-token
/// prompt whose finish leaf (3286) AND block-aligned intermediate checkpoint
/// (3264 = 204 blocks) are both registered. `lookup` must keep returning the
/// deepest anchor (the leaf); the re-anchor capped at `total - 1` must hand
/// back the intermediate instead of nothing.
#[test]
fn test_lookup_ssm_anchor_below_exact_leaf_selects_intermediate() {
    // A 3286-token EXACT-leaf hit only exists when sub-block matching is on
    // (3286 is not a multiple of 16). That arm is opt-in now, so ask for it
    // explicitly — this test is about the re-anchor, not about the arm.
    let tree = RadixTree::with_subblock_matching(true);
    let total = 3286usize;
    let tokens: Vec<u32> = (0..total as u32).collect();
    let block_table: Vec<u32> = (0..206).collect(); // 205 full + 1 partial
    tree.insert_with_snapshot(&tokens, &block_table, &[], 16, 99, 0, 0, 0);
    let tokens_at_204: Vec<u32> = (0..3264).collect();
    tree.insert_intermediate_snapshot(&tokens_at_204, &block_table[..204], &[], 16, 50, 0, 0, 0);

    // `lookup` is unchanged: the exact leaf is the deepest anchor.
    let m = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m.matched_tokens, total);
    assert_eq!(m.ssm_anchor().depth(), total);
    assert_eq!(m.ssm_snapshot, Some(99));

    // Re-anchor strictly below the prompt → the 3264 checkpoint.
    let a = tree.lookup_ssm_anchor(&tokens, total - 1, 0, 0);
    assert_eq!(a.snapshot, Some(50));
    assert_eq!(a.snapshot_tokens, 3264);
    assert_eq!(a.depth(), 3264);
    assert!(!a.is_tail);
    assert_eq!(a.tier_key, None);

    // The cap is inclusive: at exactly 3264 the checkpoint still qualifies,
    // one below it nothing does.
    assert_eq!(
        tree.lookup_ssm_anchor(&tokens, 3264, 0, 0).snapshot,
        Some(50)
    );
    assert!(!tree.lookup_ssm_anchor(&tokens, 3263, 0, 0).is_some());

    // Re-anchoring takes no KV refs: the single release balances the lookup.
    tree.release(&tokens, 16, 0);
}

/// The re-anchor applies the same candidate filter as `lookup`: a TAIL
/// snapshot below the leaf is session-gated, an exact intermediate is not.
#[test]
fn test_lookup_ssm_anchor_below_exact_leaf_session_gates_tails() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 0, 0, 0);
    // Tail at 48 for session 7; content-addressed intermediate at 32.
    let tokens_at_48: Vec<u32> = (0..48).collect();
    tree.insert_tail_snapshot(&tokens_at_48, 77, 7, 0);
    let tokens_at_32: Vec<u32> = (0..32).collect();
    tree.insert_intermediate_snapshot(&tokens_at_32, &[10, 20], &[], 16, 50, 0, 0, 0);

    // Same session: the deeper tail wins below the leaf.
    let a = tree.lookup_ssm_anchor(&tokens, 63, 7, 0);
    assert_eq!(
        (a.snapshot, a.snapshot_tokens, a.is_tail),
        (Some(77), 48, true)
    );
    // Other / no session: the tail is skipped, the intermediate is next.
    let a = tree.lookup_ssm_anchor(&tokens, 63, 8, 0);
    assert_eq!(
        (a.snapshot, a.snapshot_tokens, a.is_tail),
        (Some(50), 32, false)
    );
    let a = tree.lookup_ssm_anchor(&tokens, 63, 0, 0);
    assert_eq!(a.snapshot, Some(50));
    // Different adapter: nothing matches.
    assert!(!tree.lookup_ssm_anchor(&tokens, 63, 7, 1).is_some());
    // Divergent prefix: the hash check rejects every entry.
    let mut other: Vec<u32> = (0..48).collect();
    other.extend(500..516);
    assert_eq!(tree.lookup_ssm_anchor(&other, 63, 7, 0).snapshot, Some(77));
    let mut other: Vec<u32> = (0..16).collect();
    other.extend(500..548);
    assert!(!tree.lookup_ssm_anchor(&other, 63, 7, 0).is_some());
    tree.release(&tokens, 16, 0);
}
