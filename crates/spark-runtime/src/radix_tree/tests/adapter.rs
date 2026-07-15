// SPDX-License-Identifier: AGPL-3.0-only

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

// ── Task #24: adapter-correct KV / prefix cache ──

/// A prefix cached under adapter A must NOT be reused by an adapter-B request
/// (or a base request), because the cached K/V carry adapter A's delta. A
/// cross-adapter lookup is a MISS → recompute under the right adapter. The
/// SAME adapter still reuses its own prefix (the cache still helps).
#[test]
fn test_kv_cache_adapter_isolation() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0x1111_2222_3333_4444;
    const B: u64 = 0x5555_6666_7777_8888;

    // Adapter A caches a prefix (cache-miss insert, then the inserting seq exits).
    tree.insert(&tokens, &[10, 20], &[], 16, 0, A);
    tree.release(&tokens, 16, A);

    // Adapter B: same tokens, different adapter → cross-adapter MISS.
    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert!(
        m_b.is_empty(),
        "adapter B must NOT reuse adapter A's KV blocks"
    );

    // Base (id 0): also must not reuse an adapter's blocks.
    let m_base = tree.lookup(&tokens, 16, 0, 0);
    assert!(
        m_base.is_empty(),
        "base must NOT reuse an adapter's KV blocks"
    );

    // Same adapter A: full reuse still works.
    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(m_a.matched_tokens, 32);
    assert_eq!(m_a.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, A);
}

/// The reverse direction: base-computed blocks are keyed under the base
/// sentinel (id 0) and must not be reused by any adapter, while base requests
/// still hit them (byte-identical to pre-#24 behavior).
#[test]
fn test_base_blocks_not_reused_by_adapter() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();

    tree.insert(&tokens, &[10, 20], &[], 16, 0, 0); // base
    tree.release(&tokens, 16, 0);

    // An adapter request misses base blocks.
    let m_adapter = tree.lookup(&tokens, 16, 0, 0xABCD);
    assert!(m_adapter.is_empty());

    // Base still hits (unchanged cache behavior).
    let m_base = tree.lookup(&tokens, 16, 0, 0);
    assert_eq!(m_base.matched_tokens, 32);
    assert_eq!(m_base.matched_blocks, vec![10, 20]);
    tree.release(&tokens, 16, 0);
}

/// The reviewer's insert-collision hazard: after adapter B misses on the same
/// tokens, its own insert must create a DISJOINT node (not clobber adapter A's
/// node and steal its physical block). Both adapters must then reuse their OWN
/// blocks independently.
#[test]
fn test_adapter_insert_does_not_clobber_other_adapter() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..32).collect();
    const A: u64 = 0xAAAA;
    const B: u64 = 0xBBBB;

    // A caches blocks [10,20]; B caches DIFFERENT physical blocks [30,40] for
    // the identical token stream.
    tree.insert(&tokens, &[10, 20], &[], 16, 0, A);
    tree.release(&tokens, 16, A);
    tree.insert(&tokens, &[30, 40], &[], 16, 0, B);
    tree.release(&tokens, 16, B);

    // Each adapter reuses its OWN blocks — no cross-contamination.
    let m_a = tree.lookup(&tokens, 16, 0, A);
    assert_eq!(
        m_a.matched_blocks,
        vec![10, 20],
        "adapter A kept its blocks"
    );
    tree.release(&tokens, 16, A);

    let m_b = tree.lookup(&tokens, 16, 0, B);
    assert_eq!(
        m_b.matched_blocks,
        vec![30, 40],
        "adapter B kept its blocks"
    );
    tree.release(&tokens, 16, B);
}
