// SPDX-License-Identifier: AGPL-3.0-only

//! `PrefixCache::peek_snapshot_tokens` (requested by lane L26b for the host
//! KV tier's restore planner): the deepest SSM snapshot a lookup would accept,
//! read-only. It must apply `lookup_tiered`'s acceptance (depth bound, session
//! gate, adapter-keyed prefix hash) and leave no trace.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

const BS: usize = 16;

#[test]
fn an_empty_index_probes_zero() {
    let tree = RadixTree::new();
    let t: Vec<u32> = (0..40).collect();
    assert_eq!(tree.peek_snapshot_tokens(&t, 40, 0, 0), 0);
}

#[test]
fn the_deepest_snapshot_within_the_bound_wins() {
    let tree = RadixTree::new();
    let t: Vec<u32> = (0..48).collect();
    tree.insert_with_snapshot(&t[..16], &[1], &[], BS, 10, 0, 0, 0);
    tree.insert_with_snapshot(&t[..32], &[1, 2], &[], BS, 11, 0, 16, 0);
    tree.insert_with_snapshot(&t, &[1, 2, 3], &[], BS, 12, 0, 32, 0);
    assert_eq!(tree.peek_snapshot_tokens(&t, 48, 0, 0), 48);
    assert_eq!(
        tree.peek_snapshot_tokens(&t, 40, 0, 0),
        32,
        "deeper than the bound"
    );
    assert_eq!(
        tree.peek_snapshot_tokens(&t[..20], 48, 0, 0),
        16,
        "bounded by the prompt"
    );
    // A different continuation shares only the first snapshot's prefix.
    let mut other = t[..16].to_vec();
    other.extend(500..532);
    assert_eq!(tree.peek_snapshot_tokens(&other, 48, 0, 0), 16);
}

#[test]
fn the_probe_is_adapter_keyed_and_session_gated_for_tails() {
    let tree = RadixTree::new();
    let t: Vec<u32> = (0..32).collect();
    tree.insert_with_snapshot(&t, &[1, 2], &[], BS, 20, 0, 0, 5);
    assert_eq!(
        tree.peek_snapshot_tokens(&t, 32, 0, 0),
        0,
        "another adapter misses"
    );
    assert_eq!(tree.peek_snapshot_tokens(&t, 32, 0, 5), 32);

    let tail: Vec<u32> = (100..140).collect();
    tree.insert(&tail, &[7, 8, 9], &[], BS, 0, 0);
    let _ = tree.insert_tail_snapshot(&tail, 21, 0xabc, 0);
    assert_eq!(
        tree.peek_snapshot_tokens(&tail, 40, 0xabc, 0),
        40,
        "same session"
    );
    assert_eq!(
        tree.peek_snapshot_tokens(&tail, 40, 0xdef, 0),
        0,
        "a tail needs its session"
    );
    assert_eq!(
        tree.peek_snapshot_tokens(&tail, 40, 0, 0),
        0,
        "and a session at all"
    );
}

#[test]
fn the_probe_takes_no_reference_and_counts_no_lookup() {
    let tree = RadixTree::new();
    let t: Vec<u32> = (0..32).collect();
    tree.insert_with_snapshot(&t, &[1, 2], &[], BS, 30, 0, 0, 0);
    tree.release(&t, BS, 0);
    let before = tree.inner.lock().test_dump();
    let lookups = tree.snapshot_index.lock().stats.lookups;
    assert_eq!(tree.peek_snapshot_tokens(&t, 32, 0, 0), 32);
    assert_eq!(
        tree.inner.lock().test_dump(),
        before,
        "no radix state touched"
    );
    assert_eq!(
        tree.snapshot_index.lock().stats.lookups,
        lookups,
        "no lookup counted"
    );
}
