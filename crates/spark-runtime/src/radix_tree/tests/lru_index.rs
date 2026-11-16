// SPDX-License-Identifier: AGPL-3.0-only

//! Row `radix_lru_index`: the ordered victim index must pick exactly the
//! victim the full scan picks, so the row is victim-identical.
//!
//! The oracle drives two trees with the same randomized operations (insert,
//! lookup, copy-on-write lookup, release, batched-admission rollback, evict)
//! plus forced same-tick ties: tree A evicts by the scan with no index at all,
//! tree B by the index. Every result must match, every eviction must return
//! the same victims in the same order, and the two trees must stay identical
//! slot for slot. Independently, every pick in B asserts that the index equals
//! a fresh rebuild and agrees with the scan (`lru_index.rs`, test builds).

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

use super::force_lru_index;

const BS: usize = 4;

/// xorshift64*: deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Live {
    tokens: Vec<u32>,
    adapter: u64,
}

/// One scan tree and one index tree, driven in lockstep.
struct Pair {
    scan: RadixTree,
    index: RadixTree,
    live: Vec<Live>,
    next_block: u32,
}

impl Pair {
    fn new(cow: bool) -> Self {
        let scan = RadixTree::with_levers(false, cow);
        // Remove the test-build index entirely: tree A is the pure scan.
        scan.inner.lock().lru = None;
        let index = RadixTree::with_levers(true, cow);
        assert!(index.inner.lock().use_lru);
        Self {
            scan,
            index,
            live: Vec::new(),
            next_block: 1,
        }
    }

    /// A prompt that often shares a prefix with an earlier one: a random
    /// prefix of one of a few fixed documents, then a short random tail.
    fn prompt(rng: &mut Rng, docs: &[Vec<u32>]) -> Vec<u32> {
        let doc = &docs[rng.below(docs.len())];
        let keep = rng.below(doc.len() + 1);
        let mut t = doc[..keep].to_vec();
        for _ in 0..rng.below(2 * BS) {
            t.push(rng.below(3) as u32);
        }
        if t.is_empty() {
            t.push(rng.below(3) as u32);
        }
        t
    }

    fn admit(&mut self, rng: &mut Rng, docs: &[Vec<u32>], cow: bool) {
        let tokens = Self::prompt(rng, docs);
        let adapter = if rng.below(5) == 0 { 7 } else { 0 };
        let (ma, mb) = if cow {
            (
                self.scan.lookup_cow(&tokens, BS, 0, adapter),
                self.index.lookup_cow(&tokens, BS, 0, adapter),
            )
        } else {
            (
                self.scan.lookup(&tokens, BS, 0, adapter),
                self.index.lookup(&tokens, BS, 0, adapter),
            )
        };
        assert_eq!(ma.matched_blocks, mb.matched_blocks);
        assert_eq!(ma.matched_tokens, mb.matched_tokens);
        assert_eq!(ma.cow_tail, mb.cow_tail);
        // The sequence's own table: the shared blocks, then fresh blocks
        // (a copy-on-write tail is a fresh private block, like any other).
        let blocks = tokens.len().div_ceil(BS);
        let mut table = ma.matched_blocks.clone();
        while table.len() < blocks {
            table.push(self.next_block);
            self.next_block += 1;
        }
        let ia = self
            .scan
            .insert(&tokens, &table, &[], BS, ma.matched_tokens, adapter);
        let ib = self
            .index
            .insert(&tokens, &table, &[], BS, mb.matched_tokens, adapter);
        assert_eq!(ia.blocks, ib.blocks);
        assert_eq!(ia.released_blocks, ib.released_blocks);
        self.live.push(Live { tokens, adapter });
    }

    fn release(&mut self, rng: &mut Rng) {
        if self.live.is_empty() {
            return;
        }
        let l = self.live.swap_remove(rng.below(self.live.len()));
        self.scan.release(&l.tokens, BS, l.adapter);
        self.index.release(&l.tokens, BS, l.adapter);
    }

    /// Batched admission that acquires and then rolls back.
    fn rollback(&mut self, rng: &mut Rng, docs: &[Vec<u32>]) {
        let tokens = Self::prompt(rng, docs);
        let ma = self.scan.lookup(&tokens, BS, 0, 0);
        let mb = self.index.lookup(&tokens, BS, 0, 0);
        assert_eq!(ma.matched_blocks, mb.matched_blocks);
        self.scan.release_matched(&tokens, BS, ma.matched_tokens, 0);
        self.index
            .release_matched(&tokens, BS, mb.matched_tokens, 0);
    }

    fn evict(&mut self, n: usize) -> Vec<u32> {
        let a = self.scan.evict(n);
        let b = self.index.evict(n);
        assert_eq!(
            a.physical, b.physical,
            "the index evicted a different victim sequence than the scan"
        );
        a.physical
    }

    /// Give node `j` node `i`'s tick in both trees: a same-tick tie.
    fn force_tie(&mut self, rng: &mut Rng) {
        let live = self.scan.inner.lock().test_live_nodes();
        assert_eq!(live, self.index.inner.lock().test_live_nodes());
        if live.len() < 2 {
            return;
        }
        let i = live[rng.below(live.len())];
        let j = live[rng.below(live.len())];
        let tick = self.scan.inner.lock().test_tick(i);
        self.scan.inner.lock().test_set_last_access(j, tick);
        self.index.inner.lock().test_set_last_access(j, tick);
    }

    fn assert_identical(&self) {
        assert_eq!(
            self.scan.inner.lock().test_dump(),
            self.index.inner.lock().test_dump(),
            "scan and index trees diverged"
        );
    }
}

fn run_oracle(seed: u64, ops: usize, cow: bool) -> usize {
    let mut rng = Rng(seed | 1);
    let docs: Vec<Vec<u32>> = (0..5)
        .map(|_| (0..6 * BS).map(|_| rng.below(3) as u32).collect())
        .collect();
    let mut pair = Pair::new(cow);
    let mut victims = 0usize;
    for op in 0..ops {
        match rng.below(100) {
            // Bounded concurrency keeps the tree small enough for the
            // per-pick full cross-check in an unoptimized test build.
            0..=29 if pair.live.len() < 48 => {
                let cow_lookup = cow && rng.below(2) == 0;
                pair.admit(&mut rng, &docs, cow_lookup);
            }
            0..=54 => pair.release(&mut rng),
            55..=79 => victims += pair.evict(1 + rng.below(6)).len(),
            80..=89 => pair.rollback(&mut rng, &docs),
            _ => {
                // Many ties: force several per tie op.
                for _ in 0..3 {
                    pair.force_tie(&mut rng);
                }
            }
        }
        if op.is_multiple_of(64) {
            pair.assert_identical();
        }
    }
    // Drain: release everything, then evict the whole tree.
    while !pair.live.is_empty() {
        pair.release(&mut rng);
    }
    victims += pair.evict(usize::MAX).len();
    pair.assert_identical();
    assert_eq!(pair.scan.stats(), (0, 0), "a drained tree is empty");
    assert_eq!(pair.index.stats(), (0, 0));
    victims
}

/// The oracle: thousands of mixed operations with many same-tick ties, on
/// several seeds, with and without copy-on-write lookups.
#[test]
fn index_and_scan_evict_identical_victims_on_random_workloads() {
    let mut victims = 0;
    for seed in [1u64, 2, 3, 0x9e37_79b9, 0xdead_beef, 42] {
        victims += run_oracle(seed, 3_000, false);
        victims += run_oracle(seed ^ 0x55, 3_000, true);
    }
    assert!(
        victims > 1_000,
        "the workload must actually evict ({victims})"
    );
}

/// The index tracks every transition of the victim predicate.
#[test]
fn the_index_follows_every_transition_of_the_predicate() {
    let tree = RadixTree::with_levers(true, false);
    let indexed = |t: &RadixTree| t.inner.lock().lru.as_ref().map_or(0, |l| l.len());
    let chain: Vec<u32> = (0..3 * BS as u32).collect();
    tree.insert(&chain, &[10, 11, 12], &[], BS, 0, 0);
    // Seq-owned nodes carry ref 2 until the inserting sequence releases.
    assert_eq!(indexed(&tree), 0, "a live sequence holds every node");
    tree.release(&chain, BS, 0);
    assert_eq!(indexed(&tree), 1, "only the leaf is a candidate");

    // A lookup takes a ref on the whole path: nothing is evictable.
    let m = tree.lookup(&chain, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10, 11, 12]);
    assert_eq!(indexed(&tree), 0);
    tree.release(&chain, BS, 0);
    assert_eq!(indexed(&tree), 1);

    // A branch under the first node (looked up, then cached, then released,
    // as a sequence does): the new leaf is a candidate too.
    let mut branch = chain[..BS].to_vec();
    branch.extend([9, 9, 9, 9]);
    let m = tree.lookup(&branch, BS, 0, 0);
    assert_eq!(m.matched_tokens, BS);
    tree.insert(&branch, &[10, 20], &[], BS, m.matched_tokens, 0);
    tree.release(&branch, BS, 0);
    assert_eq!(indexed(&tree), 2);

    // Evicting the old leaf makes its parent a leaf and a candidate.
    assert_eq!(tree.evict(1).physical, vec![12]);
    assert_eq!(indexed(&tree), 2, "node 11 replaced node 12");
    assert_eq!(tree.evict(3).physical, vec![11, 20, 10]);
    assert_eq!(indexed(&tree), 0);
    assert_eq!(tree.stats(), (0, 0));
}

/// Equal ticks break toward the lowest node id in both modes, because the
/// index orders by `(tick, id)` and the scan replaces its best only on `<`.
#[test]
fn same_tick_ties_break_toward_the_lowest_node_id() {
    for on in [false, true] {
        let tree = RadixTree::new();
        force_lru_index(&tree, on);
        let a: Vec<u32> = vec![1, 1, 1, 1];
        let b: Vec<u32> = vec![2, 2, 2, 2];
        let c: Vec<u32> = vec![3, 3, 3, 3];
        tree.insert(&a, &[100], &[], BS, BS, 0);
        tree.insert(&b, &[200], &[], BS, BS, 0);
        tree.insert(&c, &[300], &[], BS, BS, 0);
        {
            // Nodes 1, 2, 3 in insertion order; give all three one tick,
            // with node 2 stamped first so the arena order is the only rule.
            let mut inner = tree.inner.lock();
            inner.test_set_last_access(2, 5);
            inner.test_set_last_access(3, 5);
            inner.test_set_last_access(1, 5);
        }
        let got: Vec<u32> = (0..3).flat_map(|_| tree.evict(1).physical).collect();
        assert_eq!(got, vec![100, 200, 300], "index={on}");
    }
}

/// A stale index entry would evict a node a sequence still uses (corrupt KV),
/// so the index victim is always checked: release builds rebuild the index and
/// pick correctly.
#[test]
fn a_stale_entry_is_rebuilt_away_rather_than_trusted() {
    let tree = RadixTree::with_levers(true, false);
    let a: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    tree.insert(&a, &[10, 11], &[], BS, 0, 0);
    // The sequence still holds both nodes. Plant the held leaf (node 2) at
    // the front of the index as if a refresh had been missed.
    let mut inner = tree.inner.lock();
    inner.lru.as_mut().unwrap().plant_stale(2, 0);
    assert_eq!(inner.index_victim(false), None, "nothing is evictable");
    inner.lru_assert_consistent();
    drop(inner);
    tree.release(&a, BS, 0);
    assert_eq!(tree.evict(2).physical, vec![11, 10]);
}

/// Debug and test builds refuse a stale entry outright.
#[test]
#[should_panic(expected = "stale entry")]
fn a_stale_entry_panics_in_strict_mode() {
    let tree = RadixTree::with_levers(true, false);
    let a: Vec<u32> = vec![1, 2, 3, 4];
    tree.insert(&a, &[10], &[], BS, 0, 0);
    let mut inner = tree.inner.lock();
    inner.lru.as_mut().unwrap().plant_stale(1, 0);
    let _ = inner.index_victim(true);
}

/// The eviction series count only while timing is on (the index row on, or
/// `AVAROK_PREFIX_EVICT_TIMING=1`), so a row-off target renders zeros.
#[test]
fn eviction_is_timed_and_counted_with_the_index_on() {
    let tree = RadixTree::with_levers(true, false);
    assert!(tree.evict_metrics);
    let a: Vec<u32> = (0..2 * BS as u32).collect();
    tree.insert(&a, &[10, 11], &[], BS, 0, 0);
    tree.release(&a, BS, 0);
    let (_, victims0, calls0, ..) = crate::radix_tree::port_metrics::snapshot();
    assert_eq!(tree.evict(2).physical, vec![11, 10]);
    let (_, victims1, calls1, ..) = crate::radix_tree::port_metrics::snapshot();
    // Other tests may evict concurrently; this tree alone adds 2 and 1.
    assert!(victims1 >= victims0 + 2);
    assert!(calls1 > calls0);
    let mut text = String::new();
    crate::radix_tree::render_prometheus(&mut text);
    for name in [
        "atlas_prefix_evict_seconds_total",
        "atlas_prefix_evict_victims_total",
        "atlas_prefix_cow_copies_total",
    ] {
        assert!(
            text.contains(&format!("# TYPE {name} counter\n{name} ")),
            "{name} missing from:\n{text}"
        );
    }
}
