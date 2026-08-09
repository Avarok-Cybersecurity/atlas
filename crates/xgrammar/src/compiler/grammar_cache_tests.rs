// SPDX-License-Identifier: AGPL-3.0-only

//! The cache exists to be BOUNDED, so most of these assert eviction happens
//! rather than that lookups work.

/// A stand-in with a controllable footprint. `CompiledGrammar` needs a real
/// tokenizer and grammar to build, which would make these tests a compilation
/// harness rather than a cache harness.
mod fake {
    pub struct Sized(pub usize);
    impl Sized {
        pub fn memory_size_bytes(&self) -> usize {
            self.0
        }
    }
}

/// Mirrors `GrammarCache` over the stand-in so the eviction policy itself is
/// under test. Kept in lockstep with the real `insert` by
/// `the_policy_mirrors_the_real_insert` below.
struct TestCache {
    max_bytes: usize,
    lru: Vec<(u32, fake::Sized)>,
}

impl TestCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            lru: Vec::new(),
        }
    }
    fn insert(&mut self, key: u32, size: usize) {
        if let Some(i) = self.lru.iter().position(|(k, _)| *k == key) {
            let e = self.lru.remove(i);
            self.lru.push(e);
            return;
        }
        self.lru.push((key, fake::Sized(size)));
        if self.max_bytes == usize::MAX {
            return;
        }
        let mut total: usize = self.lru.iter().map(|(_, v)| v.memory_size_bytes()).sum();
        while total > self.max_bytes && self.lru.len() > 1 {
            let (_, v) = self.lru.remove(0);
            total = total.saturating_sub(v.memory_size_bytes());
        }
    }
    fn keys(&self) -> Vec<u32> {
        self.lru.iter().map(|(k, _)| *k).collect()
    }
    fn get(&mut self, key: u32) -> bool {
        match self.lru.iter().position(|(k, _)| *k == key) {
            Some(i) => {
                let e = self.lru.remove(i);
                self.lru.push(e);
                true
            }
            None => false,
        }
    }
}

/// ★ The bug: without a bound, distinct keys accumulate forever. This is the
/// BFCL shape — a new tool schema on every request.
#[test]
fn distinct_keys_do_not_accumulate_without_bound() {
    let mut c = TestCache::new(1000);
    for k in 0..100 {
        c.insert(k, 100);
    }
    assert!(
        c.keys().len() <= 10,
        "bounded to the budget, got {}",
        c.keys().len()
    );
}

#[test]
fn unlimited_really_is_unlimited() {
    let mut c = TestCache::new(usize::MAX);
    for k in 0..50 {
        c.insert(k, 1_000_000);
    }
    assert_eq!(c.keys().len(), 50, "-1 stays an explicit opt-out");
}

/// Least-recently-USED, not least-recently-inserted: a re-read must protect
/// an entry from the next eviction.
#[test]
fn a_recent_read_protects_an_entry_from_eviction() {
    let mut c = TestCache::new(300);
    c.insert(1, 100);
    c.insert(2, 100);
    c.insert(3, 100);
    assert!(c.get(1), "1 is resident"); // 1 becomes most-recent
    c.insert(4, 100); // evicts the now-oldest, which is 2
    let keys = c.keys();
    assert!(keys.contains(&1), "the re-read entry survived: {keys:?}");
    assert!(!keys.contains(&2), "the genuinely-oldest went: {keys:?}");
}

/// ★ The newest entry is never evicted, even when it alone busts the budget:
/// a live request is about to use it, so evicting it would trade a memory
/// bound for a correctness bug.
#[test]
fn an_oversized_newest_entry_is_retained() {
    let mut c = TestCache::new(100);
    c.insert(1, 10_000);
    assert_eq!(c.keys(), vec![1]);
    // ...and it is evicted by the NEXT insert, once it is no longer newest.
    c.insert(2, 10);
    assert_eq!(
        c.keys(),
        vec![2],
        "the oversized entry left once superseded"
    );
}

#[test]
fn reinserting_a_present_key_refreshes_rather_than_duplicates() {
    let mut c = TestCache::new(1000);
    c.insert(1, 100);
    c.insert(2, 100);
    c.insert(1, 100);
    assert_eq!(
        c.keys(),
        vec![2, 1],
        "1 moved to most-recent, not duplicated"
    );
}

/// The real cache must agree with the policy modelled above. If `insert`
/// changes shape, this is the test that should fail.
#[test]
fn the_policy_mirrors_the_real_insert() {
    let src = include_str!("grammar_cache.rs");
    assert!(
        src.contains("while total > self.max_bytes && inner.lru.len() > 1"),
        "the real eviction loop changed shape — update TestCache to match"
    );
    assert!(
        src.contains("if self.max_bytes == UNLIMITED_BYTES"),
        "the unlimited short-circuit changed shape"
    );
}
