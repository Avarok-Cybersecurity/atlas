// SPDX-License-Identifier: AGPL-3.0-only

//! Tier-1 LRU: the compiled-grammar cache, bounded.
//!
//! # The bug this closes
//!
//! This cache was a bare `DashMap<CacheKey, CompiledGrammar>` with no
//! eviction. `cache_limit_bytes` was recorded and never enforced — the C++
//! `ThreadSafeLRUCache` simply was not ported, while the Tier-2
//! [`super::rule_cache::RuleLevelCache`] was.
//!
//! That is unbounded in the one dimension that matters for a server:
//! [`CacheKey::Schema`] is keyed by the **full tool-schema text**, and agentic
//! traffic sends a distinct schema per request. Every request therefore minted
//! a permanent entry. Measured on a GB10 serve under BFCL: **~100 MB/min of
//! host RSS, roughly 17 MB per request**, with kernel slab flat — pure
//! userspace heap. It matches the production report in issue #368, where
//! operators restart the container roughly every seven hours.
//!
//! # Why entries grow AFTER insertion
//!
//! A `CompiledGrammar` is not a fixed-size value. Each one owns a
//! `mask_cache: Mutex<AHashMap<ParserState, Arc<AdaptiveTokenMask>>>` that
//! fills lazily as that grammar is *used* (XGrammar-2 JIT). So an entry's
//! footprint is a moving target, and a size captured at insert time
//! understates it — often by orders of magnitude, since a freshly compiled
//! grammar has an empty mask cache.
//!
//! This is why the budget is re-measured on each insert rather than
//! accumulated incrementally, using the existing
//! [`CompiledGrammar::memory_size_bytes`] (which already sums the grammar plus
//! whatever masks are live): the sum of what is actually resident now is the
//! only honest number. Inserts happen on a cache MISS, which by construction
//! is the rare path, so an O(entries) walk there is not on the hot path.
//!
//! Dropping a grammar drops its mask cache with it, which is what makes
//! bounding Tier 1 sufficient to bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::compiled_grammar::CompiledGrammar;

/// Pass as the budget for no bound.
pub const UNLIMITED_BYTES: usize = usize::MAX;

struct Entry<K> {
    key: K,
    value: CompiledGrammar,
}

struct Inner<K> {
    /// Key -> index into `lru`. `lru[0]` is least-recently-used.
    index: HashMap<K, usize>,
    lru: Vec<Entry<K>>,
}

/// An LRU-bounded cache of compiled grammars.
///
/// Cheap to clone; state is shared through an `Arc`, matching
/// [`super::rule_cache::RuleLevelCache`].
pub struct GrammarCache<K> {
    max_bytes: usize,
    inner: Arc<Mutex<Inner<K>>>,
}

impl<K> Clone for GrammarCache<K> {
    fn clone(&self) -> Self {
        Self {
            max_bytes: self.max_bytes,
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K: std::hash::Hash + Eq + Clone> GrammarCache<K> {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            inner: Arc::new(Mutex::new(Inner {
                index: HashMap::new(),
                lru: Vec::new(),
            })),
        }
    }

    /// Cached grammar for `key`, marking it most-recently-used on a hit.
    pub fn get(&self, key: &K) -> Option<CompiledGrammar> {
        let mut inner = self.inner.lock().expect("grammar cache mutex poisoned");
        let idx = *inner.index.get(key)?;
        Self::touch(&mut inner, idx);
        Some(inner.lru[inner.lru.len() - 1].value.clone())
    }

    /// Drop every entry. Port target for `GrammarCompiler::ClearCache`.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("grammar cache mutex poisoned");
        inner.index.clear();
        inner.lru.clear();
    }

    /// Bytes currently resident, re-measured across live entries.
    pub fn resident_bytes(&self) -> usize {
        let inner = self.inner.lock().expect("grammar cache mutex poisoned");
        inner.lru.iter().map(|e| e.value.memory_size_bytes()).sum()
    }

    /// Insert `value` under `key`, evicting least-recently-used entries until
    /// the total fits the budget.
    ///
    /// The new entry is ALWAYS retained, even if it alone exceeds the budget:
    /// the caller is about to use it, and evicting the grammar a live request
    /// depends on would turn a memory bound into a correctness problem. An
    /// oversized entry is instead evicted by the next insert, once it is no
    /// longer the most recent.
    pub fn insert(&self, key: K, value: CompiledGrammar) {
        let mut inner = self.inner.lock().expect("grammar cache mutex poisoned");
        if let Some(&idx) = inner.index.get(&key) {
            Self::touch(&mut inner, idx);
            return;
        }

        let slot = inner.lru.len();
        inner.index.insert(key.clone(), slot);
        inner.lru.push(Entry { key, value });

        if self.max_bytes == UNLIMITED_BYTES {
            return;
        }
        // Re-measured, not accumulated: entries grow after insertion as their
        // mask caches fill (see the module docs).
        let mut total: usize = inner.lru.iter().map(|e| e.value.memory_size_bytes()).sum();
        while total > self.max_bytes && inner.lru.len() > 1 {
            let evicted = inner.lru.remove(0);
            total = total.saturating_sub(evicted.value.memory_size_bytes());
            inner.index.remove(&evicted.key);
            // Removing index 0 shifts every later index down by one.
            for v in inner.index.values_mut() {
                *v -= 1;
            }
        }
    }

    /// Move `idx` to the back (most-recently-used).
    fn touch(inner: &mut Inner<K>, idx: usize) {
        let last = inner.lru.len() - 1;
        if idx == last {
            return;
        }
        let entry = inner.lru.remove(idx);
        inner.lru.push(entry);
        // Everything after `idx` shifted down one; the moved entry is last.
        for v in inner.index.values_mut() {
            if *v > idx {
                *v -= 1;
            }
        }
        let key = inner.lru[last].key.clone();
        inner.index.insert(key, last);
    }
}

#[cfg(test)]
#[path = "grammar_cache_tests.rs"]
mod grammar_cache_tests;
