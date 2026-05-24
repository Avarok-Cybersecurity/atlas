// SPDX-License-Identifier: AGPL-3.0-only
//
// RuleLevelCache — port of `class RuleLevelCache` from
// `cpp/grammar_functor.cc` (upstream commit `bfb2a79`, "feat: support
// crossing-grammar cache", PR #526).
//
// CROSS-GRAMMAR / SUB-STRUCTURE MASK CACHE
// ----------------------------------------
// Two grammars compiled in different requests very often share
// structurally identical rules — every JSON-schema tool definition
// reuses the same `basic_string`, `basic_number`, whitespace and
// punctuation sub-rules. The expensive part of compiling a rule is the
// per-state `AdaptiveTokenMask` computation (a full sorted-vocab scan
// through an `EarleyParser`). That result depends ONLY on the rule's
// FSM structure and the tokenizer — not on which grammar the rule
// belongs to.
//
// `RuleLevelCache` keys a computed mask by a *structural* tuple:
//   (fsm_hash, fsm_new_node_id, state_cnt, edge_cnt)
// where `fsm_hash` is the `GrammarFsmHasher` per-rule structural hash,
// `fsm_new_node_id` is the canonical (renumbered) state id within that
// rule's FSM, and `state_cnt`/`edge_cnt` are the rule FSM's node/edge
// counts (a cheap collision guard, exactly as upstream). A rule seen in
// a previous request — even from an entirely different grammar — reuses
// its masks instead of recomputing them. Upstream measures 3-7x on
// multi-request tool-calling workloads.
//
// LRU EVICTION
// ------------
// Upstream bounds the cache by an approximate memory budget with an LRU
// list. This port mirrors that: `max_cache_memory_size` bounds the sum
// of `AdaptiveTokenMask::memory_size()` of cached entries; on insert,
// least-recently-used entries are evicted until the new entry fits.
// `kUnlimitedSize` (`usize::MAX`) disables the bound — entries are then
// kept until `clear`.
//
// THREAD SAFETY
// -------------
// The `GrammarCompiler` is shared across requests and its grammar-level
// cache is `dashmap`-based; this rule-level cache matches that posture
// with an inner `Mutex` guarding the map + LRU list. Lookups and
// inserts are short (a hash-map probe and an `Arc` clone / LRU splice),
// so a single `Mutex` is cheaper than `DashMap`'s sharding here and
// keeps the LRU list — which is inherently shared — trivially correct.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::mask::AdaptiveTokenMask;

/// Sentinel `max_cache_memory_size` meaning "no memory bound" — port of
/// `RuleLevelCache::kUnlimitedSize` (`static_cast<size_t>(-1)`).
pub const UNLIMITED_SIZE: usize = usize::MAX;

/// The structural cache key — port of `RuleLevelCache::Impl::NodeKey`.
///
/// Faithful to upstream's `std::tuple<uint64_t, int32_t, int32_t,
/// int32_t>`: the rule FSM's structural hash, the renumbered (canonical)
/// node id, and the rule FSM's node / edge counts. The two counts make
/// a hash collision between structurally different rules astronomically
/// unlikely without storing the full FSM in the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuleMaskKey {
    /// Per-rule FSM structural hash from [`crate::grammar::functor::
    /// GrammarFsmHasher`]. For lookahead-bearing / root rules this is
    /// the hash already combined with the lookahead hash + flags by the
    /// caller (see `compile.rs`), so the key stays a single `u64`.
    pub fsm_hash: u64,
    /// The canonical (renumbered) FSM node id of the parser state — from
    /// `per_rule_fsm_new_state_ids`.
    pub fsm_new_node_id: i32,
    /// The rule FSM's node count (`CompactFsmWithStartEnd::num_states`).
    pub state_cnt: i32,
    /// The rule FSM's edge count (`CompactFsmWithStartEnd::num_edges`).
    pub edge_cnt: i32,
}

/// An entry in the LRU intrusive list — a key plus its cached mask. The
/// list is kept newest-at-back, matching upstream's `cache_list_`.
struct Entry {
    key: RuleMaskKey,
    mask: Arc<AdaptiveTokenMask>,
}

/// The mutable interior, guarded by one `Mutex`.
struct Inner {
    /// `key -> index into `lru``. Port of `RuleLevelCache::Impl::cache_`.
    index: HashMap<RuleMaskKey, usize>,
    /// LRU order — `lru[0]` is least-recently-used, `lru.last()` is
    /// most-recently-used. A `Vec` splice is O(n) but the cache is
    /// small (bounded by the memory budget) and lookups dominate; this
    /// keeps the structure `unsafe`-free, which the crate forbids.
    lru: Vec<Entry>,
    /// Sum of `memory_size()` over every cached mask.
    current_size: usize,
}

/// Cross-grammar adaptive-token-mask cache. Cheap to clone — the inner
/// state is shared via [`Arc`], matching the C++ pimpl `shared_ptr`.
#[derive(Clone)]
pub struct RuleLevelCache {
    max_size: usize,
    inner: Arc<Mutex<Inner>>,
}

impl std::fmt::Debug for RuleLevelCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().expect("rule cache mutex poisoned");
        f.debug_struct("RuleLevelCache")
            .field("max_size", &self.max_size)
            .field("entries", &inner.lru.len())
            .field("current_size", &inner.current_size)
            .finish()
    }
}

impl RuleLevelCache {
    /// Construct a cache bounded by `max_cache_memory_size` bytes.
    /// Pass [`UNLIMITED_SIZE`] for no bound. Port of the C++
    /// `RuleLevelCache(size_t)` constructor.
    pub fn new(max_cache_memory_size: usize) -> Self {
        Self {
            max_size: max_cache_memory_size,
            inner: Arc::new(Mutex::new(Inner {
                index: HashMap::new(),
                lru: Vec::new(),
                current_size: 0,
            })),
        }
    }

    /// Look up the cached mask for `key`, marking it most-recently-used
    /// on a hit. Port of `RuleLevelCache::GetCache`.
    pub fn get(&self, key: &RuleMaskKey) -> Option<Arc<AdaptiveTokenMask>> {
        let mut inner = self.inner.lock().expect("rule cache mutex poisoned");
        let idx = *inner.index.get(key)?;
        Self::touch(&mut inner, idx);
        // After `touch`, the entry is at the back.
        Some(Arc::clone(&inner.lru[inner.lru.len() - 1].mask))
    }

    /// Insert `mask` under `key`. Returns `true` if it was stored,
    /// `false` if rejected (already present, or larger than the whole
    /// budget). Port of `RuleLevelCache::AddCache`.
    pub fn add(&self, key: RuleMaskKey, mask: Arc<AdaptiveTokenMask>) -> bool {
        let item_size = mask.memory_size();
        let mut inner = self.inner.lock().expect("rule cache mutex poisoned");

        // A mask larger than the entire budget can never be cached.
        if self.max_size != UNLIMITED_SIZE && item_size > self.max_size {
            return false;
        }
        // Already cached — upstream returns false and does not refresh.
        if inner.index.contains_key(&key) {
            return false;
        }

        // Evict least-recently-used entries until the new item fits.
        if self.max_size != UNLIMITED_SIZE {
            let budget = self.max_size.saturating_sub(item_size);
            while inner.current_size > budget && !inner.lru.is_empty() {
                let evicted = inner.lru.remove(0);
                inner.current_size -= evicted.mask.memory_size();
                inner.index.remove(&evicted.key);
                // Removing index 0 shifts every later index down by one.
                for v in inner.index.values_mut() {
                    *v -= 1;
                }
            }
        }

        inner.current_size += item_size;
        inner.lru.push(Entry { key, mask });
        let last = inner.lru.len() - 1;
        inner.index.insert(key, last);
        true
    }

    /// Get-or-compute: returns the cached mask if present, otherwise
    /// runs `compute`, inserts the result, and returns it. This is the
    /// single call site `compile.rs` uses — it keeps the "hash on
    /// compile; on hit reuse, on miss compute + insert" logic in one
    /// place. The computed value is always returned even if the insert
    /// is rejected (e.g. the mask exceeds the whole budget).
    pub fn get_or_compute<F>(&self, key: RuleMaskKey, compute: F) -> Arc<AdaptiveTokenMask>
    where
        F: FnOnce() -> AdaptiveTokenMask,
    {
        if let Some(hit) = self.get(&key) {
            return hit;
        }
        let computed = Arc::new(compute());
        self.add(key, Arc::clone(&computed));
        computed
    }

    /// Drop every cached entry. Port of `RuleLevelCache::ClearCache`.
    pub fn clear(&self) {
        let mut inner = self.inner.lock().expect("rule cache mutex poisoned");
        inner.index.clear();
        inner.lru.clear();
        inner.current_size = 0;
    }

    /// The configured memory bound in bytes ([`UNLIMITED_SIZE`] when
    /// unbounded). Port of `RuleLevelCache::GetMaxSize`.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Approximate bytes held by the cache — the sum of every cached
    /// mask's `memory_size()`. Port of `MemorySize(const RuleLevelCache&)`.
    pub fn memory_size(&self) -> usize {
        self.inner
            .lock()
            .expect("rule cache mutex poisoned")
            .current_size
    }

    /// Number of cached entries — for tests / introspection.
    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("rule cache mutex poisoned")
            .lru
            .len()
    }

    /// True if the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Move the entry at `idx` to the back of the LRU list (most-
    /// recently-used), fixing every shifted index. Port of the C++
    /// `List::MoveBack`.
    fn touch(inner: &mut Inner, idx: usize) {
        let last = inner.lru.len() - 1;
        if idx == last {
            return;
        }
        let entry = inner.lru.remove(idx);
        let moved_key = entry.key;
        inner.lru.push(entry);
        // Everything that was after `idx` shifted down by one; the moved
        // entry is now at the back.
        for v in inner.index.values_mut() {
            if *v > idx {
                *v -= 1;
            }
        }
        inner.index.insert(moved_key, inner.lru.len() - 1);
    }
}

#[cfg(test)]
#[path = "rule_cache_tests.rs"]
mod tests;
