// SPDX-License-Identifier: AGPL-3.0-only

//! Test split for radix tree — moved out of `radix_tree.rs` because
//! the combined test file exceeded the workspace 500-LoC budget.
//!
//! Both `radix_lru_index` modes run under every test here: test builds always
//! maintain the ordered victim index (`inner/lru_index.rs`), and every
//! eviction pick asserts that the index equals a fresh rebuild and that its
//! victim is the full scan's, whichever mode the tree evicts with. A test that
//! needs one mode explicitly uses [`force_lru_index`].

mod adapter;
mod basic;
mod cow_tail;
mod lru_index;
mod partial_tail;
mod snapshot;
mod snapshot_probe;
mod snapshot_reap;

/// Arm the pre-#1193 sub-block tail arms on THIS tree only.
///
/// `RadixTree` resolves the lever once at construction from
/// `AVAROK_PREFIX_SUBBLOCK` (`radix_tree::partial_tail`), and the shipped
/// value is OFF. A test binary cannot steer a process-global env read — every
/// test shares one process and the read is cached — so a test that needs the
/// legacy arms sets the resolved value on its own tree instead. This is the
/// state the production path reads; nothing here is a second code path.
pub(super) fn arm_legacy_partial_tail(tree: &super::RadixTree) {
    let mut inner = tree.inner.lock();
    inner.partial_tail_sharing = true;
    inner.cow_tail = false;
}

/// Put THIS tree in copy-on-write mode (row `prefix_cow_tail`), the state
/// `RadixTree::with_levers(_, true)` builds, for a test that started from
/// `RadixTree::new()`.
pub(super) fn arm_cow_tail(tree: &super::RadixTree) {
    let mut inner = tree.inner.lock();
    inner.cow_tail = true;
    inner.partial_tail_sharing = false;
}

/// Force THIS tree's victim source: `true` = the ordered index (row
/// `radix_lru_index` on), `false` = the full scan. The index itself stays
/// maintained either way in test builds, so the per-pick cross-check runs in
/// both modes.
pub(super) fn force_lru_index(tree: &super::RadixTree, on: bool) {
    let mut inner = tree.inner.lock();
    if inner.lru.is_none() {
        inner.lru = Some(Default::default());
        inner.lru_rebuild();
    }
    inner.use_lru = on;
}
