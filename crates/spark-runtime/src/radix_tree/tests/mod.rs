// SPDX-License-Identifier: AGPL-3.0-only

//! Test split for radix tree — moved out of `radix_tree.rs` because
//! the combined test file exceeded the workspace 500-LoC budget.

mod adapter;
mod basic;
mod partial_tail;
mod snapshot;
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
    tree.inner.lock().partial_tail_sharing = true;
}
