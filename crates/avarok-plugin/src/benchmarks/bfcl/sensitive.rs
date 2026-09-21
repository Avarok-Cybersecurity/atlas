// SPDX-License-Identifier: AGPL-3.0-only

//! The BFCL samples known to answer differently depending on what ran before
//! them.
//!
//! Issue #936, measured 2026-09-08 at pin `e897463b54` on the shipped serve
//! (temp 0, seed 42): the golden n=995 draw run WHOLE and the same draw run as
//! its four shards disagreed on exactly these twelve `sample_id`s. Every one
//! of them is a near-tied argmax whose emitted token depends on which SSM
//! snapshot anchor the request restored from — cross-request snapshot reuse,
//! the mechanism [`crate::gate::group`] documents — so the partition, not the
//! model, decides the answer.
//!
//! The certified regime keeps that reuse ON (owner decision, 2026-09-13). This
//! list exists so the fact is visible where the number is produced: the driver
//! warns on each of these it scores and the record carries the count as
//! `known_partition_sensitive`. It is a statement about the shipped engine at
//! that pin, not a whitelist — a sample being here changes nothing about how
//! it is scored.
//!
//! Ten are `live_irrelevance` (the "make no call" subset, where a flipped token
//! is the difference between a correct abstention and a spurious call), one is
//! `live_multiple`, one is `live_parallel_multiple`. Under `--hermetic` the
//! same comparison reads 0 of 995 (#981).

/// The twelve, sorted, as recorded in #936.
pub const KNOWN_PARTITION_SENSITIVE: &[&str] = &[
    "live_irrelevance_14-2-2",
    "live_irrelevance_15-2-3",
    "live_irrelevance_16-2-4",
    "live_irrelevance_2-0-2",
    "live_irrelevance_40-2-28",
    "live_irrelevance_47-2-35",
    "live_irrelevance_55-2-43",
    "live_irrelevance_71-2-59",
    "live_irrelevance_79-2-67",
    "live_irrelevance_8-0-8",
    "live_multiple_57-22-4",
    "live_parallel_multiple_3-2-1",
];

/// Is this sample one of the twelve?
pub fn is_known(sample_id: &str) -> bool {
    KNOWN_PARTITION_SENSITIVE.binary_search(&sample_id).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_is_sorted_and_deduplicated_so_binary_search_is_valid() {
        let mut sorted = KNOWN_PARTITION_SENSITIVE.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted, KNOWN_PARTITION_SENSITIVE, "keep the list sorted");
        assert_eq!(KNOWN_PARTITION_SENSITIVE.len(), 12, "#936 records twelve");
    }

    #[test]
    fn every_id_names_a_single_turn_subset() {
        for id in KNOWN_PARTITION_SENSITIVE {
            let subset = super::super::draw::SINGLE_TURN_SUBSETS
                .iter()
                .filter(|s| id.starts_with(&format!("{s}_")))
                .max_by_key(|s| s.len());
            assert!(subset.is_some(), "{id} does not belong to a scored subset");
        }
    }

    #[test]
    fn membership_is_exact() {
        assert!(is_known("live_irrelevance_2-0-2"));
        assert!(is_known("live_parallel_multiple_3-2-1"));
        // Negative controls: a neighbour, a prefix, and an unrelated id.
        assert!(!is_known("live_irrelevance_2-0-3"));
        assert!(!is_known("live_irrelevance_2-0"));
        assert!(!is_known("simple_python_0"));
    }
}
