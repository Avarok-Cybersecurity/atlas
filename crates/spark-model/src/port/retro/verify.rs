// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026).

//! Responsibility: one RetroInfer selection for the K+1 rows of a speculative
//! verify (master plan L33, report 74 R9), so draft and verify attend through
//! the same approximate model: each row selects on its own query, the
//! retrieval zone is the union of the rows' retrieval clusters, the estimation
//! zone is the first row's ranked clusters that the union did not take, and
//! every row then attends through the shared zones with its own causal extent
//! (the exact buffer ends with the verify rows' own positions, so row `j` of
//! `R` trims the last `R - 1 - j` entries; `state::ExactBuf::row_trim`).
//!
//! The union is filled rank by rank across rows (every row's best cluster,
//! then every row's second, ...), deduplicated, up to the retrieval capacity,
//! so each row's own retrieval set is contained in the union whenever the
//! union fits; a union cut short is reported.

use std::collections::HashSet;

use super::oracle_eval::Zones;

/// The shared selection of one verify step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UnionZones {
    pub zones: Zones,
    /// Retrieval clusters some row wanted that did not fit the capacity.
    pub cut: u32,
}

/// Union of the rows' selections. `rows[j]` is row `j`'s selection (its
/// retrieval ids in rank order, then its estimation ids in rank order);
/// `retrieval_cap` and `es_cap` are the zone-list capacities.
pub fn union_select(rows: &[Zones], retrieval_cap: usize, es_cap: usize) -> UnionZones {
    let mut seen = HashSet::new();
    let mut retrieval = Vec::new();
    let depth = rows.iter().map(|r| r.retrieval.len()).max().unwrap_or(0);
    let mut cut = 0u32;
    for rank in 0..depth {
        for r in rows {
            let Some(&c) = r.retrieval.get(rank) else {
                continue;
            };
            if seen.contains(&c) {
                continue;
            }
            if retrieval.len() < retrieval_cap {
                seen.insert(c);
                retrieval.push(c);
            } else {
                cut += 1;
            }
        }
    }
    // A cut cluster may still be wanted by a later row; count each id once.
    let want: HashSet<u32> = rows
        .iter()
        .flat_map(|r| r.retrieval.iter().copied())
        .collect();
    cut = cut.min(u32::try_from(want.len().saturating_sub(retrieval.len())).unwrap_or(u32::MAX));
    let estimation: Vec<u32> = rows
        .first()
        .map(|r0| {
            r0.retrieval
                .iter()
                .chain(&r0.estimation)
                .copied()
                .filter(|c| !seen.contains(c))
                .take(r0.estimation.len().min(es_cap))
                .collect()
        })
        .unwrap_or_default();
    UnionZones {
        zones: Zones {
            retrieval,
            estimation,
        },
        cut,
    }
}

/// Exact-buffer entries row `row` of `rows` verify rows ignores.
pub fn row_trim(rows: usize, row: usize) -> u32 {
    u32::try_from(rows.saturating_sub(row + 1)).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn z(r: &[u32], e: &[u32]) -> Zones {
        Zones {
            retrieval: r.to_vec(),
            estimation: e.to_vec(),
        }
    }

    #[test]
    fn union_contains_every_rows_retrieval_set_when_it_fits() {
        let rows = vec![
            z(&[5, 1, 9], &[2, 3, 4]),
            z(&[1, 7, 5], &[9, 2]),
            z(&[8, 5, 1], &[]),
        ];
        let u = union_select(&rows, 16, 8);
        for r in &rows {
            for c in &r.retrieval {
                assert!(
                    u.zones.retrieval.contains(c),
                    "{c} missing from {:?}",
                    u.zones.retrieval
                );
            }
        }
        assert_eq!(u.cut, 0);
        // Rank by rank: every row's best first.
        assert_eq!(&u.zones.retrieval[..3], &[5, 1, 8]);
        let unique: HashSet<_> = u.zones.retrieval.iter().collect();
        assert_eq!(unique.len(), u.zones.retrieval.len(), "no duplicates");
    }

    #[test]
    fn estimation_never_repeats_a_retrieval_cluster() {
        let rows = vec![z(&[5, 1], &[7, 2, 3]), z(&[7, 2], &[5])];
        let u = union_select(&rows, 16, 8);
        for c in &u.zones.estimation {
            assert!(!u.zones.retrieval.contains(c));
        }
        // Row 0's estimation size is kept: 7 and 2 moved to retrieval, so its
        // ranked list continues with 3, then nothing.
        assert_eq!(u.zones.estimation, vec![3]);
    }

    #[test]
    fn a_full_union_is_cut_and_counted() {
        let rows = vec![z(&[1, 2, 3], &[]), z(&[4, 5, 6], &[])];
        let u = union_select(&rows, 4, 0);
        assert_eq!(u.zones.retrieval, vec![1, 4, 2, 5]);
        assert_eq!(u.cut, 2);
    }

    #[test]
    fn one_row_is_its_own_selection() {
        let rows = vec![z(&[3, 1], &[2, 0])];
        let u = union_select(&rows, 8, 8);
        assert_eq!(u.zones, rows[0]);
    }

    #[test]
    fn trims_hide_later_rows() {
        assert_eq!(row_trim(4, 0), 3);
        assert_eq!(row_trim(4, 3), 0);
        assert_eq!(row_trim(1, 0), 0);
    }
}
