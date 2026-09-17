// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `verify_hc.rs`, in a sibling so the parent stays under the
//! 500-line cap. Declared there with `#[path]`, the repo's existing
//! pattern; the module keeps its name and its `super::` imports.

use super::super::async_chkpt::commit_rewind_index;
use super::hc_publish_rows;

/// THE regression this pins: every intermediate slot
/// `commit_accepted_prefix` can read on a partial accept must have been
/// published by the verify that preceded it. When the mHC verify ran as a
/// fused `1 + (K-1)` pair it published NOTHING, so every K=3 step rewound
/// 36 GDN layers onto never-written pool memory.
#[test]
fn hc_publish_covers_every_commit_rewind() {
    for k in 2..=8usize {
        let published = hc_publish_rows(k);
        // `num_accepted == k` short-circuits before any rewind; 0 bails.
        for num_accepted in 1..k {
            let idx = commit_rewind_index(num_accepted);
            assert!(
                published.contains(&idx),
                "k={k}: commit_accepted_prefix({num_accepted}) reads intermediate \
                 {idx}, which the verify never publishes ({published:?})"
            );
        }
    }
}

/// The PLE carry's snapshot boundaries must be the SAME set as the SSM
/// carry's. They are recorded in different crates' worth of code — the
/// SSM's by the conv+GDN kernels via `hc_publish_rows`, PLE's by
/// `decode_batched_inner_hc`'s row loop via `hc_verify_snapshot_rows` —
/// and a partial accept reads BOTH at the same index. One range shorter
/// than the other is a silent desync of exactly the kind this pins.
#[test]
fn hc_ple_snapshot_range_matches_the_ssm_one() {
    use crate::layers::qwen3_ssm::trait_decode_batched_hc::hc_verify_snapshot_rows;
    for k in 1..=8usize {
        assert_eq!(
            hc_verify_snapshot_rows(k),
            hc_publish_rows(k),
            "k={k}: the PLE and SSM verify carries disagree about which row \
             boundaries a commit can land on"
        );
        for num_accepted in 1..k {
            assert!(
                hc_verify_snapshot_rows(k).contains(&commit_rewind_index(num_accepted)),
                "k={k}: commit of {num_accepted} rows has no PLE snapshot"
            );
        }
    }
}

/// And nothing beyond: publishing the LAST row would cost a copy per GDN
/// layer that no rewind can reach (`num_accepted == k` short-circuits).
#[test]
fn hc_publish_stops_at_the_last_reachable_row() {
    assert_eq!(hc_publish_rows(1), 0..0);
    assert_eq!(hc_publish_rows(2), 0..1);
    assert_eq!(hc_publish_rows(3), 0..2);
    assert_eq!(hc_publish_rows(4), 0..3);
}

/// THE regression this pins, aux half: every aux snapshot a commit can
/// restore must be one the verify actually stashed, and it must be the row
/// the SSM rewind lands on.
///
/// This FAILS against the pre-fix code, which stashed exactly ONE snapshot
/// (after row 0) and restored it unconditionally: at k=3 with two rows
/// committed the correct row is 1, so the carries came back a row short of
/// the SSM state the same commit had just rewound.
#[test]
fn aux_restore_row_tracks_the_ssm_rewind_index() {
    for k in 2..=8usize {
        let published = hc_publish_rows(k);
        for num_accepted in 1..k {
            let idx = super::verify_aux_restore_row(num_accepted, k).unwrap_or_else(|| {
                panic!("k={k}: partial accept of {num_accepted} restores no aux row")
            });
            assert_eq!(
                idx,
                commit_rewind_index(num_accepted),
                "k={k}, accepted={num_accepted}: the aux carry and the SSM state \
                 would be rewound to different tokens"
            );
            assert!(
                published.contains(&idx),
                "k={k}: commit of {num_accepted} rows restores aux snapshot {idx}, \
                 which the verify never stashed ({published:?})"
            );
        }
    }
}

/// The single-snapshot behaviour this replaced, stated as the thing that is
/// now false. Row 0 is correct ONLY for a one-row commit.
#[test]
fn aux_restore_row_is_not_always_row_zero() {
    assert_eq!(super::verify_aux_restore_row(1, 2), Some(0));
    assert_eq!(super::verify_aux_restore_row(1, 3), Some(0));
    assert_eq!(super::verify_aux_restore_row(2, 3), Some(1));
    assert_eq!(super::verify_aux_restore_row(2, 4), Some(1));
    assert_eq!(super::verify_aux_restore_row(3, 4), Some(2));
}

/// A full accept discarded nothing, so it restores nothing — the same
/// short-circuit `commit_accepted_prefix` takes at `num_accepted == k`.
#[test]
fn aux_restore_row_is_none_on_a_full_accept() {
    for k in 1..=8usize {
        assert_eq!(super::verify_aux_restore_row(k, k), None, "k={k}");
    }
}

/// The stash the verify builds must be exactly as long as the restore can
/// index. `decode_verify_hc` pushes one entry per `hc_publish_rows` row, so
/// the highest valid index is `len - 1`.
#[test]
fn every_restorable_row_is_inside_the_stash() {
    for k in 2..=8usize {
        let stashed = hc_publish_rows(k).len();
        for num_accepted in 1..=k {
            if let Some(idx) = super::verify_aux_restore_row(num_accepted, k) {
                assert!(
                    idx < stashed,
                    "k={k}: restore row {idx} but the verify stashes {stashed}"
                );
            }
        }
    }
}
