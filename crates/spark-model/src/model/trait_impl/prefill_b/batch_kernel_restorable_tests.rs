// SPDX-License-Identifier: AGPL-3.0-only

//! `hybrid_match_is_restorable` is the proof the batched reservation leans on
//! when it admits a warm hybrid-SSM match as cold: `false` must imply the
//! per-stream path's own restore vote (`snap_agree::local_proposal`) is `0`
//! for EVERY value of the gates the reservation cannot see. Sibling file
//! because `batch_kernel_tests.rs` sits at the 500-LoC cap.

use spark_runtime::prefix_cache::PrefixMatch;

use super::batch_kernel::hybrid_match_is_restorable;
use super::snap_agree::{LocalGates, local_proposal};

fn warm(matched: usize, resident: usize, tier: usize) -> PrefixMatch {
    PrefixMatch {
        matched_blocks: vec![3; matched / 16],
        matched_disk_block_ids: Vec::new(),
        matched_tokens: matched,
        ssm_snapshot: (resident > 0).then_some(1),
        ssm_snapshot_tokens: resident,
        ssm_snapshot_tier_key: (tier > 0).then_some(9),
        ssm_snapshot_tier_tokens: tier,
        ssm_snapshot_is_tail: false,
    }
}

/// The concurrency sweep's measured burst: warm-up served the exact prompt
/// (200 tokens), so 192 block-aligned tokens match, but the only snapshot is
/// the 200-token leaf, which is not on the matched path.
#[test]
fn the_sweeps_warm_burst_is_unrestorable_and_admitted_as_cold() {
    assert!(!hybrid_match_is_restorable(&warm(192, 0, 0), 256));
    // A shallow intermediate checkpoint is still below the default threshold.
    assert!(!hybrid_match_is_restorable(&warm(192, 176, 0), 256));
}

#[test]
fn a_restorable_depth_keeps_the_wave_on_the_per_stream_path() {
    assert!(hybrid_match_is_restorable(&warm(512, 496, 0), 256));
    // A spilled anchor counts: the per-stream path faults it back in.
    assert!(hybrid_match_is_restorable(&warm(512, 0, 496), 256));
    // Exactly at the threshold restores (`>=` in local_proposal).
    assert!(hybrid_match_is_restorable(&warm(256, 256, 0), 256));
    // A threshold of 0 still needs a snapshot to exist.
    assert!(!hybrid_match_is_restorable(&warm(64, 0, 0), 0));
    assert!(hybrid_match_is_restorable(&warm(64, 48, 0), 0));
}

/// The soundness property, exhaustively over the gates the reservation does
/// not evaluate: whenever the predicate says "cannot restore", the per-stream
/// vote over EITHER candidate depth is 0 — so demoting to cold changes which
/// kernel runs, never whether a snapshot is restored.
#[test]
fn unrestorable_implies_the_per_stream_vote_is_zero() {
    let depths = [0usize, 16, 176, 192, 255, 256, 257, 496];
    let mins = [0usize, 1, 176, 256, 512];
    let mut checked = 0usize;
    for &resident in &depths {
        for &tier in &depths {
            for &min_tokens in &mins {
                let m = warm(512, resident, tier);
                if hybrid_match_is_restorable(&m, min_tokens) {
                    continue;
                }
                for snap_tok in [resident, tier] {
                    for bits in 0u32..64 {
                        let bit = |i: u32| bits & (1 << i) != 0;
                        let g = LocalGates {
                            snap_tok,
                            matched: m.matched_tokens,
                            total: if bit(0) { 512 } else { 600 },
                            min_tokens,
                            has_hidden: bit(1),
                            exact_enabled: bit(2),
                            is_tail: bit(3),
                            session_ok: bit(4),
                            needs_aux: bit(5),
                            has_aux: true,
                        };
                        assert_eq!(
                            local_proposal(&g),
                            0,
                            "predicate said unrestorable but the vote restores: \
                             resident={resident} tier={tier} min={min_tokens} {bits:#b}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(
        checked > 1000,
        "the sweep must actually exercise the property"
    );
}
