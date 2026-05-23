// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for Phase-C rollback + re-steer (`rollback.rs`).
//!
//! `find_last_boundary` and `rewind_buffers` are pure over plain
//! `Vec`s / scalars, so they are tested directly without constructing
//! an `ActiveSeq` (which carries channels, `Instant`s and a
//! `SequenceState`). The `ActiveSeq`-level glue in `rollback_to_boundary`
//! / `apply_rollback` is a thin composition of these two — exercising
//! both pure cores plus the cap constant covers the behavior.

use super::super::ssm_decode_ring::SsmDecodeRing;
use super::{
    RollbackFallback, RollbackOutcome, find_last_boundary, find_last_boundary_with_snapshot,
    rewind_buffers,
};

// ── boundary detection ──────────────────────────────────────────────

/// `mask` where token id `id` is a boundary iff `id` is in `boundary_ids`.
fn mask_of(boundary_ids: &[u32], vocab: usize) -> Vec<bool> {
    let mut m = vec![false; vocab];
    for &b in boundary_ids {
        m[b as usize] = true;
    }
    m
}

#[test]
fn finds_last_boundary_skipping_min_keep() {
    // tokens: [10, NL, 11, 12, NL, 13, 14, 15]  (NL = boundary id 99)
    // indices:  0    1   2   3    4   5   6   7
    let tokens = [10, 99, 11, 12, 99, 13, 14, 15];
    let mask = mask_of(&[99], 100);
    // min_keep = 2 → search region is indices 0..=5. Last boundary there
    // is index 4.
    assert_eq!(find_last_boundary(&tokens, &mask, 2), Some(4));
}

#[test]
fn boundary_inside_min_keep_window_is_ignored() {
    // The only boundary (id 99) is at index 6; with min_keep=3 the
    // search region is 0..=4, so it must not be found.
    let tokens = [10, 11, 12, 13, 14, 15, 99, 16];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 3), None);
}

#[test]
fn no_boundary_in_buffer_returns_none() {
    let tokens: Vec<u32> = (0..40).collect();
    let mask = mask_of(&[200, 201], 256); // boundary ids absent from buffer
    assert_eq!(find_last_boundary(&tokens, &mask, 4), None);
}

#[test]
fn buffer_shorter_than_min_keep_returns_none() {
    let tokens = [99, 99, 99];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 5), None);
}

#[test]
fn picks_latest_of_several_boundaries() {
    // boundaries at 2, 5, 8; min_keep=1 → region 0..=6 → latest is 5.
    let tokens = [0, 1, 99, 3, 4, 99, 6, 7, 99];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 1), Some(5));
}

// ── SSM-snapshot-aware boundary detection (hybrid models) ───────────
//
// A boundary token at index `i` is eligible only when the decode ring
// holds a snapshot taken at `token_position == i + 1` (the `keep_len`
// a rollback to that boundary would request). These tests drive the
// real `find_last_boundary_with_snapshot` against a populated ring.

#[test]
fn snapshot_aware_search_picks_latest_boundary_with_a_snapshot() {
    // boundaries (id 99) at indices 2, 5, 8; min_keep=1 → region 0..=8.
    let tokens = [0, 1, 99, 3, 4, 99, 6, 7, 99, 9];
    let mask = mask_of(&[99], 100);
    // Plain search ignores snapshots → latest boundary in 0..=8 is 8.
    assert_eq!(find_last_boundary(&tokens, &mask, 1), Some(8));
    // Ring has snapshots only at keep_len 3 (idx 2) and 6 (idx 5).
    let mut ring = SsmDecodeRing::new(3);
    ring.record(3); // boundary idx 2 → keep_len 3
    ring.record(6); // boundary idx 5 → keep_len 6
    // idx 8 has no snapshot (keep_len 9) → fall back to idx 5.
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        Some(5),
    );
}

#[test]
fn snapshot_aware_search_declines_when_no_boundary_has_a_snapshot() {
    // boundaries at indices 2 and 5; ring snapshots land between them.
    let tokens = [0, 1, 99, 3, 4, 99, 6, 7];
    let mask = mask_of(&[99], 100);
    let mut ring = SsmDecodeRing::new(3);
    ring.record(4); // keep_len 4 — no boundary at idx 3
    ring.record(7); // keep_len 7 — no boundary at idx 6
    // A plain boundary still exists, but none coincides with a snapshot.
    assert!(find_last_boundary(&tokens, &mask, 1).is_some());
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        None,
    );
}

#[test]
fn snapshot_aware_search_respects_min_keep_window() {
    // The only snapshotted boundary (id 99) is at index 6. With n=10 and
    // min_keep=4 the search region is 0..=(n-1-min_keep)=0..=5, so the
    // boundary is inside the discarded window and must NOT be selected —
    // even though the ring holds its snapshot.
    let tokens = [0, 1, 2, 3, 4, 5, 99, 7, 8, 9];
    let mask = mask_of(&[99], 100);
    let mut ring = SsmDecodeRing::new(3);
    ring.record(7); // boundary idx 6 → keep_len 7
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 4, &ring),
        None,
    );
    // With min_keep=3 the search region extends to idx 6, so the same
    // boundary IS now eligible (its snapshot is present).
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 3, &ring),
        Some(6),
    );
}

#[test]
fn snapshot_aware_search_empty_ring_declines() {
    let tokens = [0, 99, 2, 3, 4, 5];
    let mask = mask_of(&[99], 100);
    let ring = SsmDecodeRing::new(3); // enabled but empty
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        None,
    );
}

#[test]
fn snapshot_aware_search_after_eviction_only_sees_live_snapshots() {
    // boundaries at indices 1, 4, 7, 10.
    let tokens = [0, 99, 2, 3, 99, 5, 6, 99, 8, 9, 99, 11];
    let mask = mask_of(&[99], 100);
    // Capacity-2 ring: recording 4 boundary snapshots evicts the oldest 2.
    let mut ring = SsmDecodeRing::new(2);
    ring.record(2); // boundary idx 1  (evicted)
    ring.record(5); // boundary idx 4  (evicted)
    ring.record(8); // boundary idx 7  (live)
    ring.record(11); // boundary idx 10 (live)
    // Latest boundary with a *live* snapshot is idx 10 (keep_len 11).
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        Some(10),
    );
    // Truncating after keep_len 8 drops the idx-10 snapshot (keep_len
    // 11 > 8) while keeping the idx-7 snapshot (keep_len 8): the latest
    // reachable boundary falls back to idx 7.
    ring.truncate_after(8);
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        Some(7),
    );
}

// ── buffer rewind (the attention-KV rewind core) ────────────────────

#[test]
fn rewind_truncates_output_and_seq_and_lowers_seq_len() {
    // output_tokens is the generated suffix; seq_tokens is prompt+gen.
    let mut output = vec![50, 51, 52, 53, 54, 55]; // 6 generated
    let mut seq = vec![1, 2, 3, 50, 51, 52, 53, 54, 55]; // 3 prompt + 6 gen
    let seq_len = 9;
    // Keep the first 4 generated tokens → drop 2.
    let new_len = rewind_buffers(&mut output, &mut seq, seq_len, 4);
    assert_eq!(output, vec![50, 51, 52, 53]);
    assert_eq!(seq, vec![1, 2, 3, 50, 51, 52, 53]);
    assert_eq!(new_len, 7, "seq_len must drop by the 2 rewound tokens");
}

#[test]
fn rewind_keeping_all_is_a_noop() {
    let mut output = vec![50, 51, 52];
    let mut seq = vec![1, 50, 51, 52];
    let new_len = rewind_buffers(&mut output, &mut seq, 4, 3);
    assert_eq!(output, vec![50, 51, 52]);
    assert_eq!(seq, vec![1, 50, 51, 52]);
    assert_eq!(new_len, 4);
}

#[test]
fn rewind_seq_len_saturates_at_zero() {
    // Pathological: seq_len smaller than the drop count must not wrap.
    let mut output = vec![1, 2, 3, 4, 5];
    let mut seq = vec![1, 2, 3, 4, 5];
    let new_len = rewind_buffers(&mut output, &mut seq, 2, 1);
    assert_eq!(output, vec![1]);
    assert_eq!(new_len, 0, "saturating, never underflow");
}

// ── rollback cap ────────────────────────────────────────────────────

#[test]
fn rollback_cap_is_two() {
    // The per-sequence cap that flips a watchdog back to a hard stop.
    assert_eq!(atlas_kernels::ROLLBACK_RESTEER_CAP, 2);
}

#[test]
fn fallback_variants_are_distinct() {
    // Sanity: the three decline reasons are not accidentally merged.
    assert_ne!(
        RollbackOutcome::Fallback(RollbackFallback::Disabled),
        RollbackOutcome::Fallback(RollbackFallback::CapReached),
    );
    assert_ne!(
        RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
        RollbackOutcome::Fallback(RollbackFallback::CapReached),
    );
    assert_eq!(
        RollbackOutcome::RolledBack { dropped: 7 },
        RollbackOutcome::RolledBack { dropped: 7 },
    );
}

// ── ROM scaffold ────────────────────────────────────────────────────

struct StubRomHead;
impl super::RomHead for StubRomHead {
    fn repetition_onset_score(&self, recent: &[u32]) -> f32 {
        // Trivial stand-in: longer tails score higher. NOT a real ROM
        // detector — just proves the trait seam compiles + is callable.
        (recent.len() as f32 / 100.0).min(1.0)
    }
}

#[test]
fn rom_head_trait_seam_is_callable() {
    let head: std::sync::Arc<dyn super::RomHead> = std::sync::Arc::new(StubRomHead);
    assert!((head.repetition_onset_score(&[1, 2, 3]) - 0.03).abs() < 1e-6);
    assert_eq!(head.repetition_onset_score(&vec![0u32; 500]), 1.0);
}

#[test]
fn rom_head_absent_by_default() {
    // Without `set_rom_head`, the accessor must return None so callers
    // fall back to the F2 confidence heuristic. (Process-global OnceLock;
    // this test does not install a head.)
    assert!(super::rom_head().is_none());
}
