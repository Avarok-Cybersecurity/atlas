// SPDX-License-Identifier: AGPL-3.0-only

//! Issue #31 sliding-window offload-cursor helpers (pure functions).

use anyhow::{Result, bail};

/// Issue #31: validate that the block being evicted at logical position
/// `evict_pos` has been offloaded by every attention layer. The slide
/// is safe iff `disk_last_offloaded[L] > evict_pos` for all L (strictly
/// greater because `disk_last_offloaded[L]` is the count of offloaded
/// blocks, and a block at position N is "offloaded" iff the count is at
/// least N+1, i.e., > N).
///
/// Returns `Err` describing the first lagging layer if any layer hasn't
/// caught up, else `Ok(())`. Pure function — no side effects.
pub(super) fn check_safe_to_evict(
    disk_last_offloaded_per_layer: &[u32],
    evict_pos: usize,
) -> Result<()> {
    for (layer_idx, &cursor) in disk_last_offloaded_per_layer.iter().enumerate() {
        if (cursor as usize) <= evict_pos {
            bail!(
                "high-speed-swap: attempting to evict block at logical position {} \
                 from HBM, but attention layer {} only offloaded up to position {}. \
                 Eviction would lose K/V data. Per-layer cursors: {:?}",
                evict_pos,
                layer_idx,
                cursor,
                disk_last_offloaded_per_layer,
            );
        }
    }
    Ok(())
}

/// Issue #31: after a successful slide advances `window_start` to
/// `new_window_start`, advance every attention layer's offload cursor
/// to keep pace. Layers whose cursor was already ≥ `new_window_start`
/// (e.g. they offloaded more recently in this chunk) are left alone.
/// Pure mutation on a `&mut [u32]`.
pub(super) fn advance_layer_cursors_after_slide(
    disk_last_offloaded_per_layer: &mut [u32],
    new_window_start: usize,
) {
    let new_ws = new_window_start as u32;
    for cursor in disk_last_offloaded_per_layer.iter_mut() {
        if *cursor < new_ws {
            *cursor = new_ws;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Issue #31 — `check_safe_to_evict` enforces the slide invariant.

    #[test]
    fn safe_to_evict_when_all_layers_caught_up() {
        // Every layer has offloaded position 5 (cursor=6 means 0..6 are on disk),
        // evicting position 5 is safe because cursor > 5 for every layer.
        let cursors = vec![6, 6, 6];
        assert!(check_safe_to_evict(&cursors, 5).is_ok());
    }

    #[test]
    fn unsafe_to_evict_when_a_layer_lags() {
        // Layer 1 has only offloaded up to position 4 (cursor=5 = positions 0..5
        // means position 4 is the last offloaded; cursor=5 means the LIMIT is 5,
        // i.e. cursor > 5 means position 5 is offloaded). Strict-greater comparison.
        let cursors = vec![10, 5, 10];
        let err = check_safe_to_evict(&cursors, 5).unwrap_err().to_string();
        assert!(err.contains("attention layer 1"), "got: {err}");
        assert!(err.contains("position 5"), "got: {err}");
    }

    #[test]
    fn unsafe_to_evict_when_a_layer_never_offloaded() {
        // Cursor=0 means the layer has offloaded NOTHING. Evicting any
        // position fails the check.
        let cursors = vec![10, 10, 0];
        let err = check_safe_to_evict(&cursors, 0).unwrap_err().to_string();
        assert!(err.contains("attention layer 2"), "got: {err}");
    }

    #[test]
    fn safe_to_evict_with_empty_cursor_vec_is_vacuously_true() {
        // A sequence whose `disk_last_offloaded_per_layer` hasn't been
        // populated yet (e.g. fresh sequence with no attn layers run)
        // can't have un-offloaded blocks because no layer has run. The
        // production `meta.rs:180` initializes this vec to `vec![0; n_attn]`
        // so this case shouldn't fire in real workloads, but the helper
        // should be vacuously correct.
        let cursors: Vec<u32> = vec![];
        assert!(check_safe_to_evict(&cursors, 100).is_ok());
    }

    // Issue #31 — `advance_layer_cursors_after_slide` keeps cursors ≥ window_start.

    #[test]
    fn advance_after_slide_promotes_lagging_cursors() {
        let mut cursors = vec![10, 5, 8];
        advance_layer_cursors_after_slide(&mut cursors, 9);
        // Layer 0 was already at 10 ≥ 9, unchanged. Layer 1 was at 5 < 9, bumped.
        // Layer 2 was at 8 < 9, bumped.
        assert_eq!(cursors, vec![10, 9, 9]);
    }

    #[test]
    fn advance_after_slide_never_moves_cursor_backward() {
        let mut cursors = vec![100, 100, 100];
        advance_layer_cursors_after_slide(&mut cursors, 50);
        // All cursors ≥ 50 → no change.
        assert_eq!(cursors, vec![100, 100, 100]);
    }

    #[test]
    fn advance_after_slide_idempotent() {
        let mut cursors = vec![5, 5, 5];
        advance_layer_cursors_after_slide(&mut cursors, 10);
        advance_layer_cursors_after_slide(&mut cursors, 10);
        assert_eq!(cursors, vec![10, 10, 10]);
    }

    // Round-trip: a slide loop pattern — for each slide, check then advance.
    // Models the cap=4 / chunk crossing case described in issue #31.

    #[test]
    fn slide_loop_round_trip_chunk_transition() {
        // After chunk N, all 3 attn layers have offloaded blocks 0..64.
        let mut cursors = vec![64u32, 64, 64];

        // Chunk N+1's bulk alloc loop: simulate 64 slides + 64 allocs with
        // window_start advancing one step per slide. cap = 64.
        let cap = 64;
        for slide_idx in 0..cap {
            let ws_before = slide_idx; // prior to this slide, ws = slide_idx
            // Safety check: every cursor > ws_before? Initial cursors are 64.
            // All slides up to slide_idx=63 have cursors > slide_idx → safe.
            assert!(
                check_safe_to_evict(&cursors, ws_before).is_ok(),
                "slide {slide_idx} should be safe with cursors {cursors:?}"
            );
            // Advance after the slide.
            advance_layer_cursors_after_slide(&mut cursors, ws_before + 1);
        }

        // After 64 slides (ws now 64), cursors should still be [64; 3] because
        // none of the advances moved them past 64 (each step advanced ws by 1
        // up to 64, and cursors started at 64 ≥ each new_ws).
        assert_eq!(cursors, vec![64, 64, 64]);
    }
}
