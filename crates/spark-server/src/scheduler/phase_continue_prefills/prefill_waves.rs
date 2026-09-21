// SPDX-License-Identifier: AGPL-3.0-only

//! Pure wave planning for the batched-prefill step.
//!
//! `plan_prefill_waves` partitions the prefilling streams of one scheduler
//! tick into dispatch waves for `model.prefill_batch_chunk`. Under VARLEN
//! batched prefill (`--prefill-varlen-batch`) each wave concatenates ragged
//! prompt chunks into ONE forward — per-layer GEMMs launch once at
//! M = Σ tokens — so the wave must satisfy the model-side admission contract
//! (`check_kernel_batched_eligible`):
//!
//!   - every member shares `chunk_start` and `is_last_chunk` (VARLEN lifts
//!     only the equal-`chunk_len` requirement), and
//!   - Σ chunk_len stays within the wave token cap (the caller passes
//!     `min(--max-prefill-tokens, hidden-buffer arena)`), so one scheduler
//!     tick's forward never exceeds the prefill budget.
//!
//! Streams that do not fit the current wave open the next one — waves run
//! back-to-back within the tick, so every stream still advances exactly one
//! chunk per tick, matching the pre-wave behaviour.
//!
//! Flag OFF (`varlen == false`) returns a single wave containing every
//! stream in order: byte-identical dispatch behaviour to the pre-wave
//! scheduler (one `prefill_batch_chunk` call with all streams).

/// Per-stream chunk geometry the planner partitions on. A projection of
/// `PrefillSlice` — kept as plain data so the planner is unit-testable
/// without a `SequenceState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct WaveGeom {
    pub chunk_start: usize,
    pub chunk_len: usize,
    pub is_last: bool,
}

/// Partition stream indices `0..geoms.len()` into dispatch waves.
///
/// First-fit greedy in FIFO order: each stream joins the earliest wave whose
/// head shares its `(chunk_start, is_last)` geometry and whose token total
/// stays within `wave_token_cap`; otherwise it opens a new wave. Index order
/// is preserved within every wave, and every stream is assigned to exactly
/// one wave (a stream whose own chunk exceeds the cap gets a wave to itself
/// — it must still advance, and a singleton wave takes the single-stream
/// dispatch path anyway).
pub(super) fn plan_prefill_waves(
    geoms: &[WaveGeom],
    varlen: bool,
    wave_token_cap: usize,
) -> Vec<Vec<usize>> {
    if geoms.is_empty() {
        return Vec::new();
    }
    if !varlen || geoms.len() == 1 {
        return vec![(0..geoms.len()).collect()];
    }
    debug_assert!(
        wave_token_cap > 0,
        "wave_token_cap must be explicit-nonzero"
    );
    // (head geometry, Σ chunk_len, member indices)
    let mut waves: Vec<(WaveGeom, usize, Vec<usize>)> = Vec::new();
    for (i, g) in geoms.iter().enumerate() {
        let placed = waves.iter_mut().find(|(head, total, _)| {
            head.chunk_start == g.chunk_start
                && head.is_last == g.is_last
                && total + g.chunk_len <= wave_token_cap
        });
        match placed {
            Some((_, total, members)) => {
                *total += g.chunk_len;
                members.push(i);
            }
            None => waves.push((*g, g.chunk_len, vec![i])),
        }
    }
    waves.into_iter().map(|(_, _, members)| members).collect()
}

#[cfg(test)]
mod tests {
    use super::{WaveGeom, plan_prefill_waves};

    fn g(chunk_start: usize, chunk_len: usize, is_last: bool) -> WaveGeom {
        WaveGeom {
            chunk_start,
            chunk_len,
            is_last,
        }
    }

    #[test]
    fn flag_off_is_one_wave_with_every_stream_in_order() {
        // Byte-identical dispatch behaviour: the pre-wave scheduler made ONE
        // prefill_batch_chunk call with all streams, whatever their geometry
        // or total token count.
        let geoms = [g(0, 200, true), g(2048, 512, false), g(0, 4096, true)];
        assert_eq!(plan_prefill_waves(&geoms, false, 2048), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn empty_streams_no_waves() {
        assert!(plan_prefill_waves(&[], true, 2048).is_empty());
        assert!(plan_prefill_waves(&[], false, 2048).is_empty());
    }

    #[test]
    fn ragged_chunk0_wave_packs_up_to_the_budget() {
        // Ten ~200-token fresh prompts against the 2048-token budget: the
        // first ten fit (Σ = 2000), the eleventh opens wave 2 — the measured
        // C=32 case (285 ms/prompt serial) becomes ⌈32/10⌉ dispatches.
        let geoms: Vec<WaveGeom> = (0..11).map(|_| g(0, 200, true)).collect();
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves.len(), 2);
        assert_eq!(waves[0], (0..10).collect::<Vec<_>>());
        assert_eq!(waves[1], vec![10]);
    }

    #[test]
    fn budget_cap_is_exact_not_off_by_one() {
        // 1024 + 1024 == cap exactly ⇒ same wave; +1 more opens a new one.
        let geoms = [g(0, 1024, true), g(0, 1024, true), g(0, 1, true)];
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves, vec![vec![0, 1], vec![2]]);
    }

    #[test]
    fn mixed_geometry_splits_into_compatible_waves() {
        // The model-side contract: chunk_start and is_last must match across
        // a batch (check_kernel_batched_eligible). A wave mixing them would
        // be rejected wholesale and every stream would fall back to serial —
        // the planner must never emit one.
        let geoms = [
            g(0, 200, true),     // fresh single-chunk
            g(0, 2048, false),   // fresh long prompt, chunk 0 of many
            g(0, 300, true),     // fresh single-chunk → wave of stream 0
            g(2048, 512, false), // mid-prefill continuation
            g(0, 250, true),     // fresh single-chunk → wave of stream 0
        ];
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves, vec![vec![0, 2, 4], vec![1], vec![3]]);
        // Cross-check the invariant directly: uniform (chunk_start, is_last)
        // per wave, Σ ≤ cap for every multi-member wave.
        for wave in &waves {
            let head = geoms[wave[0]];
            let total: usize = wave.iter().map(|&i| geoms[i].chunk_len).sum();
            assert!(wave.len() == 1 || total <= 2048);
            for &i in wave {
                assert_eq!(geoms[i].chunk_start, head.chunk_start);
                assert_eq!(geoms[i].is_last, head.is_last);
            }
        }
    }

    #[test]
    fn oversized_stream_gets_a_singleton_wave() {
        // chunk_len > cap must still dispatch (single-stream path); it must
        // not absorb siblings past the cap.
        let geoms = [g(0, 4096, true), g(0, 100, true)];
        let waves = plan_prefill_waves(&geoms, true, 2048);
        assert_eq!(waves, vec![vec![0], vec![1]]);
    }

    #[test]
    fn every_stream_is_assigned_exactly_once() {
        let geoms: Vec<WaveGeom> = (0..37)
            .map(|i| g((i % 3) * 1024, 100 + i * 7, i % 2 == 0))
            .collect();
        let waves = plan_prefill_waves(&geoms, true, 1024);
        let mut seen = vec![0usize; geoms.len()];
        for wave in &waves {
            assert!(!wave.is_empty());
            for &i in wave {
                seen[i] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each stream in exactly one wave"
        );
        // FIFO order preserved within each wave.
        for wave in &waves {
            assert!(wave.windows(2).all(|w| w[0] < w[1]));
        }
    }
}
