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

/// Per-stream chunk geometry for ONE dispatch tick: how many tokens the
/// stream advances and whether that chunk is its last.
///
/// SSOT for `run_batched_prefill_step`'s per-stream loop, factored out so the
/// geometry a stream gets in a batched wave can be compared — in a test, with
/// no model and no GPU — against the sequence the same stream gets on the
/// per-stream path. #1002 asked that question of the round-13 shapes and the
/// answer has to stay pinned: BF16 accumulation is not associative, so a
/// stream that takes a different chunk shape depending on who it batched with
/// is a different answer to the same request at temperature 0.
///
/// * `effective_max` — `max_prefill_tokens`, or `remaining` for MLA models
///   (no paged-MLA prefill kernel ⇒ multi-chunk prefill is forced off).
/// * `tail_cut` — `Model::prefill_tail_cut`, the offset where
///   `prefill_chunk_dispatch` would split this prompt's FINAL chunk to land an
///   SSM tail checkpoint. Under VARLEN the scheduler pre-splits there so the
///   tails of a co-arriving burst share a `chunk_start` and batch.
///
/// NOTE the budget NEVER truncates a chunk-0 below the cut: `chunk_len` starts
/// at `min(remaining, effective_max)` and the wave cap is applied by the
/// PLANNER, which opens a new wave rather than shrinking a member (a stream
/// whose own chunk exceeds the cap gets a singleton wave). Pinned by
/// `chunk_zero_is_never_truncated_by_the_wave_budget`.
pub(super) fn plan_stream_chunk(
    chunk_offset: usize,
    total: usize,
    effective_max: usize,
    tail_cut: Option<usize>,
    varlen: bool,
) -> (usize, bool) {
    let remaining = total - chunk_offset;
    let mut chunk_len = remaining.min(effective_max);
    let mut is_last = chunk_offset + chunk_len >= total;
    // TAIL PRE-SPLIT (VARLEN only) — the condition mirrors
    // `prefill_chunk_dispatch`'s own (`cut > chunk_start && cut < total`, on a
    // last chunk) exactly, plus the requirement that the cut lie inside THIS
    // chunk.
    if varlen
        && is_last
        && let Some(cut) = tail_cut
        && cut > chunk_offset
        && cut < total
        && cut <= chunk_offset + chunk_len
    {
        chunk_len = cut - chunk_offset;
        is_last = false;
    }
    // Align intermediate chunks to the GDN WY4 boundary (4 tokens). The tail
    // cut is a multiple of the KV block size, itself a multiple of 4 on every
    // shipped config, so this is a no-op there.
    if !is_last && chunk_len >= 4 {
        chunk_len = (chunk_len / 4) * 4;
    }
    (chunk_len, is_last)
}

/// Does deferring these chunk-0s into `prefilling` so they can BATCH actually
/// buy anything? True only when the two smallest heads fit one wave together.
///
/// #1002 / round 13, long shape. Sixteen 4593-token prompts pre-split to
/// `4576 + 17`, and `2 x 4576 = 9152 > 8192`, so the planner emitted
/// `16 streams -> 14 wave(s), M per wave [51, 4576 x13]`. Fourteen waves for
/// sixteen streams is not batching — but the deferral was paid anyway, and it
/// is not free: waves run back-to-back inside ONE tick, so no stream is
/// promoted until every wave has run, and every TTFT collapses onto the p99.
/// Measured: long-shape C=16 TTFT 7 524.7 -> 13 756.1 ms (+82.8%) and
/// aggregate 308.86 -> 214.25 tok/s (-30.6%) against the same-binary control,
/// for zero batching. When this returns false the request keeps the inline
/// chunk-0 of `phase_start_prefills` and the staggered TTFT that comes with it.
///
/// Two heads, not `n`: the planner is first-fit, so a single pair sharing a
/// wave is the smallest win that justifies the deferral, and the two smallest
/// heads are the pair most likely to fit.
pub(in crate::scheduler) fn varlen_defer_pays<I>(heads: I, wave_token_cap: usize) -> bool
where
    I: IntoIterator<Item = usize>,
{
    let mut smallest = usize::MAX;
    let mut second = usize::MAX;
    for h in heads {
        if h < smallest {
            second = smallest;
            smallest = h;
        } else if h < second {
            second = h;
        }
    }
    if second == usize::MAX {
        // Fewer than two candidates — nothing to batch with.
        return false;
    }
    smallest.saturating_add(second) <= wave_token_cap
}

#[cfg(test)]
#[path = "prefill_waves_tests.rs"]
mod tests;
