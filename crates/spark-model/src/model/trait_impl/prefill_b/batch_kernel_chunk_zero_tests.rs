// SPDX-License-Identifier: AGPL-3.0-only

//! Chunk-ZERO batched-prefill admission tests (#927).
//!
//! Sibling of `batch_kernel_tests.rs`, which was at the 500-LoC cap. Same
//! predicate under test (`check_kernel_batched_eligible`); these cases are the
//! ones the H100 round-11 receipt named — a co-arriving burst of FRESH prompts,
//! which is the wave that fixes the serialised prefill and the wave the layer
//! used to refuse after admission had accepted it.

use super::batch_kernel::check_kernel_batched_eligible;

/// (chunk_len, eff_len, chunk_start, is_last_chunk) — `eff == chunk_len`, the
/// conservative charge used when no prefix hit is proven.
fn s(chunk_len: usize, chunk_start: usize, is_last: bool) -> (usize, usize, usize, bool) {
    (chunk_len, chunk_len, chunk_start, is_last)
}

/// Qwen3.8-27B: 8 experts per token, no MRoPE.
const TOP_K: usize = 8;
const MROPE: bool = false;

#[test]
fn admits_the_h100_c16_chunk_zero_wave() {
    // The #927 receipt shape, at the admission boundary: six fresh 1193-token
    // prompts (chunk_start 0, is_last) against the 8192-token prefill budget.
    // Admission has always said YES to this under VARLEN; what failed was the
    // attention layer saying NO a moment later, after Phase A had allocated KV
    // for all six. The pair is now one predicate
    // (`ops::prefill_batched_chunk_zero_allowed`), pinned by
    // `layers::ops::chunk_zero_ssot_tests`; this test pins the admission half.
    let streams: Vec<_> = (0..6).map(|_| s(1193, 0, true)).collect();
    let arena: usize = 8_196;
    let scratch =
        spark_runtime::buffers::q12_batched_scratch_bytes_varlen(6, 6 * 1193, 1193, TOP_K, MROPE);
    assert!(check_kernel_batched_eligible(
        streams.clone(),
        6,
        arena,
        false,
        128,
        scratch,
        TOP_K,
        MROPE,
        true, // allow_chunk_zero — the resolved chunk-zero predicate
        true, // varlen
    ));
    // And the tail wave the pre-split produces: six 25-token tails sharing
    // chunk_start == 1168. One forward, not six.
    let tails: Vec<_> = (0..6).map(|_| s(25, 1168, true)).collect();
    assert!(check_kernel_batched_eligible(
        tails, 6, arena, false, 128, scratch, TOP_K, MROPE, true, true,
    ));
    // Control: with the chunk-zero predicate OFF the same wave is refused —
    // and refused HERE, before any stream is touched, which is the whole point
    // of the admission check owning the decision.
    assert!(!check_kernel_batched_eligible(
        streams, 6, arena, false, 128, scratch, TOP_K, MROPE, false, true,
    ));
}

#[test]
fn varlen_admits_ragged_chunk_zero_lengths() {
    // The equal-`chunk_len` requirement is exactly what VARLEN lifts; a burst
    // of real prompts is never uniform. `chunk_start` and `is_last` must still
    // match — those change the model-side dispatch, not just the geometry.
    let arena: usize = 8_196;
    let scratch =
        spark_runtime::buffers::q12_batched_scratch_bytes_varlen(4, 4_000, 1_400, TOP_K, MROPE);
    let ragged = [
        s(1193, 0, true),
        s(900, 0, true),
        s(1400, 0, true),
        s(507, 0, true),
    ];
    assert!(check_kernel_batched_eligible(
        ragged, 4, arena, false, 128, scratch, TOP_K, MROPE, true, true,
    ));
    assert!(
        !check_kernel_batched_eligible(
            ragged, 4, arena, false, 128, scratch, TOP_K, MROPE, true, false,
        ),
        "without VARLEN the ragged lengths must still be refused",
    );
    // A mixed chunk_start is refused under VARLEN too — a head and a tail
    // cannot share one forward, which is why the scheduler's wave planner
    // groups on (chunk_start, is_last).
    let mixed = [s(1168, 0, false), s(25, 1168, true)];
    assert!(!check_kernel_batched_eligible(
        mixed, 2, arena, false, 128, scratch, TOP_K, MROPE, true, true,
    ));
}
