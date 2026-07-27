// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure-data eligibility predicate in
//! `batch_kernel.rs`. Kept in a sibling file to keep `batch_kernel.rs`
//! itself under the 500-LoC file-size-cap.

use super::batch_kernel::check_kernel_batched_eligible;

/// (chunk_len, chunk_start, is_last_chunk)
fn s(chunk_len: usize, chunk_start: usize, is_last: bool) -> (usize, usize, usize, bool) {
    // eff == chunk_len: the conservative charge used when no prefix hit is
    // proven, i.e. exactly the pre-`ATLAS_Q12_EFFECTIVE_ARENA` behaviour.
    (chunk_len, chunk_len, chunk_start, is_last)
}

/// Stream whose cached prefix means only `eff` of its `chunk_len` gets staged.
fn s_eff(
    chunk_len: usize,
    eff: usize,
    chunk_start: usize,
    is_last: bool,
) -> (usize, usize, usize, bool) {
    (chunk_len, eff, chunk_start, is_last)
}

// Scratch capacity large enough that the #110 footprint check never trips for
// the structural-eligibility tests below (those assert the chunk_len/start/
// is_last/arena/model gates, not the scratch fit). 8 MiB ≫ any footprint here.
const BIG_SCRATCH: usize = 8 * 1024 * 1024;
const TOP_K: usize = 8;
const MROPE: bool = false;

#[test]
fn rejects_under_two_streams() {
    assert!(!check_kernel_batched_eligible(
        std::iter::empty(),
        0,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false)],
        1,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_chunk_zero() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn accepts_chunk_zero_when_explicitly_allowed() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 0, false), s(4096, 0, false)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false, // varlen
    ));
}

#[test]
fn accepts_uniform_paged_n_2() {
    assert!(check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mismatched_chunk_len() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(2048, 4096, false)],
        2,
        16384,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mismatched_chunk_start() {
    // Scheduler stream-desync case observed 2026-05-11:
    // stream 0 at chunk_start=12288, stream 1 at chunk_start=4096.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 12288, false), s(4096, 4096, false)],
        2,
        16384,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mismatched_is_last() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, true)],
        2,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_arena_overflow() {
    // N=2 × 4096 = 8192 > 4100 arena → reject.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        4100,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_mla_model() {
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "mistral",
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_large_head_dim() {
    // Gemma-4 long-attention head_dim=512 → reject.
    assert!(!check_kernel_batched_eligible(
        vec![s(4096, 4096, false), s(4096, 4096, false)],
        2,
        8192,
        "gemma4",
        512,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn accepts_n_4_uniform() {
    assert!(check_kernel_batched_eligible(
        vec![s(2048, 2048, false); 4],
        4,
        8192,
        "qwen3_next",
        256,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn rejects_scratch_footprint_overflow() {
    // #110 regression lock: the staging footprint must fit in scratch even
    // when the token-arena check passes. The deterministic crash repro was
    // n=4, chunk_len=935, top_k=8, MRoPE → 374_352 B footprint vs a 348_840 B
    // scratch. With that exact (too-small) scratch the batch is INELIGIBLE
    // (routes to per-stream from clean state, no mid-Phase-A bail), but with
    // the #110 enlarged scratch sizing it becomes eligible again.
    let streams = [s(935, 4096, false); 4];
    let arena = 4096; // 4×935 = 3740 ≤ 4096 → arena check passes
    let too_small = 348_840;
    let enlarged = spark_runtime::buffers::q12_batched_scratch_bytes(4, 935, 8, true);
    assert!(
        !check_kernel_batched_eligible(
            streams.iter().copied(),
            4,
            arena,
            "qwen3_next",
            256,
            too_small,
            8,
            true,
            false,
            false, // varlen
        ),
        "footprint must NOT fit in the old 348_840 B scratch"
    );
    assert!(
        check_kernel_batched_eligible(
            streams.iter().copied(),
            4,
            arena,
            "qwen3_next",
            256,
            enlarged,
            8,
            true,
            false,
            false, // varlen
        ),
        "footprint must fit once scratch is sized to it"
    );
}

/// Two 8192-token chunks cannot stack in an 8200-token arena when each is
/// charged its raw length — this is the arithmetic that made concurrent prefill
/// impossible on the production config (chunk 8192, arena = 8192 + max_batch_size).
#[test]
fn raw_charge_blocks_stacking_at_production_sizes() {
    assert!(!check_kernel_batched_eligible(
        vec![s(8192, 16, false), s(8192, 16, false)],
        2,
        8200,
        "qwen3_next",
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// Same two streams, but warm: a prefix hit leaves ~400 uncached tokens each, so
/// the packed layout needs ~800 of the 8200 arena and the batch is eligible.
#[test]
fn effective_charge_allows_warm_stacking() {
    assert!(check_kernel_batched_eligible(
        vec![s_eff(8192, 424, 16, false), s_eff(8192, 400, 16, false)],
        2,
        8200,
        "qwen3_next",
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// The effective charge is still a real bound: enough warm streams to exceed the
/// arena in aggregate are rejected.
#[test]
fn effective_charge_still_rejects_when_sum_exceeds_arena() {
    let streams: Vec<_> = (0..8).map(|_| s_eff(8192, 2000, 16, false)).collect();
    assert!(!check_kernel_batched_eligible(
        streams,
        8,
        8200,
        "qwen3_next",
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// A fully-cached MIDDLE chunk stages zero tokens. It must never be admitted to
/// the batch: a zero-length stream is degenerate in the packed cu_seqlens layout
/// (empty segment, and `running_proc_off += 0` leaves it sharing an offset with
/// the next stream). Observed as a hard server hang — the batched dispatch
/// logged `n=4` and never returned.
#[test]
fn effective_charge_rejects_zero_length_stream() {
    assert!(!check_kernel_batched_eligible(
        vec![s_eff(2048, 176, 2048, false), s_eff(2048, 0, 2048, false)],
        2,
        8192,
        "qwen3_next",
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}
