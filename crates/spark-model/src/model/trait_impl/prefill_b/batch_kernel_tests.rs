// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure-data eligibility predicate in
//! `batch_kernel.rs`. Kept in a sibling file to keep `batch_kernel.rs`
//! itself under the 500-LoC file-size-cap.

use spark_runtime::prefix_cache::PrefixMatch;

use super::batch_kernel::{
    batched_reserve_hybrid_ssm_ok, cache_batch_matches_compatible, check_kernel_batched_eligible,
    config_is_mla,
};

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

fn cache_match(tokens: usize) -> PrefixMatch {
    PrefixMatch {
        matched_blocks: vec![7; tokens / 16],
        matched_disk_block_ids: Vec::new(),
        matched_tokens: tokens,
        ssm_snapshot: None,
        ssm_snapshot_tokens: 0,
        ssm_snapshot_tier_key: None,
        ssm_snapshot_tier_tokens: 0,
        ssm_snapshot_is_tail: false,
    }
}

#[test]
fn hybrid_ssm_admits_only_all_cold_reservations() {
    // The 2026-08-16 stackval blocker: a blanket num_ssm_layers!=0 veto
    // rejected COLD chunk-0 waves on the hybrid 27B, serializing the whole
    // prefill ramp. Cold (matched_tokens == 0 everywhere) must be admitted —
    // it is state-identical to the cache-inactive case.
    assert!(batched_reserve_hybrid_ssm_ok(
        &[cache_match(0), cache_match(0), cache_match(0)],
        true,
    ));
    // A warm match on a hybrid model keeps the per-stream path (KV/Marconi
    // skip interplay with recurrent state is not admitted transactionally).
    assert!(!batched_reserve_hybrid_ssm_ok(
        &[cache_match(0), cache_match(48)],
        true,
    ));
    // Attention-only models are unaffected either way.
    assert!(batched_reserve_hybrid_ssm_ok(
        &[cache_match(48), cache_match(48)],
        false,
    ));
    // No matches (cache active, empty batch guard upstream) is trivially ok.
    assert!(batched_reserve_hybrid_ssm_ok(&[], true));
}

#[test]
fn cache_batch_accepts_equal_partial_hits() {
    assert!(cache_batch_matches_compatible(
        &[cache_match(48), cache_match(48)],
        8192,
    ));
    assert!(!cache_batch_matches_compatible(&[], 8192));
}

#[test]
fn cache_batch_rejects_mixed_processing_geometry() {
    assert!(!cache_batch_matches_compatible(
        &[cache_match(0), cache_match(48)],
        8192,
    ));
    let mut fewer_blocks = cache_match(48);
    fewer_blocks.matched_blocks.pop();
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), fewer_blocks],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_restore_metadata() {
    let mut snapshot = cache_match(48);
    snapshot.ssm_snapshot = Some(3);
    snapshot.ssm_snapshot_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), snapshot],
        8192,
    ));

    let mut disk = cache_match(48);
    disk.matched_disk_block_ids = vec![9; 3];
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), disk],
        8192,
    ));

    let mut snapshot_tokens = cache_match(48);
    snapshot_tokens.ssm_snapshot_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), snapshot_tokens],
        8192,
    ));

    let mut tier_key = cache_match(48);
    tier_key.ssm_snapshot_tier_key = Some(7);
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), tier_key],
        8192,
    ));

    let mut tier_tokens = cache_match(48);
    tier_tokens.ssm_snapshot_tier_tokens = 48;
    assert!(!cache_batch_matches_compatible(
        &[cache_match(48), tier_tokens],
        8192,
    ));
}

#[test]
fn cache_batch_rejects_full_chunk_hit() {
    assert!(!cache_batch_matches_compatible(
        &[cache_match(8192), cache_match(8192)],
        8192,
    ));
}

#[test]
fn rejects_under_two_streams() {
    assert!(!check_kernel_batched_eligible(
        std::iter::empty(),
        0,
        8192,
        false,
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
        false,
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
        false,
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
        false,
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
        false,
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
        false,
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
        false,
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
        false,
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
        false,
        256,
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
        false,
        512,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        false,
        false, // varlen
    ));
}

#[test]
fn accepts_varlen_batch_when_packed_footprint_fits() {
    // Regression: the old preflight charged all four requests at 4,782 tokens
    // (19,128 tokens) instead of their packed cu_seqlens total (13,649). The
    // standard 16,388-token arena is deliberately provisioned for this
    // workload, so the oversized estimate silently serialized realistic
    // agentic/RAG traffic.
    let streams = [
        s(2051, 0, true),
        s(2953, 0, true),
        s(3863, 0, true),
        s(4782, 0, true),
    ];
    let arena: usize = 16_388;
    let scratch = spark_runtime::buffers::q12_batched_scratch_bytes(
        spark_runtime::buffers::Q12_SIZING_STREAMS,
        arena.div_ceil(spark_runtime::buffers::Q12_SIZING_STREAMS),
        TOP_K,
        MROPE,
    );
    assert!(check_kernel_batched_eligible(
        streams, 4, arena, false, 128, scratch, TOP_K, MROPE, true, true,
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
            false,
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
            false,
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
        false,
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
        false,
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
        false,
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
        false,
        128,
        BIG_SCRATCH,
        TOP_K,
        MROPE,
        true,
        false,
    ));
}

/// Config-level pin of the MLA rejection: an MLA config (mistral-shaped,
/// `kv_lora_rank = 512`) must be rejected by the batched-kernel gate THROUGH
/// the same `config_is_mla` seam the production caller reads. Review finding
/// on the capability conversion: sabotaging the caller's derivation
/// (`kv_lora_rank > 0` → `false`) left all unit tests green because every
/// test passed the bool directly. This test fails under that sabotage.
#[test]
fn mistral_config_is_rejected_as_mla() {
    let mut cfg = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    // Non-MLA baseline: the derivation says no, and an otherwise-eligible
    // batch is admitted — proving the rejection below comes from MLA alone.
    assert!(!config_is_mla(&cfg));
    let eligible = |is_mla: bool| {
        check_kernel_batched_eligible(
            vec![s(2048, 16, false), s(2048, 16, false)],
            2,
            8192,
            is_mla,
            128,
            BIG_SCRATCH,
            TOP_K,
            MROPE,
            true,
            false,
        )
    };
    assert!(eligible(config_is_mla(&cfg)));

    // Mistral-Small-4 ships kv_lora_rank = 512 in config.json; the parser
    // copies it verbatim (`parsers/mistral.rs`), so this is the config-level
    // fact the serving path sees.
    cfg.kv_lora_rank = 512;
    assert!(config_is_mla(&cfg));
    assert!(!eligible(config_is_mla(&cfg)));
}
