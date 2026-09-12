// SPDX-License-Identifier: AGPL-3.0-only

//! Exact-integer pins for the per-sequence reserve. No GPU, no checkpoint, no floats.
//!
//! Every constant here is derived from source dimensions, never from a runtime observation.

use super::*;
use crate::layers::glm5next_dsa::state::{dsa_capacity, indexer_state_bytes};

/// GLM-5.3, from the shipped config: `index_head_dim = 128`, `index_kpool = 4`,
/// `hidden_size = 4096`, `vocab_size = 154880`, 11 text DSA layers.
const HEAD_DIM: usize = 128;
const KPOOL: usize = 4;
const HIDDEN: usize = 4096;
const VOCAB: usize = 154_880;
const DSA_LAYERS: usize = 11;
const MSL: usize = 131_072;

/// `block_bytes_kv_all_layers()` for GLM-5.3's absorbed-MLA FP8 cache:
/// `2 (k + v, symmetric for fp8) * block_size 16 * num_kv_heads 1 * head_dim (kv_lora_rank) 512`
/// = 16,384 B per layer per block, over 11 DSA layers.
const KV_BLOCK_BYTES: usize = (2 * 16) * 512 * DSA_LAYERS;

#[test]
fn the_kv_block_is_180_224_bytes_exactly() {
    assert_eq!(KV_BLOCK_BYTES, 180_224);
}

#[test]
fn indexer_bytes_are_513_per_token_per_layer() {
    assert_eq!(indexer_state_bytes(1, HEAD_DIM), 4 * HEAD_DIM + 1);
    assert_eq!(indexer_state_bytes(1, HEAD_DIM), 513);
    assert_eq!(indexer_state_bytes(MSL, HEAD_DIM), 67_239_936);
}

#[test]
fn capacity_rounds_down_to_whole_pools_and_is_the_only_spelling() {
    // Exact multiples are unchanged — which is why the old drift never fired on GLM-5.3.
    assert_eq!(dsa_capacity(MSL, KPOOL), MSL);
    assert_eq!(dsa_capacity(65_536, KPOOL), 65_536);
    // A context that does not divide evenly is where the two old spellings disagreed.
    assert_eq!(dsa_capacity(4_099, KPOOL), 4_096);
    assert_eq!(dsa_capacity(3, KPOOL), 0);
    // Total even for an unvalidated config.
    assert_eq!(dsa_capacity(MSL, 0), MSL);
}

/// The reserve must equal what `Glm5NextDsaState::alloc` actually takes: it issues exactly
/// three allocations, `capacity*d*2`, `capacity*d*2`, `capacity`.
#[test]
fn the_reserve_equals_what_dsa_alloc_actually_allocates() {
    let cap = dsa_capacity(MSL, KPOOL);
    let what_alloc_takes = cap * HEAD_DIM * 2 + cap * HEAD_DIM * 2 + cap;
    assert_eq!(indexer_state_bytes(cap, HEAD_DIM), what_alloc_takes);
}

#[test]
fn target_layer_charge_at_131072_is_exact() {
    let per_layer = indexer_state_bytes(dsa_capacity(MSL, KPOOL), HEAD_DIM);
    assert_eq!(DSA_LAYERS * per_layer, 739_639_296);
}

#[test]
fn proposer_charge_at_131072_is_exact_and_is_a_separate_owner() {
    let per_layer = indexer_state_bytes(dsa_capacity(MSL, KPOOL), HEAD_DIM);
    let proposer = per_layer + 2 * HIDDEN * 2 + HIDDEN * 2 + VOCAB * 2 + 4 + 16;
    assert_eq!(proposer, 67_574_292);
    // It is NOT 1/11th of the target term, and NOT the whole 12-layer figure: the drafter owns
    // ONE indexer block plus five scalars. Folding it into the target term would charge 806.9 MB
    // of DSA against a layer sum that only owns 739.6 MB of it.
    assert_eq!(proposer - per_layer, 334_356);
    assert_eq!(12 * per_layer, 806_879_232);
    assert_ne!(
        proposer,
        12 * per_layer - DSA_LAYERS * per_layer + 334_356 + 1
    );
}

#[test]
fn total_per_sequence_and_the_batch_3_charge_are_exact() {
    let s = PerSequenceState {
        target_layers: 739_639_296,
        proposer: 67_574_292,
    };
    assert_eq!(s.total(), 807_213_588);
    assert_eq!(s.for_batch(3), 2_421_640_764);
    // `for_batch` multiplies once; a caller that also multiplies would land here.
    assert_eq!(s.for_batch(1), s.total());
    assert_eq!(
        s.for_batch(0),
        s.total(),
        "batch 0 is clamped to 1, never zero-charged"
    );
}

/// PART 4: the reserve-to-KV-block conversion, pinned exactly. This is the number BOOT-C3's
/// prediction is built on, so it is derived from the cache geometry and never from a log.
#[test]
fn the_batch_3_charge_converts_to_exactly_13_436_kv_blocks() {
    let charge = 2_421_640_764usize;
    assert_eq!(charge / KV_BLOCK_BYTES, 13_436);
    // Reachable demand at bs=3, msl=131072: `bs * ceil(msl/16) + bs + 1`.
    let reachable = 3 * MSL.div_ceil(16) + 3 + 1;
    assert_eq!(reachable, 24_580);
    // Arm A (clamp-bound) holds only above this budget; below it, arm B applies.
    assert_eq!(reachable + charge / KV_BLOCK_BYTES, 38_016);
}

/// The two measured clamp lines are reproduced by the source-derived block size, to the
/// precision the log prints (it divides by 1024^3 and labels the result "GB" — it is GiB).
#[test]
fn the_source_derived_block_size_reproduces_both_measured_clamp_lines() {
    let gib = |b: usize| (b as f64) / (1024.0 * 1024.0 * 1024.0);
    let freed_131k = (46_887 - 8_194) * KV_BLOCK_BYTES;
    let freed_65k = (47_537 - 4_098) * KV_BLOCK_BYTES;
    assert_eq!(format!("{:.2}", gib(freed_131k)), "6.49");
    assert_eq!(format!("{:.2}", gib(freed_65k)), "7.29");
}

#[test]
fn a_non_glm_config_is_charged_nothing() {
    let cfg = ModelConfig::qwen3_next_80b_nvfp4();
    assert_ne!(cfg.model_type, "glm5_next");
    let s = per_sequence_state_bytes(&cfg, MSL, true).expect("non-GLM is inert, not an error");
    assert_eq!(s, PerSequenceState::default());
    assert_eq!(s.total(), 0);
}
