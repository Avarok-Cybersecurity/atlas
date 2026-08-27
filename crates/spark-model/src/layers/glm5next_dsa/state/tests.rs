// SPDX-License-Identifier: AGPL-3.0-only

//! Indexer-cache sizing and refusal. No GPU: the arithmetic is what can be wrong.

use super::*;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
    }
}

/// 🔴 The cap is set by the top-k's PADDED pool axis, so it is the power of two below
/// the raw byte budget: 49,152/8 = 6,144 → 4,096 pools → 16,384 tokens at kpool 4.
/// Quoting 6,144 would promise 24,576 tokens that never work.
#[test]
fn the_context_cap_is_the_power_of_two_pool_bound() {
    let c = cfg();
    assert_eq!(max_dsa_context(&c), 16_384);
    // The last selection that fits, and the first that does not.
    assert!(DsaSelectGeometry::plan(&c, max_dsa_context(&c), 1).is_ok());
    assert!(DsaSelectGeometry::plan(&c, max_dsa_context(&c) + 4, 1).is_err());
}

/// The cap scales with the pool size, because the bound is on POOLS, not tokens.
#[test]
fn the_cap_scales_with_kpool() {
    let mut c = cfg();
    c.index_kpool = 8;
    c.index_topk = 8 * 256;
    assert_eq!(max_dsa_context(&c), 32_768);
}

/// Past the cap the selector cannot sort the pool axis at all. Clamping would select over
/// a prefix while the MLA cache held the full context — a wrong answer with no crash.
#[test]
fn advancing_past_the_cap_is_refused_not_clamped() {
    let c = cfg();
    let cap = max_dsa_context(&c);
    // `alloc` needs a GPU; the length bookkeeping does not, so model it directly.
    let mut s = Glm5NextDsaState {
        k_normed: spark_runtime::gpu::DevicePtr(0),
        gate: spark_runtime::gpu::DevicePtr(0),
        valid: spark_runtime::gpu::DevicePtr(0),
        len: 0,
        capacity: cap,
        index_head_dim: c.index_head_dim,
    };
    assert!(s.is_empty());
    s.advance(cap - 1).unwrap();
    assert_eq!(s.len(), cap - 1);
    s.advance(1).unwrap();
    assert_eq!(s.len(), cap, "exactly full is legal");
    let e = s.advance(1).unwrap_err().to_string();
    assert!(e.contains("segmented/radix"), "name the real fix: {e}");
    assert_eq!(s.len(), cap, "a refused advance must not move the cursor");
}

/// Row offsets are in BYTES over a flat `[capacity, index_head_dim]` BF16 buffer — the
/// indexer kernels address `k[raw * D + d]` linearly, so a paged stride would be wrong.
#[test]
fn row_offsets_are_flat_bf16_rows() {
    let c = cfg();
    let s = Glm5NextDsaState {
        k_normed: spark_runtime::gpu::DevicePtr(0),
        gate: spark_runtime::gpu::DevicePtr(0),
        valid: spark_runtime::gpu::DevicePtr(0),
        len: 0,
        capacity: max_dsa_context(&c),
        index_head_dim: c.index_head_dim,
    };
    assert_eq!(s.row_offset(0), 0);
    assert_eq!(s.row_offset(1), 128 * 2);
    assert_eq!(s.row_offset(1000), 1000 * 128 * 2);
}

/// 8 MiB per layer per sequence at the cap; ~92 MiB across the 11 text DSA layers.
/// Small enough to reserve up front, which is what makes the fixed cap workable.
#[test]
fn the_reservation_is_small_enough_to_preallocate() {
    let c = cfg();
    let cap = max_dsa_context(&c);
    let per_layer = cap * c.index_head_dim * 2 * 2 + cap; // k + gate + valid
    assert_eq!(per_layer, 8_404_992);
    assert!(
        per_layer * 11 < 100 << 20,
        "under 100 MiB for the DSA stack"
    );
}
