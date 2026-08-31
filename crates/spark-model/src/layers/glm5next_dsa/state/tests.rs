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
        max_context: 16_384,
    }
}

/// 🟢 The cap is an ALLOCATION now, not the top-k kernel's shared-memory budget: the
/// select is tiled, so `plan` succeeds at any context. ANOMALIES A62.
#[test]
fn the_context_cap_follows_max_seq_len_not_the_kernel() {
    let c = cfg();
    assert_eq!(max_dsa_context(&c), 16_384, "the fixture reserves 16,384");
    // Past the OLD 16,384 kernel ceiling, planning now succeeds — 131,072 tokens is
    // 32,768 pools, twenty-nine tiles wide, and shared memory does not move.
    let mut big = cfg();
    big.max_context = 131_072;
    assert_eq!(max_dsa_context(&big), 131_072);
    for seq in [16_384usize, 16_388, 65_536, 131_072] {
        let g = DsaSelectGeometry::plan(&big, seq, 1).unwrap_or_else(|e| {
            panic!("plan refused {seq} tokens: {e}");
        });
        assert_eq!(g.topk_np2, super::super::select::topk_tile());
        assert_eq!(
            g.topk_smem,
            super::super::select::topk_smem_for_tile(g.topk_np2)
        );
        assert!(g.topk_smem <= super::super::select::TOPK_SMEM_CEILING);
    }
}

/// A reservation is a whole number of pools — a trailing partial pool is not a pool, so
/// reserving rows it could never select over would be dead memory.
#[test]
fn the_cap_is_rounded_down_to_whole_pools() {
    let mut c = cfg();
    c.max_context = 16_386;
    assert_eq!(max_dsa_context(&c), 16_384);
    c.index_kpool = 8;
    c.max_context = 1_001;
    assert_eq!(max_dsa_context(&c), 1_000);
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
    assert!(e.contains("--max-seq-len"), "name the knob that moves it: {e}");
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

/// The capacity question has to be answerable WITHOUT writing first — `indexer_forward`
/// writes into row `len()` and only then advances, so `advance`'s refusal arrives one
/// out-of-bounds row too late. ANOMALIES A62.
#[test]
fn ensure_room_refuses_before_the_write_and_moves_nothing() {
    let c = cfg();
    let cap = max_dsa_context(&c);
    let mut s = Glm5NextDsaState {
        k_normed: spark_runtime::gpu::DevicePtr(0),
        gate: spark_runtime::gpu::DevicePtr(0),
        valid: spark_runtime::gpu::DevicePtr(0),
        len: 0,
        capacity: cap,
        index_head_dim: c.index_head_dim,
    };
    s.advance(cap).unwrap();
    assert!(s.ensure_room(0).is_ok(), "exactly full still has room for zero rows");
    let e = s.ensure_room(1).unwrap_err().to_string();
    assert!(e.contains("--max-seq-len"), "name the knob that moves it: {e}");
    assert_eq!(s.len(), cap, "a refused ensure_room must not move the cursor");
}
