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
        released: false,
    };
    assert!(s.is_empty());
    s.advance(cap - 1).unwrap();
    assert_eq!(s.len(), cap - 1);
    s.advance(1).unwrap();
    assert_eq!(s.len(), cap, "exactly full is legal");
    let e = s.advance(1).unwrap_err().to_string();
    assert!(
        e.contains("--max-seq-len"),
        "name the knob that moves it: {e}"
    );
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
        released: false,
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
        released: false,
    };
    s.advance(cap).unwrap();
    assert!(
        s.ensure_room(0).is_ok(),
        "exactly full still has room for zero rows"
    );
    let e = s.ensure_room(1).unwrap_err().to_string();
    assert!(
        e.contains("--max-seq-len"),
        "name the knob that moves it: {e}"
    );
    assert_eq!(
        s.len(),
        cap,
        "a refused ensure_room must not move the cursor"
    );
}

/// ANOMALIES A76: `LayerState` has no `Drop` and `DevicePtr` has none either, so the
/// three indexer buffers survive the sequence unless `free` releases them. A round trip
/// must return the backend to its exact baseline — "allocated fewer" is still a leak.
#[test]
fn free_returns_every_indexer_buffer_and_is_idempotent() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let base = gpu.alloc_count();
    let mut s = Glm5NextDsaState::alloc(&gpu, &c).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        base + 3,
        "k_normed + gate + valid are three live device allocations"
    );
    s.free(&gpu).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        base,
        "free returns to the baseline exactly"
    );
    assert_eq!(s.k_normed.0, 0, "a released state must not look live");

    // Two owners can reach a DSA state (the drafter's `free_state` and, since A76, the
    // target layer's). A second call must be a no-op, not a double `gpu.free`.
    let other = gpu.alloc(4096).unwrap();
    s.free(&gpu).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        base + 1,
        "the second free must not touch an unrelated allocation"
    );
    gpu.free(other).unwrap();
}

/// 100 sequence lifetimes, one backend: the per-request growth this fix exists to remove
/// has to be exactly zero, not merely small. Fails on the pre-A76 tree.
#[test]
fn a_hundred_alloc_free_cycles_leak_nothing() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let base = gpu.alloc_count();
    for _ in 0..100 {
        let mut s = Glm5NextDsaState::alloc(&gpu, &c).unwrap();
        s.advance(16).unwrap();
        s.free(&gpu).unwrap();
    }
    assert_eq!(gpu.alloc_count(), base, "no per-sequence growth");
}

/// L1 (lifecycle invariant): alloc → release returns the backend ledger to baseline in
/// **count AND bytes**.
///
/// 🔴 Bytes are the point. `live_alloc_count` alone cannot see a same-count, different-size
/// leak — a state that frees three buffers and allocates three smaller ones balances the count
/// and loses memory every cycle. The on-device G5 gate reported count only; this is the
/// unit-level half of closing that.
#[test]
fn alloc_then_release_returns_the_ledger_to_baseline_in_count_and_bytes() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let cfg = Glm5NextDsaConfig {
        max_context: 131_072,
        ..cfg()
    };

    let base_count = gpu.live_alloc_count();
    let base_bytes = gpu.live_bytes().expect("the mock keeps a ledger");

    for _ in 0..11 {
        let mut st = Glm5NextDsaState::alloc(&gpu, &cfg).expect("alloc");

        // In flight: exactly the three buffers, and exactly the bytes the reserve charges.
        assert_eq!(gpu.live_alloc_count(), base_count + 3);
        let in_flight = gpu.live_bytes().expect("ledger") - base_bytes;
        assert_eq!(
            in_flight,
            indexer_state_bytes(max_dsa_context(&cfg), cfg.index_head_dim),
            "what alloc actually takes must equal what the reserve charges"
        );

        st.free(&gpu).expect("free");
        assert_eq!(gpu.live_alloc_count(), base_count, "count back to baseline");
        assert_eq!(
            gpu.live_bytes().expect("ledger"),
            base_bytes,
            "BYTES back to baseline"
        );

        // Idempotent: a second release is a no-op, not a double free.
        st.free(&gpu).expect("second free is a no-op");
        assert_eq!(gpu.live_bytes().expect("ledger"), base_bytes);
    }
}
