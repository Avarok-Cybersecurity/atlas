// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation contract for the SSM/GDN QKVZ prefill projection. Split from
//! `tests.rs` to keep that file under the 500-LoC cap; the harness helpers it
//! uses (`native_fp8_gdn_layer`) live there and are reached through `super`.

use super::tests::native_fp8_gdn_layer;
use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;

/// H100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`, tip `5f78270dc`: with
/// `ATLAS_CUBLAS_GEMM=1` this projection routed to `ops::cublas_bf16_proj`,
/// whose cached FP8→BF16 weight dequant allocated `167772160` bytes PER LAYER
/// (`[10240,5120] + [6144,5120]` fused, x 2 B) outside the buffer ledger. One
/// 28-token prefill consumed 6120 MiB and died at layer 36 with
/// `cuMemAlloc_v2 failed: status 2`.
///
/// The assertion is deliberately taken BEFORE the first call, not between the
/// two: the old allocation was CACHED by weight pointer, so a first-vs-second
/// comparison alone would have passed it. What is wrong is that the projection
/// allocates at all — the arena already owns every buffer it needs.
///
/// The k-major scale adapter is zeroed so the arm takes the in-tree kernel:
/// a CPU test cannot enter cuBLASLt's FFI, and it is the arm a mock backend
/// can reach. The cuBLASLt arm's own clauses are pinned in
/// `prefill_w8a8_tests.rs`, and it allocates nothing by construction — its
/// activation, scales and k-major transpose are all arena buffers.
#[test]
fn ssm_qkvz_prefill_allocates_nothing_with_the_cublas_lever_armed() {
    use crate::layers::ops::CublasScope;

    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.fp8_act_scale_kmajor_k = spark_runtime::gpu::KernelHandle(0);

    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch {
        cublas: CublasScope::ALL,
        ..crate::layers::ops::GemmDispatch::defaults()
    };
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };

    let m = 28_u32; // the smoke prompt that killed the H100 run
    let qkvz_size = config.ssm_qkvz_size();
    let run = || {
        layer.prefill_qkvz_proj(
            buffers.norm_output(),
            buffers.ssm_deinterleaved(),
            m,
            qkvz_size,
            config.hidden_size,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            config.linear_num_value_heads / config.linear_num_key_heads.max(1),
            config.linear_value_head_dim,
            &ctx,
            0,
        )
    };
    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    run().expect("first QKVZ prefill projection");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the QKVZ prefill projection allocated on its FIRST call — that is the \
         167772160-byte per-layer BF16 weight dequant from the H100 receipt"
    );
    // Two launches, and the same two again: the per-token FP8 quant and the
    // block-scaled GEMM. Pinned so "allocated nothing" cannot be satisfied by
    // "did nothing" — and so a future arm that slips a dequant kernel back in
    // is visible even if it reuses a buffer.
    assert_eq!(
        gpu.launch_count() - launches,
        2,
        "expected per_token_group_quant_fp8 + fp8_gemm_t_blockscaled"
    );
    run().expect("second QKVZ prefill projection");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the QKVZ prefill projection allocated per call"
    );
    assert_eq!(gpu.launch_count() - launches, 4);
}
