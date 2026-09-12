// SPDX-License-Identifier: AGPL-3.0-only

//! Native FP8 FFN must use the existing kernels suited to the row count.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

fn run_case(
    rows: u32,
    prefill: bool,
    expected: u64,
    grid: [u32; 3],
    block: [u32; 3],
    configure: impl FnOnce(&mut DenseFfnLayer),
) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, &gpu).unwrap();
    // Real fallback weights coexist with the FP8 overlay in the loader.
    // The regression is selecting these valid but lower-precision bytes.
    let mut fallback = QuantizedWeight::null();
    fallback.weight = gpu.alloc(128 * 64).unwrap();
    fallback.weight_scale = gpu.alloc(128 * 8).unwrap();
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: fallback,
            up_proj: fallback,
            down_proj: fallback,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        &gpu,
    )
    .unwrap();
    // Mock kernel lookup deliberately returns one placeholder for every name;
    // separate these handles so the oracle identifies actual dispatch routes.
    layer.w8a16_gemm_k = KernelHandle(0xF08);
    layer.act_mul = KernelHandle(0xAC7);
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.set_fp8_weights(fp8, fp8, fp8);
    configure(&mut layer);
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        // Prefill shape: `attn_metadata` is None and the FFN under test never
        // reads the decode scalars this flag guards.
        decode_step: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let start = gpu.launch_count();
    let allocs = gpu.alloc_count();
    if prefill {
        layer
            .forward_prefill(buffers.norm_output(), rows as usize, &ctx, 7)
            .unwrap();
    } else {
        assert!(layer.can_forward_km(rows));
        layer
            .forward_km(buffers.norm_output(), rows, &ctx, 7)
            .unwrap();
    }
    assert_eq!(
        gpu.alloc_count(),
        allocs,
        "dispatch must not allocate weight copies"
    );
    let launches = gpu.launches_snapshot();
    let launches = &launches[start..];
    let projections: Vec<_> = launches
        .iter()
        .filter(|launch| launch.func != layer.act_mul.0)
        .collect();
    assert_eq!(projections.len(), 3);
    for launch in projections {
        assert_eq!(
            launch.func, expected,
            "M={rows}: gate/up/down selected the wrong native kernel"
        );
        assert_eq!(launch.grid, grid);
        assert_eq!(launch.block, block);
        assert_eq!(launch.stream, 7);
        assert_eq!(launch.args[1], MockArg::Buffer(fp8.weight));
        assert_eq!(launch.args[2], MockArg::Buffer(fp8.row_scale));
        assert_eq!(launch.args[4], MockArg::Bytes(rows.to_ne_bytes().to_vec()));
        assert_eq!(
            launch.args[5],
            MockArg::Bytes(128_u32.to_ne_bytes().to_vec())
        );
        assert_eq!(
            launch.args[6],
            MockArg::Bytes(128_u32.to_ne_bytes().to_vec())
        );
        assert!(!launch.args.contains(&MockArg::Buffer(fallback.weight)));
    }
}

#[test]
fn small_native_ffn_uses_existing_batched_gemv() {
    for rows in [1, 2, 3, 4] {
        run_case(rows, true, 0xDEAD, [32, 1, 1], [256, 1, 1], |_| {});
    }
    run_case(4, false, 0xDEAD, [32, 1, 1], [256, 1, 1], |_| {});
}

#[test]
fn larger_native_ffn_uses_existing_pipelined_gemm() {
    // 5..=32 now belong to `w8a16_gemv_batch16` (#927), so the tile GEMM's
    // first row count is 33. Zeroing the batch16 handle is how a shadow that
    // lacks the entry point behaves, and it must still land here.
    run_case(129, true, 0xDEAD, [4, 2, 1], [256, 1, 1], |_| {});
    for rows in [5, 8] {
        run_case(
            rows,
            true,
            0xDEAD,
            [4, rows.div_ceil(128), 1],
            [256, 1, 1],
            |layer| layer.w8a16_gemv_batch16_k = KernelHandle(0),
        );
    }
    run_case(5, false, 0xDEAD, [4, 1, 1], [256, 1, 1], |layer| {
        layer.w8a16_gemv_batch16_k = KernelHandle(0)
    });
}

/// #927: the 5..=16 band streams the FP8 weight ONCE through the MAX_M=16
/// twin of `w8a16_gemv_batch4` — same grid, same block, same argument layout,
/// so only the handle distinguishes the two rungs. (`run_case` asserts one
/// launch per projection, which is the other half of the contract: the tile
/// GEMM it replaces also emitted one, but over an M-padded MMA tile.)
///
/// The tier is ARMED explicitly (`layer.batch16_tier`): since the per-target
/// `[defaults]` table landed, whether it is on depends on which `kernels/<hw>`
/// tree this binary was built for, and a dispatch test must grade the route
/// rather than the build's target. Which targets declare it on is
/// `ops::target_defaults`'s tests.
#[test]
fn five_to_sixteen_row_native_ffn_uses_the_batch16_gemv() {
    let arm = |layer: &mut DenseFfnLayer| {
        layer.batch16_tier = true;
        layer.w8a16_gemv_batch16_k = KernelHandle(0xB16);
    };
    for rows in [5, 8, 16] {
        run_case(rows, true, 0xB16, [32, 1, 1], [256, 1, 1], arm);
    }
    run_case(5, false, 0xB16, [32, 1, 1], [256, 1, 1], arm);
}

#[test]
fn missing_small_batch_kernel_uses_same_precision_pipelined_fallback() {
    run_case(4, false, 0xDEAD, [4, 1, 1], [256, 1, 1], |layer| {
        layer.w8a16_gemv_batch4_k = KernelHandle(0);
    });
}

#[test]
fn missing_optional_kernels_keep_base_native_gemm() {
    for rows in [4, 5, 129] {
        run_case(
            rows,
            true,
            0xF08,
            [2, rows.div_ceil(64), 1],
            [128, 1, 1],
            |layer| {
                layer.w8a16_gemv_batch4_k = KernelHandle(0);
                layer.w8a16_gemv_batch16_k = KernelHandle(0);
                layer.w8a16_gemm_pipelined_k = KernelHandle(0);
            },
        );
    }
}
