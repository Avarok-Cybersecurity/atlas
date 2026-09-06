// SPDX-License-Identifier: AGPL-3.0-only

//! The real multi-row entry point must retain the installed weight format.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

fn run_batch(native_fp8: bool, rows: u32) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
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
    // Also prove the original native fallback when optional fast kernels are absent.
    layer.w8a16_gemv_batch4_k = KernelHandle(0);
    layer.w8a16_gemm_pipelined_k = KernelHandle(0);
    layer.act_mul = KernelHandle(0xAC7);
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    if native_fp8 {
        layer.set_fp8_weights(fp8, fp8, fp8);
    }
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
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    assert!(layer.can_forward_km(rows));
    let start = gpu.launch_count();
    layer
        .forward_km(buffers.norm_output(), rows, &ctx, 0)
        .unwrap();
    let launches = gpu.launches_snapshot();
    let launches = &launches[start..];
    let expected = if native_fp8 {
        layer.w8a16_gemm_k
    } else {
        layer.batchm_kernel(rows)
    };
    assert_eq!(
        launches
            .iter()
            .filter(|launch| launch.func == expected.0)
            .count(),
        3,
        "M={rows}: gate/up/down must all use the installed weight format"
    );
    if native_fp8 {
        assert!(
            launches
                .iter()
                .all(|launch| { !launch.args.contains(&MockArg::Buffer(fallback.weight)) }),
            "M={rows}: native FP8 dispatch read the NVFP4 fallback"
        );
    }
}

#[test]
fn native_fp8_multi_row_dispatch_preserves_checkpoint_weights() {
    for rows in [4, 5, 8] {
        run_batch(true, rows);
    }
}

#[test]
fn nvfp4_multi_row_dispatch_keeps_existing_batch_kernels() {
    for rows in [4, 5, 8] {
        run_batch(false, rows);
    }
}
