// SPDX-License-Identifier: AGPL-3.0-only

//! The native-BF16 dense FFN must work with NO NVFP4 fallback weights.
//!
//! WHY. For `Bf16Raw` checkpoints the Qwen3.5 dense loader used to quantize
//! gate/up/down to NVFP4 (plus transposed twins and the MMQ repacks) next to
//! the BF16 overlay, on the stated ground that the batched and speculative
//! paths needed a "never null" fallback. They do not: `forward` and
//! `forward_prefill_inner` branch on `bf16_weights` ahead of every NVFP4 read,
//! and `forward_k2`/`forward_k3`/`forward_km` redirect to `forward_prefill`
//! through `native_small_batch_uses_prefill`. These tests pin that, with the
//! NVFP4 weights NULL as the loader now builds them, so a path that ever read
//! them would put a null buffer into a launch (the mock's analogue of the
//! CUDA-700 the old comment feared). They mirror
//! `dense_ffn_fp8_residency_tests.rs`.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{DenseWeight, QuantizedWeight};
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

fn config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    config
}

fn null_nvfp4() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

/// Exactly what `qwen35_dense.rs` builds on the native-BF16 route: NULL NVFP4
/// weights, no transposed twins, a BF16 overlay on top.
fn native_bf16_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let mut layer = DenseFfnLayer::new(null_nvfp4(), gpu).unwrap();
    layer.act_mul = KernelHandle(0xAC7);
    let bf16 = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    layer.set_bf16_weights(
        bf16(128 * 128 * 2),
        bf16(128 * 128 * 2),
        bf16(128 * 128 * 2),
    );
    layer
}

/// Turn on every handle the two load-time MMQ finalizers check, so the test
/// exercises the arm a real Blackwell/GB10 kernel set would enable.
fn arm_the_mmq_finalizers(layer: &mut DenseFfnLayer) {
    for h in [
        &mut layer.q4k_mmq_nc_k,
        &mut layer.q4k_quant_act_k,
        &mut layer.q4k_quant_w_k,
        &mut layer.dequant_nvfp4_bf16_k,
        &mut layer.nvfp4_mmq_nc_k,
        &mut layer.nvfp4_quant_act_k,
        &mut layer.nvfp4_repack_k,
        &mut layer.nvfp4_silu_scaled_k,
    ] {
        *h = KernelHandle(0xBEEF);
    }
}

fn with_ctx(gpu: &MockGpuBackend, f: impl FnOnce(&ForwardContext, &BufferArena)) {
    let config = config();
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    f(&ctx, &buffers);
}

#[test]
fn the_verify_arm_still_selects_this_layer_without_nvfp4_weights() {
    // `multi_seq/ffn.rs` gates the 4..=8-row verify branch on this. It must
    // not change answer just because the NVFP4 fallback is gone, or dropping
    // the copies would silently reroute decode as well as shrink it.
    let gpu = MockGpuBackend::new();
    let layer = native_bf16_layer(&gpu);
    for rows in 1..=8 {
        assert!(
            layer.can_forward_km(rows),
            "native BF16 layer must stay eligible at m={rows}"
        );
    }
}

#[test]
fn a_layer_with_no_weights_at_all_is_still_not_eligible() {
    let gpu = MockGpuBackend::new();
    let mut layer = DenseFfnLayer::new(null_nvfp4(), &gpu).unwrap();
    layer.act_mul = KernelHandle(0xAC7);
    assert!(!layer.can_forward_km(4));
}

#[test]
fn the_mmq_finalizers_are_no_ops_under_a_native_bf16_overlay() {
    // `finalize_nvfp4_mmq_load` is active by DEFAULT wherever its kernels
    // exist; over NULL NVFP4 sources it would repack null pointers, and over
    // real ones it is dead work because the BF16 arm returns first.
    let gpu = MockGpuBackend::new();
    let mut layer = native_bf16_layer(&gpu);
    arm_the_mmq_finalizers(&mut layer);
    let allocs = gpu.alloc_count();
    let launches = gpu.launch_count();
    layer.finalize_q4k_load(&gpu, 128, 128, 7).unwrap();
    layer.finalize_nvfp4_mmq_load(&gpu, 128, 128, 7).unwrap();
    assert_eq!(gpu.alloc_count(), allocs, "no MMQ repack may be allocated");
    assert_eq!(gpu.launch_count(), launches, "no kernel may be launched");
}

#[test]
fn every_entry_point_runs_without_nvfp4_weights() {
    // One row, two, three, 4..=8 verify rows, a wide batch and a prefill: the
    // whole surface `FfnComponent` exposes to the decode and verify ladders.
    // Any read of a NULL NVFP4 weight would surface as a null buffer in a
    // launch's arguments.
    let gpu = MockGpuBackend::new();
    let layer = native_bf16_layer(&gpu);
    with_ctx(&gpu, |ctx, buffers| {
        let start = gpu.launch_count();
        let allocs = gpu.alloc_count();
        let input = buffers.norm_output();
        layer.forward(input, ctx, 7).unwrap();
        layer.forward_k2(input, ctx, 7).unwrap();
        layer.forward_k3(input, ctx, 7).unwrap();
        for rows in 4..=8 {
            layer.forward_km(input, rows, ctx, 7).unwrap();
        }
        layer.forward_batched(input, 16, ctx, 7).unwrap();
        layer.forward_prefill(input, 64, ctx, 7).unwrap();
        assert_eq!(
            gpu.alloc_count(),
            allocs,
            "dispatch must not allocate weight copies"
        );
        let launches = gpu.launches_snapshot();
        assert!(launches.len() > start, "the forwards must have dispatched");
        for launch in &launches[start..] {
            for arg in &launch.args {
                if let MockArg::Buffer(p) = arg {
                    assert!(
                        !p.is_null(),
                        "a NULL NVFP4 weight reached a kernel launch: {launch:?}"
                    );
                }
            }
        }
    });
}

#[test]
fn small_batch_entry_points_read_the_bf16_weights_not_a_fallback() {
    // With a REAL NVFP4 fallback present (the KEEP_NVFP4 hatch) the small-batch
    // paths must still pick the BF16 overlay; this is the property that makes
    // the fallback unreachable and the null build above sound.
    let gpu = MockGpuBackend::new();
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
    layer.act_mul = KernelHandle(0xAC7);
    layer.set_bf16_weights(
        DenseWeight {
            weight: gpu.alloc(128 * 128 * 2).unwrap(),
        },
        DenseWeight {
            weight: gpu.alloc(128 * 128 * 2).unwrap(),
        },
        DenseWeight {
            weight: gpu.alloc(128 * 128 * 2).unwrap(),
        },
    );
    with_ctx(&gpu, |ctx, buffers| {
        let start = gpu.launch_count();
        let input = buffers.norm_output();
        layer.forward(input, ctx, 7).unwrap();
        layer.forward_k2(input, ctx, 7).unwrap();
        layer.forward_k3(input, ctx, 7).unwrap();
        layer.forward_km(input, 4, ctx, 7).unwrap();
        layer.forward_prefill(input, 64, ctx, 7).unwrap();
        let launches = gpu.launches_snapshot();
        assert!(launches.len() > start, "the forwards must have dispatched");
        for launch in &launches[start..] {
            assert!(
                !launch.args.contains(&MockArg::Buffer(fallback.weight)),
                "a small-batch path read the NVFP4 fallback: {launch:?}"
            );
        }
    });
}
