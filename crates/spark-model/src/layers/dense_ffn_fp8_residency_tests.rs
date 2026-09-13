// SPDX-License-Identifier: AGPL-3.0-only

//! The native-FP8 dense FFN must work with NO NVFP4 fallback weights (#915).
//!
//! WHY. On 1xH100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`, the loader built an
//! NVFP4 gate/up/down plus a transposed twin for all 64 layers — 18.4 GiB — on
//! the strength of a comment that said the batched spec-decode paths had no FP8
//! branch. They do: `forward_k2`/`forward_k3`/`forward_km` all redirect through
//! `native_small_batch_uses_prefill`, and `w8_gemm!` binds its transposed
//! operand to a literal `None`. These tests pin that, so the loader can install
//! `QuantizedWeight::null()` and the 18.4 GiB stays unallocated.
//!
//! Every case here is RED before #915: with a NULL `gate_proj`,
//! `can_forward_km` returned false (silently rerouting the 4..8-row verify arm)
//! and `finalize_nvfp4_mmq_load` — active by DEFAULT wherever its kernels
//! exist — repacked over the null pointers.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
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

/// Exactly what `qwen35_dense.rs` now builds on the native-FP8 route: NULL
/// NVFP4 fallbacks, no transposed twins, an FP8 overlay on top.
fn native_fp8_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        gpu,
    )
    .unwrap();
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
    layer
}

/// Turn on every handle the two load-time MMQ finalizers check, so the test
/// exercises the arm a real Blackwell/GB10 kernel set would enable rather than
/// passing because the handles happen to be zero.
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
    // `multi_seq/ffn.rs:145` gates the 4..=8-row verify branch on this. It must
    // not change answer just because the NVFP4 fallback is gone, or the fix
    // would silently reroute decode as well as shrink it.
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_layer(&gpu);
    for rows in 1..=8 {
        assert!(
            layer.can_forward_km(rows),
            "native FP8 layer must stay eligible at m={rows}"
        );
    }
}

#[test]
fn a_layer_with_neither_nvfp4_nor_fp8_weights_is_not_eligible() {
    // The guard `can_forward_km` actually exists for: packed-Q2 and any other
    // route whose NVFP4 weights are NULL with nothing installed over them.
    let gpu = MockGpuBackend::new();
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        &gpu,
    )
    .unwrap();
    layer.act_mul = KernelHandle(0xAC7);
    assert!(!layer.can_forward_km(4));
}

#[test]
fn the_mmq_finalizers_are_no_ops_under_a_native_fp8_overlay() {
    // `finalize_nvfp4_mmq_load` is active by DEFAULT (no opt-in env) wherever
    // its four kernels exist. Over NULL NVFP4 sources that is a CUDA-700; and
    // even over real ones it is dead work, because `forward_prefill_inner`
    // returns from the FP8 arm long before the MMQ arm.
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_layer(&gpu);
    arm_the_mmq_finalizers(&mut layer);
    let allocs = gpu.alloc_count();
    let launches = gpu.launch_count();
    layer.finalize_q4k_load(&gpu, 128, 128, 7).unwrap();
    layer.finalize_nvfp4_mmq_load(&gpu, 128, 128, 7).unwrap();
    assert_eq!(gpu.alloc_count(), allocs, "no MMQ repack may be allocated");
    assert_eq!(gpu.launch_count(), launches, "no kernel may be launched");
}

#[test]
fn every_small_batch_entry_point_runs_without_nvfp4_weights() {
    // The claim the 18.4 GiB rested on: "the batched spec-decode forward_k2/k3
    // paths have no FP8 branch". They redirect to `forward_prefill`; if any of
    // these reached a w4a16 dispatch it would read a NULL pointer, which on the
    // mock backend surfaces as a launch carrying the null buffer.
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_layer(&gpu);
    with_ctx(&gpu, |ctx, buffers| {
        let start = gpu.launch_count();
        let allocs = gpu.alloc_count();
        let input = buffers.norm_output();
        layer.forward(input, ctx, 7).unwrap();
        layer.forward_k2(input, ctx, 7).unwrap();
        layer.forward_k3(input, ctx, 7).unwrap();
        layer.forward_km(input, 4, ctx, 7).unwrap();
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
                if let spark_runtime::gpu::mock::MockArg::Buffer(p) = arg {
                    assert!(
                        !p.is_null(),
                        "a NULL NVFP4 weight reached a kernel launch: {launch:?}"
                    );
                }
            }
        }
    });
}
