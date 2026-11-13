// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the multi-sequence decode arm on the FP8 batch-M GEMVs
//! (`fp8_decode_gemv.rs`, row `ffn_fp8_decode_gemv`): the admission rule, the
//! launches a decode step makes, and that prefill and verify passes, the row
//! off, and layers without the FP8 overlay launch nothing. The kernels' byte
//! identity with the M=1 GEMVs is pinned by `ada_kernels/verify_tests.rs`
//! (host model) and `examples/fp8_ada_batchm_microtest.rs` (device).

use super::*;
use crate::layer::MoeLoraRoute;
use crate::layers::FfnComponent;
use crate::layers::dense_ffn::{DenseFfnWeights, FfnActivation};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

const H: u32 = 256;
const INTER: u32 = 512;
const DUAL_K: KernelHandle = KernelHandle(0xD2D);
const PLAIN_K: KernelHandle = KernelHandle(0xD21);
const SILU_K: KernelHandle = KernelHandle(0xD25);
const ACT_MUL: KernelHandle = KernelHandle(0xDC7);

fn kernels() -> RowKernels {
    RowKernels {
        plain: PLAIN_K,
        dual: DUAL_K,
        silu_input: SILU_K,
    }
}

fn fp8(gpu: &MockGpuBackend, n: u32, k: u32) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc((n * k) as usize).unwrap(),
        row_scale: gpu
            .alloc(4 * n.div_ceil(128) as usize * k.div_ceil(128) as usize)
            .unwrap(),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

fn null_weights() -> DenseFfnWeights {
    DenseFfnWeights {
        gate_proj: QuantizedWeight::null(),
        up_proj: QuantizedWeight::null(),
        down_proj: QuantizedWeight::null(),
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    }
}

/// What `qwen35_dense.rs` builds on the native-FP8 route: NULL NVFP4
/// fallbacks and a block-scaled FP8 overlay.
fn fp8_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let mut layer =
        DenseFfnLayer::new_with_activation(null_weights(), FfnActivation::SiLU, gpu).unwrap();
    layer.act_mul = ACT_MUL;
    let (g, u, d) = (fp8(gpu, INTER, H), fp8(gpu, INTER, H), fp8(gpu, H, INTER));
    layer.set_fp8_weights(g, u, d);
    layer
}

/// A forward context over a small arena; `decode_step` is the one field the
/// arm reads (true on the decode forwards, false on prefill and verify).
fn with_ctx<R>(
    gpu: &MockGpuBackend,
    decode_step: bool,
    f: impl FnOnce(&ForwardContext, &BufferArena) -> R,
) -> R {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = 0;
    config.num_experts_per_tok = 0;
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, 64, 256, 256, 8, gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let mut levers = ModelLevers::defaults();
    levers.decode_split_silu = true;
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
        decode_step,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    f(&ctx, &buffers)
}

fn funcs(gpu: &MockGpuBackend, from: usize) -> Vec<u64> {
    gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| l.func)
        .collect()
}

#[test]
fn the_rule_needs_the_row_a_decode_step_an_fp8_layer_and_two_to_eight_rows() {
    for rows in MIN_ROWS..=MAX_ROWS {
        assert!(admits(true, true, true, false, rows), "rows={rows}");
    }
    assert!(!admits(false, true, true, false, 8), "row off");
    assert!(!admits(true, false, true, false, 8), "prefill or verify");
    assert!(!admits(true, true, false, false, 8), "no FP8 overlay");
    assert!(!admits(true, true, true, true, 8), "packed Q2_0");
    for rows in [0, 1, MAX_ROWS + 1, 16] {
        assert!(!admits(true, true, true, false, rows), "rows={rows}");
    }
}

#[test]
fn eight_decode_streams_are_one_pass_per_weight_into_moe_output() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let x = buffers.norm_output();
        let start = gpu.launch_count();
        assert!(
            layer
                .fp8_decode_arm_with(true, &kernels(), x, 8, ctx, 3)
                .unwrap()
        );
        // The M=1 decode forward's own arm: gate and up in one launch, the
        // SwiGLU product, then down. Three launches, each weight read once.
        assert_eq!(funcs(&gpu, start), vec![DUAL_K.0, ACT_MUL.0, PLAIN_K.0]);
        let l = gpu.launches_snapshot();
        assert_eq!(l[start].grid, [INTER.div_ceil(4), 1, 2]);
        assert_eq!(
            l[start + 1].grid[0],
            (8 * INTER).div_ceil(256),
            "SwiGLU over all 8 rows"
        );
        assert_eq!(l[start + 2].grid, [H.div_ceil(4), 1, 1]);
        assert!(l[start].args.contains(&MockArg::Buffer(x)));
        assert!(
            l[start + 2]
                .args
                .contains(&MockArg::Buffer(buffers.moe_output())),
            "the result lands where every small-batch entry point writes"
        );
        assert!(l[start..].iter().all(|l| l.stream == 3));
    });
}

#[test]
fn every_width_from_two_to_eight_takes_the_arm_and_the_edges_decline() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let x = buffers.norm_output();
        for rows in MIN_ROWS..=MAX_ROWS {
            let start = gpu.launch_count();
            assert!(
                layer
                    .fp8_decode_arm_with(true, &kernels(), x, rows, ctx, 0)
                    .unwrap(),
                "rows={rows}"
            );
            assert_eq!(gpu.launch_count() - start, 3, "rows={rows}");
        }
        for rows in [1usize, MAX_ROWS + 1] {
            let start = gpu.launch_count();
            assert!(
                !layer
                    .fp8_decode_arm_with(true, &kernels(), x, rows, ctx, 0)
                    .unwrap(),
                "rows={rows}"
            );
            assert_eq!(gpu.launch_count(), start, "rows={rows}");
        }
    });
}

#[test]
fn a_prefill_or_verify_pass_never_takes_the_decode_arm() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, false, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !layer
                .fp8_decode_arm_with(true, &kernels(), buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn the_row_off_declines_and_launches_nothing() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !layer
                .fp8_decode_arm_with(false, &kernels(), buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn a_layer_without_the_fp8_overlay_declines() {
    let gpu = MockGpuBackend::new();
    let layer = DenseFfnLayer::new(null_weights(), &gpu).unwrap();
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !layer
                .fp8_decode_arm_with(true, &kernels(), buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn a_missing_batchm_handle_declines_instead_of_mixing_kernels() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    let no_dual = RowKernels {
        dual: KernelHandle(0),
        ..kernels()
    };
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !layer
                .fp8_decode_arm_with(true, &no_dual, buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn with_the_row_off_the_process_entry_points_issue_no_lookup_and_no_launch() {
    // The default environment resolves every port row off; this test is
    // meaningful only then (an operator running the suite with the row
    // exported is testing the other polarity on purpose).
    if ops::port_rows::ffn_fp8_decode_gemv::row() || ops::port_rows::ffn_fp8_verify_gemv::row() {
        return;
    }
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let x = buffers.norm_output();
        let lookups = gpu.kernel_lookups_snapshot().len();
        let start = gpu.launch_count();
        assert!(!layer.fp8_decode_arm(x, 8, ctx, 0).unwrap());
        // The hook every small-batch entry point tries first (`w4.rs`).
        assert!(!layer.fp8_verify_arm(x, 8, ctx, 0).unwrap());
        assert_eq!(gpu.launch_count(), start);
        assert_eq!(gpu.kernel_lookups_snapshot().len(), lookups);
        // And the SSM wide arm's question, which then keeps `forward_prefill`.
        let dense = FfnComponent::Dense(fp8_layer(&gpu));
        let lookups = gpu.kernel_lookups_snapshot().len();
        assert!(!dense.try_forward_fp8_decode_rows(x, 8, ctx, 0).unwrap());
        assert_eq!(gpu.launch_count(), start);
        assert_eq!(gpu.kernel_lookups_snapshot().len(), lookups);
    });
}

#[test]
fn an_absent_ffn_component_declines() {
    let gpu = MockGpuBackend::new();
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !FfnComponent::None
                .try_forward_fp8_decode_rows(buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}
