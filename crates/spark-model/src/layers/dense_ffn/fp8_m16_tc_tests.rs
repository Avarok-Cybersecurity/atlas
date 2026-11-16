// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the 16- and 32-row tensor-core arm of a native-FP8 dense FFN
//! (`fp8_m16_tc.rs`, row `ffn_fp8_m16_tc`): the admission rule, the three
//! launches a 5 to 32 row call makes (the 16-row entries up to 16 rows, the
//! 32-row entries above) and their geometry, and that the row
//! off, the widths outside the arm, a layer without the FP8 overlay, a shape
//! that is not whole scale blocks and an absent kernel launch nothing. The
//! kernel's numerics are the device oracle's
//! (`examples/fp8_ada_m16_tc_microtest.rs`).

use super::*;
use crate::layer::MoeLoraRoute;
use crate::layers::FfnComponent;
use crate::layers::dense_ffn::{DenseFfnWeights, FfnActivation};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::QuantizedWeight;
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

const H: u32 = 256;
const INTER: u32 = 512;
const DUAL_K: KernelHandle = KernelHandle(0x7C2);
const PLAIN_K: KernelHandle = KernelHandle(0x7C1);
const DUAL32_K: KernelHandle = KernelHandle(0x7C4);
const PLAIN32_K: KernelHandle = KernelHandle(0x7C3);
const ACT_MUL: KernelHandle = KernelHandle(0x7C7);
const MIN: usize = ops::port_rows::ffn_fp8_m16_tc::DEFAULT_MIN_ROWS;

fn kernels() -> TcKernels {
    TcKernels {
        plain: PLAIN_K,
        dual: DUAL_K,
        plain32: PLAIN32_K,
        dual32: DUAL32_K,
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
/// fallbacks and a block-scaled FP8 overlay, at `[inter, h]` and `[h, inter]`.
fn fp8_layer_at(gpu: &MockGpuBackend, h: u32, inter: u32) -> DenseFfnLayer {
    let mut layer =
        DenseFfnLayer::new_with_activation(null_weights(), FfnActivation::SiLU, gpu).unwrap();
    layer.act_mul = ACT_MUL;
    let (g, u, d) = (fp8(gpu, inter, h), fp8(gpu, inter, h), fp8(gpu, h, inter));
    layer.set_fp8_weights(g, u, d);
    layer
}

fn fp8_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    fp8_layer_at(gpu, H, INTER)
}

/// A forward context over a small arena at hidden `h` and intermediate
/// `inter`. `decode_step` is passed through: the arm does not read it (its
/// call sites are decode and verify paths), and a test pins that.
fn with_ctx_at<R>(
    gpu: &MockGpuBackend,
    h: u32,
    inter: u32,
    decode_step: bool,
    f: impl FnOnce(&ForwardContext, &BufferArena) -> R,
) -> R {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = h as usize;
    config.intermediate_size = inter as usize;
    config.num_experts = 0;
    config.num_experts_per_tok = 0;
    config.moe_intermediate_size = inter as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, 64, 256, 256, 8, gpu).unwrap();
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

fn with_ctx<R>(
    gpu: &MockGpuBackend,
    decode_step: bool,
    f: impl FnOnce(&ForwardContext, &BufferArena) -> R,
) -> R {
    with_ctx_at(gpu, H, INTER, decode_step, f)
}

fn funcs(gpu: &MockGpuBackend, from: usize) -> Vec<u64> {
    gpu.launches_snapshot()[from..]
        .iter()
        .map(|l| l.func)
        .collect()
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

#[test]
fn the_rule_needs_the_row_an_fp8_layer_and_five_to_thirty_two_rows() {
    assert_eq!((TILE_ROWS, MAX_ROWS), (16, 32));
    for rows in MIN..=MAX_ROWS {
        assert!(admits(true, true, false, rows, MIN), "rows={rows}");
    }
    assert!(!admits(false, true, false, 8, MIN), "row off");
    assert!(!admits(true, false, false, 8, MIN), "no FP8 overlay");
    assert!(!admits(true, true, true, 8, MIN), "packed Q2_0");
    assert!(!admits(false, true, false, 24, MIN), "row off, 32-row tile");
    assert!(
        !admits(true, false, false, 24, MIN),
        "no overlay, 32-row tile"
    );
    for rows in [0, 1, 2, 3, 4, MAX_ROWS + 1, 33, 64] {
        assert!(!admits(true, true, false, rows, MIN), "rows={rows}");
    }
}

#[test]
fn the_lower_bound_moves_down_to_two_rows_and_never_to_one() {
    for rows in 2..=MAX_ROWS {
        assert!(admits(true, true, false, rows, 2), "rows={rows}");
    }
    // Lowering the bound never raises the ceiling.
    assert!(!admits(true, true, false, MAX_ROWS + 1, 2));
    // One row is the M=1 decode forward whatever the knob says.
    for min in [0usize, 1, 2] {
        assert!(!admits(true, true, false, 1, min), "min={min}");
    }
    assert!(!admits(true, true, false, 8, 9), "below a raised bound");
}

#[test]
fn sixteen_rows_are_one_pass_per_weight_into_moe_output() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, false, |ctx, buffers| {
        let x = buffers.norm_output();
        let start = gpu.launch_count();
        assert!(
            layer
                .fp8_m16_tc_arm_with(true, MIN, &kernels(), x, 16, ctx, 3)
                .unwrap()
        );
        // Gate and up in one launch, the SwiGLU product, then down: three
        // launches, each weight read once.
        assert_eq!(funcs(&gpu, start), vec![DUAL_K.0, ACT_MUL.0, PLAIN_K.0]);
        let l = gpu.launches_snapshot();
        assert_eq!(l[start].grid, [INTER.div_ceil(N_TILE), 1, 2]);
        assert_eq!(l[start].block, [THREADS, 1, 1]);
        assert_eq!(
            l[start + 1].grid[0],
            (16 * INTER).div_ceil(256),
            "SwiGLU over all 16 rows"
        );
        assert_eq!(l[start + 2].grid, [H.div_ceil(N_TILE), 1, 1]);
        assert_eq!(l[start + 2].block, [THREADS, 1, 1]);
        assert!(l[start].args.contains(&MockArg::Buffer(x)));
        assert!(
            l[start + 2]
                .args
                .contains(&MockArg::Buffer(buffers.moe_output())),
            "the result lands where every small-batch entry point writes"
        );
        // The kernel takes the row count as an argument: (M, N, K) close the
        // argument list of both projections.
        let tail = |i: usize| l[i].args[l[i].args.len() - 3..].to_vec();
        assert_eq!(tail(start), vec![u32_arg(16), u32_arg(INTER), u32_arg(H)]);
        assert_eq!(
            tail(start + 2),
            vec![u32_arg(16), u32_arg(H), u32_arg(INTER)]
        );
        assert!(l[start..].iter().all(|l| l.stream == 3));
    });
}

#[test]
fn every_width_from_five_to_thirty_two_takes_the_arm_and_the_edges_decline() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let x = buffers.norm_output();
        for rows in MIN..=MAX_ROWS {
            let start = gpu.launch_count();
            assert!(
                layer
                    .fp8_m16_tc_arm_with(true, MIN, &kernels(), x, rows, ctx, 0)
                    .unwrap(),
                "rows={rows}"
            );
            assert_eq!(gpu.launch_count() - start, 3, "rows={rows}");
        }
        for rows in [1usize, 2, 4, MAX_ROWS + 1, 33, 64] {
            let start = gpu.launch_count();
            assert!(
                !layer
                    .fp8_m16_tc_arm_with(true, MIN, &kernels(), x, rows, ctx, 0)
                    .unwrap(),
                "rows={rows}"
            );
            assert_eq!(gpu.launch_count(), start, "rows={rows}");
        }
    });
}

#[test]
fn decode_steps_and_verify_passes_take_the_same_launches() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    let mut seen = Vec::new();
    for decode_step in [true, false] {
        with_ctx(&gpu, decode_step, |ctx, buffers| {
            let start = gpu.launch_count();
            assert!(
                layer
                    .fp8_m16_tc_arm_with(true, MIN, &kernels(), buffers.norm_output(), 8, ctx, 0)
                    .unwrap()
            );
            seen.push(funcs(&gpu, start));
        });
    }
    assert_eq!(seen[0], seen[1]);
}

#[test]
fn the_row_off_declines_and_launches_nothing() {
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !layer
                .fp8_m16_tc_arm_with(false, MIN, &kernels(), buffers.norm_output(), 8, ctx, 0)
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
                .fp8_m16_tc_arm_with(true, MIN, &kernels(), buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn a_missing_handle_or_activation_kernel_declines_instead_of_mixing_kernels() {
    let gpu = MockGpuBackend::new();
    let no_dual = TcKernels {
        dual: KernelHandle(0),
        ..kernels()
    };
    let no_plain = TcKernels {
        plain: KernelHandle(0),
        ..kernels()
    };
    let no_m32 = [
        TcKernels {
            plain32: KernelHandle(0),
            ..kernels()
        },
        TcKernels {
            dual32: KernelHandle(0),
            ..kernels()
        },
    ];
    let mut no_act = fp8_layer(&gpu);
    no_act.act_mul = KernelHandle(0);
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let x = buffers.norm_output();
        let start = gpu.launch_count();
        for k in [no_dual, no_plain] {
            assert!(
                !layer
                    .fp8_m16_tc_arm_with(true, MIN, &k, x, 8, ctx, 0)
                    .unwrap()
            );
        }
        assert!(
            !no_act
                .fp8_m16_tc_arm_with(true, MIN, &kernels(), x, 8, ctx, 0)
                .unwrap()
        );
        // A missing 32-row handle declines a 17..32 row call and leaves a
        // 16-row one alone (and the reverse for a missing 16-row handle).
        for k in no_m32 {
            assert!(
                !layer
                    .fp8_m16_tc_arm_with(true, MIN, &k, x, 24, ctx, 0)
                    .unwrap()
            );
        }
        for k in [no_dual, no_plain] {
            assert!(
                !layer
                    .fp8_m16_tc_arm_with(true, MIN, &k, x, 16, ctx, 0)
                    .unwrap()
            );
            assert!(
                layer
                    .fp8_m16_tc_arm_with(true, MIN, &k, x, 24, ctx, 0)
                    .unwrap()
            );
        }
        assert_eq!(
            gpu.launch_count() - start,
            6,
            "two 24-row calls launch 3 each"
        );
    });
}

#[test]
fn a_reduction_that_is_not_whole_scale_blocks_declines() {
    // The kernel folds one scale per 128 K; hidden 320 has a partial block.
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer_at(&gpu, 320, INTER);
    with_ctx_at(&gpu, 320, INTER, true, |ctx, buffers| {
        let start = gpu.launch_count();
        assert!(
            !layer
                .fp8_m16_tc_arm_with(true, MIN, &kernels(), buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn with_the_row_off_the_process_entry_points_issue_no_lookup_and_no_launch() {
    // The default environment resolves every port row off; this test is
    // meaningful only then (an operator running the suite with a row exported
    // is testing the other polarity on purpose).
    if ops::port_rows::ffn_fp8_m16_tc::row()
        || ops::port_rows::ffn_fp8_decode_gemv::row()
        || ops::port_rows::ffn_fp8_verify_gemv::row()
    {
        return;
    }
    let gpu = MockGpuBackend::new();
    let layer = fp8_layer(&gpu);
    with_ctx(&gpu, true, |ctx, buffers| {
        let x = buffers.norm_output();
        let lookups = gpu.kernel_lookups_snapshot().len();
        let start = gpu.launch_count();
        assert!(!layer.fp8_m16_tc_arm(x, 8, ctx, 0).unwrap());
        // The hook every small-batch entry point tries first (`w4.rs`).
        assert!(!layer.fp8_verify_arm(x, 8, ctx, 0).unwrap());
        assert_eq!(gpu.launch_count(), start);
        assert_eq!(gpu.kernel_lookups_snapshot().len(), lookups);
        // And the wide arms' question, which then keep `forward_prefill`.
        let dense = FfnComponent::Dense(fp8_layer(&gpu));
        let lookups = gpu.kernel_lookups_snapshot().len();
        assert!(!dense.try_forward_fp8_m16_tc(x, 16, ctx, 0).unwrap());
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
                .try_forward_fp8_m16_tc(buffers.norm_output(), 8, ctx, 0)
                .unwrap()
        );
        assert_eq!(gpu.launch_count(), start);
    });
}

#[test]
fn a_target_without_the_module_has_no_handles_and_issues_no_lookup() {
    let gpu = MockGpuBackend::new();
    gpu.mark_module_absent(MODULE);
    let before = gpu.kernel_lookups_snapshot().len();
    assert!(TcKernels::cached(&gpu).is_none());
    prime(&gpu);
    assert_eq!(gpu.kernel_lookups_snapshot().len(), before);
}

#[test]
fn priming_looks_all_four_entries_up_once_and_the_forward_read_is_a_cache_hit() {
    let gpu = MockGpuBackend::new();
    prime(&gpu);
    let seen = gpu.kernel_lookups_snapshot();
    for (module, entry) in [PLAIN, DUAL, PLAIN32, DUAL32] {
        assert!(
            seen.contains(&(module.to_owned(), entry.to_owned())),
            "{entry}"
        );
    }
    let k = TcKernels::cached(&gpu).expect("the mock carries every module");
    assert!(k.plain.0 != 0 && k.dual.0 != 0 && k.plain32.0 != 0 && k.dual32.0 != 0);
    assert_eq!(gpu.kernel_lookups_snapshot().len(), seen.len());
}

#[test]
fn both_entries_are_declared_by_the_ada_source() {
    // The module name is the .cu stem and the entry is an `extern "C"` name in
    // it. This catches a rename on either side without needing nvcc.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/ada/common");
    let src = std::fs::read_to_string(root.join(format!("{MODULE}.cu")))
        .unwrap_or_else(|e| panic!("kernels/ada/common/{MODULE}.cu: {e}"));
    for (_, entry) in [PLAIN, DUAL, PLAIN32, DUAL32] {
        assert!(
            src.contains(&format!(" {entry}(")),
            "kernels/ada/common/{MODULE}.cu declares no `{entry}`"
        );
    }
}

#[path = "fp8_m16_tc_m32_tests.rs"]
mod m32;
