// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch-selection contract for the tensor-core 5..=32-row native-FP8
//! dense-FFN decode tier (`ATLAS_FFN_M16_TC`, #927). CPU tests on the mock
//! backend: they pin WHICH arm each row count takes with the lever on and off,
//! that the tier sits AHEAD of the bit-exact `w8a16_gemv_batch16` rung, the row
//! split and byte offsets of the two-launch rung, and the launch geometry
//! (`ceil(N/32)` CTAs of 128 threads, not the GEMV's `ceil(N/4)` of 256).
//! Numerics — max ULP vs the scalar `w8a16_gemv` at the real Qwen3.8-27B
//! shapes — are the GPU oracle's job
//! (`examples/native_fp8_ffn_m16_tc_microtest.rs`).

use super::{M16TcPlan, m16_tc_plan};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::ops::{W8A16_GEMM_M16_N_TILE, W8A16_GEMM_M16_N_TILE_WIDE};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Distinct per-arm handles: the mock hands the SAME placeholder to every
/// kernel name, so the handle is the only thing that separates the rungs.
const BATCH4_K: u64 = 0xB004;
const BATCH16_K: u64 = 0xB016;
const M16TC_K: u64 = 0x167C;
/// The `N_TILE=64` twin (`ATLAS_FFN_M16_TC_NTILE=64`).
const M16TC_N64_K: u64 = 0x1640;
/// Hidden/intermediate width for the mock layer. 128 is the smallest value that
/// is a whole 128-wide FP8 scale block, which is what the tier's K guard wants.
const WIDTH: u32 = 128;

// ── The pure rule ──────────────────────────────────────────────────────────

#[test]
fn m16_tc_is_off_unless_the_lever_is_set() {
    for m in [5, 8, 16, 17, 32] {
        assert_eq!(
            m16_tc_plan(m, 5120, true, false),
            None,
            "m={m}: default must stay on the bit-exact batch16 tier"
        );
    }
}

#[test]
fn m16_tc_declines_the_rows_the_batch4_rung_owns() {
    for m in [1, 2, 3, 4] {
        assert_eq!(
            m16_tc_plan(m, 5120, true, true),
            None,
            "m={m} belongs to batch4"
        );
    }
}

#[test]
fn m16_tc_claims_five_to_sixteen_in_one_launch() {
    for m in [5, 6, 8, 13, 15, 16] {
        assert_eq!(
            m16_tc_plan(m, 5120, true, true),
            Some(M16TcPlan::Single),
            "m={m} must be one weight pass"
        );
    }
}

#[test]
fn m16_tc_splits_seventeen_to_thirtytwo_into_halves_that_fit_the_m_tile() {
    for m in 17..=32u32 {
        let Some(M16TcPlan::Halves { first }) = m16_tc_plan(m, 5120, true, true) else {
            panic!("m={m} must split into halves");
        };
        assert_eq!(
            first,
            m.div_ceil(2),
            "m={m}: odd row goes to the first half"
        );
        assert!(first <= 16, "m={m}: first half {first} exceeds the M tile");
        assert!(m - first <= 16, "m={m}: second half exceeds the M tile");
        assert_eq!(first + (m - first), m, "m={m}: halves must cover every row");
    }
}

#[test]
fn m16_tc_declines_prefill_widths_above_thirtytwo() {
    for m in [33, 64, 128, 1193] {
        assert_eq!(
            m16_tc_plan(m, 5120, true, true),
            None,
            "m={m} is a prefill width"
        );
    }
}

/// The kernel folds `block_scale[n_block * (K/128) + k/128]`, so a K that is
/// not a whole number of 128-wide scale blocks has no correct scale and the
/// tier must decline rather than launch.
#[test]
fn m16_tc_declines_a_k_that_is_not_whole_scale_blocks() {
    for k in [1, 64, 127, 129, 5121, 17407] {
        assert_eq!(
            m16_tc_plan(8, k, true, true),
            None,
            "k={k} is a partial scale block"
        );
    }
    for k in [128, 5120, 17408] {
        assert_eq!(
            m16_tc_plan(8, k, true, true),
            Some(M16TcPlan::Single),
            "k={k} is whole scale blocks"
        );
    }
}

#[test]
fn m16_tc_declines_when_the_kernel_is_absent() {
    for m in [5, 8, 16, 17, 32] {
        assert_eq!(
            m16_tc_plan(m, 5120, false, true),
            None,
            "m={m} without a handle"
        );
    }
}

// ── The dispatch it produces ───────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Expect {
    /// One launch on the given handle with this row count in arg 4.
    One(u64, u32),
    /// Two launches on the given handle with these row counts, the second
    /// offset by `first` rows on both the input and the output.
    Halves(u64, u32, u32),
}

fn run(m: u32, expect: Expect, configure: impl FnOnce(&mut DenseFfnLayer)) {
    run_tiled(m, expect, W8A16_GEMM_M16_N_TILE, configure);
}

/// `tile` is the CTA N width the tensor-core arm is expected to launch with —
/// the only thing `ATLAS_FFN_M16_TC_NTILE` changes, and therefore the only
/// thing worth asserting about it.
fn run_tiled(m: u32, expect: Expect, tile: u32, configure: impl FnOnce(&mut DenseFfnLayer)) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = WIDTH as usize;
    config.intermediate_size = WIDTH as usize;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = WIDTH as usize;
    config.vocab_size = WIDTH as usize;
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, &gpu).unwrap();
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
    layer.w8a16_gemv_batch4_k = KernelHandle(BATCH4_K);
    layer.w8a16_gemv_batch16_k = KernelHandle(BATCH16_K);
    // The batch16 tier is rungs 4-5 of the ladder these tests grade, so it has
    // to be armed for "the M16 lever is unset" to mean "falls through to
    // batch16" rather than "falls through to the tile GEMMs". Per-target since
    // the `[defaults]` table landed; set explicitly for the same reason
    // `lever_on` sets `m16_tc`.
    layer.batch16_tier = true;
    layer.w8a16_gemm_m16_k = KernelHandle(M16TC_K);
    layer.w8a16_gemm_m16_n64_k = KernelHandle(M16TC_N64_K);
    layer.act_mul = KernelHandle(0xAC7);
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: WIDTH,
        k: WIDTH,
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
        decode_step: false,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let start = gpu.launch_count();
    layer
        .forward_prefill(buffers.norm_output(), m as usize, &ctx, 7)
        .unwrap();
    let all = gpu.launches_snapshot();
    let projections: Vec<_> = all[start..]
        .iter()
        .filter(|l| l.func != layer.act_mul.0)
        .collect();
    let per_proj = match expect {
        Expect::Halves(..) => 2,
        Expect::One(..) => 1,
    };
    assert_eq!(
        projections.len(),
        3 * per_proj,
        "m={m}: gate/up/down must each emit {per_proj} launch(es)"
    );
    for (i, launch) in projections.iter().enumerate() {
        assert_eq!(launch.stream, 7);
        match expect {
            Expect::One(handle, want_m) => {
                assert_eq!(launch.func, handle, "m={m}: wrong arm");
                assert_eq!(
                    launch.args[4],
                    MockArg::Bytes(want_m.to_ne_bytes().to_vec())
                );
                if handle == M16TC_K || handle == M16TC_N64_K {
                    // ceil(N/tile) CTAs of 128 threads — NOT the GEMV geometry.
                    assert_eq!(launch.grid, [WIDTH.div_ceil(tile), 1, 1]);
                    assert_eq!(launch.block, [128, 1, 1]);
                }
            }
            Expect::Halves(handle, first, second) => {
                assert_eq!(launch.func, handle, "m={m}: both halves take one arm");
                let want = if i % 2 == 0 { first } else { second };
                assert_eq!(
                    launch.args[4],
                    MockArg::Bytes(want.to_ne_bytes().to_vec()),
                    "m={m}: half {i} row count"
                );
                if handle == M16TC_K || handle == M16TC_N64_K {
                    assert_eq!(launch.grid, [WIDTH.div_ceil(tile), 1, 1]);
                    assert_eq!(launch.block, [128, 1, 1]);
                }
                // THE OFFSET the round-6 `gate/up M=32` cell was suspected of
                // (it was not — `dense_ffn_m16_tc_m32_tests.rs`): the second
                // half must start `first` rows into BOTH the activation buffer
                // (pitch K) and the output (pitch N). Asserted relative to the
                // first half's pointers, which is the whole invariant.
                if i % 2 == 1 {
                    // This mock layer is square (hidden = intermediate = WIDTH),
                    // so both pitches are WIDTH BF16 elements.
                    let rows = first as usize * WIDTH as usize * 2;
                    let lo = projections[i - 1];
                    assert_eq!(
                        launch.args[0],
                        shifted(&lo.args[0], rows),
                        "m={m}: second half reads the wrong activation rows"
                    );
                    assert_eq!(
                        launch.args[3],
                        shifted(&lo.args[3], rows),
                        "m={m}: second half writes the wrong output rows"
                    );
                }
            }
        }
    }
}

/// A launch argument's buffer, advanced by `bytes` — so a halves assertion can
/// be written against the first half's pointer rather than a base the test
/// would have to reconstruct.
fn shifted(arg: &MockArg, bytes: usize) -> MockArg {
    match arg {
        MockArg::Buffer(p) => MockArg::Buffer(p.offset(bytes)),
        other => panic!("expected a buffer argument, got {other:?}"),
    }
}

/// The lever is read ONCE into `DenseFfnLayer::m16_tc` at construction, so the
/// dispatch tests set that field instead of the environment — the env accessor
/// is a process-global `OnceLock` and a test that set the variable would leak
/// into every other test in this binary.
fn lever_on(layer: &mut DenseFfnLayer) {
    layer.m16_tc = true;
}

#[test]
fn with_the_lever_unset_five_to_thirtytwo_stay_on_the_bit_exact_batch16_tier() {
    for m in [5, 8, 16] {
        run(m, Expect::One(BATCH16_K, m), |_| {});
    }
    run(17, Expect::Halves(BATCH16_K, 9, 8), |_| {});
    run(32, Expect::Halves(BATCH16_K, 16, 16), |_| {});
}

#[test]
fn with_the_lever_set_five_to_sixteen_take_one_tensor_core_launch() {
    for m in [5, 8, 13, 16] {
        run(m, Expect::One(M16TC_K, m), lever_on);
    }
}

#[test]
fn with_the_lever_set_seventeen_to_thirtytwo_take_two_tensor_core_launches() {
    run(17, Expect::Halves(M16TC_K, 9, 8), lever_on);
    run(32, Expect::Halves(M16TC_K, 16, 16), lever_on);
}

#[test]
fn four_rows_and_under_keep_the_batch4_arm_either_way() {
    for m in [1, 4] {
        run(m, Expect::One(BATCH4_K, m), |_| {});
        run(m, Expect::One(BATCH4_K, m), lever_on);
    }
}

/// A model shadow without the entry point must land exactly where it did
/// before this tier existed, not on a zero handle — even with the lever set.
#[test]
fn the_tier_is_inert_without_its_handle() {
    for m in [5, 16] {
        run(m, Expect::One(BATCH16_K, m), |layer| {
            lever_on(layer);
            layer.w8a16_gemm_m16_k = KernelHandle(0);
        });
    }
    run(32, Expect::Halves(BATCH16_K, 16, 16), |layer| {
        lever_on(layer);
        layer.w8a16_gemm_m16_k = KernelHandle(0);
    });
}

/// The tier sits AHEAD of the batch16 rung, so with the lever set the
/// bit-exact GEMV must not be reached at all in 5..=32.
#[test]
fn the_tier_takes_precedence_over_the_batch16_rung() {
    for m in [5, 16, 32] {
        let expect = if m <= 16 {
            Expect::One(M16TC_K, m)
        } else {
            Expect::Halves(M16TC_K, 16, 16)
        };
        run(m, expect, lever_on);
    }
}

/// `ATLAS_FFN_M16_TC_NTILE=64` is a pure geometry swap: same arm, same rungs,
/// same row counts and offsets, half the CTAs.
#[test]
fn the_wide_tile_halves_the_cta_count_on_both_rungs() {
    let wide = |layer: &mut DenseFfnLayer| {
        lever_on(layer);
        layer.m16_tc_n_tile = W8A16_GEMM_M16_N_TILE_WIDE;
    };
    for m in [5, 8, 16] {
        run_tiled(
            m,
            Expect::One(M16TC_N64_K, m),
            W8A16_GEMM_M16_N_TILE_WIDE,
            wide,
        );
    }
    run_tiled(
        32,
        Expect::Halves(M16TC_N64_K, 16, 16),
        W8A16_GEMM_M16_N_TILE_WIDE,
        wide,
    );
}

/// A shadow without the wide entry point keeps the 32-wide kernel AND its
/// geometry — it must not launch a zero handle or a 64-wide grid.
#[test]
fn the_wide_tile_falls_back_to_the_default_arm_without_its_entry_point() {
    run_tiled(
        16,
        Expect::One(M16TC_K, 16),
        W8A16_GEMM_M16_N_TILE,
        |layer| {
            lever_on(layer);
            layer.m16_tc_n_tile = W8A16_GEMM_M16_N_TILE_WIDE;
            layer.w8a16_gemm_m16_n64_k = KernelHandle(0);
        },
    );
}

/// The tile lever is inert on its own: without `ATLAS_FFN_M16_TC` the arm is
/// not reached at all, so 5..=32 stay on the bit-exact batch16 tier.
#[test]
fn the_wide_tile_does_not_turn_the_tier_on_by_itself() {
    run(16, Expect::One(BATCH16_K, 16), |layer| {
        layer.m16_tc_n_tile = W8A16_GEMM_M16_N_TILE_WIDE;
    });
}
