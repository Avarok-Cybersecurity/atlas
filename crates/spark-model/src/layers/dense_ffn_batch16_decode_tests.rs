// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch-selection contract for the 5..=32-row native-FP8 dense-FFN decode
//! tier (#927). CPU tests on the mock backend: they pin WHICH arm each row
//! count takes and, for the two-launch rung, the row split and the byte
//! offsets it derives. Bit-exactness against the scalar `w8a16_gemv` at the
//! real Qwen3.8-27B shapes is the GPU oracle's job
//! (`examples/native_fp8_ffn_batch16_microtest.rs`).

use super::{Batch16Plan, batch16_plan};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Distinct per-arm handles: the mock's kernel lookup hands the SAME
/// placeholder to every name, and `w8a16_gemv_batch4` and `w8a16_gemv_batch16`
/// launch with IDENTICAL grid and block (they are one template), so the handle
/// is the only thing that can tell the two rungs apart.
const BATCH4_K: u64 = 0xB004;
const BATCH16_K: u64 = 0xB016;
const WIDTH: u32 = 128;

// ── The pure rule ──────────────────────────────────────────────────────────

#[test]
fn batch16_declines_the_rows_the_batch4_rung_owns() {
    for m in [1, 2, 3, 4] {
        assert_eq!(
            batch16_plan(m, true, false),
            None,
            "m={m} belongs to batch4"
        );
    }
}

#[test]
fn batch16_claims_five_to_sixteen_in_one_launch() {
    for m in [5, 6, 8, 12, 15, 16] {
        assert_eq!(
            batch16_plan(m, true, false),
            Some(Batch16Plan::Single),
            "m={m} must be one weight pass"
        );
    }
}

#[test]
fn batch16_splits_seventeen_to_thirtytwo_into_halves_that_fit_max_m() {
    for m in 17..=32u32 {
        let Some(Batch16Plan::Halves { first }) = batch16_plan(m, true, false) else {
            panic!("m={m} must split into halves");
        };
        assert_eq!(
            first,
            m.div_ceil(2),
            "m={m}: odd row goes to the first half"
        );
        assert!(
            first <= 16,
            "m={m}: first half {first} exceeds kernel MAX_M"
        );
        assert!(
            m - first <= 16,
            "m={m}: second half {} exceeds kernel MAX_M",
            m - first
        );
        assert_eq!(first + (m - first), m, "m={m}: halves must cover every row");
    }
}

#[test]
fn batch16_declines_prefill_widths_above_thirtytwo() {
    for m in [33, 64, 128, 1193] {
        assert_eq!(
            batch16_plan(m, true, false),
            None,
            "m={m} is a prefill width"
        );
    }
}

#[test]
fn batch16_declines_when_the_kernel_is_absent() {
    // A model shadow that does not carry the MAX_M=16 entry point must land
    // exactly where it did before #927, not on a zero handle.
    for m in [5, 8, 16, 17, 32] {
        assert_eq!(
            batch16_plan(m, false, false),
            None,
            "m={m} without a handle"
        );
    }
}

/// `ATLAS_FFN_NO_BATCH16` — injected, not read from the environment: the real
/// accessor is a process-global `OnceLock` and a test that set the variable
/// would leak into every other test in this binary.
#[test]
fn batch16_kill_switch_restores_the_previous_routing() {
    for m in [5, 8, 16, 17, 32] {
        assert_eq!(
            batch16_plan(m, true, true),
            None,
            "m={m} with the switch set"
        );
    }
}

// ── The dispatch it produces ───────────────────────────────────────────────

/// What `forward_prefill` is expected to emit for EACH of gate/up/down.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Expect {
    /// One launch on the given handle, with this row count in arg 4.
    One(u64, u32),
    /// Two launches on the batch16 handle with these row counts, the second
    /// offset by `first` rows on both the input and the output.
    Halves(u32, u32),
    /// The M-padded tile GEMM (one launch, `grid.y = ceil(m/128)`).
    Tile,
}

fn run(m: u32, expect: Expect, configure: impl FnOnce(&mut DenseFfnLayer)) {
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
    // The TIER is armed per target (`[defaults] ffn_batch16_tier`) and this
    // binary is built for whichever tree `ATLAS_TARGET_HW` selected, so the
    // dispatch tests set the field rather than depend on the build's target —
    // the same reason the M16 tests set `m16_tc`. What is graded here is the
    // ROUTE an armed tier takes; that the targets in tree declare it off is
    // `target_defaults_tests` and `atlas-kernels/tests/target_defaults.rs`.
    layer.batch16_tier = true;
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
        // The FFN under test never reads the decode scalars this flag guards.
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
    layer
        .forward_prefill(buffers.norm_output(), m as usize, &ctx, 7)
        .unwrap();
    assert_eq!(
        gpu.alloc_count(),
        allocs,
        "m={m}: dispatch must not allocate weight copies"
    );
    let all = gpu.launches_snapshot();
    let projections: Vec<_> = all[start..]
        .iter()
        .filter(|l| l.func != layer.act_mul.0)
        .collect();
    let per_proj = match expect {
        Expect::Halves(..) => 2,
        _ => 1,
    };
    assert_eq!(
        projections.len(),
        3 * per_proj,
        "m={m}: gate/up/down must each emit {per_proj} launch(es)"
    );
    let rows = |arg: &MockArg| *arg == MockArg::Bytes(m.to_ne_bytes().to_vec());
    for (i, launch) in projections.iter().enumerate() {
        assert_eq!(launch.stream, 7);
        match expect {
            Expect::One(handle, want_m) => {
                assert_eq!(launch.func, handle, "m={m}: wrong arm");
                assert_eq!(launch.grid, [WIDTH.div_ceil(4), 1, 1]);
                assert_eq!(launch.block, [256, 1, 1]);
                assert_eq!(
                    launch.args[4],
                    MockArg::Bytes(want_m.to_ne_bytes().to_vec())
                );
                assert_eq!(launch.args[1], MockArg::Buffer(fp8.weight));
            }
            Expect::Halves(first, second) => {
                assert_eq!(launch.func, BATCH16_K, "m={m}: both halves are batch16");
                let want = if i % 2 == 0 { first } else { second };
                assert_eq!(
                    launch.args[4],
                    MockArg::Bytes(want.to_ne_bytes().to_vec()),
                    "m={m}: half {i} row count"
                );
                assert_eq!(launch.grid, [WIDTH.div_ceil(4), 1, 1]);
            }
            Expect::Tile => {
                assert_ne!(launch.func, BATCH4_K, "m={m}: must not reach batch4");
                assert_ne!(launch.func, BATCH16_K, "m={m}: must not reach batch16");
                assert_eq!(launch.grid[1], m.div_ceil(128), "m={m}: M-padded tile");
                assert!(rows(&launch.args[4]), "m={m}: tile GEMM takes all rows");
            }
        }
    }
}

#[test]
fn four_rows_and_under_keep_the_batch4_arm() {
    for m in [1, 4] {
        run(m, Expect::One(BATCH4_K, m), |_| {});
    }
}

#[test]
fn five_to_sixteen_rows_take_one_batch16_launch() {
    for m in [5, 8, 16] {
        run(m, Expect::One(BATCH16_K, m), |_| {});
    }
}

#[test]
fn seventeen_to_thirtytwo_rows_take_two_batch16_launches() {
    run(17, Expect::Halves(9, 8), |_| {});
    run(32, Expect::Halves(16, 16), |_| {});
}

#[test]
fn prefill_widths_still_reach_the_tile_gemm() {
    run(64, Expect::Tile, |_| {});
}

#[test]
fn without_the_batch16_handle_the_cliff_widths_fall_back_as_before() {
    for m in [5, 8, 16, 17, 32] {
        run(m, Expect::Tile, |layer| {
            layer.w8a16_gemv_batch16_k = KernelHandle(0);
        });
    }
}
