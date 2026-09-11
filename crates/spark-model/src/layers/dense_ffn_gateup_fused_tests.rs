// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch-selection and allocation contract for the FUSED dense-FFN gate+up
//! decode GEMM (#927). CPU tests: they pin WHICH arm the native-FP8 path picks
//! at each width and that the arm allocates nothing per call. The numerics are
//! the GPU microtest's job
//! (`examples/native_fp8_ffn_gateup_fused_microtest.rs`), which asserts BYTE
//! equality of the two halves rather than a tolerance.

use super::{fused_out_bytes, gateup_fused_selected};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{self, DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::{BufferArena, GATEUP_FUSED_MAX_M};
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Qwen3.8-27B dense FFN — the shapes the round-13 trace priced.
const H: u32 = 5120;
const INTER: u32 = 17408;

/// Every clause defaulted to its SELECTING value, so each case perturbs
/// exactly one thing and a failure names the clause.
fn selected(m: u32) -> bool {
    gateup_fused_selected(m, INTER, true, true, true, true, usize::MAX)
}

/// The band. 5 because rung 1 (`w8a16_gemv_batch4`) already makes ONE weight
/// pass at m <= 4 — there is no second launch to fuse — and because the W8A8
/// rule this arm rides on starts at `m > 4`. 16 because above it the saving
/// stops: the same two GEMMs run at 68.6% of FP8 PEAK at prefill widths, where
/// a launch costs nothing measurable.
#[test]
fn the_fused_arm_claims_exactly_the_five_to_sixteen_row_decode_band() {
    for m in [5_u32, 6, 8, 12, 15, 16] {
        assert!(selected(m), "m={m} is inside the decode band");
    }
    for m in [1_u32, 2, 4] {
        assert!(!selected(m), "m={m} belongs to the batch4 GEMV tier");
    }
    for m in [17_u32, 25, 32, 1168, 4576] {
        assert!(
            !selected(m),
            "m={m} is a prefill width, where the arm is inert"
        );
    }
    assert_eq!(
        GATEUP_FUSED_MAX_M, 16,
        "the band's upper edge and the arena buffer's row extent are ONE \
         constant; changing it here without the arena is a cross-buffer write"
    );
}

/// `ATLAS_FFN_GATEUP_FUSED=0`, and every target but hopper.
#[test]
fn the_lever_off_declines_at_every_width() {
    for m in 1_u32..=32 {
        assert!(!gateup_fused_selected(
            m,
            INTER,
            false,
            true,
            true,
            true,
            usize::MAX
        ));
    }
}

/// The arm IS the W8A8 arm with a wider N. If the ladder would have put either
/// half on a W8A16 rung, fusing would change that half's ARITHMETIC and not
/// only its launch count — so the fused arm must decline, not "win the race".
#[test]
fn the_w8a8_arm_must_have_claimed_both_halves() {
    assert!(!gateup_fused_selected(
        8,
        INTER,
        true,
        false,
        true,
        true,
        usize::MAX
    ));
}

/// The two runtime absences: a checkpoint or route that built no fused weight,
/// and a hardware tree without the strided SiLU consumer. Either one declines
/// rather than launching something with a null pointer or the wrong stride.
#[test]
fn a_missing_weight_or_a_missing_consumer_declines() {
    assert!(!gateup_fused_selected(
        8,
        INTER,
        true,
        true,
        false,
        true,
        usize::MAX
    ));
    assert!(!gateup_fused_selected(
        8,
        INTER,
        true,
        true,
        true,
        false,
        usize::MAX
    ));
}

/// The output-capacity clause is a GATE and not an assert, for the reason the
/// null-scratch check in `prefill_w8a8_selected` is one: the cuBLASLt arm
/// WRITES `ceil16(m)` rows, so a buffer one row short is a cross-buffer write,
/// and declining is always sound.
#[test]
fn an_output_buffer_short_of_the_padded_extent_declines() {
    let need = fused_out_bytes(16, INTER);
    assert_eq!(need, 16 * 2 * INTER as usize * 2, "ceil16(16) == 16");
    assert!(gateup_fused_selected(
        16, INTER, true, true, true, true, need
    ));
    assert!(!gateup_fused_selected(
        16,
        INTER,
        true,
        true,
        true,
        true,
        need - 1
    ));
    // m=5 still pays for 16 rows: cuBLASLt is handed the PADDED M.
    assert_eq!(fused_out_bytes(5, INTER), need);
}

// ── the layer, against a mock backend ──

struct Harness {
    gpu: MockGpuBackend,
    layer: DenseFfnLayer,
    buffers: BufferArena,
    config: ModelConfig,
    fp8: Fp8Weight,
}

/// A dense config at the real FFN widths with the fused arm ARMED — the
/// handles and the weight are installed directly rather than resolved from the
/// process-global lever, which a CPU test cannot toggle.
fn harness(armed: bool) -> Harness {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = 0;
    config.num_experts_per_tok = 0;
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, 64, 256, 256, 8, &gpu).unwrap();
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
    layer.per_token_group_quant_fp8_k = KernelHandle(0xA8A);
    layer.fp8_gemm_t_blockscaled_k = KernelHandle(0xA88);
    // The cuBLASLt arm is unreachable from a mock backend (it is an FFI call),
    // so zero the k-major adapter and let `w8a8_gemm` take the in-tree kernel.
    // The fused arm's own rule does not branch on which GEMM runs.
    layer.fp8_act_scale_kmajor_k = KernelHandle(0);
    layer.gateup_fused = armed;
    layer.silu_mul_strided_k = if armed {
        KernelHandle(0x5171)
    } else {
        KernelHandle(0)
    };
    // One fused allocation, with gate and up as views inside it — the loader's
    // contract, reproduced so `set_fp8_gate_up_fused`'s debug assert holds.
    let fused_w = gpu.alloc(2 * INTER as usize * H as usize).unwrap();
    let grid = (INTER as usize / 128) * (H as usize / 128) * 4;
    let fused_s = gpu.alloc(2 * grid).unwrap();
    let view = |off_w: usize, off_s: usize| Fp8Weight {
        weight: fused_w.offset(off_w),
        row_scale: fused_s.offset(off_s),
        n: INTER,
        k: H,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    let fp8 = view(0, 0);
    layer.set_fp8_weights(
        fp8,
        view(INTER as usize * H as usize, grid),
        Fp8Weight {
            weight: fused_w,
            row_scale: fused_s,
            n: H,
            k: INTER,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        },
    );
    if armed {
        layer.set_fp8_gate_up_fused(Fp8Weight {
            weight: fused_w,
            row_scale: fused_s,
            n: 2 * INTER,
            k: H,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        });
    }
    Harness {
        gpu,
        layer,
        buffers,
        config,
        fp8,
    }
}

fn ctx<'a>(
    h: &'a Harness,
    dispatch: &'a GemmDispatch,
    derived: &'a DerivedWeights,
    levers: &'a ModelLevers,
    stats: &'a ModelStats,
) -> ForwardContext<'a> {
    ForwardContext {
        dispatch,
        derived,
        levers,
        stats,
        buffers: &h.buffers,
        hc_row_offset: 0,
        gpu: &h.gpu,
        config: &h.config,
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
    }
}

/// GeLU keeps the two-GEMM pair. The fused arm launches the SiLU consumer
/// itself, so an activation it does not implement must decline rather than be
/// applied silently — this is the clause that is invisible in the pure rule
/// because it is a property of the LAYER.
#[test]
fn a_gelu_layer_never_takes_the_fused_arm() {
    let mut h = harness(true);
    h.layer.activation = crate::layers::dense_ffn::FfnActivation::GeLU;
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    for m in [5_u32, 8, 16] {
        assert!(h.layer.gateup_fused_plan(&c, m, INTER, true).is_none());
    }
}

/// The arm is picked at the decode widths and nowhere else, read off the LAYER
/// rather than the pure rule — so the plan's wiring to the installed weight,
/// the resolved lever and the arena's capacity is pinned too.
#[test]
fn the_layer_plans_the_fused_arm_only_inside_the_band() {
    let h = harness(true);
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    for m in [5_u32, 8, 16] {
        assert!(h.layer.gateup_fused_plan(&c, m, INTER, true).is_some());
    }
    for m in [1_u32, 4, 17, 64] {
        assert!(h.layer.gateup_fused_plan(&c, m, INTER, true).is_none());
    }
    // Off-target: the same layer with the lever down never plans it.
    let off = harness(false);
    let c_off = ctx(&off, &d, &w, &l, &s);
    for m in [5_u32, 8, 16] {
        assert!(
            off.layer
                .gateup_fused_plan(&c_off, m, INTER, true)
                .is_none()
        );
    }
}

/// THE ALLOCATION CONTRACT, modelled on `qwen3_ssm/prefill_alloc_tests.rs`.
///
/// Taken BEFORE the first call and not between two: an allocation cached by
/// weight pointer would pass a first-vs-second comparison. What must be true
/// is that the arm allocates AT ALL — the arena already owns the fused output,
/// and a per-call `cuMemAlloc` inside a CUDA-graph capture is not a leak, it is
/// a capture failure.
#[test]
fn the_fused_arm_allocates_nothing_per_call() {
    let h = harness(true);
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    let fused = h
        .layer
        .gateup_fused_plan(&c, 16, INTER, true)
        .expect("armed at m=16");
    let (allocs, bytes) = (h.gpu.live_alloc_count(), h.gpu.live_bytes());
    let (a8, sc) = h
        .layer
        .w8a8_quant_act(&c, h.buffers.norm_output(), 16, H, 0)
        .expect("activation quant");
    for _ in 0..3 {
        h.layer
            .w8a8_gate_up_fused(
                &c,
                a8,
                sc,
                fused,
                h.buffers.expert_gate_out(),
                16,
                INTER,
                H,
                0,
            )
            .expect("fused gate+up");
    }
    assert_eq!(
        (h.gpu.live_alloc_count(), h.gpu.live_bytes()),
        (allocs, bytes),
        "the fused gate+up arm must allocate nothing — every operand is an \
         arena buffer or a weight view"
    );
}

/// The launch count IS the lever's whole claim: one GEMM plus one SiLU where
/// the pair issues two GEMMs plus one SiLU. Counted rather than argued,
/// because "one launch instead of two" is the entire 1 476 µs/step.
#[test]
fn the_fused_arm_issues_one_gemm_where_the_pair_issues_two() {
    let h = harness(true);
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    let fused = h.layer.gateup_fused_plan(&c, 16, INTER, true).unwrap();
    let (a8, sc) = h
        .layer
        .w8a8_quant_act(&c, h.buffers.norm_output(), 16, H, 0)
        .unwrap();
    let before = h.gpu.launch_count();
    h.layer
        .w8a8_gate_up_fused(
            &c,
            a8,
            sc,
            fused,
            h.buffers.expert_gate_out(),
            16,
            INTER,
            H,
            0,
        )
        .unwrap();
    assert_eq!(
        h.gpu.launch_count() - before,
        2,
        "one block-scaled GEMM + one strided SiLU"
    );

    // The un-fused pair, same layer, same widths: two GEMMs + one SiLU.
    let before = h.gpu.launch_count();
    for out in [h.buffers.expert_gate_out(), h.buffers.expert_up_out()] {
        h.layer
            .w8a8_gemm(
                &c,
                a8,
                sc,
                &h.fp8,
                out,
                h.buffers.expert_gate_out_bytes(),
                16,
                INTER,
                H,
                0,
            )
            .unwrap();
    }
    ops::silu_mul(
        &h.gpu,
        h.layer.act_mul,
        h.buffers.expert_gate_out(),
        h.buffers.expert_up_out(),
        h.buffers.expert_gate_out(),
        16 * INTER,
        0,
    )
    .unwrap();
    assert_eq!(h.gpu.launch_count() - before, 3);
}
