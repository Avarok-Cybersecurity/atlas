// SPDX-License-Identifier: AGPL-3.0-only

//! CPU tests for the fused prefill FFN-norm: the hand-off's safety rules, the
//! selection rule, the row-off launch (exactly the old one), and the kernel's
//! thread-to-group map over every legal hidden size, with a host model of the
//! fused quantization against `per_token_group_quant`. The device byte compare
//! is `examples/rmsnorm_quant_fp8_microtest.rs`.

use super::*;
use crate::layer::MoeLoraRoute;
use crate::layers::dense_ffn::DenseFfnWeights;
use crate::layers::dense_ffn::w8a8_prefill::max_m_for;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::fp8_blockscale_ref::{
    per_token_group_quant, quantize_with_scale, round_bf16,
};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

// ── The hand-off ────────────────────────────────────────────────────────

fn token(m: u32) -> Prequant {
    Prequant {
        act: DevicePtr(0x1000),
        m,
        k: 2560,
        a_fp8: DevicePtr(0x2000),
        a_scale: DevicePtr(0x3000),
        stream: 7,
    }
}

#[test]
fn a_matching_quantization_consumes_the_hand_off_once() {
    clear();
    arm(token(64));
    assert!(take_matching(token(64)));
    assert!(
        !take_matching(token(64)),
        "consumed: the second quantization runs"
    );
    assert_eq!(pending(), None);
}

#[test]
fn any_other_quantization_clears_the_hand_off_without_using_it() {
    // A quantization of something else rewrites the scratch, so the token
    // must not survive it.
    clear();
    arm(token(64));
    assert!(!take_matching(token(63)));
    assert_eq!(pending(), None);
    for other in [
        Prequant {
            act: DevicePtr(0x1001),
            ..token(64)
        },
        Prequant {
            k: 9216,
            ..token(64)
        },
        Prequant {
            a_fp8: DevicePtr(0x2001),
            ..token(64)
        },
        Prequant {
            a_scale: DevicePtr(0x3001),
            ..token(64)
        },
        Prequant {
            stream: 8,
            ..token(64)
        },
    ] {
        arm(token(64));
        assert!(!take_matching(other), "{other:?}");
        assert_eq!(pending(), None);
    }
}

#[test]
fn dropping_the_scope_clears_an_unconsumed_hand_off() {
    clear();
    let scope = PrequantScope { _private: () };
    arm(token(5));
    drop(scope);
    assert_eq!(pending(), None);
    assert!(!take_matching(token(5)));
}

// ── Selection ───────────────────────────────────────────────────────────

const H: u32 = 256;
const INTER: u32 = 512;
const NORM_K: KernelHandle = KernelHandle(0x4A4);
const FUSED_K: KernelHandle = KernelHandle(0xF5D);

struct Harness {
    gpu: MockGpuBackend,
    ffn: FfnComponent,
    buffers: BufferArena,
    config: ModelConfig,
}

/// A dense native-FP8 layer at small widths with the W8A8 handles loaded.
/// `vanilla_norm` renames the model to one that ships plain norm weights,
/// after the arena is sized, so nothing else about the harness moves.
fn harness(vanilla_norm: bool) -> Harness {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = 0;
    config.num_experts_per_tok = 0;
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, 2048, 256, 256, 8, &gpu).unwrap();
    if vanilla_norm {
        config.model_type = "laguna".to_string();
    }
    assert_eq!(crate::ships_vanilla_norm_weights(&config), vanilla_norm);
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
    layer.per_token_group_quant_fp8_k = ops::Fp8ActQuant::shared_only(KernelHandle(0xA8A));
    layer.fp8_gemm_t_blockscaled_k = KernelHandle(0xA88);
    let w = |n: u32, k: u32| Fp8Weight {
        weight: gpu.alloc((n * k) as usize).unwrap(),
        row_scale: gpu.alloc(4 * (n / 128 * k / 128) as usize).unwrap(),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.set_fp8_weights(w(INTER, H), w(INTER, H), w(H, INTER));
    Harness {
        gpu,
        ffn: FfnComponent::Dense(layer),
        buffers,
        config,
    }
}

fn with_ctx<R>(h: &Harness, f: impl FnOnce(&ForwardContext) -> R) -> R {
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &h.buffers,
        hc_row_offset: 0,
        gpu: &h.gpu,
        config: &h.config,
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
    f(&ctx)
}

/// A prefill width inside whatever W8A8 ceiling the compiled target declares
/// (gb10 declares one; the baseline is none) and above the M > 4 GEMV rung.
fn prefill_m() -> u32 {
    let m = max_m_for(INTER, H).min(max_m_for(H, INTER)).min(48);
    assert!(
        m > 4,
        "the m > 4 clause must not be what these tests measure"
    );
    m
}

#[test]
fn a_dense_fp8_ffn_on_the_w8a8_arm_takes_the_fused_norm() {
    let h = harness(false);
    with_ctx(&h, |ctx| {
        let plan =
            fused_plan_with(&h.ffn, true, ctx, prefill_m(), H, || FUSED_K).expect("selected");
        assert_eq!(plan.kernel.0, FUSED_K.0);
        assert_eq!(plan.a_fp8, ctx.buffers.ffn_act_a());
        assert_eq!(plan.a_scale, ctx.buffers.ffn_act_scale());
    });
}

#[test]
fn every_clause_declines_on_its_own() {
    let h = harness(false);
    let m = prefill_m();
    with_ctx(&h, |ctx| {
        let take = |ffn: &FfnComponent, sole: bool, m: u32, hh: u32, k: KernelHandle| {
            fused_plan_with(ffn, sole, ctx, m, hh, || k).is_some()
        };
        assert!(take(&h.ffn, true, m, H, FUSED_K));
        assert!(
            !take(&h.ffn, false, m, H, FUSED_K),
            "another consumer between norm and FFN"
        );
        assert!(
            !take(&h.ffn, true, 4, H, FUSED_K),
            "m <= 4 stays on the batch4 GEMV"
        );
        assert!(
            !take(&h.ffn, true, m, H + 128, FUSED_K),
            "not the FFN's width"
        );
        assert!(
            !take(&h.ffn, true, m, H, KernelHandle(0)),
            "kernel absent on this target"
        );
        assert!(
            !take(&FfnComponent::None, true, m, H, FUSED_K),
            "no dense FFN"
        );
    });
}

#[test]
fn a_model_with_vanilla_norm_weights_never_takes_it() {
    // The fused kernel is the offset-from-one norm only.
    let h = harness(true);
    with_ctx(&h, |ctx| {
        assert!(fused_plan_with(&h.ffn, true, ctx, prefill_m(), H, || FUSED_K).is_none());
    });
}

#[test]
fn the_shape_clauses() {
    assert!(shape_ok(1, 2560, 2560));
    assert!(shape_ok(3921, 5120, 5120));
    assert!(shape_ok(64, MAX_HIDDEN, MAX_HIDDEN as usize));
    assert!(!shape_ok(0, 2560, 2560));
    assert!(!shape_ok(64, 2560 + 64, 2624));
    assert!(!shape_ok(64, MAX_HIDDEN + 128, (MAX_HIDDEN + 128) as usize));
}

#[test]
fn with_the_row_off_the_call_site_launches_exactly_the_old_norm() {
    if ops::port_rows::rmsnorm_fp8_quant_fused::row() {
        return;
    }
    let h = harness(false);
    let weight = DenseWeight {
        weight: h.gpu.alloc(H as usize * 2).unwrap(),
    };
    with_ctx(&h, |ctx| {
        let (hid, add, out, res) = (
            DevicePtr(0x10),
            DevicePtr(0x20),
            DevicePtr(0x30),
            DevicePtr(0x40),
        );
        let m = prefill_m();
        let start = h.gpu.launch_count();
        let lookups = h.gpu.kernel_lookups_snapshot().len();
        let scope = residual_add_rms_norm(
            &h.ffn, true, ctx, NORM_K, hid, add, &weight, out, res, m, H, 1e-6, 3,
        )
        .unwrap();
        assert_eq!(pending(), None, "row off: nothing is handed off");
        drop(scope);
        let l = &h.gpu.launches_snapshot()[start..];
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].func, NORM_K.0);
        assert_eq!(l[0].grid, [m, 1, 1]);
        assert_eq!(l[0].block, [H.min(1024), 1, 1]);
        assert_eq!(l[0].stream, 3);
        let ptrs: Vec<_> = l[0]
            .args
            .iter()
            .filter(|a| matches!(a, MockArg::Buffer(_)))
            .cloned()
            .collect();
        assert_eq!(
            ptrs,
            [hid, add, weight.weight, out, res]
                .map(MockArg::Buffer)
                .to_vec(),
            "the residual_add_rms_norm argument order"
        );
        assert_eq!(h.gpu.kernel_lookups_snapshot().len(), lookups, "no lookup");
    });
}

#[test]
fn the_fused_launch_keeps_the_norms_block_and_adds_the_two_outputs() {
    let gpu = MockGpuBackend::new();
    let weight = DenseWeight {
        weight: DevicePtr(0x50),
    };
    for (hidden, block) in [(1024u32, 1024u32), (2560, 1024), (896, 896), (128, 128)] {
        let start = gpu.launch_count();
        rmsnorm_quant_fp8(
            &gpu,
            FUSED_K,
            DevicePtr(0x10),
            DevicePtr(0x20),
            &weight,
            DevicePtr(0x30),
            DevicePtr(0x40),
            DevicePtr(0x60),
            DevicePtr(0x70),
            17,
            hidden,
            1e-6,
            0,
        )
        .unwrap();
        let l = &gpu.launches_snapshot()[start];
        assert_eq!(l.grid, [17, 1, 1]);
        assert_eq!(l.block, [block, 1, 1], "hidden {hidden}");
        let ptrs: Vec<_> = l
            .args
            .iter()
            .filter(|a| matches!(a, MockArg::Buffer(_)))
            .cloned()
            .collect();
        assert_eq!(
            ptrs,
            [0x10u64, 0x20, 0x50, 0x30, 0x40, 0x60, 0x70]
                .map(|p| MockArg::Buffer(DevicePtr(p)))
                .to_vec()
        );
    }
}

// ── The kernel's thread map and the fused quantization, on the host ─────

fn src(name: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/ada/common");
    std::fs::read_to_string(root.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn the_host_constants_are_the_kernels() {
    let cu = src("rmsnorm_quant_fp8_ada.cu");
    for line in [
        format!("#define RQ_MAX_HIDDEN {MAX_HIDDEN}"),
        format!("#define RQ_GROUP_K {GROUP_K}"),
        format!("#define RQ_MAX_THREADS {MAX_THREADS}"),
    ] {
        assert!(cu.contains(&line), "rmsnorm_quant_fp8_ada.cu lost `{line}`");
    }
}

/// Elements of one warp-pass segment and pairs a thread may hold, as the
/// kernel derives them.
const PAIRS_PER_SEGMENT: u32 = 32;
const MAX_PAIRS: u32 = MAX_HIDDEN / 2 / MAX_THREADS;

#[test]
fn every_legal_hidden_size_maps_each_half_group_to_one_warp_pass() {
    for hidden in (GROUP_K..=MAX_HIDDEN).step_by(GROUP_K as usize) {
        let block = hidden.min(MAX_THREADS);
        let half = hidden / 2;
        assert!(
            block * MAX_PAIRS >= half,
            "hidden {hidden}: a pair has no slot"
        );
        let mut seen = vec![0u32; (half / PAIRS_PER_SEGMENT) as usize];
        for warp in 0..block / 32 {
            for j in 0..MAX_PAIRS {
                let first = warp * 32 + j * block;
                let live: Vec<bool> = (0..32).map(|lane| first + lane < half).collect();
                assert!(
                    live.iter().all(|&l| l == live[0]),
                    "hidden {hidden}: a split warp"
                );
                if !live[0] {
                    continue;
                }
                // Lane 0's segment index, and every lane's pair in that segment
                // and in one 128-group.
                let seg = first / PAIRS_PER_SEGMENT;
                for lane in 0..32 {
                    let pair = first + lane;
                    assert_eq!(pair / PAIRS_PER_SEGMENT, seg);
                    assert_eq!(pair / (GROUP_K / 2), seg / 2);
                }
                seen[seg as usize] += 1;
            }
        }
        assert!(
            seen.iter().all(|&n| n == 1),
            "hidden {hidden}: segments {seen:?}"
        );
    }
}

/// The fused kernel's quantization of one normed BF16 row: half-group maxima
/// combined in pairs, then each element against its group scale.
fn fused_quant_model(row: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let seg_max: Vec<f32> = row
        .chunks_exact(2 * PAIRS_PER_SEGMENT as usize)
        .map(|seg| seg.iter().fold(0.0f32, |m, x| m.max(x.abs())))
        .collect();
    let scales: Vec<f32> = seg_max
        .chunks_exact(2)
        .map(|p| {
            let s = p[0].max(p[1]) / 448.0;
            if s < 1e-12 { 1e-12 } else { s }
        })
        .collect();
    let bytes = row
        .iter()
        .enumerate()
        .map(|(i, x)| quantize_with_scale(*x, scales[i / GROUP_K as usize]))
        .collect();
    (bytes, scales)
}

#[test]
fn the_fused_quantization_equals_the_shared_quantizer_bit_for_bit() {
    let mut state = 0x15_0815u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 2.0 - 1.0
    };
    for hidden in [128usize, 1024, 2560, 5120] {
        let mut row: Vec<f32> = (0..hidden).map(|_| round_bf16(next() * 3.0)).collect();
        // Edge groups: all zero (the scale floor), a NaN, an infinity, and
        // magnitudes at both ends of BF16's range.
        row[..128].fill(0.0);
        if hidden >= 512 {
            row[130] = f32::NAN;
            row[300] = f32::INFINITY;
            for x in &mut row[384..512] {
                *x = round_bf16(*x * 1e30);
            }
        }
        let (fb, fs) = fused_quant_model(&row);
        let (rb, rs) = per_token_group_quant(&row, 1, hidden);
        assert_eq!(fb, rb, "hidden {hidden}: bytes");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&fs), bits(&rs), "hidden {hidden}: scales");
    }
}
