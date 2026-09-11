// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch-selection contract for the W8A8 block-scaled dense-FFN prefill
//! (#917/#928). These are CPU tests: they pin WHICH arm the native-FP8 prefill
//! picks and the buffer-extent arithmetic the cuBLASLt arm depends on. The
//! numerics themselves are the GPU microtest's job
//! (`examples/native_fp8_ffn_w8a8_microtest.rs`).

use super::{max_m_for, w8a8_prefill_selected};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats, cublas_fp8_m_pad};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Qwen3.8-27B dense FFN — the shapes from #917. gate/up are `[17408, 5120]`,
/// down is `[5120, 17408]`.
const H: u32 = 5120;
const INTER: u32 = 17408;
/// The prompt length in the 2026-09-11 H100 TTFT measurement (1075 ms vs
/// vLLM's 287 ms) that motivated this path.
const PROMPT_TOKENS: u32 = 1193;
const QUANT_K: KernelHandle = KernelHandle(0xA8A);
const GEMM_K: KernelHandle = KernelHandle(0xA88);

/// Every clause of the rule defaulted to its SELECTING value, so each test
/// perturbs exactly one thing and the failure names the clause.
#[allow(clippy::too_many_arguments)]
fn selected(
    m: u32,
    n: u32,
    k: u32,
    fmt: WeightQuantFormat,
    blockscaled_lever: bool,
    quant_k: KernelHandle,
    gemm_k: KernelHandle,
    w8a16_only: bool,
) -> bool {
    // `u32::MAX` — the baseline, i.e. no cap. Every case below perturbs one of
    // the OTHER clauses, so they must not also be answering the ceiling
    // question; the ceiling has its own cases at the bottom of this file.
    selected_capped(
        m,
        n,
        k,
        fmt,
        blockscaled_lever,
        quant_k,
        gemm_k,
        w8a16_only,
        u32::MAX,
    )
}

#[allow(clippy::too_many_arguments)]
fn selected_capped(
    m: u32,
    n: u32,
    k: u32,
    fmt: WeightQuantFormat,
    blockscaled_lever: bool,
    quant_k: KernelHandle,
    gemm_k: KernelHandle,
    w8a16_only: bool,
    max_m: u32,
) -> bool {
    w8a8_prefill_selected(
        m,
        n,
        k,
        fmt,
        blockscaled_lever,
        quant_k,
        gemm_k,
        w8a16_only,
        max_m,
    )
}

/// The per-arch ceiling (#917). Separate cases from every other clause,
/// because each of those is measured with NO cap so that a failure there names
/// the clause it perturbed.
///
/// The rule is `m <= max_m`, not `m < max_m`: a ceiling of 64 means M=64 is
/// still measured to win, and the value declared in `kernels/gb10` is the LOW
/// end of the measured crossover band for that shape.
#[test]
fn the_ceiling_is_inclusive_and_cuts_above_it() {
    let case = |m: u32, max_m: u32| {
        selected_capped(
            m,
            INTER,
            H,
            WeightQuantFormat::Fp8BlockScaled,
            true,
            KernelHandle(1),
            KernelHandle(1),
            false,
            max_m,
        )
    };
    assert!(case(63, 64), "below the ceiling stays on W8A8");
    assert!(
        case(64, 64),
        "AT the ceiling stays on W8A8 — the bound is <="
    );
    assert!(
        !case(65, 64),
        "one row above the ceiling falls back to W8A16"
    );
    assert!(
        !case(949, 64),
        "the served M that measured -23.4% falls back"
    );
}

/// `u32::MAX` is the baseline and must behave as no cap at all — including at
/// `u32::MAX` itself, which `m <= max_m` admits and `m < max_m` would not.
#[test]
fn the_baseline_ceiling_caps_nothing() {
    let case = |m: u32| {
        selected_capped(
            m,
            INTER,
            H,
            WeightQuantFormat::Fp8BlockScaled,
            true,
            KernelHandle(1),
            KernelHandle(1),
            false,
            u32::MAX,
        )
    };
    assert!(case(949));
    assert!(case(u32::MAX));
}

/// A ceiling of 0 is an operator saying "never take W8A8 on this shape", and
/// the resolver honours a parsed 0 rather than discarding it. The rule has to
/// agree: `m > 4` already excludes small M, so 0 must exclude everything.
#[test]
fn a_zero_ceiling_selects_nothing() {
    for m in [5, 64, 949] {
        assert!(!selected_capped(
            m,
            INTER,
            H,
            WeightQuantFormat::Fp8BlockScaled,
            true,
            KernelHandle(1),
            KernelHandle(1),
            false,
            0,
        ));
    }
}

fn gate_up(m: u32) -> bool {
    selected(
        m,
        INTER,
        H,
        WeightQuantFormat::Fp8BlockScaled,
        true,
        QUANT_K,
        GEMM_K,
        false,
    )
}

fn down(m: u32) -> bool {
    selected(
        m,
        H,
        INTER,
        WeightQuantFormat::Fp8BlockScaled,
        true,
        QUANT_K,
        GEMM_K,
        false,
    )
}

#[test]
fn selected_for_prefill_batches_at_the_real_ffn_shapes() {
    for m in [64, PROMPT_TOKENS] {
        assert!(gate_up(m), "gate/up must take W8A8 at M={m}");
        assert!(down(m), "down must take W8A8 at M={m}");
    }
}

#[test]
fn not_selected_for_small_batches_that_belong_to_the_gemv() {
    // M<=4 streams each weight once through `w8a16_gemv_batch4`; an MMA tile
    // there is mostly padding. M=5 is the first row count this path claims.
    for m in [1, 2, 3, 4] {
        assert!(!gate_up(m), "M={m} must stay on the batch4 GEMV");
    }
    assert!(gate_up(5), "M=5 is the first W8A8 row count");
}

#[test]
fn not_selected_for_per_row_scales() {
    // `row_scale` would be `[N]`, not the `[N/128, K/128]` grid the
    // block-scaled GEMM indexes — reading it as the latter is silent garbage.
    for fmt in [
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8SingleScale,
    ] {
        assert!(
            !selected(64, INTER, H, fmt, true, QUANT_K, GEMM_K, false),
            "{fmt:?} must not take the block-scaled GEMM"
        );
    }
}

#[test]
fn not_selected_for_unaligned_shapes() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    // K not a multiple of 128: the activation quantizer emits one FP32 scale
    // per 128-wide K group, so a ragged tail has no scale.
    assert!(!selected(64, INTER, H + 1, f, true, QUANT_K, GEMM_K, false));
    assert!(!selected(64, INTER, 127, f, true, QUANT_K, GEMM_K, false));
    // N not a multiple of 128: the weight scale grid is `[N/128, K/128]`.
    assert!(!selected(64, INTER + 1, H, f, true, QUANT_K, GEMM_K, false));
    assert!(!selected(64, 64, H, f, true, QUANT_K, GEMM_K, false));
}

#[test]
fn not_selected_when_the_kill_switch_is_set() {
    // ATLAS_FFN_W8A16_ONLY — injected, not read from the environment: the
    // real accessor is a process-global `OnceLock` and a test that set the
    // variable would leak into every other test in the binary.
    assert!(!selected(
        PROMPT_TOKENS,
        INTER,
        H,
        WeightQuantFormat::Fp8BlockScaled,
        true,
        QUANT_K,
        GEMM_K,
        true,
    ));
}

#[test]
fn not_selected_when_the_blockscaled_prefill_lever_is_off() {
    // ATLAS_FP8_SINGLE_SCALE clears `dispatch.fp8_blockscaled_prefill`; it
    // already governs the attention W8A8 path and must govern this one too.
    assert!(!selected(
        PROMPT_TOKENS,
        INTER,
        H,
        WeightQuantFormat::Fp8BlockScaled,
        false,
        QUANT_K,
        GEMM_K,
        false,
    ));
}

#[test]
fn not_selected_when_either_kernel_is_missing() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    assert!(!selected(
        64,
        INTER,
        H,
        f,
        true,
        KernelHandle(0),
        GEMM_K,
        false
    ));
    assert!(!selected(
        64,
        INTER,
        H,
        f,
        true,
        QUANT_K,
        KernelHandle(0),
        false
    ));
}

// ───────────────────── layer-level gates (real arena) ─────────────────────

struct Harness {
    gpu: MockGpuBackend,
    layer: DenseFfnLayer,
    buffers: BufferArena,
    config: ModelConfig,
    fp8: Fp8Weight,
}

/// A dense (num_experts = 0) config at the real FFN widths, with the arena
/// sized for `max_batch_tokens` prefill rows.
fn harness(num_experts: usize, max_batch_tokens: usize) -> Harness {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = num_experts;
    config.num_experts_per_tok = num_experts.min(1);
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, max_batch_tokens, 256, 256, 8, &gpu).unwrap();
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
    layer.per_token_group_quant_fp8_k = QUANT_K;
    layer.fp8_gemm_t_blockscaled_k = GEMM_K;
    let fp8 = Fp8Weight {
        weight: gpu.alloc(1024).unwrap(),
        row_scale: gpu.alloc(1024).unwrap(),
        n: INTER,
        k: H,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    Harness {
        gpu,
        layer,
        buffers,
        config,
        fp8,
    }
}

/// Runs `f` with a prefill-shaped `ForwardContext` over the harness.
fn with_ctx<R>(h: &Harness, dispatch: GemmDispatch, f: impl FnOnce(&ForwardContext) -> R) -> R {
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
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    f(&ctx)
}

/// The CALL SITE, which is a different question from the pure rule above: it
/// reads the ceiling the COMPILED TARGET declares, so it cannot hardcode
/// gb10's 64 — an H100 build of this same suite declares no cap and would then
/// fail for being correct.
///
/// It asserts the RELATIONSHIP instead: selected just inside whatever ceiling
/// this build carries, declined one row outside it. The values themselves are
/// pinned as data by `atlas-kernels/tests/target_defaults.rs`.
#[test]
fn layer_selects_w8a8_on_a_dense_config() {
    let h = harness(0, 2048);
    let max_m = max_m_for(INTER, H);
    let inside = max_m.min(PROMPT_TOKENS);
    assert!(
        inside > 4,
        "the m > 4 clause must not be what this test is measuring"
    );
    with_ctx(&h, GemmDispatch::defaults(), |ctx| {
        assert!(
            h.layer.prefill_w8a8_selected(ctx, inside, INTER, H, &h.fp8),
            "inside the target's ceiling ({max_m}) the dense config takes W8A8"
        );
        // No "outside" exists on a target that declares no cap, which is the
        // baseline and is what H100 declares.
        if max_m < u32::MAX {
            assert!(
                !h.layer
                    .prefill_w8a8_selected(ctx, max_m + 1, INTER, H, &h.fp8),
                "one row above the ceiling ({max_m}) must fall back to W8A16"
            );
        }
    });
}

#[test]
fn layer_declines_w8a8_without_the_shared_ffn_scratch() {
    // `ffn_act_a` / `ffn_act_scale` are 0 for MoE configs (BufferSizes). A null
    // scratch pointer is a kernel launch writing to address 0, so the absence
    // has to gate the arm rather than trip an assert.
    let h = harness(2, 2048);
    assert_eq!(
        h.buffers.ffn_act_a().0,
        0,
        "MoE arena must have no FFN scratch"
    );
    with_ctx(&h, GemmDispatch::defaults(), |ctx| {
        assert!(
            !h.layer
                .prefill_w8a8_selected(ctx, PROMPT_TOKENS, INTER, H, &h.fp8)
        );
    });
}

// ───────────────────── cuBLASLt padded-M buffer contract ─────────────────────

#[test]
fn cublas_m_pad_rounds_up_to_sixteen() {
    // cuBLASLt rejects a scale-tensor M extent that is not a multiple of 4;
    // `cublas_fp8_proj_prequant` pads to 16 and WRITES those phantom rows.
    assert_eq!(cublas_fp8_m_pad(PROMPT_TOKENS), 1200);
    assert_eq!(cublas_fp8_m_pad(64), 64);
    assert_eq!(cublas_fp8_m_pad(1), 16);
    for m in [1_u32, 5, 63, 64, 65, 1193, 8192] {
        assert!(cublas_fp8_m_pad(m) >= m);
        assert_eq!(cublas_fp8_m_pad(m) % 16, 0);
    }
}

#[test]
fn ffn_output_buffers_hold_the_padded_m_the_cublas_gemm_writes() {
    // The worst case is a prefill chunk exactly as wide as the arena: without
    // the `div_ceil(16) * 16` in BufferSizes the padded rows would land in the
    // NEXT arena buffer. Pinned here because the sizing and the writer live in
    // different crates.
    for max_batch_tokens in [256_usize, 1193, 2048] {
        let h = harness(0, max_batch_tokens);
        let m = cublas_fp8_m_pad(max_batch_tokens as u32) as usize;
        assert!(
            h.buffers.expert_gate_out_bytes() >= m * INTER as usize * 2,
            "gate/up output too small for padded M at max_batch_tokens={max_batch_tokens}"
        );
        assert!(
            h.buffers.moe_output_bytes() >= m * H as usize * 2,
            "down output too small for padded M at max_batch_tokens={max_batch_tokens}"
        );
    }
}

#[test]
fn activation_scratch_holds_the_widest_ffn_projection() {
    // gate/up contract over K=hidden, down over K=intermediate; ONE scratch
    // pair serves both, so it must be sized for the wider of the two. The
    // scale extent is the `[M, K/128]` FP32 layout `per_token_group_quant_fp8`
    // writes and `fp8_gemm_act_weight_t_blkscaled` reads as its VEC128 B-scale
    // — same element order, which is why no transpose adapter is needed.
    let max_batch_tokens = 1193_usize;
    let h = harness(0, max_batch_tokens);
    let kmax = H.max(INTER) as usize;
    let padded = cublas_fp8_m_pad(max_batch_tokens as u32) as usize;
    assert!(h.buffers.ffn_act_a_bytes() >= padded * kmax);
    assert!(h.buffers.ffn_act_scale_bytes() >= padded * (kmax / 128) * 4);
}
