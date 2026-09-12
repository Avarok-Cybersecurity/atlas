// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch-selection contract for the W8A8 block-scaled dense-FFN prefill
//! (#917/#928). These are CPU tests: they pin WHICH arm the native-FP8 prefill
//! picks and the buffer-extent arithmetic the cuBLASLt arm depends on. The
//! numerics themselves are the GPU microtest's job
//! (`examples/native_fp8_ffn_w8a8_microtest.rs`).

use super::w8a8_prefill_selected;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{
    self, DerivedWeights, GemmDispatch, ModelLevers, ModelStats, cublas_fp8_m_pad,
};
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
/// A quantizer pair with only the shared kernel — what every non-Hopper
/// target resolves. The Hopper twin changes the launch grid, never the
/// selector, so these gates are written against the shared arm.
const QUANT_K: ops::Fp8ActQuant = ops::Fp8ActQuant {
    shared: KernelHandle(0xA8A),
    hopper: KernelHandle(0),
};
/// No quantizer at all.
const NO_QUANT: ops::Fp8ActQuant = ops::Fp8ActQuant {
    shared: KernelHandle(0),
    hopper: KernelHandle(0),
};
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
    quant_k: ops::Fp8ActQuant,
    gemm_k: KernelHandle,
    w8a16_only: bool,
) -> bool {
    w8a8_prefill_selected(m, n, k, fmt, blockscaled_lever, quant_k, gemm_k, w8a16_only)
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
    assert!(!selected(64, INTER, H, f, true, NO_QUANT, GEMM_K, false));
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

#[test]
fn layer_selects_w8a8_on_a_dense_config() {
    let h = harness(0, 2048);
    with_ctx(&h, GemmDispatch::defaults(), |ctx| {
        assert!(
            h.layer
                .prefill_w8a8_selected(ctx, PROMPT_TOKENS, INTER, H, &h.fp8)
        );
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
    // set serves both, so it must be sized for the wider of the two.
    //
    // THREE buffers, not two: `ffn_act_scale` holds the `[M, K/128]` FP32
    // layout `per_token_group_quant_fp8` writes (K-group contiguous — what the
    // in-tree `fp8_gemm_t_blockscaled` indexes), and `ffn_act_scale_kmajor`
    // holds the `[K/128, ceil16(M)]` transpose `fp8_gemm_act_weight_t_blkscaled`
    // needs, because cuBLASLt reads a VEC128 B-scale tensor with the TOKEN
    // index contiguous. Feeding it the first layout is what the 2026-09-11 H100
    // run measured at rel_rms 7.7e-2 against the kernel on identical FP8 bytes.
    let max_batch_tokens = 1193_usize;
    let h = harness(0, max_batch_tokens);
    let kmax = H.max(INTER) as usize;
    let padded = cublas_fp8_m_pad(max_batch_tokens as u32) as usize;
    assert!(h.buffers.ffn_act_a_bytes() >= padded * kmax);
    assert!(h.buffers.ffn_act_scale_bytes() >= padded * (kmax / 128) * 4);
    assert!(h.buffers.ffn_act_scale_kmajor_bytes() >= padded * (kmax / 128) * 4);
}

#[test]
fn cublas_arm_requires_a_multiple_of_four_weight_scale_column_stride() {
    // cuBLASLt's BLK128x128 factors are K-major with "the stride between the
    // consecutive columns ... a multiple of 4" (cuBLAS "Scaling factors
    // layouts"), and Atlas hands over the checkpoint's `[N/128, K/128]` grid
    // as-is — so K/128 must be a multiple of 4, i.e. K % 512 == 0. Both FFN
    // contraction dims satisfy it; the gate exists for the ones that would not.
    use spark_runtime::cublaslt::scale_layout::blk128x128_stride_ok;
    assert!(blk128x128_stride_ok(H as usize));
    assert!(blk128x128_stride_ok(INTER as usize));
    assert!(!blk128x128_stride_ok(128 * 3));
    assert!(!blk128x128_stride_ok(128 * 6));
}
