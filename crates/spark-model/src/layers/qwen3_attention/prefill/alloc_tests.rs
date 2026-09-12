// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation contract for the two ATTENTION prefill projection chains that
//! `ATLAS_CUBLAS_GEMM=attn` arms: the cache-skip Q/K/V chain
//! (`cache_skip_qkv.rs`) and the paged O-projection (`paged_oproj.rs`).
//!
//! H100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8`: both chains used to carry an arm
//! guarded by nothing but `ctx.dispatch.cublas.attn && weight.as_fp8()` that
//! called `ops::cublas_bf16_proj`, whose `dequant_fp8_bf16_cached` allocates a
//! BF16 twin of the FP8 weight — 2 bytes per weight element, lazily, with no
//! entry in `spark_runtime::buffers::sizes::BufferSizes`, so
//! `--gpu-memory-utilization` cannot see it. That is the same leak the SSM
//! QKVZ arm hit at `167772160` bytes PER LAYER, where 48 layers came to
//! ~10.3 GiB and a 28-token prefill died at layer 36 with
//! `cuMemAlloc_v2 failed: status 2` — and on the same receipt the attention
//! arms are most of the gap between the 27 SSM layers' 4.3 GiB and the
//! 6120 MiB actually consumed before the OOM. #927/#928 deleted both arms in
//! favour of the W8A8 routes; these tests are what keeps them deleted, because
//! the serve recipe for the 5..16-row decode projections is
//! `ATLAS_CUBLAS_GEMM=ffn,ssm,attn` and arming `attn` for decode must not
//! re-arm a prefill arm that allocates.
//!
//! The assertion is deliberately taken BEFORE the first call, not between the
//! two: the old allocation was CACHED by weight pointer, so a first-vs-second
//! comparison alone would have passed it. What is wrong is that the projection
//! allocates at all — the arena already owns every buffer it needs.
//!
//! The k-major scale adapter is zeroed in both fixtures so each chain takes the
//! in-tree kernel: a CPU test cannot enter cuBLASLt's FFI. That is the same
//! device `qwen3_ssm::prefill_alloc_tests` uses, and it does not weaken the
//! guard — the deleted BF16 arms sat ABOVE the W8A8 arms in both dispatch
//! chains and had no k-major clause, so a reintroduced one fires here. The
//! cuBLASLt arms' own clauses are pinned in `prefill_qkv_w8a8_tests.rs`, and
//! they allocate nothing by construction (activation, scales and the k-major
//! transpose are all arena buffers).

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::FfnComponent;
use crate::layers::ops::{CublasScope, DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat,
};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

/// The smoke prompt that killed the H100 run.
const M: u32 = 28;
const QUANT_K: KernelHandle = KernelHandle(0xA101);
const BLOCKSCALED_K: KernelHandle = KernelHandle(0xA102);
const W8A16_PIPELINED_K: KernelHandle = KernelHandle(0xA103);

fn fp8(gpu: &MockGpuBackend, n: u32, k: u32) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc(n as usize * k as usize).unwrap(),
        row_scale: gpu
            .alloc((n as usize).div_ceil(128) * (k as usize).div_ceil(128) * 4)
            .unwrap(),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// A full-attention layer whose Q/K/V/O weights are block-scaled FP8 — the
/// shape class `Qwen/Qwen3.8-27B-FP8` presents and the only one that can
/// satisfy the deleted arms' `as_fp8()` gate.
fn native_fp8_attention_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3AttentionLayer {
    let dense = DenseWeight {
        weight: gpu
            .alloc(config.hidden_size * config.hidden_size * 2)
            .unwrap(),
    };
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj: QuantizedWeight::null(),
        q_norm: dense,
        k_norm: dense,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new(
        dense,
        attn,
        dense,
        FfnComponent::None,
        0,
        None,
        None,
        None,
        gpu,
        KvCacheDtype::Bf16,
        0,
        config,
    )
    .unwrap();
    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let nkv = config.num_key_value_heads as u32;
    let hd = config.head_dim as u32;
    let q_proj_dim = if layer.gated { 2 * nq * hd } else { nq * hd };
    layer.q_weight = Some(QuantWeight::Fp8(fp8(gpu, q_proj_dim, h)));
    layer.k_weight = Some(QuantWeight::Fp8(fp8(gpu, nkv * hd, h)));
    layer.v_weight = Some(QuantWeight::Fp8(fp8(gpu, nkv * hd, h)));
    layer.o_weight = Some(QuantWeight::Fp8(fp8(gpu, h, nq * hd)));
    // Transposed and FP8xFP8 twins stay absent so the chains resolve the arms
    // the H100 trace actually shows (`w8a16_gemm_pipelined` for Q/K/V).
    layer.q_fp8w_t = None;
    layer.k_fp8w_t = None;
    layer.v_fp8w_t = None;
    layer.o_fp8w_t = None;
    layer.q_fp8 = None;
    layer.k_fp8 = None;
    layer.v_fp8 = None;
    layer.o_fp8 = None;
    layer.per_token_group_quant_fp8_k = QUANT_K;
    layer.fp8_gemm_t_blockscaled_k = BLOCKSCALED_K;
    layer.w8a16_gemm_pipelined_k = W8A16_PIPELINED_K;
    // See the module header: no k-major adapter -> the in-tree kernel, which a
    // mock backend can reach and cuBLASLt's FFI cannot.
    layer.fp8_act_scale_kmajor_k = KernelHandle(0);
    layer
}

/// `ATLAS_CUBLAS_GEMM=attn` armed, which is what the decode recipe sets.
fn armed_dispatch() -> GemmDispatch {
    GemmDispatch {
        cublas: CublasScope::ALL,
        ..GemmDispatch::defaults()
    }
}

macro_rules! fwd_ctx {
    ($buffers:expr, $gpu:expr, $config:expr, $dispatch:expr, $derived:expr, $levers:expr, $stats:expr) => {
        ForwardContext {
            buffers: $buffers,
            hc_row_offset: 0,
            gpu: $gpu,
            config: $config,
            dispatch: $dispatch,
            derived: $derived,
            levers: $levers,
            stats: $stats,
            attn_metadata: None,
            decode_step: false,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    };
}

/// `paged_oproj.rs`: the O-projection must not allocate with the `attn` family
/// armed. Before #927 this chain's second arm dequantized `o_proj`
/// (`[h, nq*hd]`) to BF16 and cached it off-ledger.
#[test]
fn attention_o_projection_prefill_allocates_nothing_with_the_cublas_lever_armed() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_attention_layer(&gpu, &config);
    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let (dispatch, derived) = (armed_dispatch(), DerivedWeights::new());
    let (levers, stats) = (ModelLevers::defaults(), ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &config, &dispatch, &derived, &levers, &stats
    );

    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let hd = config.head_dim as u32;
    let run = || layer.prefill_attention_paged_oproj(buffers.attn_output(), M, h, nq, hd, &ctx, 0);
    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    // The allocation is checked BEFORE the result is unwrapped: the deleted arm
    // allocates and only then calls cuBLASLt, so on a mock backend the FFI error
    // would otherwise mask the leak this test exists to name.
    let first = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the O-projection prefill allocated on its FIRST call — that is the \
         off-ledger BF16 weight dequant from the H100 receipt (here `o_proj` \
         [h, nq*hd] x 2 B = 16777216 bytes, per layer)"
    );
    first.expect("first O-projection prefill");
    // Two launches, and the same two again: the per-token FP8 activation quant
    // and the block-scaled GEMM. Pinned so "allocated nothing" cannot be
    // satisfied by "did nothing", and so a future arm that slips a dequant
    // kernel back in is visible even if it reuses a buffer.
    assert_eq!(
        gpu.launch_count() - launches,
        2,
        "expected per_token_group_quant_fp8 + fp8_gemm_t_blockscaled"
    );
    let second = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the O-projection prefill allocated per call"
    );
    second.expect("second O-projection prefill");
    assert_eq!(gpu.launch_count() - launches, 4);
}

/// `cache_skip_qkv.rs`: the chunk-0 Q/K/V chain must not allocate with the
/// `attn` family armed. Before #927 each of the three projections had its own
/// dequant-and-cache arm, so one armed prefill left a BF16 twin of `q_proj`,
/// `k_proj` AND `v_proj` behind per layer.
#[test]
fn attention_cache_skip_qkv_prefill_allocates_nothing_with_the_cublas_lever_armed() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_attention_layer(&gpu, &config);
    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let (dispatch, derived) = (armed_dispatch(), DerivedWeights::new());
    let (levers, stats) = (ModelLevers::defaults(), ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &config, &dispatch, &derived, &levers, &stats
    );

    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let nkv = config.num_key_value_heads as u32;
    let hd = config.head_dim as u32;
    let q_dim = (nq * hd) as usize;
    let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
    let kv_dim = (nkv * hd) as usize;
    let run = || {
        layer.prefill_attention_cache_skip_qkv(
            buffers.norm_output(),
            spark_runtime::gpu::DevicePtr::NULL,
            M,
            h,
            nkv,
            hd,
            q_proj_dim,
            kv_dim,
            M as usize,
            2,
            &ctx,
            0,
        )
    };
    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    // Allocation before unwrap — see the O-projection twin above.
    let first = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the Q/K/V prefill chain allocated on its FIRST call — that is the \
         off-ledger BF16 weight dequant from the H100 receipt, once per projection"
    );
    first.expect("first Q/K/V prefill chain");
    // Three launches, one per projection: `w8a16_gemm_pipelined`, the arm the
    // round-9 nsys trace shows at 100.582 ms over 112 launches.
    assert_eq!(
        gpu.launch_count() - launches,
        3,
        "expected one w8a16_gemm_pipelined per projection"
    );
    let second = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the Q/K/V prefill chain allocated per call"
    );
    second.expect("second Q/K/V prefill chain");
    assert_eq!(gpu.launch_count() - launches, 6);
}
