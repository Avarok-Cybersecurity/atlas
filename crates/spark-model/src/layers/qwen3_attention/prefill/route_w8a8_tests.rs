// SPDX-License-Identifier: AGPL-3.0-only

//! Which kernels an attention layer built by the load-time BF16 -> FP8 route
//! (`weight_loader/qwen35_dense/attn_fp8_from_bf16.rs`) launches at prefill,
//! and which handles the constructor resolves with and without the sm_89
//! drop-ins.
//!
//! The route installs block-scaled FP8 q/k/v/o and NO transposed twins, which
//! is the fixture `alloc_tests::native_fp8_attention_layer` already builds,
//! plus the one setter the route adds (`set_w8a8_qkv_prefill`). Every handle
//! under test is a sentinel, so a launch names its kernel. The cuBLASLt arm
//! is never reached here: a mock backend cannot enter its FFI.

use super::super::Qwen3AttentionLayer;
use super::alloc_tests::native_fp8_attention_layer;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, Fp8ActQuant, GemmDispatch, ModelLevers, ModelStats};
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{DevicePtr, KernelHandle};

const M: u32 = 28;
const QUANT: KernelHandle = KernelHandle(0xD1);
const W8A8: KernelHandle = KernelHandle(0xD8);
const W8A16_PIPE: KernelHandle = KernelHandle(0xE1);

fn route_layer(gpu: &MockGpuBackend, config: &ModelConfig, armed: bool) -> Qwen3AttentionLayer {
    let mut layer = native_fp8_attention_layer(gpu, config);
    layer.set_w8a8_qkv_prefill(armed);
    layer.per_token_group_quant_fp8_k = Fp8ActQuant::shared_only(QUANT);
    layer.fp8_gemm_t_blockscaled_k = W8A8;
    layer.w8a16_gemm_pipelined_k = W8A16_PIPE;
    layer
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn launches(gpu: &MockGpuBackend, k: KernelHandle) -> Vec<spark_runtime::gpu::mock::MockLaunch> {
    gpu.launches_snapshot()
        .into_iter()
        .filter(|l| l.func == k.0)
        .collect()
}

/// Run the chunk-0 Q/K/V chain at M = 28 on `dispatch`.
fn run_qkv(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &Qwen3AttentionLayer,
    dispatch: &GemmDispatch,
) -> (u32, u32) {
    let buffers = BufferArena::new(config, 64, 4096, 16, 32, gpu).unwrap();
    let (derived, levers, stats) = (
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu,
        config,
        dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        decode_step: false,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let nkv = config.num_key_value_heads as u32;
    let hd = config.head_dim as u32;
    let q_dim = nq * hd;
    let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
    layer
        .prefill_attention_cache_skip_qkv(
            buffers.norm_output(),
            DevicePtr::NULL,
            M,
            h,
            nkv,
            hd,
            q_proj_dim as usize,
            (nkv * hd) as usize,
            M as usize,
            2,
            &ctx,
            0,
        )
        .expect("cache-skip Q/K/V prefill");
    (q_proj_dim, nkv * hd)
}

/// Route on, no `AVAROK_CUBLAS_GEMM`: ONE activation quant for all three
/// projections, then the in-tree W8A8 GEMM three times at (M, N, K = hidden),
/// and no W8A16 launch.
#[test]
fn route_qkv_prefill_takes_the_in_tree_w8a8_arm_without_cublas() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, true);
    let (q_n, kv_n) = run_qkv(&gpu, &config, &layer, &GemmDispatch::defaults());
    let h = config.hidden_size as u32;
    let quants = launches(&gpu, QUANT);
    assert_eq!(quants.len(), 1, "the three projections share one quant");
    assert_eq!(quants[0].args[3..5], [u32_arg(M), u32_arg(h)]);
    let gemms = launches(&gpu, W8A8);
    assert_eq!(gemms.len(), 3);
    for (g, n) in gemms.iter().zip([q_n, kv_n, kv_n]) {
        assert_eq!(g.args[5..8], [u32_arg(M), u32_arg(n), u32_arg(h)]);
        assert_eq!(g.grid, [n.div_ceil(128), M.div_ceil(64), 1]);
    }
    let q = layer.q_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
    assert_eq!(gemms[0].args[2], MockArg::Buffer(q.weight));
    assert_eq!(gemms[0].args[3], MockArg::Buffer(q.row_scale));
    assert!(launches(&gpu, W8A16_PIPE).is_empty());
}

/// Every other FP8 attention build (flag unset, no cuBLASLt): the chain stays
/// on W8A16 exactly as before.
#[test]
fn without_the_route_flag_the_chain_is_unchanged() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, false);
    run_qkv(&gpu, &config, &layer, &GemmDispatch::defaults());
    assert!(launches(&gpu, W8A8).is_empty() && launches(&gpu, QUANT).is_empty());
    assert_eq!(launches(&gpu, W8A16_PIPE).len(), 3);
}

/// `AVAROK_FP8_SINGLE_SCALE` turns the route's arm off as well.
#[test]
fn the_single_scale_kill_switch_returns_the_chain_to_w8a16() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, true);
    let dispatch = GemmDispatch {
        fp8_blockscaled_prefill: false,
        ..GemmDispatch::defaults()
    };
    run_qkv(&gpu, &config, &layer, &dispatch);
    assert!(launches(&gpu, W8A8).is_empty());
    assert_eq!(launches(&gpu, W8A16_PIPE).len(), 3);
}

/// The pure clause list of the in-tree arm: each clause declines on its own.
#[test]
fn the_in_tree_arm_needs_every_clause() {
    use super::super::prefill_qkv_w8a8::cache_skip_qkv_kernel_selected as sel;
    use crate::weight_map::WeightQuantFormat::{Fp8BlockScaled, Fp8PerRow};
    let blk = [Some(Fp8BlockScaled); 3];
    let (cap, scap) = (1 << 24, 1 << 20);
    let q = Fp8ActQuant::shared_only(QUANT);
    assert!(sel(
        true, true, false, blk, M, 8192, 1024, 2560, cap, scap, q, W8A8
    ));
    let declines = [
        sel(
            false, true, false, blk, M, 8192, 1024, 2560, cap, scap, q, W8A8,
        ),
        sel(
            true, false, false, blk, M, 8192, 1024, 2560, cap, scap, q, W8A8,
        ),
        sel(
            true, true, true, blk, M, 8192, 1024, 2560, cap, scap, q, W8A8,
        ),
        sel(
            true,
            true,
            false,
            [Some(Fp8BlockScaled), Some(Fp8PerRow), Some(Fp8BlockScaled)],
            M,
            8192,
            1024,
            2560,
            cap,
            scap,
            q,
            W8A8,
        ),
        sel(
            true, true, false, blk, 0, 8192, 1024, 2560, cap, scap, q, W8A8,
        ),
        sel(
            true, true, false, blk, M, 8000, 1024, 2560, cap, scap, q, W8A8,
        ),
        sel(
            true, true, false, blk, M, 8192, 1000, 2560, cap, scap, q, W8A8,
        ),
        sel(
            true, true, false, blk, M, 8192, 1024, 2500, cap, scap, q, W8A8,
        ),
        sel(
            true, true, false, blk, M, 8192, 1024, 2560, 100, scap, q, W8A8,
        ),
        sel(
            true, true, false, blk, M, 8192, 1024, 2560, cap, 100, q, W8A8,
        ),
        sel(
            true,
            true,
            false,
            blk,
            M,
            8192,
            1024,
            2560,
            cap,
            scap,
            Fp8ActQuant::shared_only(KernelHandle(0)),
            W8A8,
        ),
        sel(
            true,
            true,
            false,
            blk,
            M,
            8192,
            1024,
            2560,
            cap,
            scap,
            q,
            KernelHandle(0),
        ),
    ];
    for (i, d) in declines.iter().enumerate() {
        assert!(!d, "clause {i} must decline the arm");
    }
}

/// The constructor takes the sm_89 drop-ins where the target compiled them,
/// and the old handles where it did not.
#[test]
fn the_constructor_prefers_the_ada_drop_ins_only_where_they_exist() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let has =
        |seen: &[(String, String)], m: &str, f: &str| seen.iter().any(|(a, b)| a == m && b == f);
    let ada = MockGpuBackend::new();
    let _ = native_fp8_attention_layer(&ada, &config);
    let seen = ada.kernel_lookups_snapshot();
    assert!(has(&seen, "fp8_gemv_ada", "fp8_gemv_ada"));
    assert!(has(&seen, "fp8_gemm_ada", "fp8_gemm_ada_w8a8"));

    let other = MockGpuBackend::new();
    other.mark_module_absent("fp8_gemv_ada");
    other.mark_module_absent("fp8_gemm_ada");
    let _ = native_fp8_attention_layer(&other, &config);
    let seen = other.kernel_lookups_snapshot();
    assert!(!has(&seen, "fp8_gemv_ada", "fp8_gemv_ada"));
    assert!(!has(&seen, "fp8_gemm_ada", "fp8_gemm_ada_w8a8"));
    assert!(has(&seen, "w8a16_gemv", "w8a16_gemv"));
    assert!(has(
        &seen,
        "fp8_gemm_t_blockscaled",
        "fp8_gemm_t_blockscaled"
    ));
}
