// SPDX-License-Identifier: AGPL-3.0-only

//! Which kernels a GDN layer built by the load-time BF16 -> FP8 route
//! (`weight_loader/qwen35_dense/gdn_fp8_from_bf16.rs`) launches on each path:
//! M=1 decode, prefill QKVZ and prefill `out_proj`, and which handles the
//! constructor resolves on a target with and without the sm_89 drop-ins.
//!
//! The route builds the layer exactly as the native-FP8 GDN arm does, so the
//! fixture is `native_fp8_gdn_layer` plus the one setter the route adds. The
//! batched decode and verify arms for that layout are pinned in `tests.rs`.
//! Every handle under test is a sentinel, so a launch names its kernel.

use super::tests::{mk_state, native_fp8_gdn_layer};
use super::*;
use avarok_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

const QUANT: KernelHandle = KernelHandle(0xA1);
const W8A8: KernelHandle = KernelHandle(0xA8);
const W8A16_PIPE: KernelHandle = KernelHandle(0xB1);
const W8A16_GEMM: KernelHandle = KernelHandle(0xB2);
const GEMV: KernelHandle = KernelHandle(0xC1);
const DENSE_GEMV: KernelHandle = KernelHandle(0xC2);

/// The 80B GDN shapes: hidden 2048, value_dim 4096, so `out_proj`'s N and K
/// differ, which is what makes a swapped (N, K) visible.
fn route_layer(gpu: &MockGpuBackend, config: &ModelConfig, w8a8_out: bool) -> Qwen3SsmLayer {
    let mut layer = native_fp8_gdn_layer(gpu, config, true, true);
    layer.set_w8a8_out_proj_prefill(w8a8_out);
    layer.per_token_group_quant_fp8_k = ops::Fp8ActQuant {
        shared: QUANT,
        hopper: KernelHandle(0),
    };
    layer.fp8_gemm_t_blockscaled_k = W8A8;
    layer.w8a16_gemm_pipelined_k = W8A16_PIPE;
    layer.w8a16_gemm_k = W8A16_GEMM;
    layer.w8a16_gemv_k = GEMV;
    layer.dense_gemv_k = DENSE_GEMV;
    layer
}

fn u32_arg(v: usize) -> MockArg {
    MockArg::Bytes((v as u32).to_le_bytes().to_vec())
}

fn count(gpu: &MockGpuBackend, k: KernelHandle) -> usize {
    gpu.launches_snapshot()
        .iter()
        .filter(|l| l.func == k.0)
        .count()
}

/// Run the two prefill projections at M = 28 (the launches land on `gpu`).
fn run_prefill(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &Qwen3SsmLayer,
    dispatch: &ops::GemmDispatch,
) {
    let buffers = BufferArena::new(config, 64, 4096, 16, 32, gpu).unwrap();
    let (derived, levers) = (ops::DerivedWeights::new(), ops::ModelLevers::defaults());
    let stats = ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu,
        config,
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
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    let c = config;
    let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;
    layer
        .prefill_qkvz_proj(
            buffers.norm_output(),
            buffers.ssm_deinterleaved(),
            28,
            c.ssm_qkvz_size(),
            c.hidden_size,
            c.linear_num_key_heads,
            c.linear_key_head_dim,
            c.linear_num_value_heads / c.linear_num_key_heads.max(1),
            c.linear_value_head_dim,
            &ctx,
            0,
        )
        .expect("QKVZ prefill");
    layer
        .prefill_out_proj_dispatch(
            &ctx,
            buffers.norm_output(),
            buffers.hidden_states(),
            28,
            c.hidden_size,
            value_dim,
            0,
        )
        .expect("out_proj prefill");
}

/// Route on, defaults: both prefill projections are W8A8 (one activation
/// quant and one block-scaled FP8 GEMM each), nothing on W8A16, and the
/// `out_proj` GEMM is (M, N = hidden, K = value_dim).
#[test]
fn route_prefill_runs_both_projections_on_the_w8a8_pair() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, true);
    run_prefill(&gpu, &config, &layer, &ops::GemmDispatch::defaults());
    assert_eq!(count(&gpu, QUANT), 2, "one activation quant per projection");
    assert_eq!(count(&gpu, W8A8), 2, "one W8A8 GEMM per projection");
    assert_eq!(count(&gpu, W8A16_PIPE) + count(&gpu, W8A16_GEMM), 0);

    let (h, vd, qkvz) = (
        config.hidden_size,
        config.linear_num_value_heads * config.linear_value_head_dim,
        config.ssm_qkvz_size(),
    );
    let gemms: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == W8A8.0)
        .collect();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    let qz = layer.qkvz_fp8w.as_ref().unwrap();
    assert_eq!(gemms[0].args[2], MockArg::Buffer(qz.weight));
    assert_eq!(
        gemms[0].args[5..8],
        [u32_arg(28), u32_arg(qkvz), u32_arg(h)]
    );
    assert_eq!(gemms[1].args[2], MockArg::Buffer(out.weight));
    assert_eq!(gemms[1].args[3], MockArg::Buffer(out.row_scale));
    assert_eq!(
        gemms[1].args[5..8],
        [u32_arg(28), u32_arg(h), u32_arg(vd)],
        "out_proj is [M, value_dim] x [hidden, value_dim]^T"
    );
    assert_eq!(gemms[1].grid, [(h / 128) as u32, 1, 1]);
    // The activation quant of out_proj's input runs over K = value_dim.
    let quants: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == QUANT.0)
        .collect();
    assert_eq!(quants[1].args[3..5], [u32_arg(28), u32_arg(vd)]);
}

/// Every other FP8 GDN build (the flag unset): QKVZ keeps its W8A8 default and
/// `out_proj` keeps the W8A16 pipelined GEMM it always had.
#[test]
fn without_the_route_flag_out_proj_prefill_is_unchanged() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, false);
    run_prefill(&gpu, &config, &layer, &ops::GemmDispatch::defaults());
    assert_eq!(count(&gpu, W8A8), 1, "QKVZ only");
    assert_eq!(count(&gpu, W8A16_PIPE), 1, "out_proj on W8A16 as before");
}

/// `AVAROK_FP8_SINGLE_SCALE` (`fp8_blockscaled_prefill = false`) turns the
/// route's W8A8 off for both projections: W8A16 on the same FP8 weights.
#[test]
fn the_single_scale_kill_switch_returns_both_projections_to_w8a16() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, true);
    let dispatch = ops::GemmDispatch {
        fp8_blockscaled_prefill: false,
        ..ops::GemmDispatch::defaults()
    };
    run_prefill(&gpu, &config, &layer, &dispatch);
    assert_eq!(count(&gpu, W8A8) + count(&gpu, QUANT), 0);
    assert_eq!(count(&gpu, W8A16_PIPE), 2);
}

/// M=1 decode reads the FP8 pair through `w8a16_gemv_k` twice (QKVZ and
/// `out_proj`) and never the NULL BF16 slot through `dense_gemv`.
#[test]
fn route_decode_m1_reads_only_the_fp8_pair() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = route_layer(&gpu, &config, true);
    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let dispatch = ops::GemmDispatch::defaults();
    let (derived, levers) = (ops::DerivedWeights::new(), ops::ModelLevers::defaults());
    let stats = ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: true,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };
    let mut state = mk_state(&gpu, &layer, 0);
    layer
        .ssm_forward(buffers.norm_output(), &mut state, &ctx, 0, false)
        .expect("M=1 decode");
    let gemvs: Vec<_> = gpu
        .launches_snapshot()
        .into_iter()
        .filter(|l| l.func == GEMV.0)
        .collect();
    assert_eq!(gemvs.len(), 2, "QKVZ and out_proj");
    assert_eq!(
        gemvs[0].args[1],
        MockArg::Buffer(layer.qkvz_fp8w.as_ref().unwrap().weight)
    );
    assert_eq!(
        gemvs[1].args[1],
        MockArg::Buffer(layer.out_proj_fp8w.as_ref().unwrap().weight)
    );
    assert_eq!(count(&gpu, DENSE_GEMV), 0, "the BF16 slot is NULL");
}

/// The constructor takes the sm_89 drop-ins where the target compiled them
/// and issues exactly the old lookups where it did not.
#[test]
fn the_constructor_prefers_the_ada_drop_ins_only_where_they_exist() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let ada = MockGpuBackend::new();
    let _ = native_fp8_gdn_layer(&ada, &config, true, true);
    let seen = ada.kernel_lookups_snapshot();
    let has =
        |seen: &[(String, String)], m: &str, f: &str| seen.iter().any(|(a, b)| a == m && b == f);
    assert!(has(&seen, "fp8_gemv_ada", "fp8_gemv_ada"));
    assert!(has(&seen, "fp8_gemm_ada", "fp8_gemm_ada_w8a8"));

    let other = MockGpuBackend::new();
    other.mark_module_absent("fp8_gemv_ada");
    other.mark_module_absent("fp8_gemm_ada");
    let _ = native_fp8_gdn_layer(&other, &config, true, true);
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
