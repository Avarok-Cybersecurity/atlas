// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-dispatch tests for `NemotronMamba2Layer::decode_multi_seq`.
//!
//! The mock's `kernel()` hands out one shared handle, so launch GEOMETRY
//! (grid/block) is the per-launch identity. With the Lightning-30B shapes
//! (h=2688, d_inner=4096, d_xbc=6144, in_proj_size=10304) every asserted
//! kernel has a unique signature:
//!   batched in_proj  GEMM  ([ceil(10304/128)=81, 1, 1], [256,1,1])
//!   batched out_proj GEMM  ([ceil(2688/128)=21, 1, 1], [256,1,1])
//!   per-row conv1d_update  ([ceil(6144/256)=24, 1, 1], [256,1,1])
//!   per-row mamba2_ssm     ([num_heads=64, 1, 1], [state_size=128,1,1])
//!   batched norms          ([n, 1, 1], [1024,1,1]) — rms_norm + gated
//!   per-row in_proj GEMV   ([ceil(10304/4)=2576, 1, 1], [256,1,1]) — the
//!                          default-loop tell; must be ABSENT when batching.
//!
//! These fail without the `decode_multi_seq` override: the default loop
//! yields n GEMV-geometry launches and n single-token norm grids of (1,·).

use super::*;
use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;

/// Nemotron-3.5 Lightning-30B mamba dimensions on the stock test template.
fn lightning_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 2688;
    c.mamba_num_heads = 64; // d_inner = 64*64 = 4096
    c.mamba_head_dim = 64;
    c.ssm_state_size = 128;
    c.n_groups = 8; // gs = 1024, d_xbc = 6144, in_proj_size = 10304
    c.linear_conv_kernel_dim = 4;
    c.num_experts = 128;
    c.num_experts_per_tok = 6;
    c.moe_intermediate_size = 1024;
    c.shared_expert_intermediate_size = 1024;
    c.model_type = "nemotronh".to_string();
    c
}

fn mk_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> NemotronMamba2Layer {
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let qw = |n: usize, k: usize| QuantizedWeight {
        weight: gpu.alloc(n * k / 2).unwrap(),
        weight_scale: gpu.alloc(n * k / 16).unwrap(),
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    };
    let h = config.hidden_size;
    let d_inner = config.mamba2_d_inner();
    let d_xbc = config.mamba2_d_xbc();
    let ssm = NemotronSsmWeights {
        in_proj: qw(config.mamba2_in_proj_size(), h),
        out_proj: qw(h, d_inner),
        conv1d_weight: dw(d_xbc * config.linear_conv_kernel_dim * 2),
        conv1d_bias: dw(d_xbc * 2),
        a_log: dw(config.mamba_num_heads * 4),
        d_param: dw(config.mamba_num_heads * 2),
        dt_bias: dw(config.mamba_num_heads * 2),
        ssm_norm: dw(d_inner * 2),
    };
    NemotronMamba2Layer::new(dw(h * 2), ssm, config, gpu, 0).unwrap()
}

/// Drive `decode_multi_seq` with `n` pool-style SSM states.
fn run_multi_seq(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &NemotronMamba2Layer,
    states: &mut [&mut (dyn LayerState + 'static)],
) -> anyhow::Result<()> {
    let n = states.len();
    let buffers = BufferArena::new(config, 64, 4096, 16, 32, gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch::defaults();
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        gpu,
        config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
    };
    let kv_config = spark_runtime::kv_cache::KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 128,
        num_layers: config.num_hidden_layers,
        dtype: spark_runtime::kv_cache::KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let mut kv = spark_runtime::kv_cache::PagedKvCache::new(kv_config, 8, gpu).unwrap();
    let seq_lens = vec![1usize; n];
    let block_tables = vec![vec![0u32]; n];
    layer.decode_multi_seq(
        buffers.hidden_states(),
        buffers.residual(),
        n,
        states,
        &mut kv,
        &seq_lens,
        &block_tables,
        &ctx,
        0,
    )
}

fn count(seen: &[([u32; 3], [u32; 3])], grid: [u32; 3], block: [u32; 3]) -> usize {
    seen.iter()
        .filter(|&&(g, b)| g == grid && b == block)
        .count()
}

fn grids(gpu: &MockGpuBackend) -> Vec<([u32; 3], [u32; 3])> {
    gpu.launches_snapshot()
        .into_iter()
        .map(|l| (l.grid, l.block))
        .collect()
}

/// n=4 on the native-BF16 arm: projections batch into ONE GEMM each, the
/// norms batch to grid.x==4, and the conv/scan inner stays a 4-row loop.
#[test]
fn mamba2_multi_seq_batches_projections() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let mut layer = mk_layer(&gpu, &config);
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    layer.set_bf16_weights(
        dw(config.mamba2_in_proj_size() * config.hidden_size * 2),
        dw(config.hidden_size * config.mamba2_d_inner() * 2),
    );

    let mut owned: Vec<Box<dyn LayerState>> =
        (0..4).map(|_| layer.alloc_state(&gpu).unwrap()).collect();
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    run_multi_seq(&gpu, &config, &layer, &mut states).unwrap();

    let seen = grids(&gpu);
    // ONE batched in_proj GEMM (not 4 GEMVs) and ONE batched out_proj GEMM.
    assert_eq!(
        count(&seen, [81, 1, 1], [256, 1, 1]),
        1,
        "expected exactly one batched in_proj dense GEMM; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [21, 1, 1], [256, 1, 1]),
        1,
        "expected exactly one batched out_proj dense GEMM; grids: {seen:?}"
    );
    // Batched input norm + batched gated norm, each grid.x == 4.
    assert_eq!(
        count(&seen, [4, 1, 1], [1024, 1, 1]),
        2,
        "expected batched rms_norm_residual + gated_rms_norm at grid.x=4; grids: {seen:?}"
    );
    // Per-row conv + scan: exactly 4 of each, batch=1 geometry.
    assert_eq!(count(&seen, [24, 1, 1], [256, 1, 1]), 4, "conv1d per-row");
    assert_eq!(
        count(&seen, [64, 1, 1], [128, 1, 1]),
        4,
        "ssm_decode per-row"
    );
    // The default-loop tell (per-row in_proj GEMV) must be absent.
    assert_eq!(
        count(&seen, [2576, 1, 1], [256, 1, 1]),
        0,
        "per-row in_proj GEMV launched — batched override not engaged"
    );
}

/// n=24 (a real padded_n rung) on the NVFP4 arm must use the any-M tile GEMM,
/// never the batchm GEMV family (which silently truncates above M=16).
#[test]
fn mamba2_multi_seq_n24_uses_gemm_not_batchm() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let layer = mk_layer(&gpu, &config); // NVFP4 arm: no BF16/FP8 installed

    let mut owned: Vec<Box<dyn LayerState>> =
        (0..24).map(|_| layer.alloc_state(&gpu).unwrap()).collect();
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    run_multi_seq(&gpu, &config, &layer, &mut states).unwrap();

    let seen = grids(&gpu);
    // w4a16_gemm in_proj: ([ceil(10304/64)=161, ceil(24/64)=1, 1], [128,1,1]).
    assert_eq!(
        count(&seen, [161, 1, 1], [128, 1, 1]),
        1,
        "expected the in_proj w4a16 tile GEMM at n=24; grids: {seen:?}"
    );
    // w4a16_gemm out_proj: ([ceil(2688/64)=42, 1, 1], [128,1,1]).
    assert_eq!(
        count(&seen, [42, 1, 1], [128, 1, 1]),
        1,
        "expected the out_proj w4a16 tile GEMM at n=24; grids: {seen:?}"
    );
    // batchm GEMV geometry (== single-GEMV geometry) must be absent: at n=24
    // the batch16 kernel would compute rows 0..15 and leave 16..23 garbage.
    assert_eq!(
        count(&seen, [2576, 1, 1], [256, 1, 1]),
        0,
        "w4a16_gemv_batchm (cap 16) launched at n=24 — silent-truncation hazard"
    );
}

/// A non-SSM state in the batch must be a clean `Err`, not UB.
#[test]
fn mamba2_multi_seq_bad_state_errors() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let layer = mk_layer(&gpu, &config);

    let mut ok_state = layer.alloc_state(&gpu).unwrap();
    let mut bad_state: Box<dyn LayerState> = Box::new(EmptyLayerState);
    let mut states: Vec<&mut (dyn LayerState + 'static)> = vec![&mut *ok_state, &mut *bad_state];
    let err = run_multi_seq(&gpu, &config, &layer, &mut states)
        .expect_err("EmptyLayerState in the batch must be refused");
    assert!(
        format!("{err:#}").contains("Expected SsmLayerState for seq 1"),
        "wrong error: {err:#}"
    );
}
