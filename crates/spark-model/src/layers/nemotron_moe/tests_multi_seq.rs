// SPDX-License-Identifier: AGPL-3.0-only

//! Mock-dispatch tests for `NemotronMoeLayer::decode_multi_seq`.
//!
//! The mock's `kernel()` hands out one shared handle, so launch GEOMETRY is
//! the per-launch identity. With h=2688, E=128, top_k=6, inter=1024:
//!   batched sigmoid routing   ([1, n, 1], [256,1,1])   — unique at n>=2
//!   grouped expert GEMMs      grid.z == num_experts (128) — unique
//!   per-token expert GEMV     ([ceil(1024/4)=256, top_k=6, 1], [128,1,1])
//!   single-token routing      ([1,1,1], [256,1,1]) — shared with the gate
//!                             GEMM and the expert sort, so it is asserted
//!                             by COUNT (see each test).
//!
//! These fail without the `decode_multi_seq` override: the default loop
//! yields n single-token routing launches and n×top_k expert GEMVs.

use super::*;
use crate::layer::{ForwardContext, LayerState, TransformerLayer};
use crate::weight_map::NemotronExpertWeight;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;

fn lightning_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 2688;
    c.mamba_num_heads = 64;
    c.mamba_head_dim = 64;
    c.ssm_state_size = 128;
    c.n_groups = 8;
    c.linear_conv_kernel_dim = 4;
    c.num_experts = 128;
    c.num_experts_per_tok = 6;
    c.moe_intermediate_size = 1024;
    c.shared_expert_intermediate_size = 1024;
    c.model_type = "nemotronh".to_string();
    c
}

fn mk_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> NemotronMoeLayer {
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
    let inter = config.moe_intermediate_size;
    let shared_inter = config.shared_expert_intermediate_size;
    let experts: Vec<NemotronExpertWeight> = (0..config.num_experts)
        .map(|_| NemotronExpertWeight {
            up_proj: qw(inter, h),
            down_proj: qw(h, inter),
        })
        .collect();
    let weights = NemotronMoeWeights {
        gate: dw(config.num_experts * h * 2),
        e_score_correction_bias: dw(config.num_experts * 4),
        experts,
        shared_up: qw(shared_inter, h),
        shared_up_fp8: None,
        shared_down: qw(h, shared_inter),
        shared_down_fp8: None,
        fc1_latent_proj: None,
        fc2_latent_proj: None,
    };
    NemotronMoeLayer::new(weights, dw(h * 2), config, gpu, 0, 0).unwrap()
}

fn run_multi_seq(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &NemotronMoeLayer,
    n: usize,
) -> anyhow::Result<()> {
    run_multi_seq_impl(gpu, config, layer, n, false)
}

/// Drive the batched body DIRECTLY (`decode_multi_seq_inner`), bypassing the
/// `MOE_BATCH_DECODE_MIN_SEQS` gate — the geometry tests keep the sorted
/// dispatch alive while the gate keeps it out of production decode.
fn run_multi_seq_inner(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &NemotronMoeLayer,
    n: usize,
) -> anyhow::Result<()> {
    run_multi_seq_impl(gpu, config, layer, n, true)
}

fn run_multi_seq_impl(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &NemotronMoeLayer,
    n: usize,
    direct_inner: bool,
) -> anyhow::Result<()> {
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
        // Added on main after these tests were written (MoE-LoRA fold
        // decision). `Fold` is the default that keeps legacy
        // single-request call sites byte-identical, which is what a
        // batched-decode shape test wants.
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
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
    let mut owned: Vec<Box<dyn LayerState>> =
        (0..n).map(|_| Box::new(EmptyLayerState) as _).collect();
    let mut states: Vec<&mut (dyn LayerState + 'static)> =
        owned.iter_mut().map(|b| &mut **b).collect();
    let seq_lens = vec![1usize; n];
    let block_tables = vec![vec![0u32]; n];
    if direct_inner {
        return layer.decode_multi_seq_inner(
            buffers.hidden_states(),
            buffers.residual(),
            n,
            &ctx,
            0,
        );
    }
    layer.decode_multi_seq(
        buffers.hidden_states(),
        buffers.residual(),
        n,
        &mut states,
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

/// The batched body (driven directly — production decode gates it off, see
/// `MOE_BATCH_DECODE_MIN_SEQS`) at n=4 must take the sorted dispatch: one
/// batched routing launch, grouped GEMMs over all experts, zero per-token
/// expert GEMVs and zero single-token routing launches.
#[test]
fn moe_multi_seq_uses_sorted_path() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let layer = mk_layer(&gpu, &config);
    assert!(layer.can_batch_decode());
    run_multi_seq_inner(&gpu, &config, &layer, 4).unwrap();

    let seen = grids(&gpu);
    // Batched sigmoid routing: grid [1, n, 1].
    assert_eq!(
        count(&seen, [1, 4, 1], [256, 1, 1]),
        1,
        "expected ONE nemotron_moe_topk_sigmoid_batched launch; grids: {seen:?}"
    );
    // Grouped expert GEMMs put num_experts on grid.z — at least UP and DOWN.
    let grouped = seen.iter().filter(|(g, _)| g[2] == 128).count();
    assert!(
        grouped >= 2,
        "expected grouped UP+DOWN GEMMs (grid.z=128), saw {grouped}; grids: {seen:?}"
    );
    // No per-token expert GEMV ([ceil(inter/4), top_k, 1]).
    assert_eq!(
        count(&seen, [256, 6, 1], [128, 1, 1]),
        0,
        "per-token moe_expert_gemv launched — sorted path not engaged"
    );
    // ([1,1,1],[256,1,1]) is shared by the gate GEMM (ceil(128/128)=1,
    // ceil(4/128)=1) and the expert sort. Exactly those two — a third would
    // be a single-token `moe_topk_sigmoid` sneaking back in.
    assert_eq!(
        count(&seen, [1, 1, 1], [256, 1, 1]),
        2,
        "expected exactly gate GEMM + moe_sort at [1,1,1]x[256] (no \
         single-token routing); grids: {seen:?}"
    );
}

/// With every sorted-path kernel PRESENT, production `decode_multi_seq` at
/// n=4 (a reachable padded rung) must still take the per-seq default loop:
/// the batched body is gated off by `MOE_BATCH_DECODE_MIN_SEQS` (measured
/// net loss at every profiled rung + temp-0 answer flips from C=2 up).
/// Fails if the gate is ever lowered without re-measuring.
#[test]
fn moe_multi_seq_gated_off_uses_default_loop() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let layer = mk_layer(&gpu, &config);
    assert!(layer.can_batch_decode());
    run_multi_seq(&gpu, &config, &layer, 4).unwrap();

    let seen = grids(&gpu);
    // No batched routing, no grouped GEMMs.
    assert_eq!(count(&seen, [1, 4, 1], [256, 1, 1]), 0);
    assert_eq!(seen.iter().filter(|(g, _)| g[2] == 128).count(), 0);
    // 4 single-token routing launches + 4 per-token expert GEMVs — the
    // default-loop tells.
    assert_eq!(
        count(&seen, [1, 1, 1], [256, 1, 1]),
        4,
        "expected 4 per-seq moe_topk_sigmoid launches; grids: {seen:?}"
    );
    assert_eq!(
        count(&seen, [256, 6, 1], [128, 1, 1]),
        4,
        "expected 4 per-seq moe_expert_gemv launches; grids: {seen:?}"
    );
}

/// Zeroing one sorted-path kernel must engage the per-seq fallback loop —
/// n single-token routing launches, n×top_k expert GEMVs, no crash.
#[test]
fn moe_multi_seq_missing_kernels_falls_back() {
    let config = lightning_config();
    let gpu = MockGpuBackend::new();
    let mut layer = mk_layer(&gpu, &config);
    layer.moe_sort_k = KernelHandle(0);
    assert!(!layer.can_batch_decode());
    run_multi_seq(&gpu, &config, &layer, 4).unwrap();

    let seen = grids(&gpu);
    // No batched routing, no grouped GEMMs.
    assert_eq!(count(&seen, [1, 4, 1], [256, 1, 1]), 0);
    assert_eq!(seen.iter().filter(|(g, _)| g[2] == 128).count(), 0);
    // 4 single-token routing launches (the only [1,1,1]x[256] in
    // decode_inner — the gate is a GEMV there, grid [ceil(128/4)=32,1,1]).
    assert_eq!(
        count(&seen, [1, 1, 1], [256, 1, 1]),
        4,
        "expected 4 per-seq moe_topk_sigmoid launches; grids: {seen:?}"
    );
    // 4 per-token batched-over-top_k expert GEMVs.
    assert_eq!(
        count(&seen, [256, 6, 1], [128, 1, 1]),
        4,
        "expected 4 per-seq moe_expert_gemv launches; grids: {seen:?}"
    );
}
