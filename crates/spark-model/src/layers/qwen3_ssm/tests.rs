// SPDX-License-Identifier: AGPL-3.0-only

//! Extracted piecewise from `qwen3_ssm/mod.rs` (500-LoC cap).

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn test_ssm_state_allocation_sizes() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let nv = config.linear_num_value_heads; // 32
    let vd = config.linear_value_head_dim; // 128
    let nk = config.linear_num_key_heads; // 16
    let kd = config.linear_key_head_dim; // 128
    let d_conv = config.linear_conv_kernel_dim; // 4

    let h_bytes = nv * vd * kd * 4;
    assert_eq!(h_bytes, 32 * 128 * 128 * 4); // 2 MB

    // conv_dim = 2*key_dim + value_dim = 2*2048 + 4096 = 8192
    let conv_dim = nk * kd * 2 + nv * vd;
    let conv_bytes = conv_dim * d_conv * 4;
    assert_eq!(conv_bytes, 8192 * 4 * 4); // 128 KB

    // Verify allocations
    let gpu = MockGpuBackend::new();
    let h_state = gpu.alloc(h_bytes).unwrap();
    let conv_state = gpu.alloc(conv_bytes).unwrap();
    assert!(!h_state.is_null());
    assert!(!conv_state.is_null());
}

// ── Batched-verify QKVZ/out_proj dispatch on native-FP8-GDN checkpoints ──
//
// Regression tests for the CUDA_ERROR_ILLEGAL_ADDRESS at the FIRST n>=2
// batched MTP verify (ks=[4,3] ⇒ R=7) on nvidia/Qwen3.6-27B-NVFP4: the
// qwen35_dense.rs native-FP8 GDN arm leaves the dense/NVFP4 QKVZ and
// out_proj slots NULL (`qkvz_fp8w`/`out_proj_fp8w` are the only live
// weights), and `decode_batched_inner`'s fp8w arms stopped at
// num_tokens <= 4 — so R > 4 fell through to `dense_gemm`/`w4a16_gemm`
// on the NULL slots, destroying the CUDA context (sticky 700).
// Localized on hardware via ATLAS_K4_DIAG=1: "CUDA error after GDN phase
// `2+3:qkvz_proj+deinterleave`".

use crate::layer::TransformerLayer;
use crate::weight_map::WeightQuantFormat;
use spark_runtime::buffers::BufferArena;

/// Wire a layer exactly like the qwen35_dense.rs native-FP8 GDN arm:
/// dense QKVZ slot NULL, out_proj a null QuantizedWeight, no NVFP4 fields;
/// `with_qkvz_fp8w` / `with_out_fp8w` control the block-scaled FP8 pair.
fn native_fp8_gdn_layer(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    with_qkvz_fp8w: bool,
    with_out_fp8w: bool,
) -> Qwen3SsmLayer {
    let h = config.hidden_size;
    let qkvz_size = config.ssm_qkvz_size();
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let dw = |bytes: usize| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let ssm = SsmWeights {
        in_proj_qkvz: DenseWeight {
            weight: DevicePtr::NULL,
        },
        in_proj_ba: dw(config.ssm_ba_size() * h * 2),
        conv1d: dw(
            (2 * config.linear_num_key_heads * config.linear_key_head_dim + value_dim)
                * config.linear_conv_kernel_dim
                * 2,
        ),
        a_log: dw(config.linear_num_value_heads * 4),
        dt_bias: dw(config.linear_num_value_heads * 4),
        norm: dw(config.linear_value_head_dim * 2),
        out_proj: QuantizedWeight::null(),
    };
    let mut layer = Qwen3SsmLayer::new_sequential(
        dw(h * 2),
        ssm,
        dw(h * 2),
        FfnComponent::None,
        None,
        None,
        None,
        config,
        gpu,
    )
    .unwrap();
    let fp8 = |n: usize, k: usize| Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc((n / 128) * (k / 128) * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.set_fp8_decode_weights(
        with_qkvz_fp8w.then(|| fp8(qkvz_size, h)),
        with_out_fp8w.then(|| fp8(h, value_dim)),
    );
    layer
}

/// SSM state with `n_inter` pool-style intermediates (enough for K=4).
fn mk_state(gpu: &MockGpuBackend, layer: &Qwen3SsmLayer, n_inter: usize) -> SsmLayerState {
    let h_bytes = layer.h_state_bytes;
    let conv_bytes = layer.conv_state_bytes;
    // One contiguous slab per family, mirroring the ssm_pool layout.
    let h_slab = gpu.alloc(h_bytes * n_inter).unwrap();
    let conv_slab = gpu.alloc(conv_bytes * n_inter).unwrap();
    SsmLayerState {
        h_state: gpu.alloc(h_bytes).unwrap(),
        conv_state: gpu.alloc(conv_bytes).unwrap(),
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: (0..n_inter).map(|i| h_slab.offset(i * h_bytes)).collect(),
        conv_state_intermediates: (0..n_inter)
            .map(|i| conv_slab.offset(i * conv_bytes))
            .collect(),
        h_is_f16: false,
    }
}

/// Drive the batched verify body at ragged ks through `decode_verify_multi`
/// (the verify_e.rs entry point) and return the result.
fn run_batched_verify(
    gpu: &MockGpuBackend,
    config: &ModelConfig,
    layer: &Qwen3SsmLayer,
    ks: &[usize],
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
        // Merge-interaction (#334/#335 stack): this main-side helper postdates
        // #335's base. `Fold` is the documented default and inert on verify
        // paths (they bail via `reject_decode_lora` before the fold) — same
        // convention as `layer/tests.rs`.
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
    let mut states_own: Vec<SsmLayerState> = ks.iter().map(|_| mk_state(gpu, layer, 4)).collect();
    let mut states: Vec<&mut (dyn LayerState + 'static)> = states_own
        .iter_mut()
        .map(|s| s as &mut (dyn LayerState + 'static))
        .collect();
    layer.decode_verify_multi(
        buffers.hidden_states(),
        buffers.residual(),
        ks.len(),
        ks,
        &mut states,
        &mut kv,
        DevicePtr::NULL, // no staged WY tables → per-sequence GDN loop
        &ctx,
        0,
    )
}

/// POSITIVE: R = 4+3 = 7 on an fp8w-only layer must dispatch the
/// block-scaled W8A16 GEMM for BOTH projections and succeed. Before the
/// fix this fell through to `dense_gemm`/`w4a16_gemm` on NULL slots
/// (device 700 in production; here the fail-fast guards turn it into Err,
/// so `is_ok` is the load-bearing assertion).
#[test]
fn native_fp8_gdn_batched_verify_r7_dispatches_w8a16_gemm() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[4, 3]).unwrap();
    // Pin the arm identity: w8a16_gemm_pipelined geometry at M=7 —
    // QKVZ (N=12288): grid [ceil(12288/32)=384, ceil(7/128)=1, 1];
    // out_proj (N=2048): grid [64, 1, 1]; both block [256,1,1].
    let launches = (0..gpu.launch_count()).count();
    assert!(launches > 0);
    let seen = gpu_launch_grids(&gpu);
    assert!(
        seen.contains(&([384, 1, 1], [256, 1, 1])),
        "QKVZ w8a16_gemm_pipelined launch missing; grids seen: {seen:?}"
    );
    assert!(
        seen.contains(&([64, 1, 1], [256, 1, 1])),
        "out_proj w8a16_gemm_pipelined launch missing; grids seen: {seen:?}"
    );
}

/// POSITIVE (existing behavior guard): uniform R = 2+2 = 4 keeps the
/// M<=4 `w8a16_gemv_batch4` arm working.
#[test]
fn native_fp8_gdn_batched_verify_r4_still_ok() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[2, 2]).unwrap();
}

/// NEGATIVE: no QKVZ weight in ANY form at R=7 must fail fast with the
/// dispatch error — never launch `dense_gemm` on the NULL dense slot
/// (that launch is the production context-killer).
#[test]
fn batched_verify_null_qkvz_fails_fast() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, false, false);
    let err = run_batched_verify(&gpu, &config, &layer, &[4, 3])
        .expect_err("NULL QKVZ slot must be refused, not launched");
    assert!(
        format!("{err:#}").contains("batched GDN QKVZ dispatch"),
        "wrong error: {err:#}"
    );
}

/// NEGATIVE: QKVZ fp8w present but out_proj missing in every form at R=7
/// must fail fast on the out_proj guard.
#[test]
fn batched_verify_null_out_proj_fails_fast() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, false);
    let err = run_batched_verify(&gpu, &config, &layer, &[4, 3])
        .expect_err("null out_proj must be refused, not launched");
    assert!(
        format!("{err:#}").contains("batched GDN out_proj dispatch"),
        "wrong error: {err:#}"
    );
}

/// Grid/block pairs of every launch recorded by the mock.
fn gpu_launch_grids(gpu: &MockGpuBackend) -> Vec<([u32; 3], [u32; 3])> {
    gpu.launches_snapshot()
        .into_iter()
        .map(|l| (l.grid, l.block))
        .collect()
}
