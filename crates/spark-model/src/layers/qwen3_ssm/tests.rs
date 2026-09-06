// SPDX-License-Identifier: AGPL-3.0-only

//! Extracted piecewise from `qwen3_ssm/mod.rs` (500-LoC cap).

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

#[test]
fn ssm_state_allocation_uses_layer_sizes_and_defaults() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    let before = gpu.alloc_count();

    let state = layer.alloc_state(&gpu).unwrap();
    let state = state
        .as_any()
        .downcast_ref::<SsmLayerState>()
        .expect("Qwen3 SSM allocation must return SsmLayerState");

    assert_eq!(gpu.alloc_count(), before + 2);
    assert_eq!(
        gpu.read_alloc(state.h_state).unwrap().len(),
        layer.h_state_bytes
    );
    assert_eq!(
        gpu.read_alloc(state.conv_state).unwrap().len(),
        layer.conv_state_bytes
    );
    assert_eq!(layer.h_state_bytes, 32 * 128 * 128 * 4);
    assert_eq!(layer.conv_state_bytes, 8192 * 4 * 4);
    assert!(state.h_state_checkpoint.is_none());
    assert!(state.conv_state_checkpoint.is_none());
    assert!(state.h_state_intermediates.is_empty());
    assert!(state.conv_state_intermediates.is_empty());
    assert!(!state.h_is_f16);
    assert!(state.h_prefill_stage.is_none());
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
        h_prefill_stage: None,
        ple: None,
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
        hc_row_offset: 0,
        gpu,
        config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
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
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    assert!(
        has_fp8_projection(&gpu, qkvz, 7, 12_288, 2_048, [384, 1, 1]),
        "QKVZ must consume its block-scaled FP8 pair at M=7"
    );
    assert!(
        has_fp8_projection(&gpu, out, 7, 2_048, 4_096, [64, 1, 1]),
        "out_proj must consume its block-scaled FP8 pair at M=7"
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
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    assert!(has_fp8_projection(
        &gpu,
        qkvz,
        4,
        12_288,
        2_048,
        [3_072, 1, 1]
    ));
    assert!(has_fp8_projection(&gpu, out, 4, 2_048, 4_096, [512, 1, 1]));
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

fn scalar_u32(arg: &MockArg) -> Option<u32> {
    let MockArg::Bytes(bytes) = arg else {
        return None;
    };
    (bytes.len() == 4).then(|| u32::from_ne_bytes(bytes.as_slice().try_into().unwrap()))
}

/// Find the exact block-scaled FP8 projection call among the full layer's
/// launches. Input/output buffers vary by phase; the weight pair and M/N/K
/// identify the production consumer contract.
fn has_fp8_projection(
    gpu: &MockGpuBackend,
    weight: &Fp8Weight,
    m: u32,
    n: u32,
    k: u32,
    grid: [u32; 3],
) -> bool {
    gpu.launches_snapshot().iter().any(|launch: &MockLaunch| {
        launch.grid == grid
            && launch.block == [256, 1, 1]
            && launch.args.len() == 7
            && launch.args[1] == MockArg::Buffer(weight.weight)
            && launch.args[2] == MockArg::Buffer(weight.row_scale)
            && scalar_u32(&launch.args[4]) == Some(m)
            && scalar_u32(&launch.args[5]) == Some(n)
            && scalar_u32(&launch.args[6]) == Some(k)
    })
}

// ── Batched decode/verify QKVZ weight-copy dispatch (M>8) ──
//
// nsys on b508679e4 (unsloth/Qwen3.8-27B-NVFP4) showed the batched verify
// QKVZ taking the pre-dequanted FP8 PREFILL copy at every M>8:
// `fp8_fp8_gemm_ldmab` 42.58 ms/step = 26.3% of the whole step at C=8,
// +1.762 GB/step (+77.8%) of weight traffic over the NVFP4 twin that sits at
// the same dispatch site. The decision is a pure function of the row count,
// which weight copies the layer holds, and the kill switch — so it is pinned
// here directly rather than through a mock launch, whose one shared kernel
// handle cannot tell `fp8_gemm_n128` from `w4a16_gemm_t_k64`.

use super::trait_decode_batched::qkvz_verify_nvfp4_wins;

/// Below and AT the threshold the FP8 arm must still win: M<=8 is exactly the
/// range the weight-streaming NVFP4 GEMVs (`w4a16_gemv_batch2/3/4/8`) cover,
/// and the FP8 tile GEMM is the measured-better fallback for the shapes that
/// reach it there. This is the assertion that keeps the fix additive.
#[test]
fn qkvz_verify_keeps_fp8_at_and_below_the_threshold() {
    for m in 1..=8 {
        assert!(
            !qkvz_verify_nvfp4_wins(m, true, true, true, false),
            "M={m} must keep the FP8 arm"
        );
    }
}

/// Above the threshold, with both copies present, the NVFP4 twin wins — the
/// C=4/8/16 verify widths (R = Σ ks) are all here.
#[test]
fn qkvz_verify_takes_nvfp4_above_the_threshold() {
    for m in [9, 12, 16, 32, 64, 65, 128, 512] {
        assert!(
            qkvz_verify_nvfp4_wins(m, true, true, true, false),
            "M={m} must take the NVFP4 arm"
        );
    }
}

/// The kill switch (`ATLAS_NO_QKVZ_NVFP4_DECODE`) restores the FP8 choice at
/// every row count — the A/B must be able to reproduce today's behaviour
/// verbatim, not approximately.
#[test]
fn qkvz_verify_kill_switch_restores_the_fp8_arm() {
    for m in [9, 16, 32, 128, 512] {
        assert!(
            !qkvz_verify_nvfp4_wins(m, true, true, true, true),
            "kill switch must restore the FP8 arm at M={m}"
        );
    }
}

/// No NVFP4 twin (native-FP8-GDN builds, where `qkvz_nvfp4_t` is None): the
/// arm must decline so the chain reaches the w8a16/FP8 arms it already had.
/// Diverting to a None weight is the NULL-slot dispatch bug this file's other
/// tests exist for.
#[test]
fn qkvz_verify_declines_without_the_nvfp4_twin() {
    for m in [9, 32, 512] {
        assert!(!qkvz_verify_nvfp4_wins(m, true, false, true, false));
    }
}

/// No tile GEMM linked for this target (`deep_k_gemm` resolves to a 0 handle):
/// the arm must decline rather than hand `ms_proj_gemm` a NULL kernel. Launching
/// a 0 handle is the sticky-context-loss class this dispatcher already carries
/// three NULL-slot regression tests for.
#[test]
fn qkvz_verify_declines_without_a_tile_gemm() {
    for m in [9, 32, 512] {
        assert!(!qkvz_verify_nvfp4_wins(m, true, true, false, false));
    }
}

/// No pre-dequanted FP8 copy: the predicate must decline even though NVFP4 is
/// present. This arm exists ONLY to divert the FP8 arm; on builds without that
/// copy the chain's own NVFP4 arm (`w4a16_gemm_n128_m128_v2` at the wide
/// DFlash verify) is unmeasured against `ms_proj_gemm` and must not be moved.
#[test]
fn qkvz_verify_declines_without_the_fp8_copy() {
    for m in [9, 17, 32, 512] {
        assert!(
            !qkvz_verify_nvfp4_wins(m, false, true, true, false),
            "M={m}: no FP8 copy means nothing to divert"
        );
    }
}
