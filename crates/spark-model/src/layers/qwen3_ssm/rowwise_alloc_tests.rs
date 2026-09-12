// SPDX-License-Identifier: AGPL-3.0-only

//! Allocation contract for the two `ATLAS_FP8_ROWWISE` GDN prefill arms —
//! `trait_prefill_proj.rs`'s `in_proj_qkvz` and `trait_prefill_helper.rs`'s
//! `out_proj`.
//!
//! H100, 2026-09-11, `Qwen/Qwen3.8-27B-FP8` (#917): both arms got their BF16
//! weight from `ops::cublas_bf16_proj`, whose `dequant_fp8_bf16_cached` did a
//! `gpu.alloc` memoised by weight pointer — `167772160` B for the fused
//! `[QKV|Z]` weight PER LAYER, with no `spark_runtime::buffers::sizes::BufferSizes`
//! entry, so `--gpu-memory-utilization` could not see it. One 28-token prefill
//! consumed 6120 MiB and died at layer 36 with `cuMemAlloc_v2 failed: status 2`.
//! The attention arms that shared the defect were deleted in #927 and are held
//! deleted by `qwen3_attention/prefill/alloc_tests.rs`; these two cannot be —
//! a per-row checkpoint has no other route while
//! `cublaslt::fp8_gemm_act_weight_t_rowwise` is NOT_SUPPORTED on sm_121 — so
//! the bytes moved into the arena instead (`rowwise_bf16.rs`).
//!
//! The assertion is deliberately taken BEFORE the first call, not between the
//! two: the old allocation was CACHED by weight pointer, so a first-vs-second
//! comparison alone would have passed it. What is wrong is that the projection
//! allocates at all — the arena already owns every buffer it needs.
//!
//! WHERE THE TESTS STOP, AND WHY. These bracket the arms at their dequant
//! seam (`rowwise_qkvz_bf16` / `rowwise_out_proj_bf16`), which is the step
//! that allocated, and NOT through the cuBLASLt matmul that follows it. That
//! is not squeamishness about an FFI error: this binary also builds real
//! `AtlasCudaBackend`s (`qsa_tests.rs`, `ngram_embed/tests.rs`), so a process
//! CUDA context may well exist by the time these run, and handing
//! `cublasLtMatmul` a `MockGpuBackend`'s fabricated device pointers would
//! fault that shared context and take the rest of the suite with it. The arms
//! themselves are exercised end to end by
//! `the_arms_refuse_to_run_when_the_ledger_entry_is_absent` below, which stops
//! at the ledger check for a REASON rather than by luck.

use super::tests::native_fp8_gdn_layer;
use super::*;
use crate::weight_map::WeightQuantFormat;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::{BufferArena, BufferSizes, ssm_rowwise_w_bf16_bytes_for};
use spark_runtime::gpu::mock::MockGpuBackend;

/// The smoke prompt that killed the H100 run.
const M: u32 = 28;

/// A COMPACT GDN model — 4 layers on a 4-cycle (3 GDN, 1 full attention),
/// hidden 512, 4x64 key heads, 8x64 value heads.
///
/// Deliberately not the 27B: `MockGpuBackend::alloc` is `vec![0u8; bytes]`, so
/// the real checkpoint's 10.31 GiB slab would be 10.31 GiB of HOST memory in a
/// unit test. The EXACT 27B integers — `167772160` B of `in_proj_qkvz` per
/// layer, 48 layers — are pinned where nothing has to be allocated to check
/// them: `spark_runtime::buffers::tests::rowwise_bf16_slab_is_sized_only_when_the_lever_is_armed`.
/// What these tests pin is the property that does not depend on the shape:
/// the arms take their BF16 weight from the ledger and allocate nothing.
fn compact_gdn() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 512;
    c.num_hidden_layers = 4;
    c.linear_num_key_heads = 4;
    c.linear_key_head_dim = 64;
    c.linear_num_value_heads = 8;
    c.linear_value_head_dim = 64;
    c.full_attention_interval = 4;
    c.layer_types = (0..4)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c
}

/// A per-row FP8 pair, the `Fp8PerRow` shape a mixed-precision
/// compressed-tensors checkpoint ships and the only one
/// `set_fp8_rowwise_prefill_weights` accepts.
fn fp8_per_row(gpu: &MockGpuBackend, n: usize, k: usize) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc(n * k).unwrap(),
        row_scale: gpu.alloc(n * 4).unwrap(),
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8PerRow,
    }
}

/// `ATLAS_FP8_ROWWISE` ARMED, without touching the process environment:
/// `set_var` is unsafe and process-global and would race every other test in
/// this binary, so the ledger is built with the lever passed in and handed to
/// `BufferArena::from_sizes` — the same bytes `BufferSizes::from_config` would
/// have produced with the variable exported.
fn armed_arena(config: &ModelConfig, gpu: &MockGpuBackend) -> BufferArena {
    let mut sizes = BufferSizes::from_config(config, 64, 4096, 16, 32);
    sizes.ssm_rowwise_w_bf16 = ssm_rowwise_w_bf16_bytes_for(config, true);
    BufferArena::from_sizes(config, sizes, 64, 32, gpu).unwrap()
}

/// An arena sized WITHOUT the lever — the default recipe's ledger, in which
/// the slab is NULL.
fn unarmed_arena(config: &ModelConfig, gpu: &MockGpuBackend) -> BufferArena {
    let mut sizes = BufferSizes::from_config(config, 64, 4096, 16, 32);
    sizes.ssm_rowwise_w_bf16 = 0;
    BufferArena::from_sizes(config, sizes, 64, 32, gpu).unwrap()
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
            moe_lora_route: crate::layer::MoeLoraRoute::Fold,
        }
    };
}

/// Everything a row-wise arm needs, wired the way the loader wires it.
struct Fixture {
    config: ModelConfig,
    qkvz: Fp8Weight,
    out_proj: Fp8Weight,
}

fn fixture(gpu: &MockGpuBackend) -> (Fixture, Qwen3SsmLayer) {
    let config = compact_gdn();
    let mut layer = native_fp8_gdn_layer(gpu, &config, true, true);
    let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
    let qkvz = fp8_per_row(gpu, config.ssm_qkvz_size(), config.hidden_size);
    let out_proj = fp8_per_row(gpu, config.hidden_size, value_dim);
    layer.set_fp8_rowwise_prefill_weights(Some(qkvz), Some(out_proj));
    (
        Fixture {
            config,
            qkvz,
            out_proj,
        },
        layer,
    )
}

/// `in_proj_qkvz`: the arm's BF16 weight must come out of the arena, and the
/// FIRST call must not allocate. On the real 27B that weight is the
/// 167772160 B per layer the #917 OOM receipt names.
#[test]
fn rowwise_qkvz_bf16_weight_comes_from_the_ledger_not_a_fresh_alloc() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = armed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );

    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    let first = layer
        .rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0)
        .expect("first row-wise QKVZ bind");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the row-wise QKVZ arm allocated on its FIRST call — that is the \
         per-layer BF16 weight dequant from the #917 H100 receipt (167772160 B \
         on the 27B this fixture shrinks)"
    );
    // One launch, and no second one: the dequant kernel, then a pure load.
    // Pinned so "allocated nothing" cannot be satisfied by "did nothing".
    assert_eq!(
        gpu.launch_count() - launches,
        1,
        "expected one dequant_fp8_blockscaled_bf16 into the ledgered slab"
    );
    let second = layer
        .rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0)
        .expect("second row-wise QKVZ bind");
    assert_eq!(first, second, "the slice is per layer and must not move");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes(), gpu.launch_count()),
        (allocs, bytes, launches + 1),
        "the second call must be a load, not a second dequant"
    );
}

/// `out_proj`: same contract, and the two slices must not overlap — a bump
/// cursor that handed both arms the same offset would pass every "allocates
/// nothing" assertion while silently serving one weight as the other.
#[test]
fn rowwise_out_proj_bf16_weight_comes_from_the_ledger_not_a_fresh_alloc() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = armed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );

    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    let out = layer
        .rowwise_out_proj_bf16(&ctx, &fx.out_proj, 0)
        .expect("first row-wise out_proj bind");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the row-wise out_proj arm allocated on its FIRST call — the same \
         off-ledger BF16 weight dequant, [hidden, value_dim] x 2 B"
    );
    assert_eq!(gpu.launch_count() - launches, 1);
    layer
        .rowwise_out_proj_bf16(&ctx, &fx.out_proj, 0)
        .expect("second row-wise out_proj bind");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes(), gpu.launch_count()),
        (allocs, bytes, launches + 1)
    );

    // Both arms of ONE layer, carved from one slab: disjoint, and together
    // exactly `ssm_rowwise_w_bf16_layer_bytes`.
    let qkvz = layer
        .rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0)
        .expect("row-wise QKVZ bind");
    let out_bytes = ops::dequant_fp8_bf16_bytes(&fx.out_proj);
    assert_eq!(
        out_bytes,
        fx.out_proj.n as usize * fx.out_proj.k as usize * 2
    );
    assert_eq!(qkvz.0, out.0 + out_bytes as u64, "slices must not overlap");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "neither arm may allocate"
    );
}

/// The slab is the WHOLE budget: `num_ssm_layers` pairs and not one byte more,
/// so one layer past the model's count is an error rather than a quiet
/// `gpu.alloc` — which is precisely how the #917 leak stayed invisible.
#[test]
fn the_ledgered_slab_holds_exactly_every_gdn_layer_and_refuses_the_next() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = armed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );

    let layer_bytes =
        ops::dequant_fp8_bf16_bytes(&fx.qkvz) + ops::dequant_fp8_bf16_bytes(&fx.out_proj);
    assert_eq!(
        buffers.ssm_rowwise_w_bf16_bytes(),
        fx.config.num_ssm_layers() * layer_bytes
    );
    // Every GDN layer's worth of slices, taken directly so the test does not
    // need one layer fixture per layer. The first pair is `layer`'s; the rest
    // stand in for its siblings, which take theirs on their own first prefill.
    let _ = layer.rowwise_qkvz_bf16(&ctx, &fx.qkvz, 0).unwrap();
    let _ = layer.rowwise_out_proj_bf16(&ctx, &fx.out_proj, 0).unwrap();
    for _ in 1..fx.config.num_ssm_layers() {
        buffers.take_ssm_rowwise_w_bf16(layer_bytes).unwrap();
    }
    let over = buffers.take_ssm_rowwise_w_bf16(layer_bytes);
    assert!(
        over.is_err(),
        "one GDN layer past the model's count must be refused, not served \
         from a fresh allocation"
    );
    assert!(format!("{:#}", over.unwrap_err()).contains("exhausted"));
}

/// End to end through BOTH real arms, with the ledger entry ABSENT — the
/// configuration a `ATLAS_FP8_ROWWISE=0` arena would present to row-wise
/// weights. Each arm must refuse before it reaches cuBLASLt, and neither may
/// fall back to allocating its own BF16 twin.
#[test]
fn the_arms_refuse_to_run_when_the_ledger_entry_is_absent() {
    let gpu = MockGpuBackend::new();
    let (fx, layer) = fixture(&gpu);
    let buffers = unarmed_arena(&fx.config, &gpu);
    let (dispatch, derived) = (ops::GemmDispatch::defaults(), ops::DerivedWeights::new());
    let (levers, stats) = (ops::ModelLevers::defaults(), ops::ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &fx.config, &dispatch, &derived, &levers, &stats
    );
    let c = &fx.config;
    let value_dim = c.linear_num_value_heads * c.linear_value_head_dim;

    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let qkvz = layer.prefill_qkvz_proj(
        buffers.norm_output(),
        buffers.ssm_deinterleaved(),
        M,
        c.ssm_qkvz_size(),
        c.hidden_size,
        c.linear_num_key_heads,
        c.linear_key_head_dim,
        c.linear_num_value_heads / c.linear_num_key_heads.max(1),
        c.linear_value_head_dim,
        &ctx,
        0,
    );
    let out = layer.prefill_out_proj_dispatch(
        &ctx,
        buffers.norm_output(),
        buffers.hidden_states(),
        M,
        c.hidden_size,
        value_dim,
        0,
    );
    for (what, r) in [("in_proj_qkvz", qkvz), ("out_proj", out)] {
        let e = format!("{:#}", r.expect_err(what));
        assert!(
            e.contains("row-wise") && e.contains("slab is absent"),
            "{what}: expected the missing-ledger-entry refusal, got: {e}"
        );
    }
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "a missing ledger entry must not be papered over with a fresh allocation"
    );
}
