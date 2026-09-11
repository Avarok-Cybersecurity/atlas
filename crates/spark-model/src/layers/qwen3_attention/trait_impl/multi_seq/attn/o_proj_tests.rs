// SPDX-License-Identifier: AGPL-3.0-only

use super::super::super::ctx::MultiSeqCtx;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::qwen3_attention::attn_ncol_gemv::NcolWidth;
use crate::layers::{FfnComponent, qwen3_attention::Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat,
};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

/// Which tier the FP8 o_proj is expected to take, and the row stride the group
/// loop must walk with. `Scalar` is one `w8a16_gemv` launch per row.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Tier {
    Scalar,
    Batch4,
    Batch16,
    /// The bit-exact N-column-blocked rung (#927) — same 16-row group as
    /// `Batch16`, so only the kernel differs.
    Ncol2,
    Ncol4,
}

impl Tier {
    fn step(self) -> usize {
        match self {
            Tier::Scalar => 1,
            Tier::Batch4 => 4,
            Tier::Batch16 | Tier::Ncol2 | Tier::Ncol4 => 16,
        }
    }

    fn kernel(self) -> u64 {
        match self {
            Tier::Scalar => SCALAR_K,
            Tier::Batch4 => BATCH4_K,
            Tier::Batch16 => BATCH16_K,
            Tier::Ncol2 => NCOL2_K,
            Tier::Ncol4 => NCOL4_K,
        }
    }
}

const SCALAR_K: u64 = 0xF081;
const BATCH4_K: u64 = 0xF084;
const BATCH16_K: u64 = 0xF08C;
const NCOL2_K: u64 = 0xF0C2;
const NCOL4_K: u64 = 0xF0C4;

#[test]
fn native_fp8_attention_o_projection_batches_four_real_rows() {
    check_dispatch(
        4,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch4,
    );
}

/// #927: 5..=16 concurrent decode rows used to walk the o_proj weight in
/// ceil(n/4) batch4 groups — FOUR full weight passes at n=16. The MAX_M=16
/// twin does it in one, and is bit-identical per row (same template, same K
/// order, same reduction tree).
#[test]
fn native_fp8_attention_o_projection_batches_up_to_sixteen_rows_in_one_pass() {
    for rows in [5, 8, 12, 16] {
        check_dispatch(
            rows,
            128,
            true,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch16,
        );
    }
}

/// Above the kernel's MAX_M the loop still walks — in 16-row groups now, not
/// 4-row ones (n=20 is 2 launches, was 5).
#[test]
fn native_fp8_attention_o_projection_walks_wider_batches_in_sixteen_row_groups() {
    check_dispatch(
        20,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch16,
    );
}

fn check_dispatch(
    rows: usize,
    width: usize,
    available: bool,
    format: WeightQuantFormat,
    tier: Tier,
) {
    check_dispatch_with(rows, width, available, available, format, tier, None)
}

/// `check_dispatch` with the N-column tier opted in — injected as the layer
/// field the lever resolves to, so the test drives the rule and not the
/// process-global `OnceLock`.
fn check_dispatch_ncol(rows: usize, tier: Tier, ncol: NcolWidth) {
    check_dispatch_with(
        rows,
        128,
        true,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        tier,
        Some(ncol),
    )
}

/// `wide` is the presence of the MAX_M=16 handle, separate from `available`
/// (the MAX_M=4 one), so the "shadow lacks the new kernel" case is reachable.
fn check_dispatch_with(
    rows: usize,
    width: usize,
    available: bool,
    wide: bool,
    format: WeightQuantFormat,
    tier: Tier,
    ncol: Option<NcolWidth>,
) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = width;
    config.intermediate_size = 128;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = width;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
    let dense = DenseWeight {
        weight: gpu.alloc(128 * 128 * 2).unwrap(),
    };
    let fallback = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj: fallback,
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
        &gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .unwrap();
    layer.w8a16_gemv_k = KernelHandle(SCALAR_K);
    layer.w8a16_gemv_batch4_k = KernelHandle(if available { BATCH4_K } else { 0 });
    layer.w8a16_gemv_batch16_k = KernelHandle(if wide { BATCH16_K } else { 0 });
    layer.w8a16_gemv_ncol2_k = KernelHandle(NCOL2_K);
    layer.w8a16_gemv_ncol4_k = KernelHandle(NCOL4_K);
    layer.attn_ncol = ncol;
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: format,
    };
    layer.o_weight = Some(QuantWeight::Fp8(fp8));
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let fwd = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        // Multi-row o_proj under test never reads the decode scalars this flag guards.
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
    };
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        buffers.hidden_states(),
        buffers.residual(),
        rows,
        16,
        0,
    );
    let first = gpu.launch_count();
    let allocations = gpu.alloc_count();
    let output = layer.ms_phase_o_proj(&c, buffers.attn_output()).unwrap();
    let all = gpu.launches_snapshot();
    let launches: Vec<_> = all[first..]
        .iter()
        .filter(|l| l.args.contains(&MockArg::Buffer(fp8.weight)))
        .collect();
    let step = tier.step();
    assert_eq!(
        launches.len(),
        rows.div_ceil(step),
        "production O-projection dispatch (tier {tier:?}, rows {rows})"
    );
    assert_eq!(
        gpu.alloc_count(),
        allocations,
        "projection must reuse existing buffers"
    );
    for (group, launch) in launches.iter().enumerate() {
        let row = group * step;
        assert_eq!(launch.func, tier.kernel());
        assert_eq!(
            launch.args[0],
            MockArg::Buffer(buffers.attn_output().offset(row * width * 2))
        );
        assert_eq!(launch.args[1], MockArg::Buffer(fp8.weight));
        assert_eq!(launch.args[2], MockArg::Buffer(fp8.row_scale));
        assert_eq!(
            launch.args[3],
            MockArg::Buffer(output.offset(row * width * 2))
        );
        if tier != Tier::Scalar {
            assert_eq!(
                launch.args[4],
                MockArg::Bytes(((rows - row).min(step) as u32).to_ne_bytes().to_vec())
            );
        }
    }
}

#[test]
fn native_fp8_attention_o_projection_chunks_preserve_offsets() {
    check_dispatch(
        2,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch4,
    );
    for rows in [5, 16] {
        check_dispatch(
            rows,
            128,
            true,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch16,
        );
    }
}

#[test]
fn native_fp8_attention_o_projection_retains_scalar_fallbacks() {
    let bs = WeightQuantFormat::Fp8BlockScaled;
    check_dispatch(1, 128, true, bs, Tier::Scalar);
    check_dispatch(4, 128, false, bs, Tier::Scalar);
    check_dispatch(4, 64, true, bs, Tier::Scalar);
    check_dispatch(4, 128, true, WeightQuantFormat::Fp8PerRow, Tier::Scalar);
    // Per-row scales and unaligned dims disqualify the WIDE tier too — the
    // `block_scaled` guard is shared, not duplicated per rung.
    check_dispatch(8, 64, true, bs, Tier::Scalar);
    check_dispatch(8, 128, true, WeightQuantFormat::Fp8PerRow, Tier::Scalar);
}

/// A shadow without the MAX_M=16 entry point keeps the pre-#927 grouping
/// instead of falling off the batched tier entirely.
#[test]
fn native_fp8_attention_o_projection_without_batch16_keeps_four_row_groups() {
    for rows in [8, 16] {
        check_dispatch_with(
            rows,
            128,
            true,
            false,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch4,
            None,
        );
    }
}

/// The N-column tier serves the o_proj across the band `w8a16_gemv_batch16`
/// owns, in the same ONE 16-row group — a kernel swap, not a launch-count
/// change, and bit-exact per row (oracle:
/// `examples/native_fp8_attn_decode_batch_microtest`).
#[test]
fn native_fp8_attention_o_projection_takes_the_ncol_tier() {
    for rows in [5, 8, 12, 16] {
        check_dispatch_ncol(rows, Tier::Ncol2, NcolWidth::Two);
        check_dispatch_ncol(rows, Tier::Ncol4, NcolWidth::Four);
    }
}

/// m <= 4 keeps `w8a16_gemv_batch4`: the ALU wall the tier attacks is
/// proportional to M and that kernel is already near the bandwidth floor.
#[test]
fn native_fp8_attention_o_projection_ncol_leaves_small_batches_alone() {
    for rows in [1, 2, 4] {
        let tier = if rows == 1 {
            Tier::Scalar
        } else {
            Tier::Batch4
        };
        check_dispatch_ncol(rows, tier, NcolWidth::Two);
    }
}

/// Above MAX_M the group loop still walks in 16-row groups on the batch16
/// GEMV — the tier declines rather than clamping rows away.
#[test]
fn native_fp8_attention_o_projection_ncol_declines_above_max_m() {
    check_dispatch_ncol(20, Tier::Batch16, NcolWidth::Two);
}
