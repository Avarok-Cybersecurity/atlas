// SPDX-License-Identifier: AGPL-3.0-only

//! Dispatch-selection tests for the batched native-FP8 Q/K/V tier (O13).
//!
//! These assert LAUNCH COUNTS and argument layout on the mock backend — the
//! whole point of the tier is that it is a pure launch-count change. Numeric
//! parity against the scalar `w8a16_gemv` path is the GPU oracle
//! (`examples/native_fp8_qkv_batch_microtest`), not this file.

use super::super::ctx::MultiSeqCtx;
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

const SCALAR_K: u64 = 0xF081;
const BATCH4_K: u64 = 0xF084;
const BATCH16_K: u64 = 0xF08C;
const NCOL2_K: u64 = 0xF0C2;
const NCOL4_K: u64 = 0xF0C4;
const WIDTH: usize = 128;

/// What the tier under test is expected to emit for one projection.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Expect {
    /// One strided launch for all rows, on the given kernel.
    Batched(u64),
    /// The per-sequence scalar `w8a16_gemv` loop (one launch per row).
    Scalar,
}

struct Case {
    rows: usize,
    width: usize,
    handles: bool,
    format: WeightQuantFormat,
    enabled: bool,
    /// `ATLAS_ATTN_NCOL_GEMV` as the layer caches it — injected as a field so
    /// the test drives the rule and not the process-global `OnceLock`.
    ncol: Option<NcolWidth>,
    /// Whether the shadow carries the `_ncol*_strided` entry points.
    ncol_handles: bool,
}

impl Case {
    fn new(rows: usize) -> Self {
        Self {
            rows,
            width: WIDTH,
            handles: true,
            format: WeightQuantFormat::Fp8BlockScaled,
            enabled: true,
            ncol: None,
            ncol_handles: true,
        }
    }

    /// The same case with the N-column tier opted in at `width`.
    fn ncol(rows: usize, width: NcolWidth) -> Self {
        Self {
            ncol: Some(width),
            ..Self::new(rows)
        }
    }
}

#[test]
fn native_fp8_qkv_batches_two_to_four_rows_on_batch4() {
    for rows in [2, 3, 4] {
        check_dispatch(&Case::new(rows), Expect::Batched(BATCH4_K));
    }
}

/// #927: the band read 2..=8 against a stale comment claiming the padded_n
/// ladder was [2,4,8]. It has had rungs 12 and 16 since the C=[1,2,4,8,16]
/// concurrency work, so those two widths were dropping to the per-sequence
/// scalar loop — 3n launches and n full weight passes per layer per step.
#[test]
fn native_fp8_qkv_batches_five_to_sixteen_rows_on_batch16() {
    for rows in [5, 6, 8, 12, 16] {
        check_dispatch(&Case::new(rows), Expect::Batched(BATCH16_K));
    }
}

/// 17+ is above the kernel's MAX_M, which CLAMPS rather than erroring — rows
/// 16.. would simply never be written. The band's upper edge is that template
/// bound, so the padded_n rungs above 16 must stay on the scalar loop.
#[test]
fn native_fp8_qkv_declines_rows_above_the_kernel_max_m() {
    for rows in [17, 24] {
        check_dispatch(&Case::new(rows), Expect::Scalar);
    }
}

#[test]
fn native_fp8_qkv_retains_scalar_for_single_row() {
    check_dispatch(&Case::new(1), Expect::Scalar);
}

#[test]
fn native_fp8_qkv_retains_scalar_without_strided_kernels() {
    let mut case = Case::new(4);
    case.handles = false;
    check_dispatch(&case, Expect::Scalar);
}

#[test]
fn native_fp8_qkv_retains_scalar_for_per_row_scales() {
    let mut case = Case::new(4);
    case.format = WeightQuantFormat::Fp8PerRow;
    check_dispatch(&case, Expect::Scalar);
}

#[test]
fn native_fp8_qkv_retains_scalar_for_unaligned_dims() {
    let mut case = Case::new(4);
    case.width = 64; // hidden/kv dims not a multiple of the 128 block-scale grid
    check_dispatch(&case, Expect::Scalar);
}

/// Kill switch: `ATLAS_NO_FP8_QKV_BATCH` present ⇒ tier off. Driven through the
/// injected flag rather than the env var so the test does not race the
/// process-global `OnceLock` that caches it in production.
#[test]
fn native_fp8_qkv_kill_switch_deselects_the_tier() {
    let mut case = Case::new(4);
    case.enabled = false;
    check_dispatch(&case, Expect::Scalar);
}

/// The N-column tier takes exactly the band `w8a16_gemv_batch16` owns, and is
/// a pure kernel swap: still ONE strided launch per projection, still the same
/// argument layout (the assertions in `check_dispatch` are shared).
#[test]
fn native_fp8_qkv_ncol_tier_takes_the_batch16_band() {
    for rows in [5, 8, 12, 16] {
        check_dispatch(&Case::ncol(rows, NcolWidth::Two), Expect::Batched(NCOL2_K));
        check_dispatch(&Case::ncol(rows, NcolWidth::Four), Expect::Batched(NCOL4_K));
    }
}

/// Below the band the ALU wall the tier attacks is not there yet
/// (`w8a16_gemv_batch4` pays ~10 ops per weight byte at m<=4), so the lever
/// must not move those widths.
#[test]
fn native_fp8_qkv_ncol_leaves_small_batches_on_batch4() {
    for rows in [2, 3, 4] {
        check_dispatch(&Case::ncol(rows, NcolWidth::Two), Expect::Batched(BATCH4_K));
    }
}

/// A shadow without the `_ncol*` entry points keeps the batch16 GEMV rather
/// than falling off the batched tier.
#[test]
fn native_fp8_qkv_ncol_declines_without_its_entry_points() {
    let mut case = Case::ncol(16, NcolWidth::Two);
    case.ncol_handles = false;
    check_dispatch(&case, Expect::Batched(BATCH16_K));
}

/// The kill switch reaches the layer as `attn_ncol: None` (SSOT:
/// `attn_ncol_gemv::ncol_gemv_enabled`, where `ATLAS_NO_ATTN_DECODE_BATCH`
/// wins over `ATLAS_ATTN_NCOL_GEMV`).
#[test]
fn native_fp8_qkv_ncol_off_keeps_batch16() {
    check_dispatch(&Case::new(16), Expect::Batched(BATCH16_K));
}

/// THE per-row-loop pin. The whole Q/K/V phase — projections, the gated
/// deinterleave and the q/k norms — must cost the SAME number of launches at
/// 16 rows as at 2, on both the batch16 and the N-column route. A reintroduced
/// per-sequence loop anywhere in the phase moves this number (it was 3n + 2n
/// before the strided tiers landed).
#[test]
fn native_fp8_qkv_phase_launch_count_is_row_independent() {
    let baseline = qkv_phase_launches(&Case::new(2));
    for rows in [4, 8, 12, 16] {
        assert_eq!(
            qkv_phase_launches(&Case::new(rows)),
            baseline,
            "batch16 route, rows={rows}"
        );
        assert_eq!(
            qkv_phase_launches(&Case::ncol(rows, NcolWidth::Two)),
            baseline,
            "N-column route, rows={rows}"
        );
    }
}

fn check_dispatch(case: &Case, expect: Expect) {
    run_phase(case, Some(expect));
}

/// Launches the whole `ms_phase_qkv` costs for this case, with no expectation
/// on which kernel served the projections.
fn qkv_phase_launches(case: &Case) -> usize {
    run_phase(case, None)
}

/// Drives `ms_phase_qkv` on the mock backend. Returns the phase's launch count;
/// `expect` (when given) also pins the tier and the per-projection argument
/// layout.
fn run_phase(case: &Case, expect: Option<Expect>) -> usize {
    let gpu = MockGpuBackend::new();
    let width = case.width;
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
        weight: gpu.alloc(256 * 256 * 2).unwrap(),
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
    layer.w8a16_gemv_batch4_strided_k = KernelHandle(if case.handles { BATCH4_K } else { 0 });
    layer.w8a16_gemv_batch16_strided_k = KernelHandle(if case.handles { BATCH16_K } else { 0 });
    layer.w8a16_gemv_ncol2_strided_k = KernelHandle(if case.ncol_handles { NCOL2_K } else { 0 });
    layer.w8a16_gemv_ncol4_strided_k = KernelHandle(if case.ncol_handles { NCOL4_K } else { 0 });
    layer.attn_ncol = case.ncol;
    layer.deinterleave_qg_k = KernelHandle(0xF0D1);

    let q_dim = (config.num_attention_heads * config.head_dim) as u32;
    let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
    let kv_dim = (config.num_key_value_heads * config.head_dim) as u32;
    let fp8 = |n: u32| Fp8Weight {
        weight: gpu.alloc(n as usize * width).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n,
        k: width as u32,
        scale_format: case.format,
    };
    let (q_w, k_w, v_w) = (fp8(q_proj_dim), fp8(kv_dim), fp8(kv_dim));
    layer.q_weight = Some(QuantWeight::Fp8(q_w));
    layer.k_weight = Some(QuantWeight::Fp8(k_w));
    layer.v_weight = Some(QuantWeight::Fp8(v_w));

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
        // The QKV projections under test never read the decode scalars this guards.
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
        case.rows,
        16,
        0,
    );

    if let Some(expect) = expect {
        assert_eq!(
            layer.ms_qkv_batchm_fp8_selected(&c, case.enabled),
            expect != Expect::Scalar,
            "tier selection for rows={} width={} handles={} enabled={}",
            case.rows,
            case.width,
            case.handles,
            case.enabled
        );
    }

    if !case.enabled {
        // The kill switch is checked above, on the selection predicate. The
        // production call below reads the process-global `OnceLock` that caches
        // the env var, which a single test in a shared process cannot flip
        // without racing every other test — so stop here rather than assert a
        // launch pattern this process cannot produce.
        return 0;
    }

    let first = gpu.launch_count();
    let allocations = gpu.alloc_count();
    layer.ms_phase_qkv(&c).unwrap();
    let all = gpu.launches_snapshot();
    assert_eq!(
        gpu.alloc_count(),
        allocations,
        "projection must reuse existing buffers"
    );

    let bf16 = 2usize;
    let q_proj_bytes = q_proj_dim as usize * bf16;
    let kv_bytes = kv_dim as usize * bf16;
    let c_stride = (c.per_seq_qkv / bf16) as u32;
    let projections = [
        (layer.q_weight.as_ref().unwrap(), 0usize, q_proj_dim),
        (layer.k_weight.as_ref().unwrap(), q_proj_bytes, kv_dim),
        (
            layer.v_weight.as_ref().unwrap(),
            q_proj_bytes + kv_bytes,
            kv_dim,
        ),
    ];
    let Some(expect) = expect else {
        return all.len() - first;
    };
    for (weight, out_off, n_out) in projections {
        let w = weight.as_fp8().unwrap();
        let launches: Vec<_> = all[first..]
            .iter()
            .filter(|l| l.args.contains(&MockArg::Buffer(w.weight)))
            .collect();
        match expect {
            Expect::Batched(kernel) => {
                assert_eq!(launches.len(), 1, "one strided launch per projection");
                let l = launches[0];
                assert_eq!(l.func, kernel);
                assert_eq!(l.args[0], MockArg::Buffer(c.normed));
                assert_eq!(l.args[1], MockArg::Buffer(w.weight));
                assert_eq!(l.args[2], MockArg::Buffer(w.row_scale));
                assert_eq!(l.args[3], MockArg::Buffer(c.qkv_buf.offset(out_off)));
                assert_eq!(l.args[4], u32_arg(case.rows as u32));
                assert_eq!(l.args[5], u32_arg(n_out));
                assert_eq!(l.args[6], u32_arg(width as u32));
                assert_eq!(l.args[7], u32_arg(width as u32), "A row stride = hidden");
                assert_eq!(l.args[8], u32_arg(c_stride), "C row stride = per_seq_qkv");
            }
            Expect::Scalar => {
                assert_eq!(launches.len(), case.rows, "one scalar GEMV per row");
                for (row, l) in launches.iter().enumerate() {
                    assert_eq!(l.func, SCALAR_K);
                    assert_eq!(
                        l.args[0],
                        MockArg::Buffer(c.normed.offset(row * width * bf16))
                    );
                    assert_eq!(
                        l.args[3],
                        MockArg::Buffer(c.qkv_buf.offset(row * c.per_seq_qkv + out_off))
                    );
                }
            }
        }
    }
    all.len() - first
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_ne_bytes().to_vec())
}
