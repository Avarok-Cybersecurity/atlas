// SPDX-License-Identifier: AGPL-3.0-only

//! Which `ATLAS_CUBLAS_GEMM` families arm the attention decode W8A8 arm, and —
//! the part with teeth — that its STRIDED Q/K/V write can never leave the QKV
//! buffer (#927).
//!
//! The shape/capacity rule itself is pinned once, for both families, in
//! `ops::dispatch_proj_decode_tests`. What is attention-specific and lives
//! here: the lever bit, and the three plans this layer builds from
//! `per_seq_qkv`, `q_proj_bytes` and the K/V offsets — because those are what
//! turn a correct rule into a correct bound.

use super::*;
use crate::layers::ops::{
    self, CublasScope, DecodeW8a8Plan, DecodeW8a8Scratch, decode_w8a8_selected, parse_cublas_scope,
    strided_out_extent_elems,
};
use crate::weight_map::WeightQuantFormat;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

/// Qwen3.8-27B attention: hidden 5120, 24 q-heads / 4 kv-heads, head_dim 256,
/// output gate on. So `q_proj` is the interleaved `[Q|gate]` at 12288, k/v are
/// 1024, and one sequence's `[Q|K|V]` slot is 14336 BF16 elements.
const H: u32 = 5120;
const Q_N: u32 = 12288;
const KV_N: u32 = 1024;
const PER_SEQ_QKV: u32 = Q_N + 2 * KV_N;
/// `max_batch_size` slots — what `qkv_output` is sized for.
const SLOTS: usize = 16;
const QKV_CAPACITY: usize = SLOTS * PER_SEQ_QKV as usize * 2;

fn scratch() -> DecodeW8a8Scratch {
    DecodeW8a8Scratch {
        act_fp8: DevicePtr(0x1000),
        act_fp8_bytes: 1 << 20,
        act_scale: DevicePtr(0x2000),
        act_scale_bytes: 1 << 20,
        act_scale_kmajor: DevicePtr(0x3000),
        act_scale_kmajor_bytes: 1 << 20,
        quant_k: ops::Fp8ActQuant::shared_only(KernelHandle(0xA1)),
        scale_kmajor_k: KernelHandle(0xA2),
    }
}

/// The three plans `qkv_decode_w8a8_plans` builds, reproduced from the same
/// offsets the projection loop uses: Q at 0, K at `q_proj_bytes`, V after K.
fn qkv_plans(rows: usize, capacity: usize) -> [(usize, DecodeW8a8Plan); 3] {
    let q_bytes = Q_N as usize * 2;
    let kv_bytes = KV_N as usize * 2;
    let plan = |offset: usize, n_out: u32| {
        (
            offset,
            DecodeW8a8Plan::strided(rows, n_out, H, PER_SEQ_QKV, capacity.saturating_sub(offset)),
        )
    };
    [
        plan(0, Q_N),
        plan(q_bytes, KV_N),
        plan(q_bytes + kv_bytes, KV_N),
    ]
}

fn all_selected(raw: &str, rows: usize, disabled: bool, capacity: usize) -> bool {
    let scope = parse_cublas_scope(Some(raw)).0;
    qkv_plans(rows, capacity).iter().all(|(_, plan)| {
        decode_w8a8_selected(
            attn_decode_family_armed(scope),
            disabled,
            plan,
            WeightQuantFormat::Fp8BlockScaled,
            &scratch(),
        )
    })
}

/// `attn` (alone or in a list) and `all`/`1`/`true` arm it; NOTHING else does.
/// `ATLAS_CUBLAS_GEMM=ffn` reaching the attention projections is the #917
/// failure the family set was introduced to prevent.
#[test]
fn only_the_attn_family_arms_the_attention_decode_arm() {
    for raw in ["attn", "ffn,attn", "attn,ssm", "all", "1", "true"] {
        assert!(
            all_selected(raw, 16, false, QKV_CAPACITY),
            "ATLAS_CUBLAS_GEMM={raw:?}"
        );
    }
    for raw in ["ffn", "ssm", "head", "ffn,ssm", "off", "", "junk"] {
        assert!(
            !all_selected(raw, 16, false, QKV_CAPACITY),
            "ATLAS_CUBLAS_GEMM={raw:?}"
        );
    }
    // And the bit read is the `attn` one, stated directly.
    assert!(attn_decode_family_armed(CublasScope {
        attn: true,
        ..CublasScope::OFF
    }));
    assert!(!attn_decode_family_armed(CublasScope {
        ffn: true,
        ssm: true,
        head: true,
        attn: false
    }));
}

/// The band on the PADDED ctx `n`: 5..=16 takes cuBLASLt, 1..=4 keeps the
/// `w8a16_gemv_batch4_strided` tier (and M=1 its bit-exact scalar loop),
/// 17+ falls through.
#[test]
fn the_attention_decode_arm_takes_five_to_sixteen_rows_only() {
    for rows in [5, 8, 12, 16] {
        assert!(
            all_selected("attn", rows, false, QKV_CAPACITY),
            "rows={rows}"
        );
    }
    for rows in [1, 2, 4, 17, 24] {
        assert!(
            !all_selected("attn", rows, false, QKV_CAPACITY),
            "rows={rows}"
        );
    }
}

/// `ATLAS_NO_W8A8_DECODE_PROJ` wins over an armed family, at every rung.
#[test]
fn the_kill_switch_beats_an_armed_attn_family() {
    for rows in [5, 8, 16] {
        assert!(
            !all_selected("attn", rows, true, QKV_CAPACITY),
            "rows={rows}"
        );
        assert!(
            !all_selected("all", rows, true, QKV_CAPACITY),
            "rows={rows}"
        );
    }
}

// ─────────────── the strided write extent, per projection ───────────────

/// Q, K and V are bounded from THEIR OWN base. The arithmetic, stated as
/// numbers rather than as a formula call, so a change to either one has to
/// disagree with the other to pass: at 16 padded rows Q's write reaches
/// `15 * 14336 + 12288` elements past `qkv_output`, K's the same 15 slots plus
/// 1024, and V's likewise — each measured from a base that is already
/// `q_proj_bytes` (and `+ kv_bytes`) into the buffer.
#[test]
fn each_qkv_projection_is_bounded_from_its_own_base() {
    let plans = qkv_plans(16, QKV_CAPACITY);
    let expect = [
        (0usize, Q_N, 15 * PER_SEQ_QKV as usize + Q_N as usize),
        (
            Q_N as usize * 2,
            KV_N,
            15 * PER_SEQ_QKV as usize + KV_N as usize,
        ),
        (
            (Q_N + KV_N) as usize * 2,
            KV_N,
            15 * PER_SEQ_QKV as usize + KV_N as usize,
        ),
    ];
    for ((offset, plan), (want_off, want_n, want_extent)) in plans.iter().zip(expect) {
        assert_eq!(*offset, want_off);
        assert_eq!(plan.n, want_n);
        assert_eq!(plan.ldc, PER_SEQ_QKV, "row pitch is the slot, in elements");
        assert_eq!(plan.write_extent_bytes(), want_extent * 2);
        assert_eq!(
            strided_out_extent_elems(plan.m_pad(), plan.ldc, plan.n),
            want_extent
        );
        // Fits with one BF16 element to spare at the buffer's own base, and
        // not one element less.
        let base_room = QKV_CAPACITY - *offset;
        assert!(plan.write_extent_bytes() <= base_room);
    }
}

/// THE PHANTOM-ROW GUARD, at the row counts that actually produce phantoms.
/// `ceil16` is 16 for every rung in the band, so at n=5 ELEVEN phantom rows are
/// written — into decode slots 5..16, which belong to sequences that are not in
/// this step and whose contents are re-projected before anything reads them.
/// In-bounds and harmless, but ONLY while the buffer has 16 slots: a buffer
/// sized for the live rows must make the arm decline, not overrun.
#[test]
fn phantom_rows_never_leave_the_qkv_buffer() {
    for rows in [5usize, 8, 12, 15] {
        assert_eq!(qkv_plans(rows, QKV_CAPACITY)[0].1.m_pad(), 16);
        // All 16 slots present: accepted, and the extent is the SAME at every
        // row count because the pad is.
        assert!(
            all_selected("attn", rows, false, QKV_CAPACITY),
            "rows={rows}"
        );
        assert_eq!(
            qkv_plans(rows, QKV_CAPACITY)[0].1.write_extent_bytes(),
            qkv_plans(16, QKV_CAPACITY)[0].1.write_extent_bytes(),
            "rows={rows}: the padded write does not shrink with the live rows"
        );
        // Sized for the live rows only: refused.
        let live_only = rows * PER_SEQ_QKV as usize * 2;
        assert!(
            !all_selected("attn", rows, false, live_only),
            "rows={rows}: a {live_only}-byte buffer must refuse the padded write"
        );
    }
}

/// One BF16 element short of V's extent — the tightest of the three — must
/// take ALL of Q/K/V off the arm, because the three go together.
#[test]
fn one_element_short_for_v_declines_the_whole_group() {
    let v_extent = qkv_plans(16, QKV_CAPACITY)[2].1.write_extent_bytes();
    let v_base = (Q_N + KV_N) as usize * 2;
    assert!(all_selected("attn", 16, false, v_base + v_extent));
    assert!(!all_selected("attn", 16, false, v_base + v_extent - 2));
}

// ───────────── the FUSED [q|k|v] arm, on the layer (#927) ─────────────

/// A decode-shaped layer at the round-13 attention widths, with the fused arm
/// armed (or not) by INJECTION — the production accessor is a process-global
/// `OnceLock` a CPU test cannot toggle.
struct FusedHarness {
    gpu: spark_runtime::gpu::mock::MockGpuBackend,
    layer: Qwen3AttentionLayer,
    buffers: spark_runtime::buffers::BufferArena,
    config: atlas_core::config::ModelConfig,
}

fn fused_harness(armed: bool, installed: bool) -> FusedHarness {
    use crate::weight_map::{
        AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight,
    };
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::kv_cache::KvCacheDtype;

    let gpu = MockGpuBackend::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.num_attention_heads = 24;
    config.num_key_value_heads = 4;
    config.head_dim = 256;
    config.attn_gated = true;
    config.intermediate_size = 128;
    config.moe_intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.vocab_size = 128;
    let buffers = spark_runtime::buffers::BufferArena::new(&config, SLOTS, 4096, 16, 16, &gpu)
        .expect("decode arena");
    let dense = DenseWeight {
        weight: gpu.alloc(4096).unwrap(),
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
        crate::layers::FfnComponent::None,
        0,
        None,
        None,
        None,
        &gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .expect("attention layer");
    layer.attn_qkv_fused = armed;
    // The W8A8 arm's own preconditions, installed directly: the process-global
    // kernel lookup a CPU test cannot run, and the K-major activation-scale
    // adapter without which every cuBLASLt W8A8 arm declines.
    layer.per_token_group_quant_fp8_k = ops::Fp8ActQuant::shared_only(KernelHandle(0xA8A));
    layer.fp8_act_scale_kmajor_k = KernelHandle(0xA8B);
    // ONE fused allocation, with q/k/v as VIEWS inside it — the loader's
    // contract, reproduced so the plan is built against the same aliasing the
    // production path has.
    let fused_w = gpu.alloc(PER_SEQ_QKV as usize * H as usize).unwrap();
    let kb = H as usize / 128;
    let fused_s = gpu.alloc((PER_SEQ_QKV as usize / 128) * kb * 4).unwrap();
    let view = |off_n: u32, n: u32| Fp8Weight {
        weight: fused_w.offset(off_n as usize * H as usize),
        row_scale: fused_s.offset((off_n as usize / 128) * kb * 4),
        n,
        k: H,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.q_weight = Some(QuantWeight::Fp8(view(0, Q_N)));
    layer.k_weight = Some(QuantWeight::Fp8(view(Q_N, KV_N)));
    layer.v_weight = Some(QuantWeight::Fp8(view(Q_N + KV_N, KV_N)));
    if installed {
        layer.qkv_fp8_fused = Some(Fp8Weight {
            weight: fused_w,
            row_scale: fused_s,
            n: PER_SEQ_QKV,
            k: H,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        });
    }
    FusedHarness {
        gpu,
        layer,
        buffers,
        config,
    }
}

/// Runs `f` with a `MultiSeqCtx` at `rows` decode rows.
fn with_ctx<R>(h: &FusedHarness, rows: usize, f: impl FnOnce(&MultiSeqCtx<'_>) -> R) -> R {
    use crate::layer::MoeLoraRoute;
    use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

    // `ATLAS_CUBLAS_GEMM=attn` as the serve resolves it: the family bit is
    // what arms the three-GEMM W8A8 arm this one rides on.
    let mut dispatch = GemmDispatch::defaults();
    dispatch.cublas.attn = true;
    let (derived, levers, stats) = (
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let fwd = ForwardContext {
        buffers: &h.buffers,
        hc_row_offset: 0,
        gpu: &h.gpu,
        config: &h.config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        decode_step: true,
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
        &h.layer,
        &fwd,
        h.buffers.hidden_states(),
        h.buffers.residual(),
        rows,
        16,
        0,
    );
    f(&c)
}

/// The arm is planned at the decode widths and nowhere else, read off the
/// LAYER rather than the pure rule — so the wiring to the installed weight,
/// the cached lever and `per_seq_qkv` is pinned too.
#[test]
fn the_layer_plans_the_fused_arm_only_inside_the_band() {
    let h = fused_harness(true, true);
    for rows in [5usize, 8, 16] {
        assert!(with_ctx(&h, rows, |c| h
            .layer
            .qkv_fused_plan(c, KV_N, true)
            .is_some()));
    }
    for rows in [1usize, 4, 17, 24] {
        assert!(with_ctx(&h, rows, |c| h
            .layer
            .qkv_fused_plan(c, KV_N, true)
            .is_none()));
    }
    // Lever down, and weight absent: both keep the three-GEMM arm.
    for (armed, installed) in [(false, true), (true, false)] {
        let off = fused_harness(armed, installed);
        for rows in [5usize, 8, 16] {
            assert!(with_ctx(&off, rows, |c| off
                .layer
                .qkv_fused_plan(c, KV_N, true)
                .is_none()));
        }
    }
}

/// THE PLAN IS THE SLOT LAYOUT. One GEMM of `fused_n` columns at the slot
/// pitch, whose padded write extent is exactly the whole `qkv_output` — the
/// same bytes the three separate plans cover between them, which is what makes
/// the consumers' offsets unchanged.
#[test]
fn the_fused_plan_writes_the_slot_layout_at_the_slot_pitch() {
    let h = fused_harness(true, true);
    with_ctx(&h, 16, |c| {
        let (w, plan) = h.layer.qkv_fused_plan(c, KV_N, true).expect("armed at 16");
        assert_eq!(plan.n, PER_SEQ_QKV, "q_proj_dim + 2*kv_dim");
        assert_eq!(plan.ldc, PER_SEQ_QKV, "ldc == n: the rows are contiguous");
        assert_eq!(plan.k, H);
        assert_eq!(w.n, PER_SEQ_QKV, "the fused weight spans all three");
        assert_eq!(
            plan.write_extent_bytes(),
            SLOTS * PER_SEQ_QKV as usize * 2,
            "the padded extent is the whole 16-slot buffer"
        );
        // V's plan — the tightest of the three — ends at the same byte.
        let three = h
            .layer
            .qkv_decode_w8a8_plans(c, KV_N, h.buffers.qkv_output_bytes());
        let (v_off, v_plan) = &three[2];
        assert_eq!(
            v_off + v_plan.write_extent_bytes(),
            plan.write_extent_bytes()
        );
    });
}

/// A `qkv_output` one BF16 element short of the padded fused extent must
/// DECLINE, for the reason the three-GEMM arm's own bound exists: cuBLASLt
/// writes `ceil16(m)` rows whatever `m` is, and too small is a cross-buffer
/// write.
#[test]
fn a_qkv_buffer_short_of_the_padded_fused_extent_declines() {
    let full = SLOTS * PER_SEQ_QKV as usize * 2;
    let plan = |cap: usize| DecodeW8a8Plan::strided(16, PER_SEQ_QKV, H, PER_SEQ_QKV, cap);
    assert!(decode_w8a8_selected(
        true,
        false,
        &plan(full),
        WeightQuantFormat::Fp8BlockScaled,
        &scratch()
    ));
    assert!(!decode_w8a8_selected(
        true,
        false,
        &plan(full - 2),
        WeightQuantFormat::Fp8BlockScaled,
        &scratch()
    ));
}

/// THE ALLOCATION CONTRACT. Taken BEFORE the first call and not between two: a
/// per-call `cuMemAlloc` inside a CUDA-graph capture is not a leak, it is a
/// capture failure. The fused arm's operands are the arena's `qkv_output`, the
/// shared W8A8 scratch and a weight VIEW — nothing else exists to allocate.
#[test]
fn planning_the_fused_arm_allocates_nothing() {
    let h = fused_harness(true, true);
    let (allocs, bytes) = (h.gpu.live_alloc_count(), h.gpu.live_bytes());
    for _ in 0..3 {
        with_ctx(&h, 16, |c| {
            let (w, plan) = h.layer.qkv_fused_plan(c, KV_N, true).expect("armed");
            // The output is the arena buffer itself, and the weight is a view
            // into the one fused allocation the loader made.
            assert_eq!(c.qkv_buf, h.buffers.qkv_output());
            assert_eq!(plan.out_capacity_bytes, h.buffers.qkv_output_bytes());
            // Q is the fused buffer's head, so the fused weight and the q
            // VIEW share a pointer — the "no second copy" claim, stated.
            let q = h.layer.q_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
            assert_eq!(w.weight, q.weight);
            assert_eq!(w.row_scale, q.row_scale);
        });
    }
    assert_eq!(
        (h.gpu.live_alloc_count(), h.gpu.live_bytes()),
        (allocs, bytes),
        "the fused q/k/v arm must allocate nothing per call"
    );
}
