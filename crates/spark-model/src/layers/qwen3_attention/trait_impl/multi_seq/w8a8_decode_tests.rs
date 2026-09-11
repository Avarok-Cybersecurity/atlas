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
use spark_runtime::gpu::{DevicePtr, KernelHandle};

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
