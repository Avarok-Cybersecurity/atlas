// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise the BF16 projection used by both ordinary and mixed decode.

use super::{
    LmHeadM16Tc, bf16_batch_gemv_from_value, lm_head_m16_tc_route, m16_tc_head_route_message,
    m16_tc_n_tile_from_value, project_bf16_lm_head,
};
use crate::layers::ops;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

/// Handles for the tensor-core arm, so a case can vary presence and lever
/// independently of the two GEMV arms.
const M16TC_K: u64 = 0x167C;
const M16TC_N64_K: u64 = 0x167C_0064;

/// The arm as a target that LACKS both kernels and an operator who did not ask
/// — the state every pre-existing case is in, and the default this ships with.
fn tc_off() -> LmHeadM16Tc {
    LmHeadM16Tc {
        narrow: KernelHandle(0),
        wide: KernelHandle(0),
        enabled: false,
        n_tile: ops::DENSE_GEMM_M16_BF16_N_TILE,
    }
}

/// Both kernels present, lever as given, CTA width as given.
fn tc(enabled: bool, n_tile: u32) -> LmHeadM16Tc {
    LmHeadM16Tc {
        narrow: KernelHandle(M16TC_K),
        wide: KernelHandle(M16TC_N64_K),
        enabled,
        n_tile,
    }
}

/// `batchm_max` is the band the head is resolved WITH — passed in rather than
/// read from the process, so these cases grade the ladder for any target's
/// declaration without touching `target_defaults::resolved()`'s `OnceLock`.
fn run_band(m: u32, k: u32, present: bool, enabled: bool, batchm_max: u32, expect_batch: bool) {
    run_full(m, k, present, enabled, batchm_max, tc_off(), |launch| {
        assert_eq!(
            launch.func,
            if expect_batch { 0xBF16 } else { 0xCAFE },
            "M={m}: BF16 head selected the wrong kernel"
        );
        let mut sizes = vec![m, 67, k];
        if expect_batch {
            sizes.push(67);
        }
        assert_eq!(
            launch.args[3..],
            sizes
                .iter()
                .map(|value| MockArg::Bytes(value.to_ne_bytes().to_vec()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            launch.grid,
            if expect_batch {
                [67_u32.div_ceil(4), 1, 1]
            } else {
                [67_u32.div_ceil(16), m.div_ceil(16), 1]
            }
        );
        assert_eq!(
            launch.block,
            if expect_batch {
                [256, 1, 1]
            } else {
                [16, 16, 1]
            }
        );
    })
}

/// One dispatch, with the caller grading the single launch it produced. The
/// shared part — no weight copy, one launch, the stream, the three buffers —
/// holds for EVERY arm, which is what makes it worth sharing.
fn run_full(
    m: u32,
    k: u32,
    present: bool,
    enabled: bool,
    batchm_max: u32,
    m16_tc: LmHeadM16Tc,
    grade: impl FnOnce(&spark_runtime::gpu::mock::MockLaunch),
) {
    let gpu = MockGpuBackend::new();
    let n = 67_u32;
    let input = gpu.alloc((m * k * 2) as usize).unwrap();
    let weight = DenseWeight {
        weight: gpu.alloc((n * k * 2) as usize).unwrap(),
    };
    let output = gpu.alloc((m * n * 2) as usize).unwrap();
    let allocated = gpu.alloc_count();
    let batch = KernelHandle(if present { 0xBF16 } else { 0 });
    project_bf16_lm_head(
        &gpu,
        KernelHandle(0xCAFE),
        batch,
        input,
        &weight,
        output,
        [m, n, k],
        enabled,
        batchm_max,
        m16_tc,
        7,
    )
    .unwrap();
    assert_eq!(
        gpu.alloc_count(),
        allocated,
        "the checkpoint weight must not be copied or quantized"
    );
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(launch.stream, 7);
    assert_eq!(
        &launch.args[..3],
        &[
            MockArg::Buffer(input),
            MockArg::Buffer(weight.weight),
            MockArg::Buffer(output)
        ]
    );
    let _ = batch;
    grade(launch);
}

/// The frozen band every target in the tree declares.
fn run_case(m: u32, k: u32, present: bool, enabled: bool, expect_batch: bool) {
    run_band(
        m,
        k,
        present,
        enabled,
        crate::layers::ops::DENSE_GEMV_BATCHM_DECODE_MAX_M,
        expect_batch,
    );
}

#[test]
fn default_small_bf16_head_uses_existing_batch_gemv() {
    for m in [1, 2, 4, 8] {
        run_case(m, 128, true, bf16_batch_gemv_from_value(None), true);
    }
}

#[test]
fn opt_out_missing_kernel_and_wide_head_keep_scalar_fallback() {
    run_case(4, 128, true, bf16_batch_gemv_from_value(Some("0")), false);
    run_case(4, 128, false, true, false);
    // Existing uint4 loads need 16-byte alignment at every input/weight row.
    run_case(4, 130, true, true, false);
    for m in [9, 16] {
        run_case(m, 128, true, true, false);
    }
}

/// The band a target that declares the frozen 8 serves with. Widths 9..=16
/// keep the reassociating tile GEMM they have always used there, so those
/// targets' bits are untouched unless an operator asks.
#[test]
fn lm_head_band_defaults_to_the_frozen_decode_edge() {
    assert_eq!(
        ops::target_defaults::resolve_batchm_max(ops::DENSE_GEMV_BATCHM_DECODE_MAX_M, None).value,
        ops::DENSE_GEMV_BATCHM_DECODE_MAX_M
    );
    for m in [9, 12, 16] {
        run_case(m, 128, true, true, false);
    }
}

/// Hopper's declaration, and the environment spelling of it. 9..=16 move onto
/// the batched GEMV; 17 is still outside the kernel's compile-time row bound
/// and must not.
#[test]
fn lm_head_band_widened_to_sixteen_claims_nine_through_sixteen() {
    assert_eq!(ops::target_defaults::resolve_batchm_max(16, None).value, 16);
    assert_eq!(
        ops::target_defaults::resolve_batchm_max(ops::DENSE_GEMV_BATCHM_DECODE_MAX_M, Some("16"))
            .value,
        16
    );
    for m in [1, 8, 9, 12, 16] {
        run_band(m, 128, true, true, 16, true);
    }
    run_band(17, 128, true, true, 16, false);
}

/// A band above the kernel's `MAX_M` is CLAMPED, not honoured — whether it
/// came from a target's declaration or from the environment: the kernel
/// refuses wider launches, and an Err every decode step is worse than
/// ignoring the excess. Junk and 0 keep the target's value; the band is not a
/// switch, so there is no "off".
#[test]
fn lm_head_band_lever_is_clamped_and_defaults_on_junk() {
    assert_eq!(
        ops::target_defaults::resolve_batchm_max(ops::DENSE_GEMV_BATCHM_DECODE_MAX_M, Some("64"))
            .value,
        ops::DENSE_GEMV_BATCHM_MAX_M
    );
    assert_eq!(
        ops::target_defaults::resolve_batchm_max(64, None).value,
        ops::DENSE_GEMV_BATCHM_MAX_M
    );
    assert_eq!(
        ops::target_defaults::resolve_batchm_max(ops::DENSE_GEMV_BATCHM_DECODE_MAX_M, Some(" 12 "))
            .value,
        12
    );
    for value in [Some("0"), Some(""), Some("sixteen"), Some("-4"), None] {
        assert_eq!(
            ops::target_defaults::resolve_batchm_max(ops::DENSE_GEMV_BATCHM_DECODE_MAX_M, value)
                .value,
            ops::DENSE_GEMV_BATCHM_DECODE_MAX_M,
            "value {value:?} must keep the target's declaration"
        );
    }
}

#[test]
fn legacy_opt_out_value_is_preserved() {
    assert!(!bf16_batch_gemv_from_value(Some("0")));
    for value in [None, Some("1"), Some(""), Some("false"), Some(" 0 ")] {
        assert!(bf16_batch_gemv_from_value(value));
    }
}

/// The BAND is what selects the tier, and it is now the compiled target's
/// declaration rather than a literal.
///
/// 🔴 The band's upper edge decides which BITS a decode of that width
/// produces — above it the width lands on the reassociating tile GEMM — so
/// this is a numerics seam, not only a perf one. Every target in the tree
/// declares the frozen 8, and widths 9..=16 therefore still take the GEMM
/// (asserted above); a target that declares a wider band moves them, which is
/// what this case pins.
#[test]
fn the_declared_band_is_what_selects_the_tier() {
    for m in [9, 12, 16] {
        run_band(m, 128, true, true, 8, false);
        run_band(m, 128, true, true, 16, true);
    }
    // …and the band never overrides the other two gates: a missing kernel or
    // a K that breaks the uint4 alignment still falls through.
    run_band(12, 128, false, true, 16, false);
    run_band(12, 130, true, true, 16, false);
}

// ── The tensor-core arm (`ATLAS_LM_HEAD_M16_TC`, #927/#928) ────────────────
//
// Band x lever x handle, graded at the DISPATCH rather than only on the rule,
// so a route that resolves correctly and then launches the wrong geometry is
// still a failure.

/// One dispatch with the tensor-core arm configured, asserting the launch it
/// produced is `dense_gemm_m16_bf16` at the expected CTA width and argument
/// list (`input, weight, output, m, n, k, a_row_stride, c_row_stride`).
fn expect_tc_launch(m: u32, k: u32, m16_tc: LmHeadM16Tc, handle: u64, n_tile: u32) {
    let n = 67_u32;
    run_full(m, k, true, true, 16, m16_tc, |launch| {
        assert_eq!(launch.func, handle, "M={m}: expected the tensor-core arm");
        assert_eq!(
            launch.args[3..],
            [m, n, k, k, n]
                .iter()
                .map(|value| MockArg::Bytes(value.to_ne_bytes().to_vec()))
                .collect::<Vec<_>>(),
            "the head is contiguous: a_row_stride is k and c_row_stride is n"
        );
        assert_eq!(launch.grid, [n.div_ceil(n_tile), 1, 1]);
        assert_eq!(launch.block, [128, 1, 1]);
    });
}

/// DEFAULT IS UNCHANGED. With the lever unset, every width in the band keeps
/// the batched GEMV the round-7 nsys receipt measured — including the case
/// where both kernels are present, which is what a shipped image looks like.
#[test]
fn tc_head_is_off_until_the_lever_is_set() {
    for m in [5, 8, 13, 16] {
        assert!(lm_head_m16_tc_route(tc(false, 32), m, 5120).is_none());
    }
}

/// The band is 5..=16 and nothing else: `m <= 4` keeps today's path (it is
/// already at the memory roofline) and `m > 16` is past the kernel's M tile.
#[test]
fn tc_head_claims_five_through_sixteen_only() {
    for m in [1, 2, 3, 4, 17, 24, 32] {
        assert!(
            lm_head_m16_tc_route(tc(true, 32), m, 5120).is_none(),
            "M={m} is outside the band and must stay on the existing ladder"
        );
    }
    for m in [5, 8, 13, 16] {
        let (_, kernel, n_tile) =
            lm_head_m16_tc_route(tc(true, 32), m, 5120).expect("M={m} is in the band");
        assert_eq!(kernel.0, M16TC_K);
        assert_eq!(n_tile, ops::DENSE_GEMM_M16_BF16_N_TILE);
    }
}

/// The whole band dispatches through the kernel, at the contiguous pitches.
#[test]
fn tc_head_dispatches_the_band_at_contiguous_pitches() {
    for m in [5, 8, 13, 16] {
        expect_tc_launch(m, 5120, tc(true, 32), M16TC_K, 32);
    }
}

/// A K that is not a whole number of 64-wide pipeline steps has no correct
/// route, so the tier DECLINES rather than launching and being wrong — and the
/// dispatch then falls back to the batched GEMV, not to nothing.
#[test]
fn tc_head_declines_a_k_that_is_not_a_whole_pipeline_step() {
    for k in [96, 130, 5121] {
        assert!(lm_head_m16_tc_route(tc(true, 32), 16, k).is_none(), "K={k}");
    }
    assert!(lm_head_m16_tc_route(tc(true, 32), 16, 128).is_some());
    // 128 is a whole step AND a legal GEMV width: with the lever off the same
    // shape lands on the batched GEMV, which is the fallback this relies on.
    run_band(16, 128, true, true, 16, true);
}

/// A target whose kernel set lacks the entry point declines silently — a
/// 0 handle must never be launched.
#[test]
fn tc_head_declines_when_the_kernel_is_absent() {
    let absent = LmHeadM16Tc {
        narrow: KernelHandle(0),
        wide: KernelHandle(0),
        enabled: true,
        n_tile: 32,
    };
    for m in [5, 16] {
        assert!(lm_head_m16_tc_route(absent, m, 5120).is_none());
    }
}

/// `ATLAS_LM_HEAD_M16_TC_NTILE=64` selects the wide arm; anything unrecognised
/// falls back to 32 rather than failing the boot, and a shadow built before the
/// wide arm existed falls back to the 32-wide KERNEL rather than launching a
/// zero handle.
#[test]
fn tc_head_n_tile_lever_and_its_fallbacks() {
    assert_eq!(
        m16_tc_n_tile_from_value(Some("64")),
        ops::DENSE_GEMM_M16_BF16_N_TILE_WIDE
    );
    assert_eq!(m16_tc_n_tile_from_value(Some(" 64 ")), 64);
    for value in [None, Some("32"), Some(""), Some("128"), Some("sixty-four")] {
        assert_eq!(
            m16_tc_n_tile_from_value(value),
            ops::DENSE_GEMM_M16_BF16_N_TILE,
            "value {value:?} must keep the default tile"
        );
    }
    expect_tc_launch(16, 5120, tc(true, 64), M16TC_N64_K, 64);
    let no_wide = LmHeadM16Tc {
        narrow: KernelHandle(M16TC_K),
        wide: KernelHandle(0),
        enabled: true,
        n_tile: 64,
    };
    let (_, kernel, n_tile) = lm_head_m16_tc_route(no_wide, 16, 5120).expect("falls back");
    assert_eq!(kernel.0, M16TC_K);
    assert_eq!(n_tile, ops::DENSE_GEMM_M16_BF16_N_TILE);
}

/// The arm sits AHEAD of the batched GEMV, and takes the band even when the
/// `ATLAS_LM_HEAD_BATCHM_MAX=16` recipe would have claimed the same widths.
#[test]
fn tc_head_wins_the_band_over_the_widened_gemv() {
    expect_tc_launch(16, 5120, tc(true, 32), M16TC_K, 32);
    // …and the GEMV still serves the same width once the lever is off.
    run_band(16, 5120, true, true, 16, true);
}

// ── The route line's TEXT (H100 round 9: the "<= 2 BF16 ULP" claim was wrong) ──
//
// `log_m16_tc_head_route` itself latches on a process-global `std::sync::Once`
// (see its doc comment) rather than a per-model `ModelStats`, so calling IT
// directly from a test would only ever fire once across this whole test
// binary — order-dependent and not what these tests want to pin. Testing the
// extracted `m16_tc_head_route_message` pure function sidesteps that: it has
// no latch, so every test gets an independent read of the wording.

#[test]
fn the_route_message_no_longer_claims_a_bare_two_ulp_bound() {
    let msg = m16_tc_head_route_message(32, 32);
    // The round-7 claim this replaces (verbatim, so a future edit cannot
    // silently reintroduce it under different wording).
    assert!(
        !msg.contains("(<= 2 BF16 ULP)"),
        "round 9 measured up to 100 ordinal ULP; a bare 2-ULP parenthetical \
         is the bug this test guards against: {msg}"
    );
}

#[test]
fn the_route_message_states_the_real_budget_and_cites_the_receipt() {
    let msg = m16_tc_head_route_message(32, 32);
    assert!(
        msg.contains("within_m16_tc_budget"),
        "must point at the actual contract function, not a bare bound"
    );
    assert!(
        msg.contains("2 ordinal BF16 ULP") && msg.contains("accumulation floor"),
        "must state both halves of the real budget: 2 ULP OR the accumulation floor"
    );
    assert!(
        msg.contains("100 ordinal ULP"),
        "must cite the round-9 receipt that motivated the fix"
    );
    assert!(
        msg.contains("4.9e-6..2.6e-4"),
        "must cite where the over-budget elements sat relative to the row RMS"
    );
}

#[test]
fn the_route_message_still_names_the_lever_kernel_band_and_off_switch() {
    // Fixing the ULP claim must not have dropped any of the pre-existing
    // content a boot-log reader relies on.
    let msg = m16_tc_head_route_message(64, 64);
    assert!(msg.contains("ATLAS_LM_HEAD_M16_TC"));
    assert!(msg.contains("dense_gemm_m16_bf16"));
    assert!(msg.contains("N_TILE=64 (asked 64)"));
    assert!(msg.contains("5..=16 rows"));
    assert!(msg.contains("dense_gemv_bf16_batchm"));
    assert!(msg.contains("REASSOCIATED"));
    assert!(msg.contains("Unset it to restore the bit-exact tier (#927/#928)"));
}
