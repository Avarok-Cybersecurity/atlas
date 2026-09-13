// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `gdn_flags.rs`, in a sibling so the parent stays under the
//! 500-line cap. Declared there with `#[path]`, the repo's existing
//! pattern; the module keeps its name and its `super::` imports.

use super::{GdnFlags, row_exact_lever_from, ssm_h_dtype_bits, verify_row_exact_required};

const BASE: GdnFlags = GdnFlags {
    h_f16: false,
    h_f16_pool: false,
    fused_norm: false,
    batched_recurrent: false,
    exact_verify: false,
};

// ── The pass-scoped row-exact contract (2026-09-03) ──
//
// The mHC MTP verify re-processes an ALREADY-COMMITTED token as its row 0,
// so that row's logits must equal the serial `decode()` that produced it.
// It cannot: `decode_batched_block` dispatches every stage on ROW COUNT —
// `exl3_gemv` picks its `_m0_` instance at m == 1 and `_m1_` at 2..=8, the
// conv+GDN block takes the WY/BF16-conv arms, the MoE takes `forward_k2` —
// and those are different compiled bodies, not different spellings of one.
// Measured on qwen3.8-flash-next (native EXL3, gamma=1): row-0 logits
// matched serial decode 0/N with the batched arms.
//
// `ForwardContext::gdn_exact_replay` is that pass's declaration, and these
// pin that it — alone, with NO CLI flag — selects the row-exact chain.

/// POSITIVE, and the leg that FAILS without the fix: a pass that declares
/// `gdn_exact_replay` gets the row-exact chain even though `--exact-verify`
/// was never given. The old predicate for the same three sites was
/// `verify_exact_active()`, which is false here (asserted alongside, so
/// this test states the behaviour CHANGE and not merely a new true).
#[test]
fn a_pass_declaring_exact_replay_is_row_exact_without_the_cli_flag() {
    assert!(
        verify_row_exact_required(false, true, true, false),
        "the verify pass's own gdn_exact_replay contract must select the \
         row-exact chain"
    );
    assert!(
        !BASE.verify_exact_active(),
        "and it must do so WITHOUT --exact-verify — that is the change"
    );
}

/// NEGATIVE: every other `decode_batched` caller passes
/// `gdn_exact_replay: false` (DFlash, verify_c2/e, the batched multi-seq
/// verify, plain decode). They keep the batched arms — this widens nothing.
#[test]
fn passes_that_do_not_declare_exact_replay_keep_the_batched_arms() {
    assert!(!verify_row_exact_required(false, false, true, false));
}

/// NEGATIVE: the kill switch (`ATLAS_NO_VERIFY_ROW_EXACT`) puts the
/// declaring pass back on the batched arms, so the two are A/B-able.
#[test]
fn the_kill_switch_restores_the_batched_arms() {
    assert!(!verify_row_exact_required(false, true, false, false));
}

/// The pass-scoped lever is OPT-IN (2026-09-05 polarity): unset → the
/// batched arms; `ATLAS_VERIFY_ROW_EXACT` arms the exact chain; the kill
/// switch wins when both are present.
#[test]
fn row_exact_lever_is_opt_in_and_the_kill_switch_wins() {
    assert!(
        !row_exact_lever_from(false, false),
        "default is the batched arms"
    );
    assert!(
        row_exact_lever_from(true, false),
        "ATLAS_VERIFY_ROW_EXACT arms it"
    );
    assert!(!row_exact_lever_from(false, true));
    assert!(
        !row_exact_lever_from(true, true),
        "kill switch wins over the arm"
    );
    // Composed with the pass predicate: a declaring pass with the lever
    // unset runs batched; armed, it runs the exact chain.
    assert!(!verify_row_exact_required(
        false,
        true,
        row_exact_lever_from(false, false),
        false
    ));
    assert!(verify_row_exact_required(
        false,
        true,
        row_exact_lever_from(true, false),
        false
    ));
}

/// POSITIVE: `--exact-verify` is the WIDER opt-in and is untouched by the
/// kill switch — it still selects the exact chain for every verify body.
#[test]
fn the_cli_opt_in_is_independent_of_the_pass_scoped_lever() {
    assert!(verify_row_exact_required(true, false, false, false));
}

/// NEGATIVE: an FP16 h-state pool forces non-exact from EITHER source. The
/// exact arm's kernels are FP32 readers; reading the narrow pool through
/// them is silent garbage, not an error. Same clause as
/// `verify_exact_active`, restated here because this predicate has a
/// second way in.
#[test]
fn an_f16_h_pool_refuses_the_exact_chain_from_either_source() {
    assert!(!verify_row_exact_required(true, false, true, true));
    assert!(!verify_row_exact_required(false, true, true, true));
}

/// POSITIVE (the default): with no flags the verify pass runs the legacy
/// WY/chunkwise arms, NOT the exact chain. Exact verify became OPT-IN
/// (every surveyed production engine ships exactness opt-in; its measured
/// decode-step cost here is ~+22-36%), so the #435 divergence is the
/// documented default behaviour — this test pins that polarity.
#[test]
fn legacy_wy_verify_is_the_default() {
    assert!(
        !BASE.verify_exact_active(),
        "default must be the legacy WY arms — exact verify is opt-in"
    );
    // Orthogonal flags do not sneak exact mode on.
    assert!(
        !GdnFlags {
            fused_norm: true,
            batched_recurrent: true,
            ..BASE
        }
        .verify_exact_active()
    );
}

/// POSITIVE (the opt-in): `--exact-verify` selects the exact chain, alone
/// and beside the orthogonal GDN flags.
#[test]
fn exact_verify_flag_selects_the_exact_chain() {
    assert!(
        GdnFlags {
            exact_verify: true,
            ..BASE
        }
        .verify_exact_active()
    );
    assert!(
        GdnFlags {
            exact_verify: true,
            fused_norm: true,
            batched_recurrent: true,
            ..BASE
        }
        .verify_exact_active()
    );
}

/// The environment fallback can NEVER turn exact verify on: there is no
/// `ATLAS_*` variable for it on purpose (house rule: no new env knobs),
/// so a serve that skips `set_from_cli` still defaults to the WY arms.
/// Deterministic despite reading the process environment, because only
/// the `exact_verify` field is asserted and no variable feeds it.
#[test]
fn env_fallback_never_enables_exact_verify() {
    assert!(!GdnFlags::from_env().exact_verify);
    // Same rule for the stage-3 pool sizing: no env variable feeds it.
    // `--ssm-h-dtype f16-pool` is the ONLY way to publish it, so a
    // legacy `ATLAS_SSM_H_FP16=1` script keeps the FP32-sized pool.
    assert!(!GdnFlags::from_env().h_f16_pool);
}

/// A narrow pool holding FP32 is an out-of-bounds write, not a mode, so
/// `h_f16_pool` without `h_f16` must not be expressible from any input.
/// This is the ONE decode both the validator and the publisher use, so
/// pinning it here pins it for both.
#[test]
fn the_pool_bit_is_never_set_without_the_dtype_bit() {
    for (spelling, expected) in [
        (None, (false, false)),
        (Some("f32"), (false, false)),
        (Some("f16"), (true, false)),
        (Some("f16-pool"), (true, true)),
        (Some(""), (false, false)),
        (Some("F16-POOL"), (false, false)),
        (Some("f16 "), (false, false)),
    ] {
        assert_eq!(ssm_h_dtype_bits(spelling), expected, "{spelling:?}");
    }
}

/// NEGATIVE: an FP16 h-state forces non-exact EVEN WHEN exact was
/// requested — the exact arm's FP32 kernels must never read the FP16
/// pool. (CLI validation rejects the explicit pair; this is the
/// defense-in-depth layer beneath it.)
#[test]
fn h_f16_forces_non_exact_even_when_requested() {
    assert!(
        !GdnFlags {
            exact_verify: true,
            h_f16: true,
            ..BASE
        }
        .verify_exact_active()
    );
    assert!(
        !GdnFlags {
            h_f16: true,
            ..BASE
        }
        .verify_exact_active()
    );
}
