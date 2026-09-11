// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for [`super`] (`ssm_reserve::decode_ring`). A sibling file via
//! `#[path]` — the `ssm_reserve_tests.rs` idiom — so both stay under the
//! 500-line cap.
//!
//! Every byte figure below is the 2026-09-05 rental-H100 configuration from
//! issue #915 (Qwen/Qwen3.8-27B-FP8, one 80 GB H100, hopper recipe): 48 GDN
//! layers, 151.5 MiB of SSM state per sequence, `--max-batch-size 32`,
//! inference reserve 45,823 MiB, 57.2 GiB of weights inside a 71.3 GiB
//! budget. Pinning the arithmetic here is what stops the fit and the reserve
//! from drifting apart.
use super::*;

/// Per-sequence SSM state blob: 48 GDN layers x (h + conv) = 158,859,264 B,
/// the "151.5 MiB/seq" every #915 log line quotes. Same derivation as the
/// `H_BLOB`/`CONV_BLOB` constants in `ssm_reserve_tests.rs`.
const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));
/// One unit of ring depth at `--max-batch-size 32`.
const SLOT_BYTES: usize = 32 * PER_SEQ_BLOB;
/// The 45,823 MiB inference reserve MINUS its 8-slot ring (38,784 MiB): the
/// part of the #915 reserve that does not scale with ring depth.
const RESERVE_WITHOUT_RING: usize = 7_039 * 1024 * 1024;

#[test]
fn decode_ring_decision_matrix() {
    let decide = |layers, spec, override_value, watchdogs| {
        let decision =
            decode_rollback_ring_slots_with(layers, spec, None, override_value, watchdogs);
        (decision.slots, decision.skip_reason)
    };
    let ring = atlas_kernels::DECODE_ROLLBACK_RING_SLOTS;

    for value in ["1", "true", " TRUE "] {
        assert!(
            watchdogs_disabled_from_value(Some(value)),
            "value={value:?}"
        );
    }
    for value in [None, Some(""), Some("0"), Some("false"), Some("yes")] {
        assert!(!watchdogs_disabled_from_value(value), "value={value:?}");
    }

    assert_eq!(decide(0, false, Some("1"), false), (0, None));
    assert_eq!(decide(48, true, Some("1"), true), (ring, None));
    assert_eq!(decide(48, false, Some("0"), false), (0, None));
    assert_eq!(
        decide(48, true, None, false),
        (0, Some("speculative decode active"))
    );
    assert_eq!(
        decide(48, false, None, true),
        (0, Some("watchdogs disabled"))
    );
    assert_eq!(
        decide(48, true, Some("invalid"), true),
        (0, Some("speculative decode active"))
    );
    assert_eq!(decide(48, false, None, false), (ring, None));
}

/// A published depth is what BOTH sizing call sites must see — preflight
/// sized the reserve against it, so `TransformerModel::new` allocating
/// anything else re-opens the divergence this module exists to close.
#[test]
fn a_published_depth_outranks_the_default_the_env_and_the_skips() {
    let ring = atlas_kernels::DECODE_ROLLBACK_RING_SLOTS;
    let d = decode_rollback_ring_slots_with(48, false, Some(2), None, false);
    assert_eq!((d.slots, d.skip_reason), (2, None));
    // Against the legacy env spelling of "8" ...
    assert_eq!(
        decode_rollback_ring_slots_with(48, false, Some(2), Some("1"), false).slots,
        2
    );
    // ... and against BOTH implicit skips, exactly as ATLAS_SSM_DECODE_RING=1
    // already did: an explicit depth is an explicit depth.
    assert_eq!(
        decode_rollback_ring_slots_with(48, true, Some(4), None, true).slots,
        4
    );
    // A published 0 is an explicit off, not an implicit skip: no skip_reason,
    // so the allocating call site does not log a saving nobody asked for.
    let zero = decode_rollback_ring_slots_with(48, false, Some(0), None, false);
    assert_eq!((zero.slots, zero.skip_reason), (0, None));
    // No SSM layers still outranks everything — no recurrent state to snapshot.
    assert_eq!(
        decode_rollback_ring_slots_with(0, false, Some(ring), None, false).slots,
        0
    );
}

/// The value the CLI accepts and the value it publishes come from ONE parse.
#[test]
fn parse_accepts_auto_and_zero_through_eight() {
    assert_eq!(parse_decode_ring_slots("auto"), Ok(None));
    assert_eq!(parse_decode_ring_slots("0"), Ok(Some(0)));
    assert_eq!(
        parse_decode_ring_slots("8"),
        Ok(Some(atlas_kernels::DECODE_ROLLBACK_RING_SLOTS))
    );
    // Above the ceiling the ring is sized for, and anything non-numeric, is a
    // startup refusal — never a silent clamp to 8.
    assert!(parse_decode_ring_slots("9").is_err());
    assert!(parse_decode_ring_slots("AUTO").is_err());
    assert!(parse_decode_ring_slots("").is_err());
    assert!(parse_decode_ring_slots("-1").is_err());
}

/// The #915 boot: 7,039 MiB of non-ring reserve, 14.1 GiB free after 57.2 GiB
/// of weights inside a 71.3 GiB budget. Depth 8 asks 37.88 GiB and refuses;
/// depth 1 fits and boots.
#[test]
fn autofit_picks_the_largest_fitting_depth() {
    let free = (14.1 * 1024.0 * 1024.0 * 1024.0) as usize;
    let fitted = fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, free);
    assert_eq!(fitted, 1, "largest ladder depth that fits in 14.1 GiB");
    assert!(RESERVE_WITHOUT_RING + fitted * SLOT_BYTES <= free);
    assert!(
        RESERVE_WITHOUT_RING + 2 * SLOT_BYTES > free,
        "and 2 does not"
    );

    // With more room the ladder lands on 4 — it never picks 5, 6 or 7, so a
    // fitted depth is always a rung the operator can recognise.
    let roomier = 30 * 1024 * 1024 * 1024;
    assert_eq!(
        fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, roomier),
        4
    );
    // Already fitting at the requested depth is left alone.
    assert_eq!(
        fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, 64 * 1024 * 1024 * 1024),
        8
    );
    // The fit never RAISES a depth: a request below the ladder top is a
    // ceiling, not a target.
    assert_eq!(
        fit_decode_ring_slots(2, RESERVE_WITHOUT_RING, SLOT_BYTES, 64 * 1024 * 1024 * 1024),
        2
    );
}

/// When the rest of the reserve alone exceeds free memory the ring is not the
/// problem: the fit bottoms out at 0 and the caller must still refuse.
#[test]
fn autofit_is_zero_when_even_a_ringless_reserve_does_not_fit() {
    let free = RESERVE_WITHOUT_RING - 1;
    assert_eq!(
        fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, free),
        0
    );
    assert!(
        RESERVE_WITHOUT_RING > free,
        "0 slots is still over budget — the caller refuses rather than booting ringless"
    );
    // A zero-byte ring (pure-attention model) is a no-op, not a division trap.
    assert_eq!(fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, 0, free), 0);
}

#[test]
fn the_fit_ladder_is_descending_and_starts_at_the_wired_default() {
    assert_eq!(
        DECODE_RING_FIT_LADDER[0],
        atlas_kernels::DECODE_ROLLBACK_RING_SLOTS
    );
    assert_eq!(*DECODE_RING_FIT_LADDER.last().unwrap(), 0);
    assert!(
        DECODE_RING_FIT_LADDER.windows(2).all(|w| w[0] > w[1]),
        "`fit_decode_ring_slots` returns the FIRST fitting rung, so the ladder \
         must be strictly descending or it would return a smaller depth than fits"
    );
}
