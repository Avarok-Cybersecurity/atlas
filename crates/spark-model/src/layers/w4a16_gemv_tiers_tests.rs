// SPDX-License-Identifier: AGPL-3.0-only

//! The tier decision is the whole of this change: the kernels are the same
//! template at a different `MAX_M`, so every byte of behaviour difference is
//! "which tier did M pick".
//!
//! Tests target the PURE [`select_tier`], never the process env, so both
//! polarities of the kill switch are exercised in one run and neither depends
//! on `OnceLock` latch order.

use super::{W4A16_BATCHM_WIDTHS, W4a16BatchmTiers, select_tier};
use spark_runtime::gpu::mock::MockGpuBackend;

/// Every tier resolved — the shipping GB10 target after this change.
const ALL: [bool; 5] = [true; 5];
/// Tiers 5/6/7 absent — any target PTX built before they existed.
const LEGACY: [bool; 5] = [true, false, false, false, true];

/// Width the decision picks, or `None`. Thin wrapper so the tables below read
/// as `m -> width` and not as `m -> index`.
fn pick(m: u32, present: [bool; 5], exact_m: bool) -> Option<u32> {
    select_tier(m, present, exact_m).map(|i| W4A16_BATCHM_WIDTHS[i])
}

/// POSITIVE, and the point of the change: with every tier loaded, M=5/6/7 get
/// their EXACT tier instead of riding MAX_M=8.
///
/// PROVEN BY: this is red on the pre-change dispatch (which had no 5/6/7 to
/// pick), and red again under the kill switch — see `kill_switch_*` below.
#[test]
fn exact_m_rows_pick_their_own_tier() {
    assert_eq!(pick(5, ALL, true), Some(5));
    assert_eq!(pick(6, ALL, true), Some(6));
    assert_eq!(pick(7, ALL, true), Some(7));
}

/// The widths that already had a tier are UNCHANGED, so C=1 (M<=4) and the
/// full-width M=8 verify keep byte-identical dispatch. A regression here would
/// silently re-route the two rungs this change is not supposed to touch.
#[test]
fn pre_existing_widths_are_untouched() {
    for m in 1..=4 {
        assert_eq!(pick(m, ALL, true), Some(4), "m={m} must stay on batch4");
    }
    assert_eq!(pick(8, ALL, true), Some(8));
}

/// Above the family the caller must fall through to the tile GEMMs / wide
/// tiers, and M=0 is not a launch. Returning `Some` here would dispatch a
/// kernel that truncates rows silently (`w4a16_gemv_batchm`'s own
/// `debug_assert`), which is garbage output rather than a crash.
#[test]
fn out_of_family_widths_decline() {
    assert_eq!(pick(0, ALL, true), None);
    assert_eq!(pick(9, ALL, true), None);
    assert_eq!(pick(16, ALL, true), None);
    assert_eq!(pick(32, ALL, true), None);
}

/// KILL SWITCH: with the exact-M tiers disabled the decision is byte-for-byte
/// the shipped one — `1..=4 => batch4`, `5..=8 => batch8` — even though the
/// tiers are loaded and present.
#[test]
fn kill_switch_restores_the_shipped_decision() {
    for m in 1..=4 {
        assert_eq!(pick(m, ALL, false), Some(4), "m={m}");
    }
    for m in 5..=8 {
        assert_eq!(pick(m, ALL, false), Some(8), "m={m}");
    }
    assert_eq!(pick(9, ALL, false), None);
}

/// The kill switch and a target that never had the tiers must agree exactly —
/// otherwise the A/B control leg is not measuring the tiers, it is measuring
/// the difference between two fallbacks.
#[test]
fn kill_switch_matches_a_legacy_target() {
    for m in 0..=10 {
        assert_eq!(pick(m, ALL, false), pick(m, LEGACY, true), "m={m}");
    }
}

/// PRESENCE, not assumption: 5/6/7 are only chosen when the LOADED target
/// resolved them. A target with a partial set must widen to the next resolved
/// tier, never dispatch a zero handle.
#[test]
fn absent_tiers_widen_to_the_next_resolved_one() {
    // 5 and 6 missing, 7 and 8 present.
    let partial = [true, false, false, true, true];
    assert_eq!(pick(5, partial, true), Some(7));
    assert_eq!(pick(6, partial, true), Some(7));
    assert_eq!(pick(7, partial, true), Some(7));
    assert_eq!(pick(8, partial, true), Some(8));
    // Only the exact-M tiers present: M<=4 must still find a home (5), and the
    // legacy widths must not be invented.
    let no_legacy = [false, true, true, true, false];
    assert_eq!(pick(4, no_legacy, true), Some(5));
    assert_eq!(pick(8, no_legacy, true), None);
}

/// Nothing resolved (a non-NVFP4 build): the family declines at every width so
/// the caller keeps its tile-GEMM path. A zero handle reaching
/// `KernelLaunch::launch` is a hard failure, not a fallback.
#[test]
fn empty_family_declines_everywhere() {
    for m in 0..=9 {
        assert_eq!(pick(m, [false; 5], true), None, "m={m}");
        assert_eq!(pick(m, [false; 5], false), None, "m={m}");
    }
}

/// A default-constructed table is the "no NVFP4 kernels" state, and every
/// consumer gates on `.0 != 0`; `has_base` must agree with it.
#[test]
fn default_table_is_empty_and_declines() {
    let t = W4a16BatchmTiers::default();
    assert!(!t.has_base());
    for m in 0..=9 {
        assert_eq!(t.kernel(m).0, 0, "m={m}");
        assert_eq!(t.width(m), None, "m={m}");
    }
}

/// The width table must stay sorted ascending, and the production resolver
/// must request that same family in the same order. A typo here silently
/// produces a zero handle and widens dispatch even when the tier was shipped.
/// The batch8 slot prefers the register-tiled `w4a16_gemv_batch8_rt2` (#648);
/// with every lookup resolving (as on the mock) the rt2 hit needs no
/// fallback lookup, so exactly one request per width is still the contract.
#[test]
fn width_table_and_resolver_stay_in_lockstep() {
    assert!(W4A16_BATCHM_WIDTHS.windows(2).all(|w| w[0] < w[1]));
    let gpu = MockGpuBackend::new();
    let tiers = W4a16BatchmTiers::resolve(&gpu);
    assert!(tiers.handles.iter().all(|h| h.0 != 0));
    assert_eq!(
        gpu.kernel_lookups_snapshot(),
        W4A16_BATCHM_WIDTHS.map(|w| {
            let func = if w == 8 {
                "w4a16_gemv_batch8_rt2".to_owned()
            } else {
                format!("w4a16_gemv_batch{w}")
            };
            ("w4a16_gemv".to_owned(), func)
        })
    );
}
