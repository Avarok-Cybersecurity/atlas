// SPDX-License-Identifier: AGPL-3.0-only

//! Default-ON fused K=2 MTP-verify GDN epilogue.
//!
//! Production `--num-drafts 1` is K=2 every verify step. Without this path
//! each of the ~30 SSM layers launches `conv1d_update_l2norm` twice and
//! `copy_d2d`s the conv-state for rollback. The fused kernels
//! (`gdn_verify_fused_conv_k2` / `gdn_verify_fused_norm_k2`) fold both
//! positions into one launch each, snapshot pos-0 inline, and are
//! byte-identical to the per-token path (`gdn_verify_fused_microtest`,
//! cos ≥ 0.99999, `--fmad=false`).
//!
//! They used to sit behind opt-in `ATLAS_GDN_FUSED_VERIFY=1` (default OFF)
//! even though the PTX is linked on gb10. That polarity is inverted here:
//! fused is the shipped path; `ATLAS_NO_GDN_FUSED_VERIFY=1` restores the
//! two-launch + D2D sequence. `=0` does NOT disable (house `== "1"` kill).
//! The obsolete opt-in name is ignored (warned once).

/// Kill switch. Strict `== "1"`.
pub const DISABLE_ENV: &str = "ATLAS_NO_GDN_FUSED_VERIFY";

/// Pre-default opt-in. Presence is warned; never read for behaviour.
pub const OBSOLETE_ENV: &str = "ATLAS_GDN_FUSED_VERIFY";

/// Pure so the polarity is testable without touching process env (SBIO).
pub fn gdn_fused_verify_k2_from(no_fused: Option<&str>) -> bool {
    no_fused != Some("1")
}

/// Process-static env resolution. Kernels-present is gated at the call site.
pub fn gdn_fused_verify_k2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var(OBSOLETE_ENV).is_ok() {
            tracing::warn!(
                "{OBSOLETE_ENV} is OBSOLETE and IGNORED — GDN fused K=2 verify \
                 is ON by default. Remove it; to disable, set {DISABLE_ENV}=1."
            );
        }
        gdn_fused_verify_k2_from(std::env::var(DISABLE_ENV).ok().as_deref())
    })
}

/// One-shot serve-log smoking gun. `kernels_present` is the PTX-handle gate.
pub fn log_fused_k2_once(kernels_present: bool, env_on: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if kernels_present && env_on {
            tracing::info!(
                "GDN fused K=2 verify ENGAGED: one conv1d+L2norm launch + inline \
                 pos-0 snapshot (was two conv launches + copy_d2d per SSM layer); \
                 kill switch {DISABLE_ENV}=1"
            );
        } else if !kernels_present {
            tracing::info!(
                "GDN fused K=2 verify NOT engaged: fused kernels absent (NULL \
                 handle); per-token conv1d_update_l2norm path in use"
            );
        } else {
            tracing::info!(
                "GDN fused K=2 verify NOT engaged: {DISABLE_ENV}=1; per-token \
                 conv1d_update_l2norm path in use"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unset must be ON. The old polarity (`ATLAS_GDN_FUSED_VERIFY=1` opt-in)
    /// would fail this assertion.
    #[test]
    fn unset_enables_fused_k2() {
        assert!(gdn_fused_verify_k2_from(None));
    }

    #[test]
    fn kill_switch_exactly_one_disables() {
        assert!(!gdn_fused_verify_k2_from(Some("1")));
    }

    /// `ATLAS_NO_*=0` / empty / junk do NOT disable.
    #[test]
    fn zero_empty_and_junk_do_not_disable() {
        for v in ["0", "", "true", "yes", "2", "1 "] {
            assert!(
                gdn_fused_verify_k2_from(Some(v)),
                "{DISABLE_ENV}={v:?} must not disable"
            );
        }
    }

    #[test]
    fn obsolete_opt_in_is_not_an_input() {
        assert_eq!(OBSOLETE_ENV, "ATLAS_GDN_FUSED_VERIFY");
        assert_eq!(DISABLE_ENV, "ATLAS_NO_GDN_FUSED_VERIFY");
        assert!(gdn_fused_verify_k2_from(None));
    }
}
