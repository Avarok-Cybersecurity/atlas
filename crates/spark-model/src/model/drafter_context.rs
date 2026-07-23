// SPDX-License-Identifier: AGPL-3.0-only

//! Resolution of the MTP **drafter context** feature — ON by default.
//!
//! "Drafter context" is ONE feature with two halves that must not be resolved
//! independently:
//!
//!   * **prefill** — the target prefill captures every position's final-layer
//!     hidden into `mtp_prefill_hidden`, and the drafter's KV is batch-built
//!     over the whole prompt before the first `propose()`.
//!   * **carry** — the previous turn's drafter KV is held across turns and the
//!     new turn appends only the newly computed span.
//!
//! # Why they are ONE switch, not two
//!
//! * **Carry is INERT without prefill.** The carry block is nested inside
//!   `if !self.mtp_prefill_hidden.is_null()` (`trait_impl/speculative.rs`), and
//!   that buffer is only allocated when prefill is on. Carry alone does
//!   nothing at all.
//! * **Prefill without carry is a measured NET LOSS.** A warm turn that must
//!   rebuild the drafter from scratch costs a measured **1136 ms**
//!   (`fc` GEMM alone 874 ms over 11,947 rows, GB10 2026-07-21) against only
//!   ~211 ms/turn of decode saving on the scored workload — **−927 ms/turn**,
//!   spent on TTFT, the metric Atlas wins 1.80x. Carry pays the same context
//!   as a **~21.5 ms mean append** instead.
//!
//! So the only two configurations worth shipping are BOTH ON and BOTH OFF, and
//! `resolve` enforces `carry => prefill` as an invariant.
//!
//! # Why ON is the default
//!
//! `dpcarry` (dgx3, 2026-07-21, `results/dpcarry_20260721_123848`), MLPerf-edge
//! **compliance OVERALL PASS**, 1007/1007, against the matched `ctrl04f`
//! control on the same binary:
//!
//! | metric | ctrl04f (both OFF) | dpcarry (both ON) |
//! |---|---|---|
//! | wall | 5839.26 s | **4984.53 s (−14.6%)** |
//! | TPOT p50 | 52.04 ms | **39.60 ms (−23.9%)** |
//! | TTFT p50 | 1665.5 ms | **1557.1 ms (BETTER)** |
//! | IoU | 0.6281 | 0.6285 (noise floor 0.0028) |
//!
//! TTFT IMPROVED rather than regressed, and the "IoU cost" that had kept
//! drafter prefill opt-in does not exist. Accuracy 87.04 / 88.94 against
//! floors 83.64 / 85.32.
//!
//! # Environment
//!
//! Every switch here is read with a strict `== Some("1")`. Setting a variable
//! to `0` (or to anything else) is NOT how you turn something off in this
//! module — `ATLAS_*=0` has burned this codebase before, because several
//! unrelated flags are presence-checked and are therefore ENABLED by `=0`
//! ([[reference_atlas_env_presence_check_trap]]). Only the `ATLAS_NO_*` name
//! below disables, and only when it is exactly `1`.
//!
//! | variable | effect |
//! |---|---|
//! | *(unset)* | prefill ON, carry ON — the shipped configuration |
//! | `ATLAS_NO_MTP_DRAFTER_CONTEXT=1` | prefill OFF, carry OFF |
//! | `ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE=1` | prefill ON, carry OFF — **research arm only** |
//!
//! The prefill-only arm exists because the individual contributions of the two
//! halves are still being separated by a running e2e (`dpctrl`), and deleting
//! the capability would delete the experiment. It is named `UNSAFE`, it is not
//! a supported deployment, and it logs a warning at startup, because it is the
//! −927 ms/turn configuration described above.
//!
//! The pre-default opt-in names `ATLAS_MTP_DRAFTER_PREFILL` and
//! `ATLAS_MTP_CARRY_DRAFTER` are **obsolete and ignored**. They are not silently
//! accepted: their mere presence is reported at startup so a stale launch
//! script cannot leave an operator believing a value had an effect.

/// Kill switch — disables BOTH halves. Strict `== "1"`.
pub const DISABLE_ENV: &str = "ATLAS_NO_MTP_DRAFTER_CONTEXT";

/// Research-only arm: prefill without carry. Strict `== "1"`. See module docs
/// for why this is a measured net loss and must never be a deployment.
pub const PREFILL_ONLY_ENV: &str = "ATLAS_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE";

/// Opt-in names from before the default flip. Read only to WARN that they are
/// ignored; never read for behaviour.
pub const OBSOLETE_ENVS: [&str; 2] = ["ATLAS_MTP_DRAFTER_PREFILL", "ATLAS_MTP_CARRY_DRAFTER"];

/// Which halves of the drafter-context feature are active.
///
/// Invariant, upheld by [`resolve`] and asserted in its tests:
/// `carry` implies `prefill`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterContext {
    /// Capture prompt hiddens and batch-prefill the drafter KV.
    pub prefill: bool,
    /// Carry the drafter KV across turns of a session and append the new span.
    pub carry: bool,
}

impl DrafterContext {
    /// The shipped configuration.
    pub const BOTH: Self = Self {
        prefill: true,
        carry: true,
    };
    /// Everything off (kill switch).
    pub const OFF: Self = Self {
        prefill: false,
        carry: false,
    };
}

/// Pure resolution of the two switches, with no environment access, so the
/// policy — including the coupling that makes it safe — is unit-testable
/// (SBIO).
///
/// `disable` wins over `prefill_only`: an operator shutting the feature off
/// must not be overridden by a leftover research variable.
pub fn resolve(disable: Option<&str>, prefill_only: Option<&str>) -> DrafterContext {
    if disable == Some("1") {
        return DrafterContext::OFF;
    }
    if prefill_only == Some("1") {
        // Deliberately NOT `DrafterContext::OFF`-safe: this is the only way to
        // reach `carry == false` with `prefill == true`, and it is reachable
        // only through a variable whose name says UNSAFE.
        return DrafterContext {
            prefill: true,
            carry: false,
        };
    }
    DrafterContext::BOTH
}

/// The process-wide resolved configuration, read from the environment once and
/// logged once. Called from model construction, so the log line lands in the
/// serve log at startup — previously neither half was observable from the log
/// at all, only from the launcher's `-e` flags.
pub fn config() -> DrafterContext {
    static CFG: std::sync::OnceLock<DrafterContext> = std::sync::OnceLock::new();
    *CFG.get_or_init(|| {
        let disable = std::env::var(DISABLE_ENV).ok();
        let prefill_only = std::env::var(PREFILL_ONLY_ENV).ok();
        let cfg = resolve(disable.as_deref(), prefill_only.as_deref());

        for name in OBSOLETE_ENVS {
            if let Ok(v) = std::env::var(name) {
                tracing::warn!(
                    "{name}={v} is OBSOLETE and IGNORED — MTP drafter prefill and \
                     cross-turn carry are ON by default. Remove it; to disable \
                     both, set {DISABLE_ENV}=1.",
                );
            }
        }
        tracing::info!(
            "MTP drafter context: prefill={} carry={} ({}). Disable both with {}=1.",
            on_off(cfg.prefill),
            on_off(cfg.carry),
            if cfg == DrafterContext::BOTH {
                "default"
            } else {
                "overridden by environment"
            },
            DISABLE_ENV,
        );
        if cfg.prefill && !cfg.carry {
            tracing::warn!(
                "{PREFILL_ONLY_ENV}=1: drafter prefill is ON with cross-turn carry \
                 OFF. This is a MEASUREMENT ARM, not a deployment — a warm turn \
                 rebuilds the drafter for a measured 1136 ms against ~211 ms/turn \
                 of decode saving (net -927 ms/turn, spent on TTFT).",
            );
        }
        cfg
    })
}

fn on_off(b: bool) -> &'static str {
    if b { "ON" } else { "OFF" }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped path: nothing set at all.
    #[test]
    fn bare_environment_enables_both() {
        assert_eq!(resolve(None, None), DrafterContext::BOTH);
    }

    #[test]
    fn kill_switch_disables_both() {
        assert_eq!(resolve(Some("1"), None), DrafterContext::OFF);
    }

    #[test]
    fn prefill_only_arm_disables_carry_only() {
        assert_eq!(
            resolve(None, Some("1")),
            DrafterContext {
                prefill: true,
                carry: false
            }
        );
    }

    /// Fourth combination: both switches set. The kill switch must win, so an
    /// operator disabling the feature is never overridden by a leftover
    /// research variable.
    #[test]
    fn kill_switch_beats_the_research_arm() {
        assert_eq!(resolve(Some("1"), Some("1")), DrafterContext::OFF);
    }

    /// THE COUPLING. Carry is inert without prefill (its call site is nested
    /// inside the prefill buffer's null check), so no reachable configuration
    /// may enable carry alone. Exhaustive over every string either switch can
    /// hold, including the `=0` spelling that means "off" nowhere in this
    /// module.
    #[test]
    fn carry_never_enabled_without_prefill() {
        let values = [
            None,
            Some("1"),
            Some("0"),
            Some(""),
            Some("true"),
            Some("2"),
        ];
        for d in values {
            for p in values {
                let cfg = resolve(d, p);
                assert!(
                    !cfg.carry || cfg.prefill,
                    "carry without prefill for disable={d:?} prefill_only={p:?}",
                );
            }
        }
    }

    /// `ATLAS_*=0` does NOT disable. Only the `ATLAS_NO_*` name does, and only
    /// at exactly "1" — anything else leaves the shipped default in place.
    #[test]
    fn only_exactly_one_switches_anything() {
        for v in ["0", "", "true", "yes", "2", "1 "] {
            assert_eq!(
                resolve(Some(v), None),
                DrafterContext::BOTH,
                "{DISABLE_ENV}={v:?} must not disable",
            );
            assert_eq!(
                resolve(None, Some(v)),
                DrafterContext::BOTH,
                "{PREFILL_ONLY_ENV}={v:?} must not change anything",
            );
        }
    }

    /// The obsolete opt-in names have no behavioural effect whatsoever: they
    /// are not even inputs to `resolve`. This test pins that they stay out of
    /// the signature by pinning the only two inputs that exist.
    #[test]
    fn obsolete_opt_in_names_are_not_inputs() {
        assert_eq!(OBSOLETE_ENVS.len(), 2);
        assert_eq!(resolve(None, None), DrafterContext::BOTH);
    }
}
