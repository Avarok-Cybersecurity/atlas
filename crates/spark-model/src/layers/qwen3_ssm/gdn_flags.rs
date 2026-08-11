// SPDX-License-Identifier: AGPL-3.0-only

//! GDN / SSM decode-path flags, resolved ONCE from the serve command line.
//!
//! These three select KERNELS on the GDN decode path, and they are coupled:
//! the FP16 h-state twins only exist on the fused-norm arm, so `h_f16` without
//! `fused_norm` reaches an FP32-only kernel that would read the FP16 pool as
//! FP32 — plausible numbers, silent garbage. That coupling is checked at serve
//! time by `spark-server`'s arg validation, not discovered at the first decode
//! step.
//!
//! ## Why these are set, not read
//!
//! They were three independent `std::env::var` reads scattered across six call
//! sites, each with its own convention (`ATLAS_SSM_H_FP16` presence-gated —
//! where `=0` meant ON — and the other two `== "1"`). That is how the same
//! flag came to be decoded two different ways in one binary. They are now ONE
//! cell, written once from [`set_from_cli`] before any model is built.
//!
//! The environment variables remain honoured when the setter never runs (a
//! test, a microbenchmark example, an older script), so nothing that worked
//! before stops working; the CLI wins when both are present.
//!
//! Follow-up: this is process-scoped, so a hot-swap to a model with a
//! different recipe keeps the first model's kernel selection. The proper home
//! is `ModelLevers`, which is carried per model — deferred because the h-state
//! dtype is read from `SsmLayerState` construction sites that have no
//! `ForwardContext`.

/// The resolved flags. `None` until `set_from_cli` or the first env fallback.
static FLAGS: std::sync::OnceLock<GdnFlags> = std::sync::OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnFlags {
    /// `--ssm-h-dtype f16`: store the GDN decode h-state as FP16.
    pub h_f16: bool,
    /// `--gdn-fused-norm`: fused GDN output-norm decode kernel.
    pub fused_norm: bool,
    /// `--ssm-batched-recurrent`: one strided recurrent launch per batch.
    pub batched_recurrent: bool,
    /// `--exact-verify`: run the sequential-decode-EXACT per-token MTP-verify
    /// chain (issue #435 route (a)) instead of the default WY-chunkwise /
    /// fused BF16-conv arms. OPT-IN, default OFF: by default #435's
    /// divergence REMAINS — spec-on output is NOT bitwise-equal to spec-off
    /// at temp 0. The measured decode-step cost of exact (~+22-36% at the
    /// n=8/16/32 verify rungs) is why; see the flag's help in `serve_args.rs`.
    pub exact_verify: bool,
}

impl GdnFlags {
    /// Whether the MTP-verify pass must run the sequential-decode-exact
    /// conv+GDN chain (issue #435 route (a)). Default FALSE: exact verify is
    /// opt-in via `--exact-verify`, so with default settings spec-on output
    /// is NOT bitwise-equal to spec-off (the #435 divergence ships).
    ///
    /// Pure so it is testable without touching the process-global flags cell.
    /// `h_f16` forces non-exact even when requested, because an FP16 h-state
    /// is a whole-chain numerics change that is not bit-comparable to the
    /// FP32 reference in the first place, and the exact arm's kernels are
    /// FP32 readers (reading the FP16 pool through them would be silent
    /// garbage, not an error). CLI validation additionally REJECTS the
    /// explicit pair, so this clause is defense in depth, not the interface.
    pub fn verify_exact_active(self) -> bool {
        self.exact_verify && !self.h_f16
    }
    /// The legacy environment reading, used when the CLI never set anything.
    ///
    /// `ATLAS_SSM_H_FP16` stays PRESENCE-gated here on purpose: that is how
    /// every script and ledger in the campaign wrote it, and silently changing
    /// `=0` from ON to OFF would retroactively re-label measurements. New
    /// configuration should use `--ssm-h-dtype`.
    fn from_env() -> Self {
        Self {
            h_f16: std::env::var("ATLAS_SSM_H_FP16").is_ok(),
            fused_norm: std::env::var("ATLAS_GDN_FUSED_NORM").as_deref() == Ok("1"),
            batched_recurrent: std::env::var("ATLAS_SSM_BATCHED_RECURRENT").as_deref() == Ok("1"),
            // No legacy environment variable on purpose (house rule: CLI flags
            // or defaults, no new env knobs). Default = the legacy WY arms;
            // exact verify is CLI-opt-in only (`--exact-verify`).
            exact_verify: false,
        }
    }
}

/// Publish the command line's resolution. Call once, before the model builds.
///
/// Returns the value in force, which is the argument unless something already
/// read a flag (in which case the read wins and the caller should say so
/// rather than pretend the setting took).
pub fn set_from_cli(flags: GdnFlags) -> GdnFlags {
    let _ = FLAGS.set(flags);
    *FLAGS.get().expect("just set")
}

/// The resolved flags, falling back to the environment on first touch.
pub fn flags() -> GdnFlags {
    *FLAGS.get_or_init(GdnFlags::from_env)
}

/// `--ssm-h-dtype f16` (legacy `ATLAS_SSM_H_FP16`).
pub fn ssm_h_fp16_enabled() -> bool {
    flags().h_f16
}

/// `--gdn-fused-norm` (legacy `ATLAS_GDN_FUSED_NORM=1`).
pub fn gdn_fused_norm_enabled() -> bool {
    flags().fused_norm
}

/// `--ssm-batched-recurrent` (legacy `ATLAS_SSM_BATCHED_RECURRENT=1`).
pub fn ssm_batched_recurrent_enabled() -> bool {
    flags().batched_recurrent
}

/// `--exact-verify` given (and h-state is FP32): the MTP-verify pass runs
/// the sequential-decode-exact chain. FALSE by default — without the flag the
/// verify pass runs the WY/chunkwise arms and #435's spec-on/spec-off output
/// divergence remains. See [`GdnFlags::verify_exact_active`].
pub fn verify_exact_enabled() -> bool {
    flags().verify_exact_active()
}

#[cfg(test)]
mod tests {
    use super::GdnFlags;

    const BASE: GdnFlags = GdnFlags {
        h_f16: false,
        fused_norm: false,
        batched_recurrent: false,
        exact_verify: false,
    };

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
}
