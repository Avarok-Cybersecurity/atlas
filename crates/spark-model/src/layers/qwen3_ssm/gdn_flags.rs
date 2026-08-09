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
    /// `--verify-wy`: restore the WY-chunkwise / fused BF16-conv MTP-verify
    /// arms (the pre-#435 behaviour) for A/B. Default OFF: the verify pass
    /// runs the EXACT per-token chain that is bitwise-equal to sequential
    /// decode, because spec-on output equality at temp 0 is a correctness
    /// property, not a tunable (issue #435, PR1).
    pub verify_wy: bool,
}

impl GdnFlags {
    /// Whether the MTP-verify pass must run the sequential-decode-exact
    /// conv+GDN chain (issue #435 route (a)).
    ///
    /// Pure so it is testable without touching the process-global flags cell.
    /// `verify_wy` opts back into the WY arms; `h_f16` also disables exact
    /// mode, because an FP16 h-state is a whole-chain numerics change that is
    /// not bit-comparable to the FP32 reference in the first place, and the
    /// exact arm's kernels are FP32 readers (reading the FP16 pool through
    /// them would be silent garbage, not an error).
    pub fn verify_exact_active(self) -> bool {
        !self.verify_wy && !self.h_f16
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
            // or defaults, no new env knobs). Default = exact verify.
            verify_wy: false,
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

/// `--verify-wy` NOT given (and h-state is FP32): the MTP-verify pass runs
/// the sequential-decode-exact chain. See [`GdnFlags::verify_exact_active`].
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
        verify_wy: false,
    };

    /// POSITIVE: the default flag set (no `--verify-wy`, FP32 h-state) runs
    /// the exact verify chain — the #435 fix must be ON by default.
    #[test]
    fn verify_exact_is_the_default() {
        assert!(BASE.verify_exact_active(), "default must be exact verify");
        // Orthogonal flags do not disturb the decision.
        assert!(
            GdnFlags {
                fused_norm: true,
                batched_recurrent: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }

    /// NEGATIVE: each opt-out disables exact mode on its own — `--verify-wy`
    /// (the A/B kill switch) and `--ssm-h-dtype f16` (whose FP16 pool the
    /// exact arm's FP32 kernels must never read).
    #[test]
    fn verify_wy_and_h_f16_each_disable_exact() {
        assert!(
            !GdnFlags {
                verify_wy: true,
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
        assert!(
            !GdnFlags {
                verify_wy: true,
                h_f16: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }
}
