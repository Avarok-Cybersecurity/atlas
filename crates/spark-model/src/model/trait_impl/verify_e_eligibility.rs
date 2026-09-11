// SPDX-License-Identifier: AGPL-3.0-only

/// The generic multi-sequence verifier uses the residual-stream layout.
/// Highway models keep their own bracketed bodies, which maintain the
/// highway in place of a residual.
///
/// `hc_mult == 0` (residual models) is unconditional. Highway models are
/// admitted only under `ATLAS_HC_BATCH_VERIFY=1`, because their cross-sequence
/// verify is new: the layer body exists
/// (`qwen3_ssm/trait_decode_batched_hc_multi.rs`, the R-row analogue of the
/// K-row `trait_decode_batched_hc.rs`), and the attention side is already
/// served by `decode_multi_seq`'s highway path, but the rewind and PLE
/// snapshot contracts are per-sequence and have not been proven at N > 1.
///
/// It ships OFF for the same reason the GLM EP batched verify did: a wrong
/// verdict here does not fault, it leaves ranks or sequences holding different
/// recurrent state, which surfaces later as one sequence's logits going wrong.
/// Turn it on deliberately, with known-answer probes at C > 1.
pub(super) fn supports_verify_layout(hc_mult: usize) -> bool {
    hc_mult == 0 || hc_batch_verify_enabled()
}

/// `ATLAS_HC_BATCH_VERIFY=1` — opt in to the cross-sequence highway verify.
///
/// Read once: the route must not change under a sequence mid-flight, and a
/// per-call `var()` on the decode path is a syscall per step.
pub(super) fn hc_batch_verify_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_HC_BATCH_VERIFY").as_deref() == Ok("1"))
}

#[cfg(test)]
mod tests {
    use super::supports_verify_layout;

    /// Residual models are admitted with no env at all — the pre-existing
    /// behaviour, unchanged.
    #[test]
    fn residual_models_are_always_admitted() {
        assert!(supports_verify_layout(0));
    }

    /// Highway models are refused unless the opt-in is set. The test reads
    /// the same cached gate the route does, so it asserts the DEFAULT: this
    /// suite does not set `ATLAS_HC_BATCH_VERIFY`.
    #[test]
    fn highway_models_are_refused_by_default() {
        if super::hc_batch_verify_enabled() {
            return; // opted in for a deliberate A/B run; nothing to assert
        }
        assert!(!supports_verify_layout(4));
        assert!(!supports_verify_layout(1));
    }
}
