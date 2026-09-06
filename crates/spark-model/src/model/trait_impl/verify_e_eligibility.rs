// SPDX-License-Identifier: AGPL-3.0-only

/// The generic multi-sequence verifier uses the residual-stream layout.
/// Highway models must finish one HC forward and verdict before the next
/// sequence can reuse the shared highway and auxiliary commit span.
pub(super) fn supports_verify_layout(hc_mult: usize) -> bool {
    hc_mult == 0
}

#[cfg(test)]
mod tests {
    use super::supports_verify_layout;

    #[test]
    fn generic_batched_verify_refuses_hc_and_preserves_residual_models() {
        assert!(!supports_verify_layout(4));
        assert!(!supports_verify_layout(1));
        assert!(supports_verify_layout(0));
    }
}
