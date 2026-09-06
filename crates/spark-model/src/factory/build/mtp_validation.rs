// SPDX-License-Identifier: AGPL-3.0-only

//! Startup contract for Qwen MTP's acceptance-aware verification rows.

use anyhow::Result;

pub(super) fn validate_qwen_mtp_verify(requested: bool, batched: bool) -> Result<()> {
    anyhow::ensure!(
        !requested || batched,
        "qwen4_exp MTP verification requires ATLAS_QWEN4EXP_MTP_HC_BATCHED=1: \
         the serial verification fallback overwrites earlier logits and \
         highway rows, so it cannot safely verify draft prefixes"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_qwen_mtp_verify;

    #[test]
    fn active_qwen_mtp_requires_retained_verification_rows() {
        let error = validate_qwen_mtp_verify(true, false).unwrap_err();
        assert!(error.to_string().contains("ATLAS_QWEN4EXP_MTP_HC_BATCHED=1"));
        validate_qwen_mtp_verify(true, true).unwrap();
    }

    #[test]
    fn serial_and_shadow_do_not_require_batched_verification() {
        validate_qwen_mtp_verify(false, false).unwrap();
        validate_qwen_mtp_verify(false, true).unwrap();
    }
}
