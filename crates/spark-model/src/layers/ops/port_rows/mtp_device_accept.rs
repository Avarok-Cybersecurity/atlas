// SPDX-License-Identifier: AGPL-3.0-only
//! The `mtp_device_accept` row: pick MTP verify positions on the device (lane L12,
//! `kernels/ada/common/spec_accept_sample_ada.cu`) instead of copying the verify logits to the host,
//! for rows with no grammar, logit_bias, LZ/DRY or stateful thinking stage. Greedy picks are
//! bit-identical to the host pipeline; sampled picks are distribution-exact (decision D10). Resolution:
//! `AVAROK_MTP_DEVICE_ACCEPT` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else =
//! on), else the target's declaration. The lane has no CLI flag.
use crate::layers::ops::target_defaults::Resolved;

/// The override variable.
pub const ENV: &str = "AVAROK_MTP_DEVICE_ACCEPT";

/// Resolve the row from the target's declaration and the raw override value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The resolved row (process-wide, `OnceLock`-cached by `resolved_port`).
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .mtp_device_accept
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toggle_grammar() {
        assert!(!resolve(false, None).value);
        assert!(resolve(true, None).value);
        assert!(resolve(false, Some("1")).value);
        assert!(!resolve(true, Some("off")).value);
        assert_eq!(
            resolve(false, Some("1")).source,
            crate::layers::ops::target_defaults::Source::Env
        );
    }
}
