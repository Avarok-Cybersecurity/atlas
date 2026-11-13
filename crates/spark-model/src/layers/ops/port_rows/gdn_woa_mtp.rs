// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `gdn_woa_mtp` row: GDN write-on-accept for the batched
//! speculative verify. On, the MTP batched K-row verify asks for the
//! write-on-accept twin (uniform K = 2..4 batches) and folds every verdict
//! before its commits (`scheduler/verify_k4_batch_step.rs`), and the layers
//! may engage the Ada K-generic twin `gated_delta_rule_wyn_woa_<K>` at every
//! width other than 4 (K = 4 keeps the gb10-common pair) for any requesting
//! batched verify, DFlash included. `AVAROK_GDN_WOA=1` is still required.
//! Resolution: there is no CLI flag; `AVAROK_GDN_WOA_MTP` under the toggle
//! grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the
//! target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_GDN_WOA_MTP";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .gdn_woa_mtp
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::target_defaults::Source;

    #[test]
    fn an_unset_environment_keeps_the_declaration() {
        for declared in [false, true] {
            let r = resolve(declared, None);
            assert_eq!(r.value, declared);
            assert_eq!(r.source, Source::Target);
        }
    }

    #[test]
    fn the_toggle_grammar_overrides_both_ways() {
        for off in ["0", "false", "off", "no", " OFF "] {
            let r = resolve(true, Some(off));
            assert!(!r.value, "{off:?}");
            assert_eq!(r.source, Source::Env);
        }
        for on in ["1", "true", "on", "yes"] {
            let r = resolve(false, Some(on));
            assert!(r.value, "{on:?}");
            assert_eq!(r.source, Source::Env);
        }
    }

    #[test]
    fn the_env_name_is_the_registry_name() {
        assert_eq!(ENV, "AVAROK_GDN_WOA_MTP");
    }

    /// Every target declares the row off in this sprint, so a binary with an
    /// empty environment never asks for write-on-accept from the MTP path.
    /// `TARGET_DEFAULTS` is a generated const, so the check is evaluated at
    /// compile time (a declaration flipped on fails the test build).
    #[test]
    fn the_baked_declaration_is_off() {
        const { assert!(!avarok_kernels::TARGET_DEFAULTS.port.gdn_woa_mtp) };
    }
}
