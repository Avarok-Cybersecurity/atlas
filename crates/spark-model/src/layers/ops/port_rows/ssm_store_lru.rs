// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `ssm_store_lru` row: the host-RAM SSM blob store (`model/ssm_tier/
//! store.rs`, `MemBlobStore`) evicts the least recently used blob (a fault-in hit and a put make
//! a key the most recent) instead of the oldest insert, and the model removes the tier blobs the
//! snapshot index orphaned when it re-homed or swept a spilled prefix
//! (`spark_runtime::radix_tree::snapshot_insert`, drained from `sequence/state_io.rs`).
//! Resolution: there is no CLI flag; `AVAROK_SSM_STORE_LRU` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_SSM_STORE_LRU";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .ssm_store_lru
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
    fn every_target_declares_off() {
        // The declaration is a const, so the check binds at compile time
        // (a runtime assert on it is vacuous: clippy::assertions_on_constants).
        const { assert!(!avarok_kernels::TARGET_DEFAULTS.port.ssm_store_lru) };
        assert_eq!(ENV, "AVAROK_SSM_STORE_LRU");
    }
}
