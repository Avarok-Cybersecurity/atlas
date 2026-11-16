// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `kv_batched_swap` row: save and restore a swapped-out sequence's image
//! (KV blocks, then SSM state) with batched async copies through one reused page-locked staging
//! buffer instead of one blocking copy per block per layer (`model/trait_impl/sequence/
//! state_io.rs`, `spark_runtime::kv_cache::block_io`). The file is byte-identical either way.
//! Resolution: there is no CLI flag; `AVAROK_KV_BATCHED_SWAP` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_KV_BATCHED_SWAP";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .kv_batched_swap
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
        // The sprint never flips this row on any target before its GPU receipt (3.5).
        // The declaration is a const, so the check binds at compile time
        // (a runtime assert on it is vacuous: clippy::assertions_on_constants).
        const { assert!(!avarok_kernels::TARGET_DEFAULTS.port.kv_batched_swap) };
        assert_eq!(ENV, "AVAROK_KV_BATCHED_SWAP");
    }
}
