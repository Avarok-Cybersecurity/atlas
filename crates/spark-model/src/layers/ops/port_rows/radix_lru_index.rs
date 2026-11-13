// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `radix_lru_index` row: the radix prefix cache picks its
//! eviction victim from an ordered `(last_access, node id)` index of evictable
//! leaves instead of scanning every node per evicted block
//! (`spark-runtime/src/radix_tree/inner/lru_index.rs`). Same victim as the scan,
//! by construction and by the randomized oracle. Resolution: there is no CLI
//! flag; `AVAROK_RADIX_LRU_INDEX` under the toggle grammar (`0`/`false`/`off`/
//! `no` = off, anything else = on), else the target's declaration.
//! spark-runtime never reads rows: `port_apply` passes [`row`] into
//! `spark_runtime::radix_tree::set_lru_index_enabled` before the model (and its
//! prefix cache) is built.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_RADIX_LRU_INDEX";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .radix_lru_index
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
        assert_eq!(ENV, "AVAROK_RADIX_LRU_INDEX");
    }
}
