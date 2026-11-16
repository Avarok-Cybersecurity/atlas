// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `prefix_cow_tail` row: a prefix-cache hit may end
//! inside a KV block, and the requester then receives a private COPY of that
//! block (every attention layer's K and V, D2D on the prefill stream before
//! the block table is used) instead of the donor's writable tail, which is the
//! #1193 aliasing bug (`spark-runtime/src/radix_tree/inner/cow_walk.rs`,
//! `model/trait_impl/prefill_b/prefix_cow.rs`). Resolution: there is no CLI
//! flag; `AVAROK_PREFIX_COW_TAIL` under the toggle grammar (`0`/`false`/`off`/
//! `no` = off, anything else = on), else the target's declaration. The older
//! `AVAROK_PREFIX_SUBBLOCK` lever wins when set (`0` off, `1` the legacy
//! sharing arms, `cow` copy-on-write). spark-runtime never reads rows:
//! `port_apply` passes [`row`] into
//! `spark_runtime::radix_tree::set_cow_tail_enabled` before the model (and its
//! prefix cache) is built.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_PREFIX_COW_TAIL";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .prefix_cow_tail
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
        assert_eq!(ENV, "AVAROK_PREFIX_COW_TAIL");
    }
}
