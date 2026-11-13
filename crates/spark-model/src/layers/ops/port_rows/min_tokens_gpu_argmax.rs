// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `min_tokens_gpu_argmax` row: a greedy decode step whose
//! sequences still have a `min_tokens` floor to reach stays on the device
//! argmax, and only a step whose argmax is an EOS id is redone on the host
//! (`spark-server/src/scheduler/min_floor_fast.rs`). Resolution: there is no
//! CLI flag; `AVAROK_MIN_TOKENS_GPU_ARGMAX` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's
//! declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_MIN_TOKENS_GPU_ARGMAX";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .min_tokens_gpu_argmax
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
        assert_eq!(ENV, "AVAROK_MIN_TOKENS_GPU_ARGMAX");
    }
}
