// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `weights_exl3` row: EXL3 trellis matrices (ExLlamaV3's
//! format, read by `spark_runtime::weights::exl3`) are served from their packed
//! codes on the Ada EXL3 kernels through `port::exl3::from_exl3_tensors` and
//! the `W4Linear` contract. Off: `from_exl3_tensors` refuses and
//! `port::exl3::route_enabled` is false, so no loader route builds an EXL3
//! weight and nothing else changes. Resolution: there is no CLI flag;
//! `AVAROK_WEIGHTS_EXL3` under the toggle grammar (`0`/`false`/`off`/`no` =
//! off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_WEIGHTS_EXL3";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .weights_exl3
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
        assert_eq!(ENV, "AVAROK_WEIGHTS_EXL3");
    }
}
