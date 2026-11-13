// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `lm_head_verify_gemv` row: a BF16 LM head's
//! speculative-verify rows above 8 run on the batched BF16 GEMV, 16 rows a
//! weight pass and each row bit-identical to the M=1 decode head, instead of
//! the scalar GEMM (`layers/ops/lm_head_verify_gemv.rs`). Resolution: there is
//! no CLI flag; `AVAROK_LM_HEAD_VERIFY_GEMV` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's
//! declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_LM_HEAD_VERIFY_GEMV";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .lm_head_verify_gemv
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
        assert_eq!(ENV, "AVAROK_LM_HEAD_VERIFY_GEMV");
    }
}
