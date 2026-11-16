// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `marlin_w4a16` row: a 4-bit BF16 load route that is on
//! (`dense_affine4_from_bf16`, `gdn_affine4_from_bf16`, `attn_affine4_from_bf16`)
//! quantizes every projection the Marlin kernels serve by round-to-nearest int4
//! at group 128 and decodes it on vLLM's Marlin W4A16 GEMM
//! (`port::marlin::route`, kernels `kernels/ada/common/marlin_ada.cu`).
//! Resolution: there is no CLI flag; `AVAROK_MARLIN_W4A16` under the toggle
//! grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the
//! target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_MARLIN_W4A16";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .marlin_w4a16
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
        assert_eq!(ENV, "AVAROK_MARLIN_W4A16");
    }
}
