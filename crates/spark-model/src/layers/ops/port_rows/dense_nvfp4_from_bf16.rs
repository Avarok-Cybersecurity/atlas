// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `dense_nvfp4_from_bf16` row: on a plain-BF16 dense checkpoint, quantize the dense-FFN
//! gate, up and down to NVFP4 once at load and serve them through the `W4Linear` wrapper over the existing
//! `w4a16_gemv` family (`weight_loader/qwen35_dense/nvfp4_from_bf16.rs`). The FP8 FFN route and the affine
//! 4-bit FFN route both win over it for the same layers. Resolution: there is no CLI flag;
//! `AVAROK_DENSE_NVFP4_FROM_BF16` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else =
//! on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_DENSE_NVFP4_FROM_BF16";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row, with its source (the loader's boot line names it).
pub fn resolved() -> Resolved<bool> {
    crate::layers::ops::target_defaults_port::resolved_port().dense_nvfp4_from_bf16
}

/// The process-wide resolved row.
pub fn row() -> bool {
    resolved().value
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
        assert_eq!(ENV, "AVAROK_DENSE_NVFP4_FROM_BF16");
    }

    /// Every target declares the row off in this sprint (numerics; the receipt is Q2 >= 85% of the measured
    /// bandwidth plus the D7 KL gate), so a binary with an empty environment never quantizes to NVFP4 here.
    #[test]
    fn the_baked_declaration_is_off() {
        let baked = avarok_kernels::TARGET_DEFAULTS.port.dense_nvfp4_from_bf16;
        assert!(
            !baked,
            "dense_nvfp4_from_bf16 is a numerics row and never defaults on"
        );
    }
}
