// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `embed_host_offload` row: the input embedding table in
//! MAPPED (zero-copy) pinned host memory instead of device memory, read by the
//! existing gather kernels through a device-visible pointer
//! (`port::embed_host`). Bit-identical by construction: the same bytes are
//! read, only from host RAM over PCIe. It frees the whole table from VRAM
//! (2.37 GiB on Qwen3.8-27B, 1.18 GiB on the 4B when untied). Resolution:
//! there is no CLI flag; `AVAROK_EMBED_HOST_OFFLOAD` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's
//! declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_EMBED_HOST_OFFLOAD";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .embed_host_offload
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
        assert_eq!(ENV, "AVAROK_EMBED_HOST_OFFLOAD");
    }

    /// Every target declares the row off in this sprint, so a binary with an
    /// empty environment keeps every embedding table on the device. The baked
    /// declaration goes through the resolver (a runtime call), which checks
    /// the same value and also that an empty environment keeps it.
    #[test]
    fn the_baked_declaration_is_off() {
        let r = resolve(
            avarok_kernels::TARGET_DEFAULTS.port.embed_host_offload,
            None,
        );
        assert!(
            !r.value,
            "the baked embed_host_offload declaration must be off"
        );
        assert_eq!(r.source, Source::Target);
    }
}
