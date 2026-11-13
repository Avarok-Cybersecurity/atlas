// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `gguf_native_iq4` row: the GGUF loader keeps a dense
//! Qwen3.5-family checkpoint's IQ4_XS, Q4_K and Q6_K projections as the file's
//! raw super-blocks in VRAM (tagged `WeightDtype::{Iq4Xs, Q4K, Q6K}`) for the
//! Ada keep-packed GEMV and MMQ kernels (`port::gguf_q4`), expands IQ4 and Q5_K
//! tensors on the GPU, and maps the file's MTP head (`blk.<L>.nextn.*`).
//! spark-runtime never reads the row: spark-server's `port_apply` publishes
//! [`row`] into `spark_runtime::weights::keep_packed_iq4::set_enabled` before
//! any model loads. Resolution: there is no CLI flag; `AVAROK_GGUF_NATIVE_IQ4`
//! under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on),
//! else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_GGUF_NATIVE_IQ4";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .gguf_native_iq4
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
        assert_eq!(ENV, "AVAROK_GGUF_NATIVE_IQ4");
    }
}
