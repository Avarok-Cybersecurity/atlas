// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `attn_page_sparse` row: Quest page-sparse decode attention
//! (`port/sparse_quest`, `kernels/ada/common/attn_page_sparse_ada.cu`), an APPROXIMATE method
//! that attends only to the first sink pages, the top-k pages by the Quest bound and the recent
//! window once the context passes `--attn-sparse-min-ctx`. Never default on, on any target.
//! Resolution: there is no on/off CLI flag (the `--attn-sparse-*` flags only shape the selector);
//! `AVAROK_ATTN_PAGE_SPARSE` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything
//! else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_ATTN_PAGE_SPARSE";

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .attn_page_sparse
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
        // Approximate attention is never default on (master plan 3.1, 3.5). The declaration
        // is bound to a local first so the check reads the compiled-in table at run time
        // instead of being a constant assertion (clippy::assertions_on_constants).
        let declared = avarok_kernels::TARGET_DEFAULTS.port.attn_page_sparse;
        assert!(
            !declared,
            "attn_page_sparse is APPROXIMATE and never defaults on"
        );
        assert_eq!(ENV, "AVAROK_ATTN_PAGE_SPARSE");
    }
}
