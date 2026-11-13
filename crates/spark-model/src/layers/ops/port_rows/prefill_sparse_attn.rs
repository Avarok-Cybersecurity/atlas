// SPDX-License-Identifier: AGPL-3.0-only
//! The `prefill_sparse_attn` row: APPROXIMATE block-sparse prefill attention, XAttention-style (lane L35,
//! `port::sparse_prefill`, kernels `kernels/ada/common/prefill_block_sparse_ada.cu`). A prefill chunk
//! that attends to at least `--prefill-sparse-min-tokens` keys scores 128-token key blocks by strided
//! antidiagonal sums, keeps the blocks holding `--prefill-sparse-threshold` of each query block's mass
//! (plus the sink block and the diagonal band) and runs attention over those only. It changes outputs, so
//! it is off on every target and never defaults on; the module exists only in `kernels/ada`, so elsewhere
//! the row only logs that the arm is absent. Resolution: `AVAROK_PREFILL_SPARSE_ATTN` under the toggle
//! grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration; the
//! knobs have their own flags and variables (`port::sparse_prefill::config`).
use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_PREFILL_SPARSE_ATTN";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .prefill_sparse_attn
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::target_defaults::Source;

    #[test]
    fn the_toggle_grammar_and_the_declaration() {
        assert!(!resolve(false, None).value);
        assert_eq!(resolve(false, None).source, Source::Target);
        assert!(resolve(false, Some("1")).value);
        assert_eq!(resolve(false, Some("1")).source, Source::Env);
        for off in ["0", "false", "off", "NO"] {
            assert!(!resolve(true, Some(off)).value, "{off:?}");
        }
    }

    #[test]
    fn no_target_declares_it_on() {
        let d = avarok_kernels::TARGET_DEFAULTS.port.prefill_sparse_attn;
        assert!(
            !d,
            "prefill_sparse_attn is APPROXIMATE and never defaults on"
        );
    }
}
