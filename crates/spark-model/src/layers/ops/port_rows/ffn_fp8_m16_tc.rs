// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the `ffn_fp8_m16_tc` row: a native-FP8 dense FFN over 5 to
//! 32 decode or verify rows runs on the sm_89 16- or 32-row tensor-core GEMM
//! (`layers/dense_ffn/fp8_m16_tc.rs`) instead of the prefill ladder (the W8A8
//! arm at those widths). Resolution: there is no CLI flag;
//! `AVAROK_FFN_FP8_M16_TC` under the toggle grammar (`0`/`false`/`off`/`no` =
//! off, anything else = on), else the target's declaration.
//!
//! One knob rides with it, for measurement only: `AVAROK_FFN_FP8_M16_TC_MIN_ROWS`
//! moves the fewest rows the arm takes ([`min_rows`]).
use crate::layers::ops::target_defaults::Resolved;

/// The environment override.
pub const ENV: &str = "AVAROK_FFN_FP8_M16_TC";

/// The lower-bound knob.
pub const MIN_ROWS_ENV: &str = "AVAROK_FFN_FP8_M16_TC_MIN_ROWS";

/// Fewest rows the arm takes by default: 2 to 4 rows keep `w8a16_gemv_batch4`,
/// which is bit-identical to the decode GEMV and already at the
/// weight-bandwidth floor.
pub const DEFAULT_MIN_ROWS: usize = 5;

/// The row from the target's declaration and the raw environment value.
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

/// The process-wide resolved row.
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .ffn_fp8_m16_tc
        .value
}

/// The lower bound from the raw knob: an integer in 2..=16, else the default.
/// One row is the M=1 decode forward and never this arm's.
pub fn resolve_min_rows(raw: Option<&str>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|v| (2..=16).contains(v))
        .unwrap_or(DEFAULT_MIN_ROWS)
}

/// The process-wide lower bound, read once: the selector runs per layer per
/// step, and a per-call read could change the captured launch set across
/// CUDA-graph replays.
pub fn min_rows() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| resolve_min_rows(std::env::var(MIN_ROWS_ENV).ok().as_deref()))
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
    fn the_env_names_are_the_registry_names() {
        assert_eq!(ENV, "AVAROK_FFN_FP8_M16_TC");
        assert_eq!(MIN_ROWS_ENV, "AVAROK_FFN_FP8_M16_TC_MIN_ROWS");
    }

    #[test]
    fn the_lower_bound_defaults_to_five_and_takes_two_to_sixteen() {
        assert_eq!(resolve_min_rows(None), DEFAULT_MIN_ROWS);
        assert_eq!(DEFAULT_MIN_ROWS, 5);
        assert_eq!(resolve_min_rows(Some("2")), 2);
        assert_eq!(resolve_min_rows(Some(" 9 ")), 9);
        assert_eq!(resolve_min_rows(Some("16")), 16);
        for bad in ["0", "1", "17", "", "x", "-3", "2.5"] {
            assert_eq!(resolve_min_rows(Some(bad)), DEFAULT_MIN_ROWS, "{bad:?}");
        }
    }
}
