// SPDX-License-Identifier: AGPL-3.0-only

//! The `dense_fp8_from_bf16` row of the target table.
//!
//! A child of `tests`, like `target_defaults_gateup_tests.rs`, so the `GB10` /
//! `HOPPER` fixtures and the `with` / `empty` helpers are the parent's.

use super::*;

/// Neither fixture declares the route: it is a numerics change, and no target
/// in this tree has the receipt for it. A target that wants it declares it.
#[test]
fn no_target_here_takes_the_route_with_an_empty_environment() {
    for t in [&GB10, &HOPPER] {
        let r = empty(t).dense_fp8_from_bf16;
        assert!(!r.value, "{}", t.hw);
        assert!(!r.from_env(), "{}", t.hw);
    }
}

/// `=1` arms the A/B where the target says no, `=0` keeps BF16 where a target
/// says yes, and both report `(env)`.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    let on = with(&GB10, &[("AVAROK_DENSE_FP8_FROM_BF16", "1")]).dense_fp8_from_bf16;
    assert_eq!(on, Resolved::env(true));
    let declaring = TargetDefaults {
        dense_fp8_from_bf16: true,
        ..GB10
    };
    assert_eq!(
        empty(&declaring).dense_fp8_from_bf16,
        Resolved::target(true)
    );
    for off in ["0", "false", "off", "no"] {
        assert_eq!(
            with(&declaring, &[("AVAROK_DENSE_FP8_FROM_BF16", off)]).dense_fp8_from_bf16,
            Resolved::env(false),
            "AVAROK_DENSE_FP8_FROM_BF16={off:?} must keep BF16 weights",
        );
    }
}

#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("dense_fp8_from_bf16=off"), "{line}");
    assert!(!line.contains("dense_fp8_from_bf16=off (env)"), "{line}");
    let line = format_levers(&with(&GB10, &[("AVAROK_DENSE_FP8_FROM_BF16", "1")]));
    assert!(line.contains("dense_fp8_from_bf16=on (env)"), "{line}");
}
