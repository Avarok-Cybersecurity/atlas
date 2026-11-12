// SPDX-License-Identifier: AGPL-3.0-only

//! The `memory_fit_consumer` row of the target table.
//!
//! A child of `tests`, like `target_defaults_fp8_from_bf16_tests.rs`, so the
//! `GB10` / `HOPPER` fixtures and the `with` / `empty` helpers are the
//! parent's.

use super::*;

/// Neither datacenter fixture declares the row, so neither changes sizing.
#[test]
fn no_datacenter_target_takes_the_consumer_fits() {
    for t in [&GB10, &HOPPER] {
        let r = empty(t).memory_fit_consumer;
        assert!(!r.value, "{}", t.hw);
        assert!(!r.from_env(), "{}", t.hw);
    }
}

/// `=1` takes the fits where the target says no, `=0` returns a declaring
/// target to the shared sizing, and both report `(env)`.
#[test]
fn the_environment_overrides_the_row_in_both_directions() {
    let on = with(&GB10, &[("AVAROK_MEMORY_FIT_CONSUMER", "1")]).memory_fit_consumer;
    assert_eq!(on, Resolved::env(true));
    let declaring = TargetDefaults {
        memory_fit_consumer: true,
        ..GB10
    };
    assert_eq!(
        empty(&declaring).memory_fit_consumer,
        Resolved::target(true)
    );
    for off in ["0", "false", "off", "no"] {
        assert_eq!(
            with(&declaring, &[("AVAROK_MEMORY_FIT_CONSUMER", off)]).memory_fit_consumer,
            Resolved::env(false),
            "AVAROK_MEMORY_FIT_CONSUMER={off:?} must restore the shared sizing",
        );
    }
}

#[test]
fn the_serve_line_names_the_row_and_marks_an_override() {
    let line = format_levers(&empty(&GB10));
    assert!(line.contains("memory_fit_consumer=off"), "{line}");
    assert!(!line.contains("memory_fit_consumer=off (env)"), "{line}");
    let line = format_levers(&with(&GB10, &[("AVAROK_MEMORY_FIT_CONSUMER", "1")]));
    assert!(line.contains("memory_fit_consumer=on (env)"), "{line}");
}
