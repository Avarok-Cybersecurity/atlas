// SPDX-License-Identifier: AGPL-3.0-only
//! The `model_swap_timing` row: time the hot-swap steps (drain, teardown, load, total), keep the last few swap durations per model, expose `x_atlas.resident` and `x_atlas.swap_seconds_estimate` on `/v1/models`, and feed the `atlas_model_swap_*` and `atlas_model_resident_*` series (observability only; it never changes what a swap does). Resolution: the CLI value if the lane has one (see its `cli()` setter), else `AVAROK_MODEL_SWAP_TIMING` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_MODEL_SWAP_TIMING";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .model_swap_timing
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_the_environment_turns_it_on() {
        assert!(!resolve(false, None).value);
        assert!(resolve(false, Some("1")).value);
        assert!(resolve(false, Some("on")).value);
        // The toggle grammar: these spellings are off, anything else is on.
        for off in ["0", "false", "off", "no", " OFF "] {
            assert!(!resolve(true, Some(off)).value, "{off}");
        }
    }

    #[test]
    fn the_environment_wins_over_the_target_and_says_so() {
        use crate::layers::ops::target_defaults::Source;
        assert_eq!(resolve(false, None).source, Source::Target);
        assert_eq!(resolve(false, Some("1")).source, Source::Env);
    }
}
