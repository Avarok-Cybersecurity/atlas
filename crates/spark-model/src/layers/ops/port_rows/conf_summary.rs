// SPDX-License-Identifier: AGPL-3.0-only
//! The `conf_summary` row: collect the per-request logprob confidence summary for every chat request and feed the `atlas_conf_*` metric series (it never changes sampling; a request that sets `atlas_confidence` gets the summary in its usage block whatever this row says). Resolution: the CLI value if the lane has one (see its `cli()` setter), else `AVAROK_CONF_SUMMARY` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_CONF_SUMMARY";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .conf_summary
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
