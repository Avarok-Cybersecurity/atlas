// SPDX-License-Identifier: AGPL-3.0-only
//! The `retro_attn` row: RetroInfer retrieval attention for long-context decode (APPROXIMATE; `port::retro`, lane L33). Resolution: the CLI value if the lane has one (`--retro-attn`, read by `port::retro::api::output_enabled`), else `AVAROK_RETRO_ATTN` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration (`false` on every target, never flipped by default).
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_RETRO_ATTN";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .retro_attn
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_the_environment_turns_it_on() {
        assert!(!resolve(false, None).value);
        assert!(resolve(false, Some("1")).value);
        for off in ["0", "false", "off", "no"] {
            assert!(!resolve(true, Some(off)).value, "{off}");
        }
    }

    #[test]
    fn the_environment_wins_over_the_target_and_says_so() {
        use crate::layers::ops::target_defaults::Source;
        assert_eq!(resolve(false, None).source, Source::Target);
        assert_eq!(resolve(false, Some("on")).source, Source::Env);
    }
}
