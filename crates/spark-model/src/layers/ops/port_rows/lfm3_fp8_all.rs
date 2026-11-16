// SPDX-License-Identifier: AGPL-3.0-only
//! The `lfm3_fp8_all` row: convert an LFM3 BF16 checkpoint's FFN, attention and conv projections to 128x128 block-scaled FP8 (E4M3) at load, so decode streams 1 byte per weight instead of 2 (`weight_loader/lfm3/fp8.rs`). It touches LFM3 checkpoints only (a BF16 `LiquidAI/LFM3-1.7B`; the vendor `-FP8` checkpoint is E4M3 already and its conv projections are the only thing the row converts). Resolution: the CLI value if the lane has one (see its `cli()` setter), else `AVAROK_LFM3_FP8_ALL` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_LFM3_FP8_ALL";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .lfm3_fp8_all
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::target_defaults::Source;

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
        assert_eq!(resolve(false, None).source, Source::Target);
        assert_eq!(resolve(false, Some("1")).source, Source::Env);
        assert_eq!(ENV, "AVAROK_LFM3_FP8_ALL");
    }
}
