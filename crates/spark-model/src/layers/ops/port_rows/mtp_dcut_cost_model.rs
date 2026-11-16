// SPDX-License-Identifier: AGPL-3.0-only
//! The `mtp_dcut_cost_model` row: D-Cut v2, which sizes the batched MTP verify by the budget that maximizes expected tokens per second under an online cost table and calibrated draft survival, instead of D-Cut v1's fixed 0.75 ratio (it only chooses how many drafts to verify, so accept and reject stay exact; every sequence keeps at least one draft and the width cap stays). Resolution: the CLI value if the lane has one (see its `cli()` setter), else `AVAROK_MTP_DCUT_COST` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_MTP_DCUT_COST";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .mtp_dcut_cost_model
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_the_environment_turns_it_on() {
        assert!(!resolve(false, None).value);
        assert!(resolve(false, Some("1")).value);
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
        assert_eq!(ENV, "AVAROK_MTP_DCUT_COST");
    }
}
