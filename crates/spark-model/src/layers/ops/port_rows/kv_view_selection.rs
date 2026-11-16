// SPDX-License-Identifier: AGPL-3.0-only
//! The `kv_view_selection` row: the bounded KV view (lanes L31 and L32, `port/kv_view/`), an
//! APPROXIMATE attention mode that, once per user turn, keeps only a budget of selected blocks
//! plus the recent window in the full-attention layers' decode attention. It serves only with
//! this row on AND `--kv-view-budget-tokens` above zero (the CLI side; this row has no CLI
//! setter). Resolution: `AVAROK_KV_VIEW` under the toggle grammar (`0`/`false`/`off`/`no` = off,
//! anything else = on), else the target's declaration (off on every target; never default on).
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_KV_VIEW";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .kv_view_selection
        .value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::ops::target_defaults::Source;

    #[test]
    fn off_unless_the_environment_turns_it_on() {
        assert!(!resolve(false, None).value);
        assert_eq!(resolve(false, None).source, Source::Target);
        assert!(resolve(false, Some("1")).value);
        assert_eq!(resolve(false, Some("1")).source, Source::Env);
        for off in ["0", "false", "off", "no", " OFF "] {
            assert!(!resolve(true, Some(off)).value, "{off:?}");
        }
    }
}
