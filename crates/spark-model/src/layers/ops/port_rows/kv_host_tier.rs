// SPDX-License-Identifier: AGPL-3.0-only
//! The `kv_host_tier` row: keep a pinned host-RAM copy of prefix-cached paged-KV blocks (lane
//! L26a's tier, driven by lane L26b's glue in `model/kv_tier_glue.rs`) and restore a prompt's
//! evicted prefix from it by one batched H2D instead of recomputing it. The tier serves only
//! with this row on AND a `--kv-host-tier-gb` budget AND the prefix cache on. Resolution:
//! `AVAROK_KV_HOST_TIER` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else =
//! on), else the target's declaration (the budget flag is the CLI side; this row has no CLI
//! setter). `port_apply` publishes the value to `spark_runtime::kv_host_tier::config`.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_KV_HOST_TIER";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .kv_host_tier
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
