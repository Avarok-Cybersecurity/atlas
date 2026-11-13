// SPDX-License-Identifier: AGPL-3.0-only
//! The `kv_tier_disk` row: the host KV tier demotes the blocks it evicts to one record file each under `--kv-tier-dir` (budget `--kv-tier-disk-gb`, least recently used first), promotes them back on a later hit, and a restarted server adopts the records a previous run wrote for the same model fingerprint (`spark-runtime` `kv_host_tier/disk.rs`, lane L27). Refused under `--hermetic`. Resolution: the CLI value if the lane has one (see its `cli()` setter), else `AVAROK_KV_TIER_DISK` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_KV_TIER_DISK";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .kv_tier_disk
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_the_environment_turns_it_on() {
        assert!(!resolve(false, None).value);
        assert!(resolve(false, Some("yes")).value);
        for off in ["0", "false", "off", "no"] {
            assert!(!resolve(true, Some(off)).value, "{off}");
        }
    }

    #[test]
    fn the_environment_wins_over_the_target_and_says_so() {
        use crate::layers::ops::target_defaults::Source;
        assert_eq!(resolve(false, None).source, Source::Target);
        assert_eq!(resolve(true, Some("0")).source, Source::Env);
        assert_eq!(ENV, "AVAROK_KV_TIER_DISK");
    }
}
