// SPDX-License-Identifier: AGPL-3.0-only
//! The `kv_restore_async` row: a request whose prefix the host KV tier restores is parked while its host-to-device copy runs on the tier's copy stream, so every other sequence keeps decoding, and it re-enters admission at the top of the tick after the copy finished (`spark-server` `scheduler/restore_queue.rs`, lane L27). Off: the restore is waited for synchronously at admission, as before. Resolution: the CLI value if the lane has one (see its `cli()` setter), else `AVAROK_KV_RESTORE_ASYNC` under the toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_KV_RESTORE_ASYNC";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .kv_restore_async
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_the_environment_turns_it_on() {
        assert!(!resolve(false, None).value);
        assert!(resolve(false, Some("1")).value);
        for off in ["0", "false", "off", "no", " OFF "] {
            assert!(!resolve(true, Some(off)).value, "{off}");
        }
    }

    #[test]
    fn the_environment_wins_over_the_target_and_says_so() {
        use crate::layers::ops::target_defaults::Source;
        assert_eq!(resolve(false, None).source, Source::Target);
        assert_eq!(resolve(false, Some("1")).source, Source::Env);
        assert_eq!(ENV, "AVAROK_KV_RESTORE_ASYNC");
    }
}
