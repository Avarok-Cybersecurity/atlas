// SPDX-License-Identifier: AGPL-3.0-only
//! The `kv_window_desk` row: the host-resident window desk (lane L36, `port/kv_window/`), an
//! APPROXIMATE attention mode for contexts past what VRAM holds. A long sequence keeps on the GPU
//! only its head sinks, its recent window and its recalled pages; every other full KV block moves
//! to host RAM and the full-attention layers read the desk through a compacted block table. Never
//! on by default on any target. Resolution: `AVAROK_KV_WINDOW_DESK` under the toggle grammar
//! (`0`/`false`/`off`/`no` = off, anything else = on), else the target's declaration. The sizes
//! are CLI flags (`--kv-window-tokens`, `--kv-window-head-tokens`), not part of this row; this row
//! has no CLI setter.
use crate::layers::ops::target_defaults::Resolved;
pub const ENV: &str = "AVAROK_KV_WINDOW_DESK";
pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}
pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .kv_window_desk
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
        for on in ["1", "on", "true", "yes"] {
            let r = resolve(false, Some(on));
            assert!(r.value, "{on}");
            assert_eq!(r.source, Source::Env);
        }
        for off in ["0", "false", "off", "no", " OFF "] {
            assert!(!resolve(true, Some(off)).value, "{off}");
        }
    }
}
