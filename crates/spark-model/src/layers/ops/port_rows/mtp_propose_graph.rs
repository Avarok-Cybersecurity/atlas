// SPDX-License-Identifier: AGPL-3.0-only
//! The `mtp_propose_graph` row: the MTP propose chain runs device-resident
//! (`layers/mtp_head/propose_chain.rs`): each draft position gathers its input
//! embeddings on the device from the previous position's ids and the round
//! reads every draft back once, instead of one blocking readback per position.
//! Drafts are bit-identical. Resolution: `AVAROK_MTP_PROPOSE_GRAPH` under the
//! toggle grammar (`0`/`false`/`off`/`no` = off, anything else = on), else the
//! target's declaration. No CLI flag.

use crate::layers::ops::target_defaults::Resolved;

pub const ENV: &str = "AVAROK_MTP_PROPOSE_GRAPH";

pub fn resolve(default_on: bool, raw: Option<&str>) -> Resolved<bool> {
    crate::layers::ops::target_defaults::resolve_toggle(default_on, raw, false)
}

pub fn row() -> bool {
    crate::layers::ops::target_defaults_port::resolved_port()
        .mtp_propose_graph
        .value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_unless_declared_or_overridden() {
        assert!(!resolve(false, None).value);
        assert!(resolve(false, Some("1")).value);
        assert!(!resolve(true, Some("off")).value);
        assert!(resolve(true, None).value);
    }
}
