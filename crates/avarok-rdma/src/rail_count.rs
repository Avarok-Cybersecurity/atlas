// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail};

/// Guard for `RailSet::complete`'s rail/param pairing. A bare `zip` would
/// silently truncate and leave unmatched rails disconnected.
pub(crate) fn check_rail_count(server_len: usize, rails_len: usize, peer: &str) -> Result<()> {
    if server_len != rails_len {
        bail!(
            "{peer}: server returned {server_len} rail params for {rails_len} client rails — \
             refusing to leave rails unconnected (a zip would silently truncate)"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "railset_tests.rs"]
mod tests;
