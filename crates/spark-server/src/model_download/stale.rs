// SPDX-License-Identifier: AGPL-3.0-only

//! Is the local copy of a model current?
//!
//! "Current" means `refs/main` on disk equals the sha the Hub's `main` points
//! at today. Nothing else on disk records a revision, so this is the only
//! comparison available — and it is the right one, because `refs/main` is also
//! exactly what the resolver reads.

use std::path::Path;

use super::{DownloadError, hf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Freshness {
    /// Not checked, or the check could not reach the Hub.
    ///
    /// **Renders as no badge at all.** Offline is a normal state, not an error
    /// screen — the same rule `recipe::fetch` already states — and a "stale"
    /// badge shown because the network was down would be a lie.
    Unknown,
    Current,
    Stale {
        local: String,
        remote: String,
    },
    /// Nothing on disk to compare.
    Missing,
}

impl Freshness {
    pub fn is_stale(&self) -> bool {
        matches!(self, Self::Stale { .. })
    }
}

/// Compare the local revision against the Hub's current `main`.
///
/// One request. Callers must not run this across a whole library: the Hub
/// throttles, and a list of forty models is forty requests nobody asked for.
pub fn check(repo: &str, cache_root: &Path) -> Result<Freshness, DownloadError> {
    let Some(local) = hf::local_revision(cache_root, repo) else {
        return Ok(Freshness::Missing);
    };
    let (remote, _) = hf::repo_info(repo, hf::token().as_deref())?;
    Ok(if local == remote {
        Freshness::Current
    } else {
        Freshness::Stale { local, remote }
    })
}

/// The short form of a revision, for display.
pub fn short(revision: &str) -> String {
    revision.chars().take(7).collect()
}

#[cfg(test)]
#[path = "stale_tests.rs"]
mod tests;
