// SPDX-License-Identifier: AGPL-3.0-only

//! A startup warning when the disk is nearly full.
//!
//! A full disk on this box does not present as "disk full". It presents as a
//! download that dies two hours in, a snapshot that writes half a shard, or
//! page-cache thrashing that makes a benchmark look like a regression — all of
//! which cost more to diagnose than the one line below costs to emit.
//!
//! Deliberately a WARNING and nothing else. Refusing to start would be worse:
//! serving an already-downloaded model needs no free space at all, and a
//! server that will not come up because the disk is 98% full is a bigger
//! problem than the disk being 98% full.

use std::path::Path;

/// Warn at or above this fraction of the filesystem in use.
///
/// 97% rather than something rounder because the numbers that matter here are
/// absolute: on the 3.7 TB NVMe in these boxes, 3% is ~110 GB — still room for
/// one more checkpoint, and enough warning to act before a download starts
/// failing. On a small disk 3% is little, but so is the thing being protected.
const WARN_AT: f64 = 0.97;

/// The reading, so the decision can be tested without a full disk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Usage {
    pub free: u64,
    pub total: u64,
}

impl Usage {
    pub fn used_fraction(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        1.0 - (self.free as f64 / self.total as f64)
    }

    /// The warning this usage deserves, or `None` when there is nothing to say.
    ///
    /// Pure: [`usage`] does the syscall and hands the result here.
    pub fn warning(&self, path: &Path) -> Option<String> {
        if self.used_fraction() < WARN_AT {
            return None;
        }
        Some(format!(
            "Disk {:.0}% full at {} — only {:.1} GB free of {:.1} GB. \
             Model downloads will fail and page-cache thrashing can make \
             benchmarks look like regressions. Free space before starting a \
             long run.",
            self.used_fraction() * 100.0,
            path.display(),
            self.free as f64 / 1e9,
            self.total as f64 / 1e9,
        ))
    }
}

/// Read the filesystem holding `path`.
pub fn usage(path: &Path) -> Option<Usage> {
    crate::model_download::hf::disk_usage(path).map(|(free, total)| Usage { free, total })
}

/// Emit the startup warning if the disk is nearly full.
///
/// Checks the HuggingFace cache root, because that is where the large writes
/// land; it is also usually the same filesystem as everything else on these
/// boxes. Silent when the disk is fine, when the path cannot be resolved, and
/// on platforms without `statvfs` — a missing reading is not a reason to
/// complain at someone.
pub fn warn_if_nearly_full(cache_dir: Option<&Path>) {
    let Ok(root) = crate::model_resolver::resolve_cache_root(cache_dir) else {
        return;
    };
    let Some(u) = usage(&root) else { return };
    if let Some(msg) = u.warning(&root) {
        tracing::warn!("{msg}");
    }
}

#[cfg(test)]
#[path = "disk_guard_tests.rs"]
mod tests;
