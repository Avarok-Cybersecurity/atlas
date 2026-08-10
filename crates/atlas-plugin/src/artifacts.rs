// SPDX-License-Identifier: AGPL-3.0-only

//! `~/.atlas` — where a plugin keeps everything it had to fetch or build.
//!
//! Layout:
//! ```text
//!   ~/.atlas/
//!     artifacts/<plugin-id>/     downloaded + provisioned material (venvs, datasets)
//!     runs/<benchmark-id>/       persisted run frames, read by the History pane
//! ```
//!
//! Nothing here writes into the repo or the CWD: a benchmark run must not
//! mutate the tree it is measuring.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Handle to the on-disk artifact area. Cheap to clone.
#[derive(Clone, Debug)]
pub struct ArtifactStore {
    root: PathBuf,
}

impl ArtifactStore {
    /// Resolve the Atlas home. `ATLAS_HOME` wins when set (the escape hatch for
    /// a read-only or shared `$HOME`); otherwise `$HOME/.atlas`. A missing
    /// `$HOME` is an error, not a fallback to `/tmp` — a benchmark silently
    /// provisioning several GB somewhere unexpected is worse than a clear stop.
    pub fn discover() -> Result<Self> {
        if let Some(explicit) = std::env::var_os("ATLAS_HOME") {
            let root = PathBuf::from(explicit);
            if root.as_os_str().is_empty() {
                bail!("ATLAS_HOME is set but empty");
            }
            return Ok(Self { root });
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|h| !h.as_os_str().is_empty())
            .context("neither ATLAS_HOME nor HOME is set — cannot place ~/.atlas")?;
        Ok(Self {
            root: home.join(".atlas"),
        })
    }

    /// Point the store at an explicit root (tests, and the `ATLAS_HOME` path).
    pub fn with_root(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// `~/.atlas/artifacts/<plugin_id>`, created.
    pub fn plugin_dir(&self, plugin_id: &str) -> Result<PathBuf> {
        let dir = self.root.join("artifacts").join(plugin_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating artifact dir {}", dir.display()))?;
        Ok(dir)
    }

    /// `~/.atlas/runs/<benchmark_id>`, created.
    pub fn runs_dir(&self, benchmark_id: &str) -> Result<PathBuf> {
        let dir = self.root.join("runs").join(benchmark_id);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating runs dir {}", dir.display()))?;
        Ok(dir)
    }
}

/// Write a compiled-in asset into `dir`, but only when the bytes differ.
///
/// Provisioned scripts must track the binary that ships them — an Atlas upgrade
/// that changes the BFCL scorer has to overwrite the copy in `~/.atlas`, or the
/// run would be scored by the previous release. Comparing content (rather than
/// checking existence) keeps mtimes stable so downstream stamps stay valid.
///
/// Returns `true` when the file was written.
pub fn write_asset(dir: &Path, name: &str, contents: &str) -> Result<bool> {
    let path = dir.join(name);
    if let Ok(existing) = std::fs::read_to_string(&path)
        && existing == contents
    {
        return Ok(false);
    }
    std::fs::write(&path, contents).with_context(|| format!("writing {}", path.display()))?;
    Ok(true)
}

/// A provisioning stamp: a marker file whose contents identify the inputs that
/// produced the artifact. Provisioning is skipped iff the stamp matches, so a
/// changed pin (requirements, script, dataset digest) re-provisions by itself
/// instead of needing anyone to remember to clear a cache.
pub struct Stamp {
    path: PathBuf,
    expected: String,
}

impl Stamp {
    pub fn new(dir: &Path, name: &str, expected: impl Into<String>) -> Self {
        Self {
            path: dir.join(name),
            expected: expected.into(),
        }
    }

    pub fn is_current(&self) -> bool {
        std::fs::read_to_string(&self.path).is_ok_and(|s| s.trim() == self.expected.trim())
    }

    /// Record that provisioning succeeded. Call this LAST — a stamp written
    /// before the work completes turns a half-provisioned directory into a
    /// permanent "already done".
    pub fn commit(&self) -> Result<()> {
        std::fs::write(&self.path, &self.expected)
            .with_context(|| format!("writing stamp {}", self.path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("atlas-plugin-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn dirs_are_created_under_the_configured_root() {
        let root = tmp("dirs");
        let store = ArtifactStore::with_root(&root);
        let p = store.plugin_dir("bfcl").unwrap();
        let r = store.runs_dir("bfcl-subset").unwrap();
        assert!(p.is_dir() && r.is_dir());
        assert_eq!(p, root.join("artifacts/bfcl"));
        assert_eq!(r, root.join("runs/bfcl-subset"));
    }

    #[test]
    fn write_asset_rewrites_only_on_change() {
        let dir = tmp("asset");
        assert!(write_asset(&dir, "s.py", "print(1)").unwrap());
        assert!(!write_asset(&dir, "s.py", "print(1)").unwrap());
        assert!(write_asset(&dir, "s.py", "print(2)").unwrap());
        assert_eq!(
            std::fs::read_to_string(dir.join("s.py")).unwrap(),
            "print(2)"
        );
    }

    #[test]
    fn stamp_is_stale_until_committed_and_tracks_its_inputs() {
        let dir = tmp("stamp");
        let s = Stamp::new(&dir, ".provisioned", "v1");
        assert!(!s.is_current());
        s.commit().unwrap();
        assert!(s.is_current());
        // A changed pin invalidates it without anyone clearing a cache.
        assert!(!Stamp::new(&dir, ".provisioned", "v2").is_current());
    }
}
