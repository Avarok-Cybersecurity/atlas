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
        Ok(Self {
            root: AtlasHome::resolve()?.root,
        })
    }

    /// The resolved home together with WHERE it came from.
    ///
    /// `discover()` throws the provenance away, which is why nothing could ever
    /// tell an operator that two commands had disagreed about the root.
    pub fn discover_with_provenance() -> Result<AtlasHome> {
        AtlasHome::resolve()
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

/// Where the Atlas home came from.
///
/// Kept alongside the path because the path alone has never been enough. On
/// 2026-09-05 three boxes ran one campaign with two different `ATLAS_HOME`
/// values between them; each minted its own signing identity
/// (`<root>/identity/ed25519.pk8`), CI rejected the record set for spanning
/// signers, and seven gates were re-measured. Nothing in the run output had
/// said which root was in use, because nothing carried the provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeSource {
    /// `ATLAS_HOME` was set.
    Env,
    /// Derived from `$HOME`.
    HomeDefault,
}

impl HomeSource {
    /// How to say it to an operator.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Env => "from ATLAS_HOME",
            Self::HomeDefault => "default, $HOME/.atlas",
        }
    }
}

/// A resolved Atlas home and its provenance.
#[derive(Clone, Debug)]
pub struct AtlasHome {
    /// The directory itself.
    pub root: PathBuf,
    /// Which rule produced it.
    pub source: HomeSource,
}

/// What is wrong with an Atlas home, if anything.
///
/// Every variant is a condition that has actually cost time here, and each is
/// reported as itself rather than collapsing into "0 recipes cached" — the
/// symptom every one of them used to present as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HomeFault {
    /// The directory exists but this process cannot write into it. The usual
    /// cause is a `sudo` or container run leaving root-owned files behind.
    NotWritable {
        /// The owning uid, when it could be read.
        owner_uid: Option<u32>,
        /// The uid this process runs as, when it could be read.
        process_uid: Option<u32>,
    },
    /// The path exists and is not a directory.
    NotADirectory,
    /// The directory does not exist and could not be created.
    Uncreatable(String),
}

impl std::fmt::Display for HomeFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotWritable {
                owner_uid: Some(o),
                process_uid: Some(p),
            } if o != p => write!(
                f,
                "not writable: owned by uid {o}, this process is uid {p}. A `sudo` \
                 or container run most likely created it. Fix with \
                 `sudo chown -R {p} <path>`, or point ATLAS_HOME somewhere this \
                 user owns."
            ),
            Self::NotWritable { .. } => write!(
                f,
                "not writable by this process. Fix the permissions, or point \
                 ATLAS_HOME somewhere this user owns."
            ),
            Self::NotADirectory => write!(f, "exists but is not a directory"),
            Self::Uncreatable(e) => write!(f, "does not exist and could not be created: {e}"),
        }
    }
}

impl AtlasHome {
    /// Resolve without touching the filesystem.
    pub fn resolve() -> Result<Self> {
        if let Some(explicit) = std::env::var_os("ATLAS_HOME") {
            let root = PathBuf::from(explicit);
            if root.as_os_str().is_empty() {
                bail!("ATLAS_HOME is set but empty");
            }
            return Ok(Self {
                root,
                source: HomeSource::Env,
            });
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .filter(|h| !h.as_os_str().is_empty())
            .context("neither ATLAS_HOME nor HOME is set — cannot place ~/.atlas")?;
        Ok(Self {
            root: home.join(".atlas"),
            source: HomeSource::HomeDefault,
        })
    }

    /// Can this process actually use the home? `None` means yes.
    ///
    /// Probes by WRITING, not by reading a mode bit: ownership, ACLs, a
    /// read-only mount and a full disk all present differently in the metadata
    /// and identically to the thing that matters, which is whether the next
    /// benchmark can put a file there.
    pub fn fault(&self) -> Option<HomeFault> {
        check_usable(&self.root)
    }

    /// One line naming the root and where it came from.
    pub fn describe(&self) -> String {
        format!("{} ({})", self.root.display(), self.source.describe())
    }
}

fn check_usable(root: &Path) -> Option<HomeFault> {
    if root.exists() && !root.is_dir() {
        return Some(HomeFault::NotADirectory);
    }
    if !root.exists()
        && let Err(e) = std::fs::create_dir_all(root)
    {
        return Some(HomeFault::Uncreatable(e.to_string()));
    }
    let probe = root.join(".atlas-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            None
        }
        Err(_) => Some(HomeFault::NotWritable {
            owner_uid: owner_uid_of(root),
            process_uid: current_uid(),
        }),
    }
}

#[cfg(unix)]
fn owner_uid_of(p: &Path) -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(p).ok().map(|m| m.uid())
}

#[cfg(not(unix))]
fn owner_uid_of(_p: &Path) -> Option<u32> {
    None
}

/// This process's uid, read from `/proc/self` rather than through `libc`.
///
/// The owner of `/proc/self` IS the process's effective uid, so this is exact
/// on Linux and costs no new dependency — `libc` is not in this crate's
/// manifest, and adding it would touch `Cargo.lock`, a measured input, for one
/// integer.
#[cfg(unix)]
fn current_uid() -> Option<u32> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self").ok().map(|m| m.uid())
}

#[cfg(not(unix))]
fn current_uid() -> Option<u32> {
    None
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
    write_asset_bytes(dir, name, contents.as_bytes())
}

/// [`write_asset`] for assets that are not text.
///
/// Same contract — compare content, write only on a difference, return whether
/// it wrote — but over bytes, because `read_to_string` fails on any file that
/// is not valid UTF-8 and would therefore report every binary asset as
/// "differs" and rewrite it on each `load()`, churning mtimes that downstream
/// stamps depend on. The vision benchmark provisions PNGs.
pub fn write_asset_bytes(dir: &Path, name: &str, contents: &[u8]) -> Result<bool> {
    let path = dir.join(name);
    if let Ok(existing) = std::fs::read(&path)
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
        let path = dir.join("s.py");
        assert!(write_asset(&dir, "s.py", "print(1)").unwrap());

        let old = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(old))
            .unwrap();
        let pinned_mtime = std::fs::metadata(&path).unwrap().modified().unwrap();

        assert!(!write_asset(&dir, "s.py", "print(1)").unwrap());
        assert_eq!(
            std::fs::metadata(&path).unwrap().modified().unwrap(),
            pinned_mtime,
            "unchanged contents must not rewrite the asset"
        );
        assert!(write_asset(&dir, "s.py", "print(2)").unwrap());
        assert_eq!(std::fs::read_to_string(path).unwrap(), "print(2)");
    }

    #[test]
    fn binary_assets_are_compared_as_bytes() {
        let dir = tmp("binary-asset");
        let path = dir.join("image.bin");
        let bytes = [0xff, 0x00, 0xfe];

        assert!(write_asset_bytes(&dir, "image.bin", &bytes).unwrap());
        assert!(!write_asset_bytes(&dir, "image.bin", &bytes).unwrap());
        assert_eq!(std::fs::read(path).unwrap(), bytes);
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
