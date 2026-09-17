// SPDX-License-Identifier: AGPL-3.0-only

//! Which directory is the Avarok home, and whether this process can use it.
//!
//! Split out of `artifacts.rs` because the two answer different questions: this
//! file decides WHERE the home is (and says where the answer came from, which
//! is what a campaign needs when two boxes disagree), while its parent uses the
//! directory once it has one.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

/// Where the Avarok home came from.
///
/// Kept alongside the path because the path alone has never been enough. On
/// 2026-09-05 three boxes ran one campaign with two different `AVAROK_HOME`
/// values between them; each minted its own signing identity
/// (`<root>/identity/ed25519.pk8`), CI rejected the record set for spanning
/// signers, and seven gates were re-measured. Nothing in the run output had
/// said which root was in use, because nothing carried the provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HomeSource {
    /// `AVAROK_HOME` was set.
    Env,
    /// Derived from `$HOME`.
    HomeDefault,
    /// Derived from `$HOME`, but pointing at the directory this home had
    /// before the AVAROK to AVAROK rename. See [`AvarokHome::resolve`].
    LegacyHomeDefault,
}

impl HomeSource {
    /// How to say it to an operator.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Env => "from AVAROK_HOME",
            Self::HomeDefault => "default, $HOME/.avarok",
            Self::LegacyHomeDefault => {
                "pre-rename default, $HOME/.avarok (rename it to ~/.avarok to end this fallback)"
            }
        }
    }
}

/// A resolved Avarok home and its provenance.
#[derive(Clone, Debug)]
pub struct AvarokHome {
    /// The directory itself.
    pub root: PathBuf,
    /// Which rule produced it.
    pub source: HomeSource,
}

/// What is wrong with an Avarok home, if anything.
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
                 `sudo chown -R {p} <path>`, or point AVAROK_HOME somewhere this \
                 user owns."
            ),
            Self::NotWritable { .. } => write!(
                f,
                "not writable by this process. Fix the permissions, or point \
                 AVAROK_HOME somewhere this user owns."
            ),
            Self::NotADirectory => write!(f, "exists but is not a directory"),
            Self::Uncreatable(e) => write!(f, "does not exist and could not be created: {e}"),
        }
    }
}

impl AvarokHome {
    /// Resolve from the environment, probing the filesystem only for whether a
    /// directory is already there.
    ///
    /// `AVAROK_HOME` wins, then `$HOME/.avarok`, with one exception: a box
    /// installed before the AVAROK to AVAROK rename keeps its state in
    /// `$HOME/.avarok`, so when `$HOME/.avarok` does not exist and
    /// `$HOME/.avarok` is a directory, the old one is resolved instead. That
    /// directory holds the certification signing identity, `artifacts/` and
    /// `runs/`; ignoring it would mint a second signer and re-provision every
    /// benchmark on a machine that had already done both.
    ///
    /// Beyond the `is_dir()` probe this creates nothing, moves nothing and
    /// writes nothing. Migrating is the operator's call, and `mv ~/.avarok
    /// ~/.avarok` is what ends the fallback: doing it here would relocate
    /// several GB as a side effect of reading a path.
    pub fn resolve() -> Result<Self> {
        resolve_from(std::env::var_os("AVAROK_HOME"), std::env::var_os("HOME"))
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

/// [`AvarokHome::resolve`] over explicit inputs, so the rules can be tested.
///
/// Pure over the ENVIRONMENT for the reason `gate::record::resolve_perf_env`
/// gives: `set_var` is unsafe and process-global, and a test that mutated
/// `HOME` could race another test's read and produce exactly the intermittent
/// this crate works to make impossible. The filesystem probe stays real,
/// because which directory already exists IS the question being asked.
pub(super) fn resolve_from(
    avarok_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Result<AvarokHome> {
    if let Some(explicit) = avarok_home {
        let root = PathBuf::from(explicit);
        if root.as_os_str().is_empty() {
            bail!("AVAROK_HOME is set but empty");
        }
        return Ok(AvarokHome {
            root,
            source: HomeSource::Env,
        });
    }
    let home = home
        .map(PathBuf::from)
        .filter(|h| !h.as_os_str().is_empty())
        .context("neither AVAROK_HOME nor HOME is set — cannot place ~/.avarok")?;
    let root = home.join(".avarok");
    // Only while the current name is absent: once `~/.avarok` exists it always
    // wins, so a migrated box never reads the directory it left behind.
    if !root.exists() && home.join(".avarok").is_dir() {
        return Ok(AvarokHome {
            root: home.join(".avarok"),
            source: HomeSource::LegacyHomeDefault,
        });
    }
    Ok(AvarokHome {
        root,
        source: HomeSource::HomeDefault,
    })
}

pub(super) fn check_usable(root: &Path) -> Option<HomeFault> {
    if root.exists() && !root.is_dir() {
        return Some(HomeFault::NotADirectory);
    }
    if !root.exists()
        && let Err(e) = std::fs::create_dir_all(root)
    {
        return Some(HomeFault::Uncreatable(e.to_string()));
    }
    let probe = root.join(".avarok-write-probe");
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
