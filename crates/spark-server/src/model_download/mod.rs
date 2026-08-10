// SPDX-License-Identifier: AGPL-3.0-only

//! Downloading a model from HuggingFace, and telling whether the local copy is
//! current.
//!
//! Atlas could previously only *consume* a model that was already in the HF
//! cache; a recipe naming a checkpoint you did not have dead-ended with advice
//! to go and run `huggingface-cli`. This is the missing half.
//!
//! # Where this lives
//!
//! Beside `model_resolver`, not under `tui/`, because it is a server capability
//! and because the resolver owns the cache-layout knowledge this must agree
//! with — [`hf::publish`] calls the resolver's own `snapshot_has_weights`
//! rather than reimplementing the check that decides whether a download counts
//! as finished.
//!
//! # Threading
//!
//! The dashboard's rule holds here: the render thread never polls a future, it
//! only `try_recv`s. The work runs on a plain `std::thread` named
//! `atlas-download`, and progress arrives on a `std::sync::mpsc`. See
//! `recipe/fetch.rs` for the full statement and
//! `.github/workflows/tui-threading.yml` for its enforcement.
//!
//! # The ordering that matters
//!
//! `refs/main` is written **last**, and only after the snapshot verifies. It
//! is the sole marker that a download finished: until it exists, the resolver
//! refuses the model outright, so a process killed at shard 2 of 9 leaves a
//! resumable partial rather than a model that looks loadable and is not.

pub mod hf;
pub mod plan;
pub mod stale;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender, channel};

pub(crate) const HOST: &str = "https://huggingface.co";
pub(crate) const AGENT: &str = concat!("atlas-spark/", env!("CARGO_PKG_VERSION"));

/// Progress from a running download. Terminal messages are `Done`, `Failed`
/// and `Cancelled`; exactly one of them is sent.
#[derive(Clone, Debug)]
pub enum DownloadMsg {
    /// Resolution finished: this is the size of the job.
    Planned {
        revision: String,
        files: usize,
        total_bytes: u64,
        already_have: u64,
    },
    /// Starting a file. `index` is 1-based for display.
    File {
        index: usize,
        of: usize,
        name: String,
    },
    /// Job-wide bytes, including files completed earlier in this run.
    Progress {
        done: u64,
        total: u64,
    },
    Done {
        snapshot: PathBuf,
        revision: String,
    },
    Failed(DownloadError),
    Cancelled {
        completed: usize,
        of: usize,
    },
}

/// Why a download stopped. An enum rather than a string so the UI can say
/// something specific, and so a gated repo never reads as a network fault.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadError {
    Offline(String),
    /// 401 without a token, 403 with one that lacks entitlement.
    Gated {
        repo: String,
        had_token: bool,
    },
    NotFound {
        repo: String,
    },
    RateLimited,
    DiskFull,
    /// Needs more space than the filesystem has; refused before any bytes move.
    NotEnoughSpace {
        need: u64,
        free: u64,
    },
    /// The repo publishes nothing Atlas can load.
    NoSafetensors {
        repo: String,
    },
    Http {
        repo: String,
        status: u16,
    },
    Io(String),
}

impl DownloadError {
    /// What the reader should do about it. Same idea as `error_hints`: a
    /// diagnostic without a fix is half a diagnostic.
    pub fn hint(&self) -> String {
        match self {
            Self::Offline(e) => format!("no route to huggingface.co ({e}) — check the network"),
            Self::Gated {
                repo,
                had_token: false,
            } => format!(
                // `hf auth login`, not `huggingface-cli login`: the old binary
                // was renamed in huggingface_hub 0.34 and removed in 1.0, so
                // the advice a gated repo hands out has to be the current one.
                "{repo} requires credentials — set HF_TOKEN, or run `hf auth login` \
                 (`huggingface-cli login` before huggingface_hub 1.0)"
            ),
            Self::Gated {
                repo,
                had_token: true,
            } => format!(
                "this account has not accepted the licence for {repo} — visit https://huggingface.co/{repo}"
            ),
            Self::NotFound { repo } => format!("no such model on the Hub: {repo}"),
            Self::RateLimited => "the Hub is rate-limiting this IP — wait a few minutes".into(),
            Self::DiskFull => "the disk filled up mid-download — completed files were kept".into(),
            Self::NotEnoughSpace { need, free } => format!(
                "needs {:.1} GB, {:.1} GB free",
                *need as f64 / 1e9,
                *free as f64 / 1e9
            ),
            Self::NoSafetensors { repo } => {
                format!("{repo} publishes no safetensors — Atlas cannot load it")
            }
            Self::Http { repo, status } => format!("the Hub answered {status} for {repo}"),
            Self::Io(e) => format!("could not write to the cache: {e}"),
        }
    }
}

/// A running download, from the caller's side.
pub struct Handle {
    pub repo: String,
    pub rx: Receiver<DownloadMsg>,
    cancel: Arc<AtomicBool>,
}

impl Handle {
    /// Ask the worker to stop. Takes effect within a megabyte — the flag is
    /// checked once per chunk, not once per file — and whatever has been
    /// written stays as resume credit.
    pub fn cancel(&self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Start downloading `repo` into `cache_root`.
///
/// Returns immediately; progress arrives on the handle's channel.
pub fn start(repo: &str, cache_root: PathBuf) -> Handle {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let owned = repo.to_string();
    let spawned = std::thread::Builder::new()
        .name("atlas-download".into())
        .spawn({
            let tx = tx.clone();
            let cancel = Arc::clone(&cancel);
            let repo = owned.clone();
            move || {
                if let Err(e) = run(&repo, &cache_root, &tx, &cancel) {
                    let _ = tx.send(DownloadMsg::Failed(e));
                }
            }
        });
    if let Err(e) = spawned {
        let _ = tx.send(DownloadMsg::Failed(DownloadError::Io(format!(
            "could not start the download thread: {e}"
        ))));
    }
    Handle {
        repo: repo.to_string(),
        rx,
        cancel,
    }
}

fn run(
    repo: &str,
    cache_root: &std::path::Path,
    tx: &Sender<DownloadMsg>,
    cancel: &AtomicBool,
) -> Result<(), DownloadError> {
    // Resolved once, for the whole job.
    let tok = hf::token();
    let (revision, listing) = hf::repo_info(repo, tok.as_deref())?;

    // Sizes come from a second endpoint; failing to get them is not fatal.
    let sizes = hf::sizes(repo, &revision, tok.as_deref());
    let with_sizes: Vec<plan::RemoteFile> = listing
        .into_iter()
        .map(|mut f| {
            f.size = sizes.iter().find(|(p, _)| *p == f.name).map(|(_, s)| *s);
            f
        })
        .collect();

    let files = plan::select(&with_sizes);
    if !plan::has_weights(&files) {
        return Err(DownloadError::NoSafetensors { repo: repo.into() });
    }
    let total = plan::total_bytes(&files);

    let snapshot = hf::repo_dir(cache_root, repo)
        .join("snapshots")
        .join(&revision);
    // Anything already complete from an earlier run counts immediately, so the
    // bar starts where the last attempt stopped rather than at zero.
    let already: u64 = files
        .iter()
        .filter_map(|f| {
            std::fs::metadata(snapshot.join(&f.name))
                .ok()
                .map(|m| m.len())
        })
        .sum();

    // Refuse before the first byte rather than filling the disk and failing
    // two hours in. The syscall and the decision are separate so the decision
    // can be tested without a full disk to hand.
    if let Some(free) = hf::free_bytes(cache_root) {
        fits(total, already, free)?;
    }

    let _ = tx.send(DownloadMsg::Planned {
        revision: revision.clone(),
        files: files.len(),
        total_bytes: total,
        already_have: already,
    });

    let mut done_bytes = 0u64;
    for (i, f) in files.iter().enumerate() {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tx.send(DownloadMsg::Cancelled {
                completed: i,
                of: files.len(),
            });
            return Ok(());
        }
        let dest = snapshot.join(&f.name);
        if let Ok(m) = std::fs::metadata(&dest) {
            // Already here from a previous run.
            done_bytes += m.len();
            let _ = tx.send(DownloadMsg::Progress {
                done: done_bytes,
                total,
            });
            continue;
        }
        let _ = tx.send(DownloadMsg::File {
            index: i + 1,
            of: files.len(),
            name: f.name.clone(),
        });
        let base = done_bytes;
        let mut last_sent = 0u64;
        let completed = hf::fetch_file(
            repo,
            &revision,
            &f.name,
            &dest,
            tok.as_deref(),
            cancel,
            &mut |n| {
                // One message per megabyte of movement, not per read: the UI ticks
                // at 10 Hz and a channel full of byte counts is just work.
                if n.saturating_sub(last_sent) >= 1024 * 1024 {
                    last_sent = n;
                    let _ = tx.send(DownloadMsg::Progress {
                        done: base + n,
                        total,
                    });
                }
            },
        )?;
        if !completed {
            let _ = tx.send(DownloadMsg::Cancelled {
                completed: i,
                of: files.len(),
            });
            return Ok(());
        }
        done_bytes = base + std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
        let _ = tx.send(DownloadMsg::Progress {
            done: done_bytes,
            total,
        });
    }

    // Only now is the model real. See the module doc.
    hf::publish(cache_root, repo, &revision).map_err(|e| DownloadError::Io(format!("{e:#}")))?;
    let _ = tx.send(DownloadMsg::Done { snapshot, revision });
    Ok(())
}

/// Is there room for what is left to fetch?
///
/// Pure, so the interesting cases — exactly enough, one byte short, a resumed
/// download that now fits — are testable without arranging a full filesystem.
fn fits(total: u64, already: u64, free: u64) -> Result<(), DownloadError> {
    let need = total.saturating_sub(already);
    if need > free {
        return Err(DownloadError::NotEnoughSpace { need, free });
    }
    Ok(())
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "net_tests.rs"]
mod net_tests;
