// SPDX-License-Identifier: AGPL-3.0-only

//! The dashboard's view of a model download.
//!
//! Held on `App` beside the Library state. One job at a time, deliberately:
//! two concurrent multi-gigabyte pulls saturate the same NVMe and halve each
//! other, the row has one line of progress space, and both would be writing
//! into the same cache directory. A second request is refused with a message
//! naming the running one rather than queued.
//!
//! Threading is the dashboard's standing rule — the render thread only
//! `try_recv`s; the work is on `model_download`'s own thread.

use std::collections::HashMap;
use std::time::Instant;

use crate::model_download::stale::Freshness;
use crate::model_download::{DownloadError, DownloadMsg, Handle};

/// What a running job looks like to the renderer.
pub struct Job {
    pub repo: String,
    handle: Handle,
    /// Set once the user has asked to stop. The UI must say so: cancellation
    /// is honoured within a megabyte, but "stopping…" is still more honest
    /// than a bar that keeps moving.
    pub cancelling: bool,
    pub file: Option<(usize, usize, String)>,
    pub done: u64,
    pub total: u64,
    pub started: Instant,
    /// (when, bytes) of the last rate sample.
    last: (Instant, u64),
    pub rate_bps: f64,
    /// Bytes already on disk when the plan landed.
    ///
    /// ★ This is what tells "we fetched an 18 GB model" apart from "every file
    /// was already here and we fetched nothing". Both used to end on the same
    /// `{repo} downloaded` toast, and for an already-complete model the whole
    /// job — list, stat, publish — finishes inside one frame, so the user saw
    /// a toast flash with no bar and read it as a broken download. Reported
    /// exactly that way: "When i pressed d, a toast flashed, but no progress
    /// bar."
    pub already_have: u64,
}

impl Job {
    /// Fraction complete, when the total is known.
    ///
    /// `None` when the Hub did not report sizes — the UI then shows bytes
    /// moved without a percentage rather than a bar stuck at zero.
    pub fn fraction(&self) -> Option<f64> {
        (self.total > 0).then(|| (self.done as f64 / self.total as f64).clamp(0.0, 1.0))
    }
}

#[derive(Default)]
pub struct DownloadState {
    pub job: Option<Job>,
    /// Freshness by model id, for this session only.
    ///
    /// Deliberately not persisted: a stale-check that lies after a restart is
    /// worse than one that says "unknown".
    pub freshness: HashMap<String, Freshness>,
    /// The model whose freshness is being checked, for the skeleton.
    pub checking: Option<String>,
    pending_check: Option<std::sync::mpsc::Receiver<(String, Freshness)>>,
    /// The last terminal outcome, for a toast.
    pub last_message: Option<(String, bool)>,
}

/// What `pump` observed, so the caller can react without re-deriving it.
pub enum Settled {
    /// A download finished; the cache changed and wants re-scanning.
    Finished(String),
    /// Stopped without completing. The cache changed too — a partial file
    /// alters the reported size — but nothing new is loadable.
    Stopped(String),
}

/// What a finished job says, as ONE function.
///
/// "Nothing needed fetching" and "we moved 18 GB" are different outcomes and
/// must read differently: saying "downloaded" for the first describes work that
/// did not happen, and for an already-complete model the whole job finishes
/// inside a single frame, so that toast was the ONLY thing the user saw — no
/// bar, because no bytes moved. Reported as "a toast flashed, but no progress
/// bar". `u` is named in the already-complete case because it invites the
/// obvious next question (is it the LATEST?), which is a different command.
///
/// Shared with the test seam deliberately: a second copy of this in the tests
/// would assert the copy rather than the code.
fn done_message(job: &Job) -> (String, bool) {
    if job.total > 0 && job.already_have >= job.total {
        return (
            format!(
                "{} is already complete — {} on disk, nothing to download (u checks for updates)",
                job.repo,
                crate::tui::format::bytes(job.total)
            ),
            false,
        );
    }
    // The receipt: what moved and how long it took. A bare "downloaded" leaves
    // out the one number that proves the transfer was real.
    let moved = job.total.saturating_sub(job.already_have);
    (
        format!(
            "{} downloaded — {} in {}",
            job.repo,
            crate::tui::format::bytes(moved),
            fmt_elapsed(job.started.elapsed().as_secs())
        ),
        false,
    )
}

/// `58s`, `12m 40s`, `1h 03m` — a transfer duration for a receipt.
///
/// Not in `format` because this is its only caller; move it there the moment a
/// second one appears rather than guessing now.
fn fmt_elapsed(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s}s"),
        s if s < 3_600 => format!("{}m {:02}s", s / 60, s % 60),
        s => format!("{}h {:02}m", s / 3_600, s / 60 % 60),
    }
}

impl DownloadState {
    /// Run exactly `pump`'s `Done` bookkeeping, for tests.
    ///
    /// A test cannot hand the worker a `Done` without a live thread, and
    /// re-implementing the message in the test would assert the copy rather
    /// than the code. This shares the one path.
    #[cfg(test)]
    pub(crate) fn settle_done_for_test(&mut self) {
        if let Some(job) = self.job.as_ref() {
            self.last_message = Some(done_message(job));
        }
        self.job = None;
    }

    /// Is anything running for this model?
    pub fn is_downloading(&self, repo: &str) -> bool {
        self.job.as_ref().is_some_and(|j| j.repo == repo)
    }

    /// Begin a download. Returns the message to show.
    pub fn start(&mut self, repo: &str, cache_root: std::path::PathBuf) -> (String, bool) {
        if let Some(j) = &self.job {
            return (
                format!("already downloading {} — press x to stop it", j.repo),
                true,
            );
        }
        let handle = crate::model_download::start(repo, cache_root);
        self.job = Some(Job {
            repo: repo.to_string(),
            handle,
            cancelling: false,
            file: None,
            done: 0,
            total: 0,
            started: Instant::now(),
            last: (Instant::now(), 0),
            rate_bps: 0.0,
            already_have: 0,
        });
        (format!("resolving {repo}…"), false)
    }

    /// Ask the running job to stop; a second call abandons it outright.
    pub fn cancel(&mut self) -> Option<(String, bool)> {
        let job = self.job.as_mut()?;
        if job.cancelling {
            // Second press: stop listening. The worker sees the flag and exits
            // after its current chunk; the bytes it wrote stay as resume
            // credit. `Disconnected` is already this codebase's idiom for "the
            // producer is gone", so dropping the handle needs no new handling.
            let repo = job.repo.clone();
            self.job = None;
            return Some((format!("abandoned the {repo} download"), false));
        }
        job.cancelling = true;
        job.handle.cancel();
        Some(("stopping — finishing the current chunk".into(), false))
    }

    /// Drain progress. Called once per tick.
    pub fn pump(&mut self) -> Option<Settled> {
        let mut settled = None;
        if let Some(job) = self.job.as_mut() {
            // Drained one message at a time, so a disconnect can only be seen
            // AFTER everything the worker managed to send. The first version
            // used `try_iter` plus a second `try_recv` as a liveness probe —
            // and a terminal message sent between the two was consumed by the
            // probe and dropped, reporting a finished download as "stopped
            // unexpectedly".
            let mut msgs = Vec::new();
            let mut disconnected = false;
            loop {
                match job.handle.rx.try_recv() {
                    Ok(m) => msgs.push(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
            for msg in msgs {
                match msg {
                    DownloadMsg::Planned {
                        total_bytes,
                        already_have,
                        ..
                    } => {
                        job.total = total_bytes;
                        job.done = already_have;
                        job.already_have = already_have;
                    }
                    DownloadMsg::File { index, of, name } => job.file = Some((index, of, name)),
                    DownloadMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                        // Rate over a moving window rather than since the
                        // start, so a stall is visible instead of averaged
                        // away.
                        let dt = job.last.0.elapsed().as_secs_f64();
                        if dt >= 0.5 {
                            job.rate_bps = (done.saturating_sub(job.last.1)) as f64 / dt;
                            job.last = (Instant::now(), done);
                        }
                    }
                    DownloadMsg::Done { .. } => {
                        settled = Some(Settled::Finished(job.repo.clone()));
                        self.last_message = Some(done_message(job));
                    }
                    DownloadMsg::Cancelled { completed, of } => {
                        settled = Some(Settled::Stopped(job.repo.clone()));
                        self.last_message = Some((
                            format!(
                                "{} stopped after {completed}/{of} files — press d to resume",
                                job.repo
                            ),
                            false,
                        ));
                    }
                    DownloadMsg::Failed(e) => {
                        settled = Some(Settled::Stopped(job.repo.clone()));
                        self.last_message = Some((describe(&job.repo, &e), true));
                    }
                }
            }
            // A worker that died WITHOUT a terminal message — a panic, since
            // `run`'s Err path sends `Failed` — must still settle the job.
            // Left tracked, `is_downloading` stays true forever and every
            // later `d` is refused with "already downloading", which reads as
            // downloads being broken outright.
            if settled.is_none() && disconnected {
                settled = Some(Settled::Stopped(job.repo.clone()));
                self.last_message = Some((
                    format!(
                        "{} download stopped unexpectedly — press d to resume",
                        job.repo
                    ),
                    true,
                ));
            }
        }
        if settled.is_some() {
            self.job = None;
        }
        // A freshness answer, if one arrived.
        if let Some(rx) = &self.pending_check {
            match rx.try_recv() {
                Ok((id, f)) => {
                    self.pending_check = None;
                    self.checking = None;
                    // The answer says something in EVERY case, not only the
                    // stale one. The badge renders only for a confirmed
                    // mismatch, so for a current model — or an unreachable
                    // Hub — the skeleton simply vanished and nothing else
                    // changed: indistinguishable from the feature being
                    // broken. Same shape as the completed download that
                    // looked like a failed one; same fix (`done_message`).
                    self.last_message = Some(checked_message(&id, &f));
                    self.freshness.insert(id, f);
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.pending_check = None;
                    self.checking = None;
                    // A worker that died without answering must still settle
                    // out loud — the download pump learned this first.
                    self.last_message = Some((
                        "the freshness check ended unexpectedly — u retries".into(),
                        true,
                    ));
                }
            }
        }
        settled
    }

    /// Start a freshness check for one model, off the render thread.
    ///
    /// One at a time and only on request: the Hub throttles, and a list of
    /// forty models is forty requests nobody asked for.
    pub fn check(&mut self, repo: &str, cache_root: std::path::PathBuf) -> (String, bool) {
        if self.pending_check.is_some() {
            return ("already checking a model".into(), true);
        }
        let owned = repo.to_string();
        let fallback = repo.to_string();
        let rx = crate::tui::worker::spawn(
            "atlas-freshness",
            move || {
                // An unreachable Hub is `Unknown`, which draws no badge —
                // never a "stale" claim the network could not support.
                let f = crate::model_download::stale::check(&owned, &cache_root)
                    .unwrap_or(Freshness::Unknown);
                (owned, f)
            },
            |_| (fallback, Freshness::Unknown),
        );
        self.pending_check = Some(rx);
        self.checking = Some(repo.to_string());
        (format!("checking {repo}…"), false)
    }
}

/// A failure, worded so the reader can act on it.
fn describe(repo: &str, e: &DownloadError) -> String {
    format!("{repo}: {}", e.hint())
}

/// What a finished freshness check says, as ONE function — the same rule as
/// [`done_message`], and for the same reason: a second copy in the tests
/// would assert the copy rather than the code. Words carry the outcome
/// (toasts are already glyph-marked), so NO_COLOR changes nothing.
fn checked_message(id: &str, f: &Freshness) -> (String, bool) {
    match f {
        Freshness::Current => (format!("{id} is up to date with the Hub"), false),
        Freshness::Stale { local, remote } => (
            format!(
                "{id} has an update — local {} vs Hub {}; d downloads it",
                crate::model_download::stale::short(local),
                crate::model_download::stale::short(remote)
            ),
            false,
        ),
        Freshness::Missing => (
            format!("{id} has nothing on disk to compare — d downloads it"),
            false,
        ),
        // Error tone: the question was not answered. Never a "stale" claim
        // the network could not support — the badge rule, said out loud.
        Freshness::Unknown => (
            format!("could not reach the Hub — {id} freshness unknown"),
            true,
        ),
    }
}

#[cfg(test)]
#[path = "download_state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "download_state_more_tests.rs"]
mod more_tests;
