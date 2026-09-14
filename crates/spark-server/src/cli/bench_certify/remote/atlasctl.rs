// SPDX-License-Identifier: AGPL-3.0-only
//! `atlasctl bench … --json`, as a trait, and the subprocess that speaks it.
//!
//! The wire contract is atlasctl's (atlas-recipes `docs/BENCH.md`): every
//! subcommand writes exactly one JSON document to stdout (one event per line
//! for `attach`), errors are an `ErrorObj` on stdout, and the exit code says
//! which class of failure it was. Nothing here parses prose. The shapes are
//! in [`wire`](super::wire) and re-exported here.

use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, bail};

pub use super::wire::{
    AttachEnd, ErrorObj, Exit, FetchedFile, NodeInfo, NodeRow, StreamEvent, SubmitSpec, Submitted,
};

/// What atlasctl answered when it did not do the thing: the exit class and
/// the error document, if it wrote one.
pub type Refusal = (Exit, Option<ErrorObj>);

/// The five verbs the driver needs.
pub trait Atlasctl: Send + Sync {
    fn nodes(&self, addrs: &[String]) -> Result<Vec<NodeRow>>;
    fn submit(&self, node: &str, spec: &SubmitSpec) -> Result<Result<Submitted, Refusal>>;
    /// Stream events from `from_seq` until the job ends or the link is lost
    /// for longer than `reconnect_for`. `cancel` set → the child is killed and
    /// `Cancelled` returned (the remote job is NOT cancelled here).
    fn attach(
        &self,
        node: &str,
        job: &str,
        from_seq: u64,
        reconnect_for: Duration,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(&StreamEvent),
    ) -> Result<AttachEnd>;
    fn cancel(&self, node: &str, job: &str) -> Result<()>;
    fn fetch(
        &self,
        node: &str,
        job: &str,
        out_dir: &Path,
    ) -> Result<Result<Vec<FetchedFile>, Refusal>>;
}

/// The real thing: `atlasctl` on PATH or as named.
pub struct SubprocessAtlasctl {
    pub exe: PathBuf,
}

impl SubprocessAtlasctl {
    /// Find the binary. Absent → an error naming `--atlasctl`.
    pub fn locate(explicit: Option<&Path>) -> Result<Self> {
        let exe = match explicit {
            Some(p) => p.to_path_buf(),
            None => PathBuf::from("atlasctl"),
        };
        let probe = Command::new(&exe)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        match probe {
            Ok(o) if o.status.success() => Ok(Self { exe }),
            Ok(o) => bail!(
                "{} --version failed: {}",
                exe.display(),
                String::from_utf8_lossy(&o.stderr).trim()
            ),
            Err(e) => bail!(
                "cannot run {} ({e}); install atlasctl or pass --atlasctl PATH",
                exe.display()
            ),
        }
    }

    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(&self.exe);
        c.arg("bench").args(args).arg("--json").stdin(Stdio::null());
        c
    }

    /// Run to completion; the last stdout line is the document.
    fn one_shot(&self, args: &[&str]) -> Result<(Exit, String)> {
        let out = self
            .cmd(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .with_context(|| format!("running {} bench {}", self.exe.display(), args.join(" ")))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let last = stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .to_owned();
        Ok((Exit::from_code(out.status.code()), last))
    }
}

fn error_of(line: &str) -> Option<ErrorObj> {
    serde_json::from_str(line).ok()
}

impl Atlasctl for SubprocessAtlasctl {
    fn nodes(&self, addrs: &[String]) -> Result<Vec<NodeRow>> {
        let joined = addrs.join(",");
        let out = self
            .cmd(&["nodes", &joined])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("running atlasctl bench nodes")?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        // Rows are one array on one line; any exit code still carries them
        // (a partial failure exits non-zero with every row present).
        let rows_line = stdout
            .lines()
            .find(|l| l.trim_start().starts_with('['))
            .with_context(|| {
                format!(
                    "atlasctl bench nodes wrote no rows (exit {:?}): {}",
                    out.status.code(),
                    stdout.trim()
                )
            })?;
        serde_json::from_str(rows_line).context("parsing atlasctl bench nodes output")
    }

    fn submit(&self, node: &str, spec: &SubmitSpec) -> Result<Result<Submitted, Refusal>> {
        let max_run = spec.max_run_s.map(|s| s.to_string());
        let mut args = vec![
            "submit",
            node,
            "--sha",
            &spec.sha,
            "--gate",
            &spec.gate,
            "--hardware",
            &spec.hardware,
            "--job-key",
            &spec.job_key,
            "--note",
            &spec.note,
        ];
        if let Some(m) = &max_run {
            args.push("--max-run-s");
            args.push(m);
        }
        let (exit, line) = self.one_shot(&args)?;
        if exit == Exit::Done {
            let s: Submitted = serde_json::from_str(&line)
                .with_context(|| format!("parsing submit reply {line:?}"))?;
            return Ok(Ok(s));
        }
        Ok(Err((exit, error_of(&line))))
    }

    fn attach(
        &self,
        node: &str,
        job: &str,
        from_seq: u64,
        reconnect_for: Duration,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(&StreamEvent),
    ) -> Result<AttachEnd> {
        let from = from_seq.to_string();
        let reconnect = reconnect_for.as_secs().to_string();
        let mut child = self
            .cmd(&[
                "attach",
                node,
                job,
                "--from-seq",
                &from,
                "--reconnect-for",
                &reconnect,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("spawning atlasctl bench attach")?;
        let stdout = child.stdout.take().context("no stdout")?;
        // Reader thread: the main thread watches the cancel flag, so a cancel
        // interrupts a blocked read by killing the child.
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout)
                .lines()
                .map_while(Result::ok)
            {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let mut last_seq = from_seq.saturating_sub(1);
        let mut last_line = String::new();
        let mut done: Option<StreamEvent> = None;
        let killed = Arc::new(AtomicBool::new(false));
        loop {
            if cancel.load(Ordering::SeqCst) {
                let _ = child.kill();
                killed.store(true, Ordering::SeqCst);
                break;
            }
            match rx.recv_timeout(Duration::from_millis(250)) {
                Ok(line) => {
                    last_line = line.clone();
                    if let Ok(ev) = serde_json::from_str::<StreamEvent>(&line) {
                        last_seq = ev.seq.max(last_seq);
                        if ev.kind == "done" {
                            done = Some(ev.clone());
                        }
                        on_event(&ev);
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let status = child.wait().context("waiting for atlasctl bench attach")?;
        let _ = reader.join();
        if killed.load(Ordering::SeqCst) {
            return Ok(AttachEnd::Cancelled);
        }
        let exit = Exit::from_code(status.code());
        Ok(match exit {
            Exit::Done => AttachEnd::Passed {
                exit_code: done.as_ref().and_then(|d| d.exit_code),
            },
            Exit::JobFailed => AttachEnd::JobFailed {
                outcome: done.as_ref().and_then(|d| d.outcome.clone()),
                exit_code: done.as_ref().and_then(|d| d.exit_code),
                detail: error_of(&last_line).map_or_else(
                    || {
                        done.as_ref()
                            .and_then(|d| d.reason.clone())
                            .unwrap_or_else(|| "the job did not pass".into())
                    },
                    |e| e.message,
                ),
            },
            Exit::Cancelled => AttachEnd::Cancelled,
            Exit::StreamLost => AttachEnd::StreamLost { last_seq },
            other => AttachEnd::Failed {
                exit: other,
                error: error_of(&last_line),
            },
        })
    }

    fn cancel(&self, node: &str, job: &str) -> Result<()> {
        let (exit, line) = self.one_shot(&["cancel", node, job])?;
        if exit == Exit::Done {
            return Ok(());
        }
        bail!(
            "atlasctl bench cancel {node} {job} exited {exit:?}: {}",
            error_of(&line).map_or(line.clone(), |e| e.to_string())
        )
    }

    fn fetch(
        &self,
        node: &str,
        job: &str,
        out_dir: &Path,
    ) -> Result<Result<Vec<FetchedFile>, Refusal>> {
        let out = out_dir.display().to_string();
        let (exit, line) = self.one_shot(&["fetch", node, job, "--out-dir", &out])?;
        if exit == Exit::Done {
            let files: Vec<FetchedFile> = serde_json::from_str(&line)
                .with_context(|| format!("parsing fetch reply {line:?}"))?;
            return Ok(Ok(files));
        }
        Ok(Err((exit, error_of(&line))))
    }
}

#[cfg(test)]
#[path = "atlasctl_tests.rs"]
pub(super) mod atlasctl_tests;
