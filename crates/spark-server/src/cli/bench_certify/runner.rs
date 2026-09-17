// SPDX-License-Identifier: AGPL-3.0-only

//! Running one unit and reading what it produced.
//!
//! A gate runs as a CHILD `spark benchmark run … --pull-request-gate`: the
//! self-start machinery serves one model per process and its shutdown latch is
//! one-way, so an in-process loop cannot serve gate two. The child is exactly
//! what an operator runs by hand, which is also the point.
//!
//! The evidence is the RECORD the child wrote — the same file the coverage
//! check will judge — never its exit code and never a printed line. An `Info`
//! verdict exits 0 and is not a pass; a shard's record is `Info` by design
//! (the group owns the verdict) and is a completed member only when it carries
//! its shard identity and tallies.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::plan::Unit;

/// What the campaign hands the runner for every unit.
pub struct RunCtx<'a> {
    pub root: &'a Path,
    pub anchor: &'a str,
    pub hardware: &'a str,
    pub yes: bool,
    pub deadline: Duration,
    pub log_dir: &'a Path,
}

/// The facts about a record that decide a unit's outcome. Read by a
/// [`Records`] so the classification is testable without a real record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecordFacts {
    pub path: PathBuf,
    pub git_sha: String,
    pub verdict_passes: bool,
    pub frame_completed: bool,
    /// `shard.index` present and per-subset tallies present.
    pub is_shard_with_tallies: bool,
}

/// Where a unit's newest record is read from.
pub trait Records: Send {
    /// The newest record for `id` and this `shard` recorded at or after
    /// `since` (unix secs). The shard identity is matched, never assumed:
    /// two shards of one group at one commit land in one directory.
    fn newest_since(
        &self,
        root: &Path,
        id: &str,
        shard: Option<(usize, usize)>,
        since: u64,
    ) -> Option<RecordFacts>;
}

/// The real `.benchmarks/<id>/` reader.
pub struct RepoRecords;

impl Records for RepoRecords {
    fn newest_since(
        &self,
        root: &Path,
        id: &str,
        shard: Option<(usize, usize)>,
        since: u64,
    ) -> Option<RecordFacts> {
        use avarok_plugin::gate;
        gate::records_newest_first(root, id)
            .into_iter()
            .filter_map(|path| gate::read_record(&path).ok().map(|r| (path, r)))
            .find(|(_, r)| r.benchmark_id == id && r.shard() == shard && r.recorded_at >= since)
            .map(|(path, r)| facts_of(path, &r))
    }
}

/// The facts the classifier reads, from a parsed record.
pub fn facts_of(path: PathBuf, r: &avarok_plugin::gate::GateRecord) -> RecordFacts {
    let tallies =
        avarok_plugin::benchmarks::bfcl::aggregate::tallies_from_metrics(&r.metrics).is_some();
    RecordFacts {
        is_shard_with_tallies: r.shard().is_some() && tallies,
        verdict_passes: r.verdict_passes(),
        frame_completed: !r.frame_status_failed(),
        git_sha: r.git_sha.clone(),
        path,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    /// A passing record at the anchor.
    Passed {
        record: PathBuf,
    },
    /// A shard's record at the anchor; the group's verdict comes later.
    MemberDone {
        record: PathBuf,
    },
    /// The run completed and said no (or said nothing, which is not yes).
    VerdictFail {
        record: Option<PathBuf>,
        reason: String,
    },
    /// The harness failed: no record, or a record for the wrong commit.
    Harness {
        reason: String,
        retryable: bool,
    },
    TimedOut,
    Cancelled,
}

/// Classify a finished child. `exit` is `None` when it was killed.
pub fn classify(
    unit: &Unit,
    anchor: &str,
    exit: Option<i32>,
    record: Option<RecordFacts>,
    killed_for: Option<RunOutcome>,
) -> RunOutcome {
    if let Some(k) = killed_for {
        return k;
    }
    let Some(r) = record else {
        return RunOutcome::Harness {
            reason: format!(
                "the child exited {} and wrote no record for {} at this commit",
                exit.map_or("by signal".to_string(), |c| c.to_string()),
                unit.label()
            ),
            retryable: exit != Some(0),
        };
    };
    if !(r.git_sha.starts_with(anchor) || anchor.starts_with(&r.git_sha)) {
        return RunOutcome::Harness {
            reason: format!(
                "{} names commit {}, not the anchor {anchor}: the tree moved under the run",
                r.path.display(),
                r.git_sha
            ),
            retryable: false,
        };
    }
    if !r.frame_completed {
        return RunOutcome::VerdictFail {
            record: Some(r.path),
            reason: "the run itself failed (frame status Failed)".into(),
        };
    }
    if r.verdict_passes {
        return RunOutcome::Passed { record: r.path };
    }
    if unit.shard.is_some() && r.is_shard_with_tallies {
        return RunOutcome::MemberDone { record: r.path };
    }
    RunOutcome::VerdictFail {
        record: Some(r.path),
        reason: match exit {
            Some(2) => "the gate said no (exit 2)".into(),
            Some(0) => "the run recorded an Info verdict, which is not a pass".into(),
            other => format!("verdict is not PASS (exit {other:?})"),
        },
    }
}

pub trait GateRunner {
    /// Run one unit; `on_line` receives the child's stderr as it arrives.
    fn run(&mut self, unit: &Unit, ctx: &RunCtx, on_line: &mut dyn FnMut(&str)) -> RunOutcome;
}

/// Spawns `<exe> benchmark run <id> --pull-request-gate …` and reads the
/// record it leaves behind.
pub struct LocalChild {
    pub exe: PathBuf,
    pub records: Box<dyn Records>,
    /// Set by a signal handler; checked between reads.
    pub cancel: Arc<AtomicBool>,
    /// Extra arguments after the standard ones (tests point the child at a
    /// script and pass nothing).
    pub extra_args: Vec<String>,
}

impl LocalChild {
    /// The extra arguments a local child gets when consecutive units may
    /// share a server: the child verifies the leased one is what it would
    /// have started, replaces it otherwise, and leaves it up for the next
    /// unit; the lease names THIS driver so a dead campaign's server is
    /// reclaimed by the next (`bench_lease`). Empty under `--no-serve-reuse`.
    pub fn reuse_args(no_serve_reuse: bool) -> Vec<String> {
        if no_serve_reuse {
            vec![]
        } else {
            vec![
                "--serve-reuse".to_string(),
                "--serve-lease-owner".to_string(),
                std::process::id().to_string(),
            ]
        }
    }

    pub fn argv(&self, unit: &Unit, ctx: &RunCtx) -> Vec<String> {
        let mut v = vec![
            "benchmark".to_string(),
            "run".to_string(),
            unit.id.to_string(),
            "--pull-request-gate".to_string(),
            "--hardware".to_string(),
            ctx.hardware.to_string(),
        ];
        if ctx.yes {
            v.push("--yes".into());
        }
        if let Some(p) = unit.shard_param() {
            v.push("--param".into());
            v.push(p);
        }
        v.extend(self.extra_args.iter().cloned());
        v
    }
}

/// How long to wait, after the child exits, for its output readers to reach
/// EOF before the log is closed. A child's own output is a few hundred lines
/// and arrives at once; the bound exists for a lingering grandchild (a
/// self-started server) that keeps the pipes open.
const READER_DRAIN: Duration = Duration::from_secs(5);

/// Wait for the child while streaming its stderr, enforcing the deadline and
/// the cancel flag. Returns the exit code, or the reason it was killed.
fn supervise(
    mut child: std::process::Child,
    deadline: Duration,
    cancel: &AtomicBool,
    log: &mut std::fs::File,
    on_line: &mut dyn FnMut(&str),
) -> (Option<i32>, Option<RunOutcome>) {
    use std::io::{BufRead, BufReader, Write};
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let stderr = child.stderr.take().expect("stderr is piped");
    let stdout = child.stdout.take().expect("stdout is piped");
    let tx2 = tx.clone();
    let readers = [
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if tx.send(line).is_err() {
                    break;
                }
            }
        }),
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if tx2.send(format!("stdout: {line}")).is_err() {
                    break;
                }
            }
        }),
    ];
    let started = Instant::now();
    let mut killed: Option<RunOutcome> = None;
    let mut kill_at: Option<Instant> = None;
    loop {
        while let Ok(line) = rx.try_recv() {
            let _ = writeln!(log, "{line}");
            on_line(&line);
        }
        if let Ok(Some(status)) = child.try_wait() {
            // Drain what arrived after exit: the readers finish once the
            // pipes reach EOF, so wait for THEM rather than for a quiet gap —
            // a gap is what a loaded box produces while a reader is merely
            // unscheduled, and a line lost that way made a verdict line
            // vanish from the log. Bounded, because a server the child
            // started and left behind holds the pipes open.
            let drain_until = Instant::now() + READER_DRAIN;
            while readers.iter().any(|r| !r.is_finished()) && Instant::now() < drain_until {
                std::thread::sleep(Duration::from_millis(20));
            }
            while let Ok(line) = rx.try_recv() {
                let _ = writeln!(log, "{line}");
                on_line(&line);
            }
            return (status.code(), killed);
        }
        if killed.is_none() {
            if cancel.load(Ordering::SeqCst) {
                killed = Some(RunOutcome::Cancelled);
            } else if started.elapsed() > deadline {
                killed = Some(RunOutcome::TimedOut);
            }
            if killed.is_some() {
                terminate(&mut child);
                kill_at = Some(Instant::now() + Duration::from_secs(30));
            }
        } else if kill_at.is_some_and(|t| Instant::now() > t) {
            let _ = child.kill();
            kill_at = None;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// SIGTERM the child's whole process group (it may have started a server),
/// falling back to the child alone.
#[cfg(unix)]
fn terminate(child: &mut std::process::Child) {
    let pid = child.id();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .status();
    let _ = std::process::Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
}

/// No process groups here: the child alone, and at once — there is no
/// graceful signal to send first. Certification runs on Linux boxes; this
/// arm exists so the binary still builds where `spark serve` does.
#[cfg(not(unix))]
fn terminate(child: &mut std::process::Child) {
    let _ = child.kill();
}

/// Put the child in its own process group so `terminate` can reach the
/// server it starts. A no-op where process groups do not exist.
fn in_own_group(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0)
    }
    #[cfg(not(unix))]
    {
        cmd
    }
}

impl GateRunner for LocalChild {
    fn run(&mut self, unit: &Unit, ctx: &RunCtx, on_line: &mut dyn FnMut(&str)) -> RunOutcome {
        let since = super::lockfile::now_unix();
        let _ = std::fs::create_dir_all(ctx.log_dir);
        let log_path = ctx.log_dir.join(format!("{}.log", unit.file_stem()));
        let mut log = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            Ok(f) => f,
            Err(e) => {
                return RunOutcome::Harness {
                    reason: format!("cannot open {}: {e}", log_path.display()),
                    retryable: false,
                };
            }
        };
        let mut cmd = std::process::Command::new(&self.exe);
        cmd.args(self.argv(unit, ctx))
            .current_dir(ctx.root)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let child = in_own_group(&mut cmd).spawn();
        let child = match child {
            Ok(c) => c,
            Err(e) => {
                return RunOutcome::Harness {
                    reason: format!("cannot spawn {}: {e}", self.exe.display()),
                    retryable: false,
                };
            }
        };
        let (exit, killed) = supervise(child, ctx.deadline, &self.cancel, &mut log, on_line);
        let record = self
            .records
            .newest_since(ctx.root, unit.id, unit.shard, since);
        classify(unit, ctx.anchor, exit, record, killed)
    }
}

#[cfg(all(test, unix))]
#[path = "runner_tests.rs"]
mod runner_tests;
