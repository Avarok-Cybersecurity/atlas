// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::cli::bench_certify::plan::{Estimate, Unit};
use atlas_plugin::hardware::policy::Sensitivity;

fn plain() -> Unit {
    Unit {
        id: "decode-floor",
        group: None,
        shard: None,
        class: Sensitivity::Speed,
        estimate: Estimate::Declared(180),
        needs_confirmation: false,
        serve_allowance_s: 600,
    }
}

fn shard() -> Unit {
    Unit {
        id: "bfcl-subset",
        group: Some("bfcl-subset"),
        shard: Some((0, 4)),
        class: Sensitivity::Correctness,
        estimate: Estimate::Declared(1500),
        needs_confirmation: false,
        serve_allowance_s: 600,
    }
}

fn facts(sha: &str, passes: bool, completed: bool, shard: bool) -> RecordFacts {
    RecordFacts {
        path: PathBuf::from("/r/x.json"),
        git_sha: sha.into(),
        verdict_passes: passes,
        frame_completed: completed,
        is_shard_with_tallies: shard,
    }
}

const ANCHOR: &str = "1a0dc88a8c";

#[test]
fn exit_zero_with_a_passing_record_at_the_anchor_passes() {
    let out = classify(
        &plain(),
        ANCHOR,
        Some(0),
        Some(facts(ANCHOR, true, true, false)),
        None,
    );
    assert!(matches!(out, RunOutcome::Passed { .. }), "{out:?}");
}

/// NEGATIVE CONTROL: exit 0 is not evidence. No record, no pass.
#[test]
fn exit_zero_without_a_record_is_a_harness_failure() {
    let out = classify(&plain(), ANCHOR, Some(0), None, None);
    assert!(
        matches!(
            out,
            RunOutcome::Harness {
                retryable: false,
                ..
            }
        ),
        "{out:?}"
    );
    let out = classify(&plain(), ANCHOR, Some(1), None, None);
    assert!(
        matches!(
            out,
            RunOutcome::Harness {
                retryable: true,
                ..
            }
        ),
        "{out:?}"
    );
}

/// NEGATIVE CONTROL: a record for another commit is not this campaign's.
#[test]
fn a_record_for_another_commit_is_refused() {
    let out = classify(
        &plain(),
        ANCHOR,
        Some(0),
        Some(facts("deadbeef00", true, true, false)),
        None,
    );
    match out {
        RunOutcome::Harness { reason, retryable } => {
            assert!(!retryable);
            assert!(reason.contains("deadbeef00"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
}

/// The record's sha may be the long form of a short anchor, or vice versa.
#[test]
fn short_and_long_shas_match_either_way() {
    let long = "1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18";
    let out = classify(
        &plain(),
        ANCHOR,
        Some(0),
        Some(facts(long, true, true, false)),
        None,
    );
    assert!(matches!(out, RunOutcome::Passed { .. }));
    let out = classify(
        &plain(),
        long,
        Some(0),
        Some(facts(ANCHOR, true, true, false)),
        None,
    );
    assert!(matches!(out, RunOutcome::Passed { .. }));
}

/// NEGATIVE CONTROL: an Info verdict on a plain gate exits 0 and is NOT a pass.
#[test]
fn an_info_verdict_on_a_plain_gate_is_a_fail() {
    let out = classify(
        &plain(),
        ANCHOR,
        Some(0),
        Some(facts(ANCHOR, false, true, false)),
        None,
    );
    match out {
        RunOutcome::VerdictFail { reason, .. } => assert!(reason.contains("Info"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

/// A shard's Info record with tallies is a completed member.
#[test]
fn an_info_shard_with_tallies_is_a_member_done() {
    let out = classify(
        &shard(),
        ANCHOR,
        Some(0),
        Some(facts(ANCHOR, false, true, true)),
        None,
    );
    assert!(matches!(out, RunOutcome::MemberDone { .. }), "{out:?}");
    // …but a shard record WITHOUT tallies is not (an old binary wrote it).
    let out = classify(
        &shard(),
        ANCHOR,
        Some(0),
        Some(facts(ANCHOR, false, true, false)),
        None,
    );
    assert!(matches!(out, RunOutcome::VerdictFail { .. }), "{out:?}");
}

#[test]
fn exit_two_is_a_verdict_fail() {
    let out = classify(
        &plain(),
        ANCHOR,
        Some(2),
        Some(facts(ANCHOR, false, true, false)),
        None,
    );
    match out {
        RunOutcome::VerdictFail { reason, .. } => assert!(reason.contains("exit 2"), "{reason}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_failed_frame_is_a_verdict_fail_even_if_marked_pass() {
    let out = classify(
        &plain(),
        ANCHOR,
        Some(0),
        Some(facts(ANCHOR, true, false, false)),
        None,
    );
    assert!(matches!(out, RunOutcome::VerdictFail { .. }), "{out:?}");
}

#[test]
fn a_kill_reason_wins_over_everything() {
    let out = classify(
        &plain(),
        ANCHOR,
        None,
        Some(facts(ANCHOR, true, true, false)),
        Some(RunOutcome::TimedOut),
    );
    assert_eq!(out, RunOutcome::TimedOut);
}

/// Every local child — the local-only path and the fleet's local node alike —
/// is told to reuse the leased server and whose lease it is; `--no-serve-reuse`
/// yields nothing. One spelling, so the two paths cannot disagree (the first
/// campaign on #1089 ran the fleet path without it).
#[test]
fn the_reuse_args_name_this_driver_and_vanish_under_no_serve_reuse() {
    let me = std::process::id().to_string();
    assert_eq!(
        LocalChild::reuse_args(false),
        ["--serve-reuse", "--serve-lease-owner", me.as_str()]
    );
    assert!(LocalChild::reuse_args(true).is_empty());
}

/// The real reader matches the SHARD, not just the id: two shards of one
/// group at one commit sit in one directory, and the newest file there is
/// whichever sibling finished last. Built from a committed record so the
/// fixture is the shape the driver writes.
#[test]
fn the_repo_reader_returns_the_record_of_this_shard_not_the_newest_sibling() {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let source = atlas_plugin::gate::records_newest_first(ws, "bfcl-subset")
        .into_iter()
        .find(|p| p.extension().is_some_and(|e| e == "json"))
        .expect("a committed bfcl-subset record");
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&source).unwrap()).unwrap();
    let (dir, _) = script("exit 0");
    let root = dir.path();
    let gate_dir = root.join(".benchmarks/bfcl-subset");
    std::fs::create_dir_all(&gate_dir).unwrap();
    let since = 1_800_000_000;
    for (i, at) in [(0usize, since + 10), (1, since + 20)] {
        v["git_sha"] = serde_json::json!(ANCHOR);
        v["recorded_at"] = serde_json::json!(at);
        v["metrics"]["shard.index"] = serde_json::json!(i as f64);
        v["metrics"]["shard.count"] = serde_json::json!(2.0);
        v["metrics"]["subset.simple_python.hits"] = serde_json::json!(90.0);
        v["metrics"]["subset.simple_python.n"] = serde_json::json!(100.0);
        std::fs::write(
            gate_dir.join(format!("2027-01-01-{ANCHOR}-s{i}of2.json")),
            serde_json::to_string_pretty(&v).unwrap(),
        )
        .unwrap();
    }
    let r = RepoRecords;
    let s0 = r
        .newest_since(root, "bfcl-subset", Some((0, 2)), since)
        .unwrap();
    assert!(
        s0.path.to_string_lossy().ends_with("-s0of2.json"),
        "{}",
        s0.path.display()
    );
    assert!(s0.is_shard_with_tallies);
    let s1 = r
        .newest_since(root, "bfcl-subset", Some((1, 2)), since)
        .unwrap();
    assert!(
        s1.path.to_string_lossy().ends_with("-s1of2.json"),
        "{}",
        s1.path.display()
    );
    // NEGATIVE CONTROLS: a slice nobody ran, the whole draw, a record older
    // than the run.
    assert_eq!(
        r.newest_since(root, "bfcl-subset", Some((1, 4)), since),
        None
    );
    assert_eq!(r.newest_since(root, "bfcl-subset", None, since), None);
    assert_eq!(
        r.newest_since(root, "bfcl-subset", Some((0, 2)), since + 15),
        None
    );
}

struct Scripted(Option<RecordFacts>);
impl Records for Scripted {
    fn newest_since(
        &self,
        _: &Path,
        _: &str,
        _: Option<(usize, usize)>,
        _: u64,
    ) -> Option<RecordFacts> {
        self.0.clone()
    }
}

fn script(body: &str) -> (tempfile_dir::Dir, PathBuf) {
    let dir = tempfile_dir::Dir::new("certify-runner");
    let exe = dir.path().join("fake-spark");
    std::fs::write(&exe, format!("#!/bin/sh\n{body}\n")).unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&exe, std::fs::Permissions::from_mode(0o755)).unwrap();
    (dir, exe)
}

mod tempfile_dir {
    pub struct Dir(std::path::PathBuf);
    impl Dir {
        pub fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
        pub fn path(&self) -> &std::path::Path {
            &self.0
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn ctx<'a>(root: &'a Path, secs: u64) -> RunCtx<'a> {
    RunCtx {
        root,
        anchor: ANCHOR,
        hardware: "gb10",
        yes: false,
        deadline: Duration::from_secs(secs),
        log_dir: root,
    }
}

/// The child's stderr is streamed to the callback and to the log file, and
/// the argv is the operator's command line.
#[test]
fn the_child_is_supervised_and_its_lines_are_streamed() {
    let (dir, exe) = script("echo '  [   1.0s] warmup [0/3]' >&2; echo 'stdout junk'; exit 0");
    let mut r = LocalChild {
        exe,
        records: Box::new(Scripted(Some(facts(ANCHOR, true, true, false)))),
        cancel: Arc::new(AtomicBool::new(false)),
        extra_args: vec![],
    };
    let c = ctx(dir.path(), 30);
    assert_eq!(
        r.argv(&plain(), &c),
        [
            "benchmark",
            "run",
            "decode-floor",
            "--pull-request-gate",
            "--hardware",
            "gb10"
        ]
    );
    // A shard's child is told which slice to run, and nothing else differs.
    assert_eq!(
        r.argv(&shard(), &c),
        [
            "benchmark",
            "run",
            "bfcl-subset",
            "--pull-request-gate",
            "--hardware",
            "gb10",
            "--param",
            "shard=0/4"
        ]
    );
    let mut lines = Vec::new();
    let out = r.run(&plain(), &c, &mut |l| lines.push(l.to_string()));
    assert!(matches!(out, RunOutcome::Passed { .. }), "{out:?}");
    assert!(lines.iter().any(|l| l.contains("warmup")), "{lines:?}");
    let log = std::fs::read_to_string(dir.path().join("decode-floor.log")).unwrap();
    assert!(
        log.contains("warmup") && log.contains("stdout: stdout junk"),
        "{log}"
    );
}

/// NEGATIVE CONTROL: a child that runs past its deadline is killed and the
/// outcome is TimedOut even though it would have exited 0.
#[test]
fn a_child_past_its_deadline_is_killed() {
    let (dir, exe) = script("sleep 30; exit 0");
    let mut r = LocalChild {
        exe,
        records: Box::new(Scripted(Some(facts(ANCHOR, true, true, false)))),
        cancel: Arc::new(AtomicBool::new(false)),
        extra_args: vec![],
    };
    let started = Instant::now();
    let out = r.run(&plain(), &ctx(dir.path(), 1), &mut |_| {});
    assert_eq!(out, RunOutcome::TimedOut);
    assert!(
        started.elapsed() < Duration::from_secs(20),
        "the kill was prompt"
    );
}

#[test]
fn a_cancel_flag_stops_the_child() {
    let (dir, exe) = script("sleep 30; exit 0");
    let cancel = Arc::new(AtomicBool::new(false));
    let mut r = LocalChild {
        exe,
        records: Box::new(Scripted(None)),
        cancel: cancel.clone(),
        extra_args: vec![],
    };
    let flag = cancel.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(600));
        flag.store(true, Ordering::SeqCst);
    });
    let out = r.run(&plain(), &ctx(dir.path(), 60), &mut |_| {});
    assert_eq!(out, RunOutcome::Cancelled);
}

#[test]
fn a_missing_executable_is_a_harness_failure_not_a_panic() {
    let dir = tempfile_dir::Dir::new("certify-noexe");
    let mut r = LocalChild {
        exe: dir.path().join("does-not-exist"),
        records: Box::new(Scripted(None)),
        cancel: Arc::new(AtomicBool::new(false)),
        extra_args: vec![],
    };
    let out = r.run(&plain(), &ctx(dir.path(), 5), &mut |_| {});
    assert!(
        matches!(
            out,
            RunOutcome::Harness {
                retryable: false,
                ..
            }
        ),
        "{out:?}"
    );
}
