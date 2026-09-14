// SPDX-License-Identifier: AGPL-3.0-only
//! The remote runner over a scripted atlasctl: what each way a node can
//! answer turns into, in the campaign's own outcome vocabulary.
use super::super::super::plan::{Estimate, Unit};
use super::super::atlasctl::{
    Atlasctl, AttachEnd, ErrorObj, Exit, FetchedFile, NodeRow, Refusal, StreamEvent, SubmitSpec,
    Submitted,
};
use super::*;
use anyhow::Result;
use atlas_plugin::hardware::equivalence::HardwareFingerprint;
use atlas_plugin::hardware::policy::Sensitivity;
use std::collections::VecDeque;
use std::sync::Mutex;

fn unit() -> Unit {
    Unit {
        id: "decode-floor",
        group: None,
        class: Sensitivity::Speed,
        estimate: Estimate::Declared(180),
        needs_confirmation: false,
    }
}

fn node(built: bool) -> Node {
    Node {
        addr: "10.10.10.2".into(),
        name: "dgx2".into(),
        node_id: "1730e1bea1873a8a7abcdef".into(),
        signer: "a27dbc8ed2fc2a31".into(),
        hardware: HardwareFingerprint {
            gpu: "NVIDIA GB10".into(),
            driver_major: Some(580),
            sm_clock_max_mhz: Some(3003.0),
            mem_total_kb: Some(1),
            thermal_alert: Some(false),
            hottest_chassis_c: Some(65.0),
            postcheck_valid: None,
        },
        free_fraction: Some(0.9),
        built,
        local: false,
    }
}

/// A scratch repository with the signer registry and the newest committed
/// decode-floor record staged where a fetch would put it.
struct Scratch {
    root: PathBuf,
    files: Vec<FetchedFile>,
    sha: String,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn scratch(tag: &str) -> Scratch {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .to_path_buf();
    let root = std::env::temp_dir().join(format!("certify-remote-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join(".github/record-signers")).unwrap();
    for e in std::fs::read_dir(ws.join(".github/record-signers"))
        .unwrap()
        .flatten()
    {
        std::fs::copy(
            e.path(),
            root.join(".github/record-signers").join(e.file_name()),
        )
        .unwrap();
    }
    let newest = std::fs::read_dir(ws.join(".benchmarks/decode-floor"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .max()
        .unwrap();
    let name = newest.file_name().unwrap().to_string_lossy().into_owned();
    let fetched = root.join("fetched");
    std::fs::create_dir_all(fetched.join(".benchmarks/decode-floor")).unwrap();
    let rec = fetched.join(".benchmarks/decode-floor").join(&name);
    let sig = fetched
        .join(".benchmarks/decode-floor")
        .join(format!("{name}.sig"));
    std::fs::copy(&newest, &rec).unwrap();
    std::fs::copy(format!("{}.sig", newest.display()), &sig).unwrap();
    let sha = atlas_plugin::gate::read_record(&newest).unwrap().git_sha;
    let f = |n: String, rel: String, path: PathBuf| FetchedFile {
        name: n,
        relative_path: rel,
        path,
        bytes: 0,
        sha256: String::new(),
    };
    let files = vec![
        f(
            name.clone(),
            format!(".benchmarks/decode-floor/{name}"),
            rec,
        ),
        f(
            format!("{name}.sig"),
            format!(".benchmarks/decode-floor/{name}.sig"),
            sig,
        ),
    ];
    Scratch { root, files, sha }
}

struct Script {
    submit: Result<Submitted, Refusal>,
    attach: Mutex<VecDeque<AttachEnd>>,
    fetch: Result<Vec<FetchedFile>, Refusal>,
    calls: Mutex<Vec<String>>,
}

impl Script {
    fn new(attach: Vec<AttachEnd>, files: Vec<FetchedFile>) -> Self {
        Self {
            submit: Ok(Submitted {
                node_id: "1730".into(),
                job_id: "jb-1-deadbeef".into(),
                existing: false,
            }),
            attach: Mutex::new(attach.into()),
            fetch: Ok(files),
            calls: Mutex::new(vec![]),
        }
    }
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

impl Atlasctl for Script {
    fn nodes(&self, _: &[String]) -> Result<Vec<NodeRow>> {
        unreachable!()
    }
    fn submit(&self, node: &str, spec: &SubmitSpec) -> Result<Result<Submitted, Refusal>> {
        self.calls.lock().unwrap().push(format!(
            "submit {node} {} {} {} sha40={}",
            spec.gate,
            spec.job_key,
            spec.hardware,
            spec.sha.len() == 40
        ));
        Ok(self.submit.clone())
    }
    fn attach(
        &self,
        node: &str,
        job: &str,
        from_seq: u64,
        _reconnect_for: Duration,
        cancel: &AtomicBool,
        on_event: &mut dyn FnMut(&StreamEvent),
    ) -> Result<AttachEnd> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("attach {node} {job} from {from_seq}"));
        on_event(&StreamEvent {
            seq: from_seq,
            kind: "progress".into(),
            phase: Some("isl 512".into()),
            detail: Some("[1/8]".into()),
            lines: vec![],
            outcome: None,
            exit_code: None,
            reason: None,
            stage: None,
            verdict: None,
            cached: None,
        });
        if cancel.load(Ordering::SeqCst) {
            return Ok(AttachEnd::Cancelled);
        }
        Ok(self
            .attach
            .lock()
            .unwrap()
            .pop_front()
            .expect("script has an attach answer"))
    }
    fn cancel(&self, node: &str, job: &str) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("cancel {node} {job}"));
        Ok(())
    }
    fn fetch(
        &self,
        node: &str,
        job: &str,
        _out: &Path,
    ) -> Result<Result<Vec<FetchedFile>, Refusal>> {
        self.calls
            .lock()
            .unwrap()
            .push(format!("fetch {node} {job}"));
        Ok(self.fetch.clone())
    }
}

fn run(
    script: Arc<Script>,
    s: &Scratch,
    cancel: Arc<AtomicBool>,
    built: bool,
) -> (RunOutcome, Vec<String>) {
    let mut r = RemoteRunner {
        atlasctl: script.clone(),
        node: node(built),
        run_id: "r1".into(),
        // The record names the commit abbreviated; the wire wants the full
        // 40 hex, which is what the driver resolves before building runners.
        anchor_full: format!("{:0<40}", s.sha),
        cancel,
        scratch: s.root.join("scratch"),
    };
    let log_dir = s.root.join("logs");
    std::fs::create_dir_all(&log_dir).unwrap();
    let ctx = RunCtx {
        root: &s.root,
        anchor: &s.sha,
        hardware: "gb10",
        yes: false,
        deadline: Duration::from_secs(600),
        log_dir: &log_dir,
    };
    let mut lines = vec![];
    let out = r.run(&unit(), &ctx, &mut |l| lines.push(l.to_owned()));
    let calls = script.calls();
    // Whenever an attach happened, its progress reached the line stream.
    if calls.iter().any(|c| c.starts_with("attach")) {
        assert!(
            lines.iter().any(|l| l.contains("[isl 512] [1/8]")),
            "{lines:?}"
        );
    }
    (out, calls)
}

#[test]
fn a_passing_job_is_submitted_followed_fetched_placed_and_classified() {
    let s = scratch("happy");
    let script = Arc::new(Script::new(
        vec![AttachEnd::Passed { exit_code: Some(0) }],
        s.files.clone(),
    ));
    let (out, calls) = run(script, &s, Arc::new(AtomicBool::new(false)), true);
    match out {
        RunOutcome::Passed { record } => {
            assert!(record.starts_with(s.root.join(".benchmarks/decode-floor")));
            assert!(record.exists());
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(calls.len(), 3, "{calls:?}");
    assert!(
        calls[0]
            .starts_with("submit 10.10.10.2 decode-floor certify-r1-1730e1be-decode-floor gb10"),
        "{}",
        calls[0]
    );
    // The node is told the full commit, never the campaign's abbreviation
    // (the wire refuses anything but 40 hex; the first real run found this).
    assert!(calls[0].ends_with("sha40=true"), "{}", calls[0]);
    assert_eq!(calls[1], "attach 10.10.10.2 jb-1-deadbeef from 1");
    assert_eq!(calls[2], "fetch 10.10.10.2 jb-1-deadbeef");
}

#[test]
fn a_lost_stream_is_reattached_from_the_last_seq_and_bounded() {
    let s = scratch("lost");
    let script = Arc::new(Script::new(
        vec![
            AttachEnd::StreamLost { last_seq: 17 },
            AttachEnd::StreamLost { last_seq: 40 },
            AttachEnd::Passed { exit_code: Some(0) },
        ],
        s.files.clone(),
    ));
    let (out, calls) = run(script, &s, Arc::new(AtomicBool::new(false)), true);
    assert!(matches!(out, RunOutcome::Passed { .. }), "{out:?}");
    assert_eq!(calls[1], "attach 10.10.10.2 jb-1-deadbeef from 1");
    assert_eq!(calls[2], "attach 10.10.10.2 jb-1-deadbeef from 18");
    assert_eq!(calls[3], "attach 10.10.10.2 jb-1-deadbeef from 41");
    // NEGATIVE CONTROL: more than MAX_REATTACH losses is a retryable
    // harness failure, and nothing is fetched.
    let s = scratch("lost-forever");
    let script = Arc::new(Script::new(
        (0..=MAX_REATTACH as usize + 1)
            .map(|i| AttachEnd::StreamLost { last_seq: i as u64 })
            .collect(),
        s.files.clone(),
    ));
    let (out, calls) = run(script, &s, Arc::new(AtomicBool::new(false)), true);
    match out {
        RunOutcome::Harness { reason, retryable } => {
            assert!(retryable);
            assert!(reason.contains("lost the stream"), "{reason}");
        }
        other => panic!("{other:?}"),
    }
    assert!(!calls.iter().any(|c| c.starts_with("fetch")));
}

#[test]
fn refusals_and_failures_before_a_record_are_harness_failures_with_the_right_retryability() {
    let s = scratch("refused");
    let busy = Arc::new(Script {
        submit: Err((
            Exit::Refused,
            Some(ErrorObj {
                code: "refused:busy".into(),
                message: "spark pid 1".into(),
                retryable: true,
                ..Default::default()
            }),
        )),
        ..Script::new(vec![], vec![])
    });
    let (out, _) = run(busy, &s, Arc::new(AtomicBool::new(false)), true);
    assert!(
        matches!(out, RunOutcome::Harness { retryable: true, ref reason } if reason.contains("refused:busy")),
        "{out:?}"
    );
    let unpaired = Arc::new(Script {
        submit: Err((
            Exit::NotPaired,
            Some(ErrorObj {
                code: "not_paired".into(),
                message: "no".into(),
                ..Default::default()
            }),
        )),
        ..Script::new(vec![], vec![])
    });
    let (out, _) = run(unpaired, &s, Arc::new(AtomicBool::new(false)), true);
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
    // The build failed on the node: nothing measured, another node may do it.
    let build = Arc::new(Script::new(
        vec![AttachEnd::JobFailed {
            outcome: Some("failed".into()),
            exit_code: None,
            detail: "cargo exited 101".into(),
        }],
        vec![],
    ));
    let (out, calls) = run(build, &s, Arc::new(AtomicBool::new(false)), true);
    assert!(
        matches!(out, RunOutcome::Harness { retryable: true, ref reason } if reason.contains("cargo exited 101")),
        "{out:?}"
    );
    assert!(!calls.iter().any(|c| c.starts_with("fetch")));
    let timed = Arc::new(Script::new(
        vec![AttachEnd::JobFailed {
            outcome: Some("timed_out".into()),
            exit_code: None,
            detail: String::new(),
        }],
        vec![],
    ));
    assert_eq!(
        run(timed, &s, Arc::new(AtomicBool::new(false)), true).0,
        RunOutcome::TimedOut
    );
}

#[test]
fn a_cancel_here_cancels_the_job_there() {
    let s = scratch("cancel");
    let script = Arc::new(Script::new(
        vec![AttachEnd::Passed { exit_code: Some(0) }],
        s.files.clone(),
    ));
    let cancel = Arc::new(AtomicBool::new(true));
    let (out, calls) = run(script, &s, cancel, true);
    assert_eq!(out, RunOutcome::Cancelled);
    assert!(
        calls.iter().any(|c| c == "cancel 10.10.10.2 jb-1-deadbeef"),
        "{calls:?}"
    );
    assert!(!calls.iter().any(|c| c.starts_with("fetch")));
}

#[test]
fn a_record_that_is_not_what_was_asked_for_is_refused_for_good() {
    let s = scratch("wrong");
    // The node hands back files for another gate.
    let mut wrong = s.files.clone();
    for f in &mut wrong {
        f.relative_path = f.relative_path.replace("decode-floor", "ttft-warm-gate");
    }
    let script = Arc::new(Script::new(
        vec![AttachEnd::Passed { exit_code: Some(0) }],
        wrong,
    ));
    let (out, _) = run(script, &s, Arc::new(AtomicBool::new(false)), true);
    match out {
        RunOutcome::Harness { reason, retryable } => {
            assert!(!retryable, "a wrong record will be wrong again");
            assert!(
                reason.contains("record from 10.10.10.2 refused"),
                "{reason}"
            );
        }
        other => panic!("{other:?}"),
    }
    assert!(!s.root.join(".benchmarks").exists());
}

#[test]
fn the_deadline_pays_for_a_build_only_on_a_cold_node() {
    let local = Duration::from_secs(600);
    assert_eq!(
        deadline_for(local, &node(true), Duration::from_secs(1800)),
        local
    );
    assert_eq!(
        deadline_for(local, &node(false), Duration::from_secs(1800)),
        Duration::from_secs(2400)
    );
    // The job key is bounded and names the campaign, the node and the gate.
    let r = RemoteRunner {
        atlasctl: Arc::new(Script::new(vec![], vec![])),
        node: node(true),
        run_id: "1a0dc88a8c-1757770000".into(),
        anchor_full: "1a0dc88a8c9083bb956bd84cafa2cccbdb8e6e18".into(),
        cancel: Arc::new(AtomicBool::new(false)),
        scratch: PathBuf::new(),
    };
    let k = r.job_key(&unit());
    assert!(
        k.starts_with("certify-1a0dc88a8c-1757770000-1730e1be-decode-floor"),
        "{k}"
    );
    assert!(k.len() <= 64);
}
