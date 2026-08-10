// SPDX-License-Identifier: AGPL-3.0-only

//! The download state machine, driven without the Hub.
//!
//! `DownloadState::start` is the only way to get a `Job`, and it spawns the
//! real worker — so every job here is started for a repo id containing a
//! SPACE, which cannot form a URL. The worker fails at the request builder,
//! before a socket is opened, and reports `Offline`. That gives a real job, a
//! real terminal transition, and no network.

use super::*;

/// A repo id that cannot be turned into a URL — see the module doc.
const UNREACHABLE: &str = "org/not a real model";

fn root() -> std::path::PathBuf {
    let p = std::env::temp_dir().join("atlas-dlstate-more");
    std::fs::create_dir_all(&p).expect("temp root");
    p
}

/// Pump until the job settles, so a test never depends on worker timing.
fn settle(s: &mut DownloadState) -> Settled {
    for _ in 0..600 {
        if let Some(x) = s.pump() {
            return x;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("the job never reached a terminal state");
}

#[test]
fn starting_a_download_names_the_repo_and_is_not_an_error() {
    let mut s = DownloadState::default();
    let (text, error) = s.start(UNREACHABLE, root());
    assert!(!error);
    assert!(text.contains(UNREACHABLE), "{text}");
    assert!(s.is_downloading(UNREACHABLE));
    let job = s.job.as_ref().expect("a job");
    assert_eq!(job.done, 0);
    assert_eq!(job.total, 0);
    assert!(!job.cancelling);
    assert_eq!(job.rate_bps, 0.0, "no bytes, no rate");
}

#[test]
fn a_download_that_cannot_start_settles_as_stopped_and_says_why() {
    // The transition that matters: the bar has to stop, the job has to be
    // cleared, and the message has to be something the reader can act on.
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    match settle(&mut s) {
        Settled::Stopped(repo) => assert_eq!(repo, UNREACHABLE),
        Settled::Finished(_) => panic!("a failure must not report success"),
    }
    assert!(s.job.is_none(), "the job is cleared so the bar stops");
    let (text, error) = s.last_message.take().expect("it says what went wrong");
    assert!(error, "and marks it as an error: {text}");
    assert!(text.contains(UNREACHABLE), "naming the model: {text}");
    assert!(s.pump().is_none(), "and it settles once, not every tick");
}

#[test]
fn a_settled_download_frees_the_slot_for_the_next_one() {
    // The refusal names the RUNNING job; once nothing is running the same
    // model must be startable again, which is what "press d to resume" means.
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    settle(&mut s);
    let (_, error) = s.start(UNREACHABLE, root());
    assert!(!error, "the slot is free again");
    assert!(s.is_downloading(UNREACHABLE));
}

#[test]
fn abandoning_a_download_frees_the_slot_immediately() {
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    s.cancel().expect("asks it to stop");
    s.cancel().expect("abandons it");
    let (_, error) = s.start(UNREACHABLE, root());
    assert!(!error, "a second press means the UI has let go");
}

#[test]
fn the_fraction_is_monotone_in_the_bytes_moved() {
    // A bar that can go backwards reads as a restart. `done` only ever grows
    // within a job, so the derived fraction must too.
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    let job = s.job.as_mut().expect("a job");
    job.total = 1_000;
    let mut last = 0.0;
    for done in [0u64, 1, 250, 499, 500, 999, 1_000] {
        job.done = done;
        let f = job.fraction().expect("a known total");
        assert!(f >= last, "{done} bytes went backwards: {f} < {last}");
        last = f;
    }
    assert_eq!(last, 1.0, "and it reaches one exactly");
}

#[test]
fn an_unknown_total_reports_no_fraction_rather_than_dividing_by_zero() {
    // The Hub's size endpoint is best-effort; a job can run with `total == 0`
    // from the first byte to the last.
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    let job = s.job.as_mut().expect("a job");
    for done in [0u64, 1, 4_000_000_000] {
        job.done = done;
        job.total = 0;
        assert_eq!(job.fraction(), None, "{done} bytes of an unknown total");
    }
}

#[test]
fn a_job_that_is_already_complete_reports_exactly_one() {
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    let job = s.job.as_mut().expect("a job");
    job.total = 7;
    job.done = 7;
    assert_eq!(job.fraction(), Some(1.0));
}

#[test]
fn cancelling_marks_the_job_before_the_worker_has_answered() {
    // "stopping…" is more honest than a bar that keeps moving: cancellation is
    // honoured within a megabyte, not instantly.
    let mut s = DownloadState::default();
    s.start(UNREACHABLE, root());
    let (text, error) = s.cancel().expect("something is running");
    assert!(!error, "stopping on purpose is not a failure");
    assert!(text.contains("stopping"), "{text}");
    let job = s.job.as_ref().expect("still tracked");
    assert!(job.cancelling);
    assert_eq!(job.repo, UNREACHABLE, "and it is the same job");
}

#[test]
fn a_second_freshness_check_is_refused_rather_than_queued() {
    // The Hub throttles, and the answer is per-model; two in flight is two
    // requests for one badge.
    let mut s = DownloadState::default();
    let (_tx, rx) = std::sync::mpsc::channel();
    s.pending_check = Some(rx);
    s.checking = Some("org/first".into());

    let (text, error) = s.check("org/second", root());
    assert!(error, "{text}");
    assert_eq!(
        s.checking.as_deref(),
        Some("org/first"),
        "and the running one is untouched"
    );
}

#[test]
fn a_freshness_answer_is_stored_under_the_id_the_worker_echoed_back() {
    // By the time it lands the user may have moved on, so applying it to
    // whatever is selected would be silently wrong.
    let mut s = DownloadState::default();
    let (tx, rx) = std::sync::mpsc::channel();
    tx.send(("org/asked".to_string(), Freshness::Current))
        .expect("send");
    s.pending_check = Some(rx);
    s.checking = Some("org/asked".into());

    assert!(s.pump().is_none(), "a freshness answer is not a settlement");
    assert!(s.checking.is_none(), "the skeleton clears");
    assert!(matches!(
        s.freshness.get("org/asked"),
        Some(Freshness::Current)
    ));
    assert!(!s.freshness.contains_key("org/other"));
}

#[test]
fn a_dead_freshness_worker_clears_the_skeleton_and_claims_nothing() {
    let mut s = DownloadState::default();
    let (tx, rx) = std::sync::mpsc::channel::<(String, Freshness)>();
    drop(tx);
    s.pending_check = Some(rx);
    s.checking = Some("org/asked".into());

    assert!(s.pump().is_none());
    assert!(s.checking.is_none(), "the skeleton must not spin forever");
    assert!(s.pending_check.is_none());
    assert!(s.freshness.is_empty(), "an unanswered check states nothing");
}

#[test]
fn a_freshness_check_in_flight_does_not_block_a_download() {
    // Different slots: they are refused independently, or checking one model
    // would stop you downloading another.
    let mut s = DownloadState::default();
    let (_tx, rx) = std::sync::mpsc::channel();
    s.pending_check = Some(rx);
    let (_, error) = s.start(UNREACHABLE, root());
    assert!(!error);
}

#[test]
fn every_failure_is_described_with_something_to_do_about_it() {
    // A diagnostic without a fix is half a diagnostic, and a gated repo must
    // never read as a network fault.
    let cases = [
        DownloadError::Offline("dns".into()),
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: false,
        },
        DownloadError::Gated {
            repo: "org/m".into(),
            had_token: true,
        },
        DownloadError::NotFound {
            repo: "org/m".into(),
        },
        DownloadError::RateLimited,
        DownloadError::DiskFull,
        DownloadError::NotEnoughSpace {
            need: 40_000_000_000,
            free: 1_000_000_000,
        },
        DownloadError::NoSafetensors {
            repo: "org/m".into(),
        },
        DownloadError::Http {
            repo: "org/m".into(),
            status: 500,
        },
        DownloadError::Io("permission denied".into()),
    ];
    for e in cases {
        let text = describe("org/m", &e);
        assert!(text.starts_with("org/m: "), "names the model: {text}");
        assert!(text.len() > "org/m: ".len(), "and says something: {e:?}");
    }
    assert!(
        describe(
            "org/m",
            &DownloadError::Gated {
                repo: "org/m".into(),
                had_token: false
            }
        )
        .contains("HF_TOKEN"),
        "a gated repo names the credential, not the network"
    );
}
