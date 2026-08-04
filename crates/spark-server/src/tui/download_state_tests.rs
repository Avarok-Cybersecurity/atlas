// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn root() -> std::path::PathBuf {
    let p = std::env::temp_dir().join("atlas-dlstate");
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn a_second_download_is_refused_by_name_not_queued() {
    // Two multi-gigabyte pulls saturate the same disk and halve each other,
    // and the row has one line of progress space. Refusing is the honest
    // answer, but it has to say WHICH one is running or the message is
    // useless.
    let mut s = DownloadState::default();
    let (_, err) = s.start("org/first", root());
    assert!(!err, "the first one starts");

    let (text, err) = s.start("org/second", root());
    assert!(err, "the second is refused");
    assert!(text.contains("org/first"), "names the running one: {text}");
    assert!(text.contains('x'), "and how to stop it: {text}");
    assert!(s.is_downloading("org/first"));
    assert!(!s.is_downloading("org/second"));
}

#[test]
fn cancelling_twice_abandons_the_job() {
    // First press asks; the worker finishes its chunk so the bytes become
    // resume credit. Second press stops listening.
    let mut s = DownloadState::default();
    s.start("org/m", root());

    let (text, _) = s.cancel().expect("something is running");
    assert!(text.contains("stopping"), "{text}");
    assert!(
        s.job.as_ref().is_some_and(|j| j.cancelling),
        "still tracked, now marked"
    );

    let (text, _) = s.cancel().expect("still running");
    assert!(text.contains("abandoned"), "{text}");
    assert!(s.job.is_none(), "no longer tracked");
    assert!(s.cancel().is_none(), "nothing left to cancel");
}

#[test]
fn cancelling_nothing_is_not_an_error() {
    let mut s = DownloadState::default();
    assert!(s.cancel().is_none());
}

#[test]
fn a_job_without_sizes_reports_no_fraction_rather_than_zero() {
    // The Hub does not always report sizes. A bar pinned at 0% reads as a
    // stall; "no fraction" lets the renderer show motion instead.
    let mut s = DownloadState::default();
    s.start("org/m", root());
    let job = s.job.as_mut().unwrap();
    job.total = 0;
    job.done = 5_000;
    assert_eq!(job.fraction(), None);

    job.total = 10_000;
    job.done = 2_500;
    assert_eq!(job.fraction(), Some(0.25));
}

#[test]
fn the_fraction_never_exceeds_one() {
    // Resume accounting can briefly overshoot when a `.part` is larger than
    // the server's declared remainder. A 130% bar is a rendering bug.
    let mut s = DownloadState::default();
    s.start("org/m", root());
    let job = s.job.as_mut().unwrap();
    job.total = 100;
    job.done = 250;
    assert_eq!(job.fraction(), Some(1.0));
}

#[test]
fn freshness_defaults_to_nothing_rather_than_current() {
    // An unchecked model must not claim to be up to date.
    let s = DownloadState::default();
    assert!(!s.freshness.contains_key("org/m"));
}

#[test]
fn pump_with_no_job_is_quiet() {
    let mut s = DownloadState::default();
    assert!(s.pump().is_none());
}

/// A failing download must settle, clear the job, and leave an actionable
/// message — not spin forever showing a bar for work that has stopped.
#[test]
#[ignore = "network"]
fn a_failed_download_settles_and_explains_itself() {
    let mut s = DownloadState::default();
    s.start("definitely-not/a-real-model-xyz", root());

    let mut settled = None;
    for _ in 0..600 {
        if let Some(x) = s.pump() {
            settled = Some(x);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let settled = settled.expect("the job must reach a terminal state");
    match settled {
        Settled::Stopped(r) => assert_eq!(r, "definitely-not/a-real-model-xyz"),
        Settled::Finished(_) => panic!("a nonexistent repo must not report success"),
    }
    assert!(s.job.is_none(), "the job is cleared so the bar stops");
    let (text, error) = s.last_message.take().expect("it says what went wrong");
    assert!(error, "and marks it as an error: {text}");
    assert!(
        text.contains("definitely-not/a-real-model-xyz"),
        "naming the model: {text}"
    );
}

/// A real download, driven exactly as the dashboard drives it, ending in a
/// rescan signal — the thing that makes a downloaded model appear without a
/// restart.
#[test]
#[ignore = "network"]
fn a_real_download_settles_as_finished() {
    let cache = std::env::temp_dir().join("atlas-dlstate-real");
    let _ = std::fs::remove_dir_all(&cache);
    std::fs::create_dir_all(&cache).unwrap();

    let mut s = DownloadState::default();
    s.start("hf-internal-testing/tiny-random-gpt2", cache.clone());

    let mut settled = None;
    let mut saw_progress = false;
    for _ in 0..1200 {
        if let Some(job) = s.job.as_ref()
            && job.done > 0
        {
            saw_progress = true;
        }
        if let Some(x) = s.pump() {
            settled = Some(x);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    match settled.expect("the download must finish") {
        Settled::Finished(r) => assert_eq!(r, "hf-internal-testing/tiny-random-gpt2"),
        Settled::Stopped(_) => panic!("a real model should download"),
    }
    assert!(saw_progress, "progress must be observable while it runs");
    assert!(s.job.is_none());
    let _ = std::fs::remove_dir_all(&cache);
}

#[test]
fn a_download_for_a_model_not_being_downloaded_reads_as_not_downloading() {
    let mut s = DownloadState::default();
    s.start("org/m", root());
    assert!(!s.is_downloading("org/other"));
    assert!(!DownloadState::default().is_downloading("org/m"));
}
