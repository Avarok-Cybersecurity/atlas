// SPDX-License-Identifier: AGPL-3.0-only

//! The report pipeline's invariants, driven key by key through fake workers —
//! no thread, no socket. The one that matters most is stated as a property:
//! **only `Created` clears the draft**; every other transition keeps it.

use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};

use crossterm::event::{KeyCode, KeyEvent};

use super::*;
use crate::tui::report_http::{SubmitJob, Workers};

/// Records every spawn and hands the test the sender side of each channel.
#[derive(Default)]
struct FakeWorkers {
    device_calls: Arc<Mutex<Vec<String>>>,
    submit_jobs: Arc<Mutex<Vec<(String, String, String)>>>, // (repo, title, body)
    txs: Arc<Mutex<Vec<Sender<ReportEvent>>>>,
    cancels: Arc<Mutex<Vec<Arc<std::sync::atomic::AtomicBool>>>>,
}

impl Workers for FakeWorkers {
    fn device_flow(
        &self,
        client_id: String,
        cancel: Arc<std::sync::atomic::AtomicBool>,
    ) -> Receiver<ReportEvent> {
        self.device_calls.lock().unwrap().push(client_id);
        self.cancels.lock().unwrap().push(cancel);
        let (tx, rx) = channel();
        self.txs.lock().unwrap().push(tx);
        rx
    }
    fn submit(&self, job: SubmitJob) -> Receiver<ReportEvent> {
        self.submit_jobs
            .lock()
            .unwrap()
            .push((job.repo, job.title, job.body));
        let (tx, rx) = channel();
        self.txs.lock().unwrap().push(tx);
        rx
    }
}

struct Rig {
    state: HelpState,
    workers: FakeWorkers,
}

fn rig() -> Rig {
    let workers = FakeWorkers::default();
    let mut state = HelpState {
        sub: HelpSub::Report,
        workers: Box::new(FakeWorkers {
            device_calls: workers.device_calls.clone(),
            submit_jobs: workers.submit_jobs.clone(),
            txs: workers.txs.clone(),
            cancels: workers.cancels.clone(),
        }),
        ..Default::default()
    };
    state.title = "chat pane locks up".into();
    state
        .body
        .insert_str("press Esc mid-stream, then send again");
    Rig { state, workers }
}

fn ctx() -> ReportCtx {
    ReportCtx {
        model: "test-model".into(),
        engine_ready: true,
        tee: Some("/tee/log"),
    }
}

fn tap(s: &mut HelpState, code: KeyCode) {
    s.on_key(KeyEvent::from(code), &ctx());
}

/// The worker's sender for the most recent spawn.
fn last_tx(r: &Rig) -> Sender<ReportEvent> {
    r.workers
        .txs
        .lock()
        .unwrap()
        .last()
        .expect("a worker was spawned")
        .clone()
}

fn send(r: &mut Rig, e: ReportEvent) {
    last_tx(r).send(e).expect("state still listening");
    r.state.pump();
}

fn tokens() -> ReportEvent {
    ReportEvent::Authorized {
        access: SecretString::new("ghu_T".into()),
        refresh: Some(SecretString::new("ghr_T".into())),
    }
}

// ── The preview gate ──

#[test]
fn with_logs_attached_submit_is_reachable_only_through_the_preview() {
    let mut r = rig();
    assert!(r.state.attach_logs, "the checkbox defaults ON");
    tap(&mut r.state, KeyCode::Char('s'));
    assert!(
        matches!(r.state.phase, ReportPhase::Preview),
        "s must land on the preview"
    );
    assert!(
        r.workers.device_calls.lock().unwrap().is_empty(),
        "nothing sent yet"
    );
    assert!(r.workers.submit_jobs.lock().unwrap().is_empty());
    // Only the preview has a send key.
    tap(&mut r.state, KeyCode::Char('y'));
    assert!(matches!(r.state.phase, ReportPhase::RequestingCode));
    assert_eq!(r.workers.device_calls.lock().unwrap().len(), 1);
}

#[test]
fn with_logs_off_the_preview_is_skipped_and_no_log_section_ships() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('a')); // uncheck
    tap(&mut r.state, KeyCode::Char('s'));
    assert!(matches!(r.state.phase, ReportPhase::RequestingCode));
    let body = &r.state.pending.as_ref().expect("pending").body;
    assert!(!body.contains("## Server log"), "{body}");
    assert!(body.contains("## Environment"));
    assert!(body.contains(crate::tui::report::MARKER));
}

#[test]
fn an_empty_title_is_refused_before_anything_is_composed() {
    let mut r = rig();
    r.state.title.clear();
    tap(&mut r.state, KeyCode::Char('s'));
    assert!(matches!(r.state.phase, ReportPhase::Compose));
    let (msg, error) = r.state.take_message().expect("a refusal message");
    assert!(error);
    assert!(msg.contains("title"), "{msg}");
}

// ── The draft-survival property ──

fn draft_of(s: &HelpState) -> (String, String) {
    (s.title.clone(), s.body.lines().join("\n"))
}

/// One way a report attempt can die, injected mid-flight.
type Failure = Box<dyn Fn(&mut Rig)>;

#[test]
fn every_failure_keeps_the_draft_only_created_clears_it() {
    // Walk one failure of each kind; after each, the draft must be intact
    // and `s` must still lead somewhere.
    let failures: Vec<Failure> = vec![
        Box::new(|r| {
            send(
                r,
                ReportEvent::AuthFailed {
                    message: "declined".into(),
                },
            )
        }),
        Box::new(|r| {
            send(
                r,
                ReportEvent::SubmitFailed {
                    message: "503".into(),
                    drop_auth: false,
                },
            )
        }),
        // Worker panicked: every sender is gone and the channel dies silently.
        Box::new(|r| r.workers.txs.lock().unwrap().clear()),
    ];
    for start in failures {
        let mut r = rig();
        let before = draft_of(&r.state);
        tap(&mut r.state, KeyCode::Char('s'));
        tap(&mut r.state, KeyCode::Char('y'));
        start(&mut r);
        r.state.pump();
        assert!(
            matches!(r.state.phase, ReportPhase::Failed { .. }),
            "the pipeline must settle into Failed, not spin"
        );
        assert_eq!(
            draft_of(&r.state),
            before,
            "a failure must never cost the draft"
        );
        assert!(
            r.state.pending.is_some(),
            "the composed bytes are kept for retry"
        );
    }
}

#[test]
fn created_is_the_one_clearing_site() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    send(&mut r, tokens());
    // Authorization triggered the submit with the previewed bytes.
    let jobs = r.workers.submit_jobs.lock().unwrap();
    assert_eq!(jobs.len(), 1);
    assert_eq!(jobs[0].1, "chat pane locks up");
    assert!(jobs[0].2.contains("press Esc mid-stream"));
    drop(jobs);
    send(
        &mut r,
        ReportEvent::Created {
            number: 214,
            url: "https://x/214".into(),
        },
    );
    assert!(matches!(
        r.state.phase,
        ReportPhase::Done { number: 214, .. }
    ));
    assert!(
        !r.state.has_draft(),
        "the draft clears on Created and nowhere else"
    );
    assert!(r.state.pending.is_none());
}

#[test]
fn a_stopped_worker_settles_into_failed_not_a_stuck_spinner() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    drop(last_tx(&r));
    r.workers.txs.lock().unwrap().clear(); // no sender left anywhere
    r.state.pump();
    let ReportPhase::Failed { message } = &r.state.phase else {
        panic!("expected Failed");
    };
    assert!(message.contains("s retries"), "{message}");
}

// ── Cancellation and staleness ──

#[test]
fn esc_during_auth_flags_the_worker_and_keeps_the_draft() {
    let mut r = rig();
    let before = draft_of(&r.state);
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    send(
        &mut r,
        ReportEvent::CodeReady {
            user_code: "WDJB-MJHT".into(),
            verification_uri: "https://github.com/login/device".into(),
            expires_in: std::time::Duration::from_secs(899),
        },
    );
    assert!(matches!(r.state.phase, ReportPhase::WaitingAuth { .. }));
    tap(&mut r.state, KeyCode::Esc);
    assert!(matches!(r.state.phase, ReportPhase::Compose));
    assert_eq!(draft_of(&r.state), before);
    let cancels = r.workers.cancels.lock().unwrap();
    assert!(
        cancels[0].load(Ordering::Relaxed),
        "the polling thread must be told to stop"
    );
}

#[test]
fn returning_to_compose_drops_pending_so_a_retry_cannot_ship_stale_bytes() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    send(
        &mut r,
        ReportEvent::AuthFailed {
            message: "x".into(),
        },
    );
    tap(&mut r.state, KeyCode::Esc); // back to the composer to edit
    assert!(
        r.state.pending.is_none(),
        "an edit after Esc must not race a stale pending body"
    );
    // Editing then re-submitting rebuilds from the current text.
    r.state.title = "a different bug".into();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    assert_eq!(
        r.state.pending.as_ref().expect("pending").title,
        "a different bug"
    );
}

#[test]
fn retry_after_failure_reuses_the_exact_previewed_bytes() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    send(&mut r, tokens());
    send(
        &mut r,
        ReportEvent::SubmitFailed {
            message: "503".into(),
            drop_auth: false,
        },
    );
    let body_before = r.state.pending.as_ref().expect("pending").body.clone();
    tap(&mut r.state, KeyCode::Char('s')); // retry — auth is held, so straight to submit
    assert!(matches!(r.state.phase, ReportPhase::Submitting));
    let jobs = r.workers.submit_jobs.lock().unwrap();
    assert_eq!(jobs.len(), 2);
    assert_eq!(
        jobs[1].2, body_before,
        "the retry must post what was previewed, byte for byte"
    );
}

#[test]
fn drop_auth_forces_reauthorization_on_the_next_attempt() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    send(&mut r, tokens());
    send(
        &mut r,
        ReportEvent::SubmitFailed {
            message: "revoked".into(),
            drop_auth: true,
        },
    );
    tap(&mut r.state, KeyCode::Char('s'));
    assert!(
        matches!(r.state.phase, ReportPhase::RequestingCode),
        "dead tokens must lead back to the device flow, not to a replay"
    );
    assert_eq!(r.workers.device_calls.lock().unwrap().len(), 2);
    assert_eq!(r.workers.submit_jobs.lock().unwrap().len(), 1);
}

#[test]
fn a_mid_submit_token_rotation_does_not_double_post() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    send(&mut r, tokens());
    assert_eq!(r.workers.submit_jobs.lock().unwrap().len(), 1);
    // The submit worker refreshed and reports rotated tokens mid-flight.
    send(&mut r, tokens());
    assert!(
        matches!(r.state.phase, ReportPhase::Submitting),
        "still the SAME submission"
    );
    assert_eq!(
        r.workers.submit_jobs.lock().unwrap().len(),
        1,
        "no second POST"
    );
}

// ── Composer editing ──

#[test]
fn esc_keeps_the_title_text_unlike_the_filter_boxes() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Enter); // edit the Title field
    assert!(r.state.is_editing());
    tap(&mut r.state, KeyCode::Char('!'));
    tap(&mut r.state, KeyCode::Esc);
    assert!(!r.state.is_editing());
    assert_eq!(
        r.state.title, "chat pane locks up!",
        "Esc must not clear a draft title"
    );
}

#[test]
fn body_editing_owns_every_key_until_esc() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('j')); // field: Title -> Body
    tap(&mut r.state, KeyCode::Enter);
    assert!(r.state.is_editing());
    // 's' while editing is TEXT, not "submit".
    tap(&mut r.state, KeyCode::Char('s'));
    assert!(matches!(r.state.phase, ReportPhase::Compose));
    tap(&mut r.state, KeyCode::Esc);
    assert!(r.state.body.lines().join("\n").ends_with('s'));
}

#[test]
fn preview_can_drop_the_attachment_and_the_counts_follow() {
    let mut r = rig();
    tap(&mut r.state, KeyCode::Char('s'));
    let with_logs = r.state.preview.as_ref().expect("preview").body.clone();
    assert!(with_logs.contains("## Environment"));
    tap(&mut r.state, KeyCode::Char('a')); // drop the logs from inside the preview
    let without = r.state.preview.as_ref().expect("preview rebuilt");
    assert!(!without.body.contains("## Server log"), "{}", without.body);
    assert_eq!(without.logs_total, 0);
    assert!(!r.state.attach_logs);
}

#[test]
fn work_in_flight_answers_for_drafts_and_flights() {
    let mut r = rig();
    assert!(r.state.has_draft());
    assert!(!r.state.report_in_flight());
    tap(&mut r.state, KeyCode::Char('s'));
    tap(&mut r.state, KeyCode::Char('y'));
    assert!(r.state.report_in_flight());
}
