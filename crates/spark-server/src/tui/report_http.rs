// SPDX-License-Identifier: AGPL-3.0-only

//! The I/O half of issue reporting: the transport seam, the two flow runners,
//! and the thread spawns. Everything protocol-shaped lives in
//! [`report`](super::report) as pure functions; this file owns the parts that
//! touch a socket or a clock.
//!
//! # Threading
//!
//! Both flows are blocking `ureq` on a named `std::thread` (`atlas-report`),
//! answering over a `std::sync::mpsc` the tick drains — the same contract as
//! every other worker in this tree, and the reason no `block_on` exists here
//! (CI enforces that under `tui/`). The device flow is a STREAMING producer
//! (code first, verdict later), so like `atlas-download` it hand-rolls its
//! spawn with an explicit on-failure send instead of using `worker::spawn` —
//! a receiver that can never resolve renders as a spinner that spins forever.
//!
//! # TLS (CWE-319)
//!
//! Every endpoint is `https://` through ureq's rustls with certificate
//! verification on — there is no toggle here to turn it off and no plain-HTTP
//! fallback to fall back to.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use super::report::{
    DEVICE_CODE_URL, NETWORK_FAILED, PollOutcome, ReportEvent, SecretString, TOKEN_URL,
    describe_issue_failure, issues_url, parse_device_code, parse_issue_created, parse_poll,
    parse_refresh,
};

/// One HTTP exchange, reduced to what the parsers need. `Err` is transport
/// failure (DNS, TLS, refused) — a STATUS is always `Ok`, whatever the code,
/// because GitHub's error bodies carry the message the user is shown.
pub struct HttpReply {
    pub status: u16,
    pub body: String,
    pub retry_after: Option<u64>,
}

pub type HttpResult = Result<HttpReply, String>;

/// The transport seam (SBIO): flow runners take `&dyn Http`, tests hand in a
/// scripted fake, and only [`Live`] ever opens a socket.
pub trait Http {
    /// Form-encoded POST to a github.com OAuth endpoint. No token — these are
    /// the endpoints that PRODUCE tokens.
    fn post_form(&self, url: &str, form: &[(&str, &str)]) -> HttpResult;
    /// JSON POST to the API, authenticated. The token travels ONLY in the
    /// `Authorization` header built inside the transport — no caller can
    /// accidentally place it in a URL or the body.
    fn post_json(&self, url: &str, token: &str, json: &serde_json::Value) -> HttpResult;
}

/// The real transport.
pub struct Live {
    agent: ureq::Agent,
}

impl Default for Live {
    fn default() -> Self {
        Self::new()
    }
}

impl Live {
    pub fn new() -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            // Non-2xx must come back as a reply, not an error: the failure
            // table is built from GitHub's OWN error bodies.
            .http_status_as_error(false)
            // Bounded, so a stalled POST settles into the failure path
            // instead of pinning the Submitting spinner for the process
            // lifetime. 30 s is generous for a <64 KB request.
            .timeout_global(Some(Duration::from_secs(30)))
            .build()
            .into();
        Self { agent }
    }

    fn reply(res: ureq::http::Response<ureq::Body>) -> HttpResult {
        let status = res.status().as_u16();
        let retry_after = res
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        let mut res = res;
        let body = res.body_mut().read_to_string().map_err(|e| e.to_string())?;
        Ok(HttpReply {
            status,
            body,
            retry_after,
        })
    }
}

const AGENT_HEADER: &str = concat!("atlas-spark/", env!("CARGO_PKG_VERSION"));

impl Http for Live {
    fn post_form(&self, url: &str, form: &[(&str, &str)]) -> HttpResult {
        let res = self
            .agent
            .post(url)
            .header("Accept", "application/json")
            .header("User-Agent", AGENT_HEADER)
            .send_form(form.iter().copied())
            .map_err(|e| e.to_string())?;
        Self::reply(res)
    }

    fn post_json(&self, url: &str, token: &str, json: &serde_json::Value) -> HttpResult {
        let res = self
            .agent
            .post(url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", AGENT_HEADER)
            .header("Authorization", &format!("Bearer {token}"))
            .send_json(json)
            .map_err(|e| e.to_string())?;
        Self::reply(res)
    }
}

// ── Device flow ──

/// Run the whole grant: request a code, hand it to the UI, poll until a
/// verdict. `wait` sleeps AND answers whether to keep going — `false` means
/// the user cancelled, and the flow ends in silence because the UI that would
/// hear a message already moved on.
///
/// The device code stays inside this function: it is the secret half of the
/// grant while pending, so it is never sent home, displayed, or logged.
pub fn run_device_flow(
    http: &dyn Http,
    client_id: &str,
    wait: &mut dyn FnMut(Duration) -> bool,
    tx: &Sender<ReportEvent>,
) {
    let fail = |tx: &Sender<ReportEvent>, message: String| {
        let _ = tx.send(ReportEvent::AuthFailed { message });
    };
    let grant = match http.post_form(DEVICE_CODE_URL, &[("client_id", client_id)]) {
        Ok(reply) => match parse_device_code(reply.status, &reply.body) {
            Ok(g) => g,
            Err(message) => return fail(tx, message),
        },
        Err(e) => {
            // The transport detail goes to the log ring for the maintainer;
            // the user-facing string stays actionable. This runs on the
            // worker thread, not the render thread.
            tracing::warn!("device-code request failed: {e}");
            return fail(tx, NETWORK_FAILED.to_string());
        }
    };
    let _ = tx.send(ReportEvent::CodeReady {
        user_code: grant.user_code.clone(),
        verification_uri: grant.verification_uri.clone(),
        expires_in: grant.expires_in,
    });
    let mut interval = grant.interval.max(Duration::from_secs(1));
    loop {
        if !wait(interval) {
            return;
        }
        let reply = match http.post_form(
            TOKEN_URL,
            &[
                ("client_id", client_id),
                ("device_code", &grant.device_code),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ],
        ) {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("device-flow poll failed: {e}");
                return fail(tx, NETWORK_FAILED.to_string());
            }
        };
        match parse_poll(reply.status, &reply.body) {
            PollOutcome::Pending => {}
            PollOutcome::SlowDown => interval += Duration::from_secs(5),
            PollOutcome::Authorized { access, refresh } => {
                let _ = tx.send(ReportEvent::Authorized { access, refresh });
                return;
            }
            PollOutcome::Expired => {
                return fail(
                    tx,
                    "the code expired before it was entered — s requests a fresh one".to_string(),
                );
            }
            PollOutcome::Denied => {
                return fail(
                    tx,
                    "authorization was declined on github.com — nothing was sent".to_string(),
                );
            }
            PollOutcome::Fatal(message) => return fail(tx, message),
        }
    }
}

// ── Submit ──

/// Everything one submission needs, bundled so the spawn signature stays
/// readable and the tokens travel together.
pub struct SubmitJob {
    pub client_id: String,
    pub repo: String,
    pub access: SecretString,
    pub refresh: Option<SecretString>,
    pub title: String,
    pub body: String,
}

/// POST the issue; on a 401 try one silent refresh (secret-free for
/// device-flow tokens) and retry once. Every outcome is a single terminal
/// event except the refresh success, which first sends `Authorized` so the
/// state keeps the rotated tokens.
pub fn run_submit(http: &dyn Http, job: SubmitJob, tx: &Sender<ReportEvent>) {
    let url = issues_url(&job.repo);
    let payload = serde_json::json!({ "title": job.title, "body": job.body });
    // NO labels/assignees/milestone: the API silently drops them for callers
    // without push access, so setting them would work for maintainers and
    // vanish for everyone else — the body marker is the labelling mechanism.
    if post_issue(http, &url, job.access.expose(), &payload, &job.repo, tx).is_none() {
        return; // terminal event already sent
    }
    // From here on the first POST answered 401 — the only status that earns
    // a second attempt, and only through a fresh token.
    let Some(refresh) = job.refresh.as_ref() else {
        return drop_auth_failed(tx);
    };
    let refreshed = http.post_form(
        TOKEN_URL,
        &[
            ("client_id", &job.client_id),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh.expose()),
        ],
    );
    let (access, new_refresh) = match refreshed {
        Ok(reply) => match parse_refresh(reply.status, &reply.body) {
            Some(pair) => pair,
            None => return drop_auth_failed(tx),
        },
        Err(e) => {
            tracing::warn!("token refresh failed: {e}");
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: NETWORK_FAILED.to_string(),
                drop_auth: false,
            });
            return;
        }
    };
    let _ = tx.send(ReportEvent::Authorized {
        access: access.clone(),
        refresh: new_refresh,
    });
    if post_issue(http, &url, access.expose(), &payload, &job.repo, tx) == Some(401) {
        drop_auth_failed(tx);
    }
}

/// One POST. Returns `Some(status)` when the caller still owes the user an
/// answer (only ever 401); `None` when a terminal event was already sent.
fn post_issue(
    http: &dyn Http,
    url: &str,
    token: &str,
    payload: &serde_json::Value,
    repo: &str,
    tx: &Sender<ReportEvent>,
) -> Option<u16> {
    let reply = match http.post_json(url, token, payload) {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("issue POST failed: {e}");
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: NETWORK_FAILED.to_string(),
                drop_auth: false,
            });
            return None;
        }
    };
    match parse_issue_created(reply.status, &reply.body) {
        Ok((number, url)) => {
            let _ = tx.send(ReportEvent::Created { number, url });
            None
        }
        Err(f) if f.status == 401 => Some(401),
        Err(f) => {
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: describe_issue_failure(&f, repo, reply.retry_after),
                drop_auth: false,
            });
            None
        }
    }
}

fn drop_auth_failed(tx: &Sender<ReportEvent>) {
    let _ = tx.send(ReportEvent::SubmitFailed {
        message: "GitHub no longer accepts this authorization — s re-authorizes".to_string(),
        drop_auth: true,
    });
}

// ── Spawns ──

/// The worker-thread boundary the state machine talks to, injectable so the
/// reducer's full flow is testable with no thread and no socket.
pub trait Workers {
    fn device_flow(&self, client_id: String, cancel: Arc<AtomicBool>) -> Receiver<ReportEvent>;
    fn submit(&self, job: SubmitJob) -> Receiver<ReportEvent>;
}

/// Real threads, real sockets.
pub struct LiveWorkers;

impl Workers for LiveWorkers {
    fn device_flow(&self, client_id: String, cancel: Arc<AtomicBool>) -> Receiver<ReportEvent> {
        let (tx, rx) = channel();
        let spawned = std::thread::Builder::new()
            .name("atlas-report".into())
            .spawn({
                let tx = tx.clone();
                move || {
                    // Sleep in slices so a cancel takes effect within ~200 ms of
                    // Esc rather than at the end of a 5 s (or slowed-down 10 s+)
                    // poll interval — the download worker's "cancellation
                    // honoured within a chunk" rule, applied to time.
                    let mut wait = |d: Duration| {
                        let deadline = std::time::Instant::now() + d;
                        while std::time::Instant::now() < deadline {
                            if cancel.load(Ordering::Relaxed) {
                                return false;
                            }
                            std::thread::sleep(Duration::from_millis(200));
                        }
                        !cancel.load(Ordering::Relaxed)
                    };
                    run_device_flow(&Live::new(), &client_id, &mut wait, &tx);
                }
            });
        if let Err(e) = spawned {
            // The always-answer rule: a spawn that failed silently leaves the
            // UI polling a receiver that never resolves.
            let _ = tx.send(ReportEvent::AuthFailed {
                message: format!("could not start the report worker: {e}"),
            });
        }
        rx
    }

    fn submit(&self, job: SubmitJob) -> Receiver<ReportEvent> {
        let (tx, rx) = channel();
        let spawned = std::thread::Builder::new()
            .name("atlas-report".into())
            .spawn({
                let tx = tx.clone();
                move || run_submit(&Live::new(), job, &tx)
            });
        if let Err(e) = spawned {
            let _ = tx.send(ReportEvent::SubmitFailed {
                message: format!("could not start the report worker: {e}"),
                drop_auth: false,
            });
        }
        rx
    }
}

#[cfg(test)]
#[path = "report_http_tests.rs"]
mod tests;
