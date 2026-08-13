// SPDX-License-Identifier: AGPL-3.0-only

//! The Help section's state: the Guide/Report split, the issue composer, and
//! the report pipeline's phase machine.
//!
//! # The one invariant worth stating twice
//!
//! **The draft is cleared in exactly one place: on `Created`.** Every failure
//! — network, declined authorization, expired code, 401, 422, a worker that
//! died — keeps the composed title and body intact. A long bug report lost to
//! a failed POST is the feature teaching users not to use it.
//!
//! # Why the preview is not skippable while logs are attached
//!
//! The attach checkbox defaults ON, and the attachment goes to a PUBLIC
//! tracker (CWE-200). The default is only defensible because submission with
//! logs is unreachable except THROUGH the preview of the exact final bytes —
//! `s` on the composer leads to the preview, and only the preview has a send
//! key. Unchecking the box is the one way to skip it, because then nothing
//! ships that the user did not type.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use super::report::{Composed, ReportEvent, SecretString, Target};
use super::report_http::{LiveWorkers, SubmitJob, Workers};

#[derive(Clone, Copy, PartialEq)]
pub enum HelpSub {
    Guide,
    Report,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ComposerField {
    Title,
    Body,
    Attach,
}

pub enum ReportPhase {
    Compose,
    Preview,
    /// Device code requested, not yet issued.
    RequestingCode,
    /// The user has a code to type into github.com; a worker is polling.
    WaitingAuth {
        user_code: String,
        verification_uri: String,
        expires_at: Instant,
    },
    Submitting,
    Done {
        number: u64,
        url: String,
    },
    Failed {
        message: String,
    },
}

/// The in-memory authorization. Lives here and nowhere else — not on disk,
/// not in a keyring (these are headless SSH boxes with no Secret Service),
/// not in any log (CWE-522). Forgotten when the process exits, by design.
pub(super) struct Auth {
    pub(super) access: SecretString,
    pub(super) refresh: Option<SecretString>,
}

/// A composed submission waiting on auth or a retry. Kept until `Created` so
/// `s` after a failure resubmits the exact previewed bytes.
pub(super) struct Pending {
    pub(super) title: String,
    pub(super) body: String,
    pub(super) target: Target,
}

/// What the composer needs from `App` to build a body — passed in, so this
/// file never reaches back into the host.
pub struct ReportCtx {
    pub model: String,
    pub engine_ready: bool,
    pub tee: Option<&'static str>,
}

pub struct HelpState {
    pub sub: HelpSub,
    pub phase: ReportPhase,
    pub title: String,
    pub title_editing: bool,
    pub body: tui_textarea::TextArea<'static>,
    pub(super) body_editing: bool,
    pub field: ComposerField,
    /// Defaults ON (see the module doc for why that is survivable).
    pub attach_logs: bool,
    pub preview: Option<Composed>,
    pub preview_scroll: usize,
    /// Published by the renderer — the `chat_scroll_max` pattern.
    pub preview_scroll_max: std::cell::Cell<usize>,
    pub(super) auth: Option<Auth>,
    pub(super) pending: Option<Pending>,
    pub(super) rx: Option<Receiver<ReportEvent>>,
    pub(super) cancel: Option<Arc<AtomicBool>>,
    pub(super) workers: Box<dyn Workers + Send>,
    pub(super) last_message: Option<(String, bool)>,
}

fn fresh_body() -> tui_textarea::TextArea<'static> {
    let mut body = tui_textarea::TextArea::default();
    body.set_cursor_line_style(ratatui::style::Style::default());
    body.set_cursor_style(ratatui::style::Style::default());
    body.set_placeholder_text("What happened, what you expected, steps to reproduce…");
    body
}

impl Default for HelpState {
    fn default() -> Self {
        Self {
            sub: HelpSub::Guide,
            phase: ReportPhase::Compose,
            title: String::new(),
            title_editing: false,
            body: fresh_body(),
            body_editing: false,
            field: ComposerField::Title,
            attach_logs: true,
            preview: None,
            preview_scroll: 0,
            preview_scroll_max: std::cell::Cell::new(0),
            auth: None,
            pending: None,
            rx: None,
            cancel: None,
            workers: Box::new(LiveWorkers),
            last_message: None,
        }
    }
}

impl HelpState {
    /// True while a text field owns the keyboard (feeds `App::in_input`).
    pub fn is_editing(&self) -> bool {
        self.title_editing || self.body_editing
    }

    /// Unsubmitted words that `q` would destroy.
    pub fn has_draft(&self) -> bool {
        !self.title.trim().is_empty() || self.body.lines().iter().any(|l| !l.trim().is_empty())
    }

    /// An authorization or submission is mid-flight.
    pub fn report_in_flight(&self) -> bool {
        matches!(
            self.phase,
            ReportPhase::RequestingCode | ReportPhase::WaitingAuth { .. } | ReportPhase::Submitting
        )
    }

    /// A toast for the event loop, if one is owed.
    pub fn take_message(&mut self) -> Option<(String, bool)> {
        self.last_message.take()
    }

    pub(super) fn say(&mut self, text: impl Into<String>, error: bool) {
        self.last_message = Some((text.into(), error));
    }

    /// Every failure lands here: phase, toast, AND the log ring — a toast
    /// that fired while the operator looked elsewhere is otherwise gone
    /// without a trace.
    pub(super) fn fail(&mut self, message: String) {
        tracing::warn!(target: "atlas_tui", "issue report: {message}");
        self.say(message.clone(), true);
        self.phase = ReportPhase::Failed { message };
    }

    pub(super) fn body_text(&self) -> String {
        self.body.lines().join("\n")
    }

    pub(super) fn set_body_editing(&mut self, editing: bool) {
        self.body_editing = editing;
        // The cursor cell is reversed only while the textarea owns the
        // keyboard — a visible cursor in an unfocused field claims focus the
        // field does not have.
        self.body.set_cursor_style(if editing {
            ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::REVERSED)
        } else {
            ratatui::style::Style::default()
        });
    }

    // ── Pipeline ──

    /// Back to the composer. `pending` is dropped deliberately: the user is
    /// about to edit, and a kept `pending` would let a later `s` submit bytes
    /// that no longer match what is on screen.
    pub(super) fn back_to_compose(&mut self) {
        self.phase = ReportPhase::Compose;
        self.preview = None;
        self.pending = None;
    }

    pub(super) fn review(&mut self, ctx: &ReportCtx) {
        if self.title.trim().is_empty() {
            self.say("a title is required — k to the Title row, ⏎ to edit", true);
            return;
        }
        match self.compose(ctx) {
            Err(e) => self.say(e, true),
            Ok(c) => {
                if self.attach_logs {
                    self.preview = Some(c);
                    self.preview_scroll = 0;
                    self.phase = ReportPhase::Preview;
                } else {
                    // No attachment: everything in the body is what the user
                    // typed (plus the environment line), so there is nothing
                    // to preview that they have not already seen.
                    self.proceed(c);
                }
            }
        }
    }

    /// Build the exact body a submit would post — one function feeds preview
    /// and POST, so they cannot diverge.
    pub(super) fn compose(&self, ctx: &ReportCtx) -> Result<Composed, String> {
        let env = super::report::env_line(&ctx.model, ctx.engine_ready);
        let logs: Option<Vec<String>> = self.attach_logs.then(|| {
            let redact_ctx = super::redact::RedactCtx::from_env();
            super::log_ring::tail(10_000)
                .into_iter()
                .map(|l| {
                    super::redact::redact_line(
                        &format!("{:>5} {} {}", l.level, l.target, l.message),
                        &redact_ctx,
                    )
                })
                .collect()
        });
        super::report::compose_body(&self.body_text(), &env, logs.as_deref(), ctx.tee)
    }

    pub(super) fn proceed(&mut self, composed: Composed) {
        match super::report::target() {
            Err(m) => self.fail(m.to_string()),
            Ok(target) => {
                self.pending = Some(Pending {
                    title: self.title.trim().to_string(),
                    body: composed.body,
                    target,
                });
                self.auth_or_submit();
            }
        }
    }

    pub(super) fn auth_or_submit(&mut self) {
        let Some(p) = &self.pending else { return };
        if let Some(auth) = &self.auth {
            self.rx = Some(self.workers.submit(SubmitJob {
                client_id: p.target.client_id.clone(),
                repo: p.target.repo.clone(),
                access: auth.access.clone(),
                refresh: auth.refresh.clone(),
                title: p.title.clone(),
                body: p.body.clone(),
            }));
            self.phase = ReportPhase::Submitting;
        } else {
            let cancel = Arc::new(AtomicBool::new(false));
            self.rx = Some(
                self.workers
                    .device_flow(p.target.client_id.clone(), cancel.clone()),
            );
            self.cancel = Some(cancel);
            self.phase = ReportPhase::RequestingCode;
        }
    }

    // ── Event ingress (tick) ──

    pub fn pump(&mut self) {
        let Some(rx) = &self.rx else { return };
        let mut events: Vec<ReportEvent> = rx.try_iter().collect();
        // `try_iter` cannot distinguish "nothing yet" from "the producer
        // died" — the chat pump's lesson. Undetected, a panicked worker pins
        // the spinner for the rest of the process.
        let mut disconnected = false;
        if events.is_empty() && self.report_in_flight() {
            match rx.try_recv() {
                Ok(e) => events.push(e),
                Err(std::sync::mpsc::TryRecvError::Disconnected) => disconnected = true,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        if disconnected {
            self.rx = None;
            self.cancel = None;
            self.fail("the report worker stopped unexpectedly — s retries".to_string());
            return;
        }
        for e in events {
            self.apply(e);
        }
    }

    fn apply(&mut self, event: ReportEvent) {
        match event {
            ReportEvent::CodeReady {
                user_code,
                verification_uri,
                expires_in,
            } => {
                self.phase = ReportPhase::WaitingAuth {
                    user_code,
                    verification_uri,
                    expires_at: Instant::now() + expires_in,
                };
            }
            ReportEvent::Authorized { access, refresh } => {
                self.auth = Some(Auth { access, refresh });
                // From the device flow: the grant is done, submit what waited.
                // From a mid-submit refresh rotation (phase Submitting): keep
                // listening — Created/SubmitFailed follows on the same channel,
                // and respawning here would double-post the issue.
                if matches!(
                    self.phase,
                    ReportPhase::RequestingCode | ReportPhase::WaitingAuth { .. }
                ) {
                    self.cancel = None;
                    self.auth_or_submit();
                }
            }
            ReportEvent::AuthFailed { message } => {
                self.rx = None;
                self.cancel = None;
                self.fail(message);
            }
            ReportEvent::Created { number, url } => {
                self.rx = None;
                // THE clearing site — the only one. See the module doc.
                self.title.clear();
                self.body = fresh_body();
                self.title_editing = false;
                self.body_editing = false;
                self.field = ComposerField::Title;
                self.preview = None;
                self.pending = None;
                self.say(format!("issue #{number} opened"), false);
                self.phase = ReportPhase::Done { number, url };
            }
            ReportEvent::SubmitFailed { message, drop_auth } => {
                self.rx = None;
                if drop_auth {
                    self.auth = None;
                }
                self.fail(message);
            }
        }
    }
}

#[cfg(test)]
#[path = "help_state_tests.rs"]
mod tests;
