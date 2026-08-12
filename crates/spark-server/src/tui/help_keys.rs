// SPDX-License-Identifier: AGPL-3.0-only

//! Key handling for Help ▸ Report Issue — one function per pipeline phase,
//! split from `help_state.rs` at the repository's 500-LoC cap. It moves as a
//! unit: every function here answers the same question, "what does this
//! keystroke mean on the report screen the user is looking at".

use std::sync::atomic::Ordering;

use crossterm::event::{KeyCode, KeyEvent};

use super::help_state::{ComposerField, HelpState, HelpSub, ReportCtx, ReportPhase};

impl HelpState {
    pub(super) fn on_key(&mut self, key: KeyEvent, ctx: &ReportCtx) {
        if self.sub == HelpSub::Guide {
            return;
        }
        match self.phase {
            ReportPhase::Compose => self.compose_key(key, ctx),
            ReportPhase::Preview => self.preview_key(key, ctx),
            ReportPhase::RequestingCode | ReportPhase::WaitingAuth { .. } => self.auth_key(key),
            // A POST that may already have created the issue cannot be
            // cancelled honestly; the 30 s transport timeout bounds the wait.
            ReportPhase::Submitting => {}
            ReportPhase::Done { .. } => self.done_key(key),
            ReportPhase::Failed { .. } => self.failed_key(key),
        }
    }

    fn compose_key(&mut self, key: KeyEvent, ctx: &ReportCtx) {
        if self.title_editing {
            match key.code {
                // Esc KEEPS the text — the filter-box "Esc clears" grammar
                // would destroy a draft title here.
                KeyCode::Esc | KeyCode::Enter => self.title_editing = false,
                KeyCode::Backspace => {
                    self.title.pop();
                }
                KeyCode::Char(c) => self.title.push(c),
                _ => {}
            }
            return;
        }
        if self.body_editing {
            if key.code == KeyCode::Esc {
                self.set_body_editing(false);
            } else {
                self.body.input(key);
            }
            return;
        }
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.field = match self.field {
                    ComposerField::Title => ComposerField::Body,
                    _ => ComposerField::Attach,
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.field = match self.field {
                    ComposerField::Attach => ComposerField::Body,
                    _ => ComposerField::Title,
                }
            }
            KeyCode::Enter => match self.field {
                ComposerField::Title => self.title_editing = true,
                ComposerField::Body => self.set_body_editing(true),
                ComposerField::Attach => self.attach_logs = !self.attach_logs,
            },
            KeyCode::Char(' ') if self.field == ComposerField::Attach => {
                self.attach_logs = !self.attach_logs;
            }
            KeyCode::Char('a') => self.attach_logs = !self.attach_logs,
            KeyCode::Char('s') => self.review(ctx),
            _ => {}
        }
    }

    fn preview_key(&mut self, key: KeyEvent, ctx: &ReportCtx) {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.scroll_preview(1),
            KeyCode::Up | KeyCode::Char('k') => self.scroll_preview(-1),
            KeyCode::PageDown => self.scroll_preview(10),
            KeyCode::PageUp => self.scroll_preview(-10),
            KeyCode::Char('g') | KeyCode::Home => self.preview_scroll = 0,
            KeyCode::Char('G') | KeyCode::End => {
                self.preview_scroll = self.preview_scroll_max.get();
            }
            // The attachment can be dropped even from inside the preview —
            // deciding "actually, no logs" after reading them is the point.
            KeyCode::Char('a') => {
                self.attach_logs = !self.attach_logs;
                match self.compose(ctx) {
                    Ok(c) => {
                        self.preview = Some(c);
                        self.preview_scroll = 0;
                    }
                    Err(e) => self.say(e, true),
                }
            }
            KeyCode::Char('y') | KeyCode::Enter => {
                if let Some(c) = self.preview.take() {
                    self.proceed(c);
                }
            }
            KeyCode::Esc => self.back_to_compose(),
            _ => {}
        }
    }

    fn auth_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Esc => {
                // Stop the polling thread (it notices within ~200 ms) and
                // walk back. The draft is untouched by construction.
                if let Some(c) = self.cancel.take() {
                    c.store(true, Ordering::Relaxed);
                }
                self.rx = None;
                self.back_to_compose();
            }
            KeyCode::Char('c') => {
                if let ReportPhase::WaitingAuth { user_code, .. } = &self.phase {
                    let said = match super::clipboard::copy(user_code) {
                        Ok(_) => (
                            "code sent to the terminal clipboard (OSC 52)".to_string(),
                            false,
                        ),
                        Err(e) => (e, true),
                    };
                    self.say(said.0, said.1);
                }
            }
            _ => {}
        }
    }

    fn done_key(&mut self, key: KeyEvent) {
        match (key.code, &self.phase) {
            (KeyCode::Char('c'), ReportPhase::Done { url, .. }) if !url.is_empty() => {
                let said = match super::clipboard::copy(url) {
                    Ok(_) => (
                        "link sent to the terminal clipboard (OSC 52)".to_string(),
                        false,
                    ),
                    Err(e) => (e, true),
                };
                self.say(said.0, said.1);
            }
            (KeyCode::Esc | KeyCode::Enter, _) => self.phase = ReportPhase::Compose,
            _ => {}
        }
    }

    fn failed_key(&mut self, key: KeyEvent) {
        match key.code {
            // Retry resubmits the exact previewed bytes (or re-authorizes
            // first if the tokens were dropped) — no silent re-compose.
            KeyCode::Char('s') if self.pending.is_some() => self.auth_or_submit(),
            KeyCode::Char('s') | KeyCode::Esc => self.back_to_compose(),
            _ => {}
        }
    }

    pub(super) fn scroll_preview(&mut self, rows: i32) {
        let max = self.preview_scroll_max.get() as i32;
        self.preview_scroll = (self.preview_scroll as i32 + rows).clamp(0, max.max(0)) as usize;
    }
}

impl super::app::App {
    /// Route a key into the Help section with the context the composer needs.
    pub(super) fn on_help_key(&mut self, key: KeyEvent) {
        let ctx = ReportCtx {
            model: super::render::live_model_name(self),
            engine_ready: self.progress.ready,
            tee: super::init::tee_file_path(),
        };
        self.help.on_key(key, &ctx);
    }
}
