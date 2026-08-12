// SPDX-License-Identifier: AGPL-3.0-only

//! Text-entry routing for the [`App`] reducer.
//!
//! Split from `app.rs` only to stay under the repository's per-file cap. It is
//! one concern — deciding which of the several text buffers owns a keystroke —
//! so it moves as a unit, unchanged.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use super::app::{App, Focus, Section, TermSub};

/// Minimal single-line editor for the two filter boxes.
///
/// Lives here rather than in `app.rs` because this file is the one that decides
/// which buffer owns a keystroke, and this is what those buffers are edited
/// with — the split is by concern, not only by the 500-LoC cap.
pub(super) fn edit_line(buf: &mut String, key: KeyEvent, editing: &mut bool) {
    match key.code {
        KeyCode::Esc => {
            buf.clear();
            *editing = false;
        }
        KeyCode::Enter => *editing = false,
        KeyCode::Backspace => {
            buf.pop();
        }
        KeyCode::Char(c) => buf.push(c),
        _ => {}
    }
}

impl App {
    pub(super) fn on_input_key(&mut self, key: KeyEvent) {
        // Which buffer?
        if self.log_filter_editing {
            edit_line(&mut self.log_filter, key, &mut self.log_filter_editing);
            return;
        }
        if self.section == Section::Library {
            self.on_library_key(key);
            return;
        }
        if self.section == Section::Benchmarks {
            self.on_bench_key(key);
            return;
        }
        // Terminal input.
        match self.term_sub {
            TermSub::Ops => match key.code {
                KeyCode::Esc => self.focus = Focus::Content,
                KeyCode::Enter => {
                    let line = std::mem::take(&mut self.ops.input);
                    if !line.trim().is_empty() {
                        self.ops.history.push(line.clone());
                        self.ops.history_pos = None;
                        super::commands::execute(&line, self);
                    }
                }
                // The one completion affordance on screen — the "⇥ accept"
                // hint beside the ghost text — pressed. It used to fall into
                // `_ => {}` here while the global Tab handler sat unreachable
                // behind `in_input()`: the advertised key did nothing.
                KeyCode::Tab => {
                    if let Some(ghost) = super::commands::complete(&self.ops.input) {
                        self.ops.input = ghost.to_string();
                    }
                }
                KeyCode::Up => {
                    let h = &self.ops.history;
                    if !h.is_empty() {
                        let pos = match self.ops.history_pos {
                            None => h.len() - 1,
                            Some(p) => p.saturating_sub(1),
                        };
                        self.ops.history_pos = Some(pos);
                        self.ops.input = h[pos].clone();
                    }
                }
                // Up's missing other half: history could be walked back and
                // never forward again. Past the newest entry the line
                // returns to empty — the readline contract fingers expect.
                KeyCode::Down => {
                    if let Some(p) = self.ops.history_pos {
                        if p + 1 < self.ops.history.len() {
                            self.ops.history_pos = Some(p + 1);
                            self.ops.input = self.ops.history[p + 1].clone();
                        } else {
                            self.ops.history_pos = None;
                            self.ops.input.clear();
                        }
                    }
                }
                // Scrollback stays reachable while typing; Up/Down are spent
                // on history here, so the page pair does the moving (Chat
                // makes the same trade the other way round).
                KeyCode::PageUp => self.scroll(-10),
                KeyCode::PageDown => self.scroll(10),
                KeyCode::Backspace => {
                    self.ops.input.pop();
                }
                KeyCode::Char(c) => self.ops.input.push(c),
                _ => {}
            },
            TermSub::Chat => match key.code {
                KeyCode::Esc => {
                    self.chat.cancel();
                    self.focus = Focus::Content;
                }
                // Enter sends; a trailing backslash continues onto a new
                // line (Ctrl+Enter is indistinguishable from Enter in legacy
                // terminal protocols, so it cannot be the only send chord).
                KeyCode::Enter => {
                    if let Some(stripped) = self.chat.input.strip_suffix('\\') {
                        self.chat.input = format!("{stripped}\n");
                    } else {
                        self.chat.send(self.args.port);
                    }
                }
                KeyCode::Backspace => {
                    self.chat.input.pop();
                }
                // Transcript scrollback stays live while the input holds focus —
                // that is where you are while a reply streams, and Up/Down are
                // otherwise unused here (unlike Ops, which spends them on history).
                KeyCode::Up => self.chat.scroll_by(1),
                KeyCode::Down => self.chat.scroll_by(-1),
                KeyCode::PageUp => self.chat.scroll_by(10),
                KeyCode::PageDown => self.chat.scroll_by(-10),
                KeyCode::End => self.chat.follow(),
                // Home is not text, so the input keeps it even while focused —
                // the other half of the End pair above.
                KeyCode::Home => self.chat_jump_top(),
                // The two thinking toggles, in their chorded forms. They come
                // BEFORE the catch-all: a bare `t` is text, and `Ctrl+T`
                // arrives as `Char('t')` with a modifier, so an unguarded
                // catch-all would type a `t` for it instead.
                KeyCode::Char('t') => match self.chat.on_view_key(key, true) {
                    Some(said) => self.toast(said, false),
                    None => self.chat.input.push('t'),
                },
                // Same trap as Ctrl+T above: unguarded, the catch-all types an
                // `n` for it. Works with the input focused because starting
                // over is most wanted mid-conversation, which is where the
                // cursor lives.
                KeyCode::Char('n') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    self.request_chat_clear();
                }
                KeyCode::Char(c) => self.chat.input.push(c),
                _ => {}
            },
        }
    }

    /// Chat keys when the transcript, not the input box, has focus. Bare
    /// letters are free here, so the toggles get their unchorded forms too.
    pub(super) fn on_chat_content_key(&mut self, key: KeyEvent) {
        if key.code == KeyCode::Char('n') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.request_chat_clear();
            return;
        }
        // `g`/Home need the renderer-published ceiling, which `ChatState`
        // cannot see; the rest of the pair (`G`/End) is handled inside it.
        if matches!(key.code, KeyCode::Char('g') | KeyCode::Home) {
            self.chat_jump_top();
            return;
        }
        if let Some(said) = self.chat.on_content_key(key) {
            self.toast(said, false);
        }
    }

    /// `Ctrl+N`: start a new chat session — after a confirmation whenever
    /// there is a conversation to lose.
    ///
    /// The gate is conditional for the same reason `on_quit_key`'s is: a
    /// prompt over an empty transcript protects nothing, and a prompt that is
    /// usually pointless trains the reflex that dismisses the one that
    /// matters.
    pub(super) fn request_chat_clear(&mut self) {
        if self.chat.transcript.is_empty() && !self.chat.streaming {
            self.toast("chat is already empty", false);
        } else {
            self.confirm_chat_clear = true;
        }
    }

    /// Answer the clear-chat prompt. Always consumes the key, like
    /// `answer_quit_prompt`, and for the same reason: a prompt that lets keys
    /// through makes dismissing it navigate somewhere as a side effect.
    ///
    /// Only an affirmative clears — `y`, or `Ctrl+N` again, the same
    /// double-press grammar the quit prompt taught with `q`. A bare `n` is
    /// NOT the trigger key here: it reads as "no" and must cancel.
    pub(super) fn answer_chat_clear(&mut self, key: KeyEvent) -> bool {
        self.confirm_chat_clear = false;
        let again = key.code == KeyCode::Char('n') && key.modifiers.contains(KeyModifiers::CONTROL);
        if again || matches!(key.code, KeyCode::Char('y') | KeyCode::Char('Y')) {
            let turns = self.chat.transcript.len();
            self.chat.reset();
            self.toast(format!("chat cleared — {turns} turns discarded"), false);
        }
        true
    }
}

#[cfg(test)]
#[path = "app_input_tests.rs"]
mod tests;
