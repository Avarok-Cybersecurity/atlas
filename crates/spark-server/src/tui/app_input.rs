// SPDX-License-Identifier: AGPL-3.0-only

//! Text-entry routing for the [`App`] reducer.
//!
//! Split from `app.rs` only to stay under the repository's per-file cap. It is
//! one concern — deciding which of the several text buffers owns a keystroke —
//! so it moves as a unit, unchanged.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::{App, Focus, Section, TermSub, edit_line};

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
                KeyCode::Char(c) => self.chat.input.push(c),
                _ => {}
            },
        }
    }
}
