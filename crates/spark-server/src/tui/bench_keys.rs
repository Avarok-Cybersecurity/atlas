// SPDX-License-Identifier: AGPL-3.0-only

//! Keyboard handling for the Benchmarks section.
//!
//! Split from the state so the reducer stays readable: this file only decides
//! what a key means in each of the three views, and everything it calls lives
//! in [`super::bench_state`].

use crossterm::event::{KeyCode, KeyEvent};

use super::app::BenchSub;
use super::bench_state::{BenchState, View};

/// What the section wants the app to do afterwards.
pub enum Outcome {
    None,
    /// Show a toast — a refused start, or a started run.
    Toast {
        text: String,
        error: bool,
    },
}

impl BenchState {
    pub fn on_key(&mut self, key: KeyEvent, sub: BenchSub) -> Outcome {
        if sub == BenchSub::History {
            return self.history_key(key);
        }
        match self.view {
            View::List => self.list_key(key),
            View::Params => self.params_key(key),
            View::Run => self.run_key(key),
        }
    }

    fn list_key(&mut self, key: KeyEvent) -> Outcome {
        let n = atlas_plugin::registry::all().len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.select((self.selected + 1).min(n - 1));
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.select(self.selected.saturating_sub(1));
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                self.view = View::Params;
            }
            // A run stays reachable after you navigate away from it.
            KeyCode::Char('v') if self.frame.is_some() || self.is_running() => {
                self.view = View::Run;
            }
            _ => {}
        }
        Outcome::None
    }

    fn params_key(&mut self, key: KeyEvent) -> Outcome {
        if self.confirm_open {
            return self.confirm_key(key);
        }
        if self.editing {
            return self.edit_key(key);
        }
        let rows = self.row_count();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if rows > 0 => {
                self.row = (self.row + 1).min(rows - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => self.row = self.row.saturating_sub(1),
            KeyCode::Enter => self.editing = true,
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            // Reset the form to the schema's defaults.
            KeyCode::Char('d') => {
                let selected = self.selected;
                self.select(selected);
            }
            KeyCode::Char('s') => return self.request_start(),
            _ => {}
        }
        Outcome::None
    }

    /// The one benchmark that runs model-authored shell asks first. The prompt
    /// is deliberately not a yes/no keypress on the same key that started it.
    fn request_start(&mut self) -> Outcome {
        let needs_confirmation = self.descriptor().is_some_and(|d| d.needs_confirmation);
        if needs_confirmation && !self.confirm_open {
            self.confirm_open = true;
            return Outcome::None;
        }
        self.confirm_open = false;
        match self.start() {
            Ok(()) => Outcome::Toast {
                text: format!(
                    "started {}",
                    self.descriptor().map(|d| d.name).unwrap_or("benchmark")
                ),
                error: false,
            },
            Err(e) => Outcome::Toast {
                text: e,
                error: true,
            },
        }
    }

    fn confirm_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Char('y') | KeyCode::Char('Y') => self.request_start(),
            _ => {
                self.confirm_open = false;
                Outcome::None
            }
        }
    }

    fn edit_key(&mut self, key: KeyEvent) -> Outcome {
        let row = self.row;
        match key.code {
            KeyCode::Enter => {
                self.commit_row(row);
                self.editing = false;
            }
            KeyCode::Esc => {
                // Restore what the value actually is, so a cancelled edit
                // cannot leave a half-typed string on screen.
                self.reset_row_buffer(row);
                self.editing = false;
            }
            KeyCode::Backspace => {
                if let Some(buf) = self.edit.get_mut(row) {
                    buf.pop();
                }
            }
            KeyCode::Char(c) => {
                if let Some(buf) = self.edit.get_mut(row) {
                    buf.push(c);
                }
            }
            _ => {}
        }
        Outcome::None
    }

    fn reset_row_buffer(&mut self, row: usize) {
        let current = match self.specs.get(row) {
            Some(spec) => self
                .values
                .get(spec.key)
                .map(|v| v.to_edit_string())
                .unwrap_or_else(|| spec.default.to_edit_string()),
            None if row == self.specs.len() => self.target.base_url.clone(),
            _ => self.target.model.clone(),
        };
        if let Some(buf) = self.edit.get_mut(row) {
            *buf = current;
        }
    }

    fn run_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Char('c') => {
                if self.is_running() {
                    self.cancel();
                    return Outcome::Toast {
                        text: "cancelling — the server keeps serving".into(),
                        error: false,
                    };
                }
            }
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            KeyCode::Down | KeyCode::Char('j') => {
                self.table_scroll = self.table_scroll.saturating_add(1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.table_scroll = self.table_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') | KeyCode::Home => self.table_scroll = 0,
            _ => {}
        }
        Outcome::None
    }

    fn history_key(&mut self, key: KeyEvent) -> Outcome {
        let n = self.history.len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.history_row = (self.history_row + 1).min(n - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.history_row = self.history_row.saturating_sub(1);
            }
            _ => {}
        }
        Outcome::None
    }
}
