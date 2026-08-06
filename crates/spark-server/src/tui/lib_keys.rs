// SPDX-License-Identifier: AGPL-3.0-only

//! Library key handling.
//!
//! Deliberately the same contract as the benchmark section: `j/k` moves, `⏎`
//! opens or edits, `Esc` steps back one level, `/` searches, `d` restores
//! defaults. Two browsable sections with different reflexes would make both
//! harder to learn.

use crossterm::event::{KeyCode, KeyEvent};

use super::lib_state::{LibState, View};

/// What the section wants the app to do about a keypress.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum Outcome {
    #[default]
    None,
    /// Show a message; `error` picks the colour.
    Toast { text: String, error: bool },
    /// Start the configured recipe. The reducer cannot do it itself — it has no
    /// handle to the server — so the app layer performs it and reports back.
    Launch,
    /// Download, resume, or update the selected model. Same division of labour
    /// as `Launch`: the reducer has no cache-dir handle, so `App` performs it.
    Download,
    /// Stop the running download. A second press abandons it.
    CancelDownload,
    /// Check whether the selected model is behind the Hub.
    CheckFresh,
}

impl LibState {
    /// True while a text field owns the keyboard, so global bindings stand down.
    pub fn is_editing(&self) -> bool {
        self.editing || self.filter_editing
    }

    pub fn on_key(&mut self, key: KeyEvent) -> Outcome {
        if self.filter_editing {
            return self.filter_key(key);
        }
        match self.view {
            View::List => self.list_key(key),
            View::Cards => self.cards_key(key),
            View::Config => self.config_key(key),
        }
    }

    fn filter_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Esc => {
                self.filter.clear();
                self.filter_editing = false;
            }
            KeyCode::Enter => self.filter_editing = false,
            KeyCode::Backspace => {
                self.filter.pop();
            }
            KeyCode::Char(c) => self.filter.push(c),
            _ => return Outcome::None,
        }
        // The selection is an index into the FILTERED list, so it has to be
        // re-clamped on every keystroke or it can point past the end.
        self.selected = 0;
        Outcome::None
    }

    fn list_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => self.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => self.move_selection(-1),
            KeyCode::Char('/') => self.filter_editing = true,
            // `d` is "restore defaults" in the CONFIG pane — a different
            // screen, so no conflict, and `d` for download is the obvious
            // mnemonic here. `x` rather than `c` for cancel because the help
            // modal already teaches `c` as "cancel the running benchmark", and
            // one key with two meanings in adjacent sections is exactly what
            // this module's doc warns against.
            KeyCode::Char('d') => return Outcome::Download,
            KeyCode::Char('x') => return Outcome::CancelDownload,
            KeyCode::Char('u') => return Outcome::CheckFresh,
            KeyCode::Char('r') => {
                if self.fetching {
                    return Outcome::None;
                }
                self.refresh();
                // Also re-scan the local cache: `r` reads as "bring this list
                // up to date", and a model that appeared on disk since the
                // last scan is exactly what a reader pressing it expects to
                // show up. The scan is local and cheap; the fetch is neither,
                // which is why only this key does both.
                self.mark_dirty = true;
                // Report what actually happened. `refresh` is a no-op without a
                // store to cache into, and announcing a fetch that never
                // started would leave the user waiting for nothing.
                return if self.fetching {
                    Outcome::Toast {
                        text: "fetching recipes…".into(),
                        error: false,
                    }
                } else {
                    Outcome::Toast {
                        text: "no artifact store — recipes cannot be fetched or cached".into(),
                        error: true,
                    }
                };
            }
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Err(e) = self.open_cards() {
                    return Outcome::Toast {
                        text: e,
                        error: true,
                    };
                }
            }
            _ => {}
        }
        Outcome::None
    }

    /// The recipe cards for one model. Same contract as the list: j/k moves,
    /// Enter descends, Esc goes back one level.
    fn cards_key(&mut self, key: KeyEvent) -> Outcome {
        let n = self.cards().len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.card = (self.card + 1).min(n - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => self.card = self.card.saturating_sub(1),
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if let Err(e) = self.open_config() {
                    return Outcome::Toast {
                        text: e,
                        error: true,
                    };
                }
            }
            // The same three the list offers, because Cards is where you are
            // when you have picked a model and are reading its recipes — which
            // is exactly when you discover you do not have the weights. Making
            // the user step back to the list first is a step for nothing, and
            // it is what made the config pane's "Esc, then d" advice wrong by
            // one keypress.
            KeyCode::Char('d') => return Outcome::Download,
            KeyCode::Char('x') => return Outcome::CancelDownload,
            KeyCode::Char('u') => return Outcome::CheckFresh,
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            _ => {}
        }
        Outcome::None
    }

    fn config_key(&mut self, key: KeyEvent) -> Outcome {
        if self.editing {
            return self.edit_key(key);
        }
        let rows = self.config_rows().len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if rows > 0 => {
                self.row = (self.row + 1).min(rows - 1);
            }
            KeyCode::Up | KeyCode::Char('k') => self.row = self.row.saturating_sub(1),
            KeyCode::Enter => {
                // Seed the buffer with the current value: editing a setting is
                // usually adjusting it, not retyping it.
                if let Some((_, value, _)) = self.config_rows().into_iter().nth(self.row) {
                    self.edit_buffer = value;
                    self.editing = true;
                }
            }
            KeyCode::Char('d') => {
                self.reset_overrides();
                return Outcome::Toast {
                    text: "restored the recipe's own values".into(),
                    error: false,
                };
            }
            KeyCode::Char('s') => return Outcome::Launch,
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => {
                self.view = View::Cards;
                self.error = None;
            }
            _ => {}
        }
        Outcome::None
    }

    fn edit_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Enter => self.commit_edit(),
            KeyCode::Esc => {
                // Cancel discards the buffer and leaves the committed value —
                // and any error from a PREVIOUS commit, which is still true.
                self.editing = false;
                self.edit_buffer.clear();
            }
            KeyCode::Backspace => {
                self.edit_buffer.pop();
            }
            KeyCode::Char(c) => self.edit_buffer.push(c),
            _ => {}
        }
        Outcome::None
    }
}

#[cfg(test)]
#[path = "lib_keys_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_keys_more_tests.rs"]
mod more_tests;
