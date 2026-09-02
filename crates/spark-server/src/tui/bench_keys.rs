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
            View::Variants => {
                self.variants_key(key);
                Outcome::None
            }
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
            // Enter on the benchmark that is CURRENTLY RUNNING goes to its
            // run, not to a form it cannot start from. Anything else goes
            // through the variant step when the benchmark declares variants,
            // and straight to the form when it does not.
            KeyCode::Enter | KeyCode::Right | KeyCode::Char('l') => {
                if self.is_running() && self.running_id == self.descriptor().map(|d| d.id) {
                    self.view = View::Run;
                } else {
                    self.enter_selected();
                }
            }
            // A finished run stays reachable after you navigate away from it.
            KeyCode::Char('v') if self.frame.is_some() => self.view = View::Run,
            // The pair the help overlay advertises globally. The list scrolls
            // by keeping the selection in view, so first/last ARE top/bottom.
            KeyCode::Char('g') | KeyCode::Home if n > 0 => self.select(0),
            KeyCode::Char('G') | KeyCode::End if n > 0 => self.select(n - 1),
            // Paged by the renderer-published viewport and clamped at both
            // ends; `select` re-clamps, so a stale page size from a resize
            // cannot walk the selection off the registry.
            KeyCode::PageDown if n > 0 => {
                let page = self.suite_page.get().max(1);
                self.select((self.selected + page).min(n - 1));
            }
            KeyCode::PageUp => {
                let page = self.suite_page.get().max(1);
                self.select(self.selected.saturating_sub(page));
            }
            _ => {}
        }
        Outcome::None
    }

    fn params_key(&mut self, key: KeyEvent) -> Outcome {
        // The pre-flight owns the keyboard while it is up: it is a decision,
        // and letting the form underneath react would edit a field the user
        // cannot currently see.
        if self.preflight.is_some() {
            return self.preflight_key(key);
        }
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
            // Back retraces the way in: through the variant step when this
            // benchmark has one, to the list when it does not.
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => {
                self.view = if self.variants.is_empty() {
                    View::List
                } else {
                    View::Variants
                };
            }
            // Reset the form to the schema's defaults.
            KeyCode::Char('d') => {
                let selected = self.selected;
                self.select(selected);
            }
            // Toggle the pre-run coherence probe. A base (non-instruct)
            // checkpoint cannot answer the questions but is still a valid
            // latency target.
            KeyCode::Char('p') => {
                self.coherence = match self.coherence {
                    atlas_plugin::CoherencePolicy::Probe => atlas_plugin::CoherencePolicy::Skip,
                    atlas_plugin::CoherencePolicy::Skip => atlas_plugin::CoherencePolicy::Probe,
                };
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
        match self.begin_start() {
            // No toast on success: starting switches to the Run view, which
            // already names the benchmark and its phase. A toast here was both
            // redundant and drawn on top of the progress bar it was announcing.
            Ok(()) => Outcome::None,
            Err(e) => Outcome::Toast {
                text: e,
                error: true,
            },
        }
    }

    /// While checking, only Esc does anything — there is nothing to decide yet.
    /// Once there is a concern, `p` proceeds and anything else goes back.
    fn preflight_key(&mut self, key: KeyEvent) -> Outcome {
        let checking = self
            .preflight
            .as_ref()
            .is_some_and(crate::tui::bench_preflight::Preflight::is_checking);
        if checking {
            if key.code == KeyCode::Esc {
                self.cancel_preflight();
            }
            return Outcome::None;
        }
        match key.code {
            KeyCode::Char('p') | KeyCode::Char('P') | KeyCode::Enter => {
                match self.accept_preflight() {
                    Ok(()) => Outcome::None,
                    Err(e) => Outcome::Toast {
                        text: e,
                        error: true,
                    },
                }
            }
            _ => {
                self.cancel_preflight();
                Outcome::None
            }
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
            // No toast: `cancel` sets the pane's own status line, and a toast
            // drawn at the top of the content area lands on the progress bar.
            // Toasts here are reserved for what the pane cannot already say.
            KeyCode::Char('c') => self.cancel(),
            KeyCode::Esc | KeyCode::Left | KeyCode::Char('h') => self.view = View::List,
            // Clamped to the renderer-published ceiling: `draw_table` clamps
            // the DISPLAY, so an unclamped offset banked invisible presses
            // that were paid back one dead `k` at a time.
            KeyCode::Down | KeyCode::Char('j') => {
                self.table_scroll = (self.table_scroll + 1).min(self.table_scroll_max.get());
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.table_scroll = self.table_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') | KeyCode::Home => self.table_scroll = 0,
            KeyCode::Char('G') | KeyCode::End => {
                self.table_scroll = self.table_scroll_max.get();
            }
            _ => {}
        }
        Outcome::None
    }

    fn history_key(&mut self, key: KeyEvent) -> Outcome {
        let n = self.history.len();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') if n > 0 => {
                self.history_row = (self.history_row + 1).min(n - 1);
                // The offset described ANOTHER run's table; carrying it over
                // would open the next run scrolled to a random depth.
                self.history_table_scroll = 0;
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.history_row = self.history_row.saturating_sub(1);
                self.history_table_scroll = 0;
            }
            // A stored table is the same 40-row sweep the live Run view
            // scrolls with j/k; here j/k are spent on run selection, so the
            // page pair does the moving. Without it, rows below the pane
            // height were unreadable anywhere in the TUI.
            KeyCode::PageDown => {
                self.history_table_scroll =
                    (self.history_table_scroll + 5).min(self.history_table_scroll_max.get());
            }
            KeyCode::PageUp => {
                self.history_table_scroll = self.history_table_scroll.saturating_sub(5);
            }
            // Make a shareable card of the selected run.
            //
            // The card is rendered from the COMMITTED gate record for this
            // benchmark, not from the history entry beside it. A `RunRecord`
            // carries no hardware and no commit sha, and a card exists to print
            // exactly those beside the number — inventing them would produce the
            // artefact the card was designed to prevent.
            KeyCode::Char('c') if n > 0 => return self.export_card(),
            _ => {}
        }
        Outcome::None
    }
}

#[cfg(test)]
#[path = "bench_keys_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bench_keys_more_tests.rs"]
mod more_tests;
