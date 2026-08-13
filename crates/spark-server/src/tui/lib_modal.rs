// SPDX-License-Identifier: AGPL-3.0-only

//! The Config form's pickers: which one is open, and the keys it owns.
//!
//! Split from `lib_config` at the 500-LoC cap when the borrow pickers joined
//! the option and add lists. The split line is state-versus-verbs: the modal
//! enum and its keyboard dispatch live here, while the form edits they commit
//! into (`try_set`, `add_field`'s target rows, removal) stay in `lib_config`.

use crossterm::event::{KeyCode, KeyEvent};

use crate::tui::lib_fields::FieldSpec;

/// Columns of help TEXT in the add-picker's side panel. One number shared by
/// the key handler (scroll clamping) and the renderer (wrapping): if the two
/// wrapped at different widths they would disagree about how far the text
/// scrolls.
pub(crate) const HELP_PANEL_TEXT_W: usize = 30;
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_state::LibState;

/// A picker drawn over the form. While one is open it owns the keyboard,
/// the same contract `editing` has.
#[derive(Clone, Debug)]
pub enum ConfigModal {
    /// Every valid value for `key`.
    Options {
        key: String,
        options: Vec<String>,
        selected: usize,
    },
    /// Every serve flag the form does not already carry.
    Add {
        fields: Vec<&'static FieldSpec>,
        selected: usize,
        /// The side panel's viewport into the highlighted flag's FULL help,
        /// in wrapped lines. J/K moves it; any cursor move resets it, because
        /// a panel still scrolled into the previous flag's help would caption
        /// one flag with another's paragraphs.
        help_scroll: usize,
    },
    /// The recipes whose parameters can be applied over this form.
    Borrow {
        donors: Vec<crate::recipe::Recipe>,
        selected: usize,
    },
    /// What applying the chosen donor would change — shown, then confirmed.
    /// The donor list rides along so Esc steps back to it with the cursor
    /// where it was, rather than throwing the user out of a half-made choice.
    Preview {
        donors: Vec<crate::recipe::Recipe>,
        donor: usize,
        changes: Vec<crate::tui::lib_borrow::BorrowChange>,
        scroll: usize,
    },
}

impl ConfigModal {
    fn len(&self) -> usize {
        match self {
            ConfigModal::Options { options, .. } => options.len(),
            ConfigModal::Add { fields, .. } => fields.len(),
            ConfigModal::Borrow { donors, .. } => donors.len(),
            ConfigModal::Preview { changes, .. } => changes.len(),
        }
    }

    fn selected(&self) -> usize {
        match self {
            ConfigModal::Options { selected, .. }
            | ConfigModal::Add { selected, .. }
            | ConfigModal::Borrow { selected, .. } => *selected,
            // The preview has no cursor, only a viewport: j/k drives `scroll`
            // through the same helpers so the dialect stays one dialect.
            ConfigModal::Preview { scroll, .. } => *scroll,
        }
    }

    fn set_selected(&mut self, i: usize) {
        let n = self.len();
        let clamped = i.min(n.saturating_sub(1));
        match self {
            ConfigModal::Options { selected, .. } | ConfigModal::Borrow { selected, .. } => {
                *selected = clamped;
            }
            ConfigModal::Add {
                selected,
                help_scroll,
                ..
            } => {
                *selected = clamped;
                *help_scroll = 0;
            }
            ConfigModal::Preview { scroll, .. } => *scroll = clamped,
        }
    }

    fn move_selection(&mut self, delta: isize) {
        let next = (self.selected() as isize + delta).max(0) as usize;
        self.set_selected(next);
    }

    /// Scroll the add-picker's help panel. A no-op on every other picker —
    /// none of them has one.
    fn scroll_help(&mut self, delta: isize) {
        let ConfigModal::Add {
            fields,
            selected,
            help_scroll,
        } = self
        else {
            return;
        };
        let Some(spec) = fields.get(*selected) else {
            return;
        };
        // Clamped against the WRAPPED line count at the panel's fixed text
        // width, so K always has something to undo: an unclamped J would bank
        // presses past the end that a reader then pays back one dead K at a
        // time.
        let max = crate::tui::format::wrap_help(&spec.help_full, HELP_PANEL_TEXT_W)
            .len()
            .saturating_sub(1);
        let next = (*help_scroll as isize + delta).clamp(0, max as isize);
        *help_scroll = next as usize;
    }
}

impl LibState {
    /// Keys while a picker is open. Same dialect as every list in the
    /// dashboard: j/k moves, g/G jumps, Enter selects, Esc cancels.
    pub fn modal_key(&mut self, key: KeyEvent) -> Outcome {
        let Some(modal) = self.modal.as_mut() else {
            return Outcome::None;
        };
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => modal.move_selection(1),
            KeyCode::Up | KeyCode::Char('k') => modal.move_selection(-1),
            KeyCode::Char('g') => modal.set_selected(0),
            KeyCode::Char('G') => modal.set_selected(usize::MAX),
            // Shift-j/k: the help panel's own scroll. Lower-case j/k belongs
            // to the list, so the panel takes the shifted pair — the same
            // motion, one modifier over.
            KeyCode::Char('J') => modal.scroll_help(1),
            KeyCode::Char('K') => modal.scroll_help(-1),
            KeyCode::Enter => return self.modal_pick(),
            KeyCode::Esc => self.close_modal(),
            _ => {}
        }
        Outcome::None
    }

    /// Esc: close the picker — except the borrow preview, which steps back to
    /// the donor list it came from. Dropping the user on the form there would
    /// discard a half-made choice a second `b` can only rebuild from scratch.
    fn close_modal(&mut self) {
        self.modal = match self.modal.take() {
            Some(ConfigModal::Preview { donors, donor, .. }) => Some(ConfigModal::Borrow {
                donors,
                selected: donor,
            }),
            _ => None,
        };
    }

    fn modal_pick(&mut self) -> Outcome {
        match self.modal.take() {
            Some(ConfigModal::Options {
                key,
                options,
                selected,
            }) => {
                let Some(value) = options.get(selected) else {
                    return Outcome::None;
                };
                // A failure leaves `error` set and the value out of the form,
                // exactly like a rejected typed commit.
                if self.try_set(&key.clone(), value) {
                    self.select_row(&key);
                }
                Outcome::None
            }
            Some(ConfigModal::Add {
                fields, selected, ..
            }) => match fields.get(selected) {
                Some(spec) => self.add_field(spec),
                None => Outcome::None,
            },
            Some(ConfigModal::Borrow { donors, selected }) => self.pick_donor(donors, selected),
            Some(ConfigModal::Preview {
                donors,
                donor,
                changes,
                ..
            }) => match donors.get(donor) {
                Some(d) => self.apply_borrow(&d.clone(), &changes),
                None => Outcome::None,
            },
            None => Outcome::None,
        }
    }

    fn add_field(&mut self, spec: &'static FieldSpec) -> Outcome {
        match (&spec.default, spec.options.is_empty()) {
            // clap declares a default: land on it, then edit like any row.
            (Some(default), _) => {
                if self.try_set(&spec.key, default) {
                    self.select_row(&spec.key);
                    Outcome::Toast {
                        text: format!("{} added at its default, {default}", spec.key),
                        error: false,
                    }
                } else {
                    Outcome::None
                }
            }
            // No default but a closed set: chain straight into the picker.
            (None, false) => {
                self.modal = Some(ConfigModal::Options {
                    key: spec.key.clone(),
                    options: spec.options.clone(),
                    selected: 0,
                });
                Outcome::None
            }
            // No default, free text: ask for the value before anything is
            // created — an empty override would render `--flag ""`.
            (None, true) => {
                self.pending_add = Some(spec.key.clone());
                self.edit_buffer.clear();
                self.editing = true;
                Outcome::None
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_modal_tests.rs"]
mod tests;
