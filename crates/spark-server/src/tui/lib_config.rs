// SPDX-License-Identifier: AGPL-3.0-only

//! The Config form's editing model: rows, the option picker, adding and
//! removing settings. Split from `lib_state` at the 500-LoC cap; the fields
//! stay on [`LibState`], the behaviour lives here.
//!
//! Three verbs beyond plain typing:
//! - **Pick**: Enter on a field with a closed value set opens a picker
//!   instead of a text buffer. The set comes from `lib_fields`, which reads
//!   the same lists the CLI validator enforces — the picker can offer only
//!   what a launch accepts, by construction rather than by review.
//! - **Add**: `a` lists every serve flag the form does not already carry,
//!   read out of clap at runtime. An added setting lands at clap's own
//!   default; a flag with no default asks for a value before anything is
//!   created, because inventing one here would be a second, silently
//!   divergent default.
//! - **Remove**: `x` un-pins a recipe setting. The flag is NOT PASSED and the
//!   server's default applies — which is not "disabled", and the form says
//!   so. The row stays visible, greyed, and `x` (or Enter) restores it.

use crate::tui::lib_fields::{self, FieldSpec};
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::lib_state::{LibState, problem_line};

/// One row of the form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigRow {
    pub key: String,
    /// The effective value: override if edited, recipe value otherwise. For
    /// a removed row this is the recipe value a restore would return to.
    pub value: String,
    pub changed: bool,
    pub removed: bool,
    /// Not in the recipe's `defaults:` — added in this form.
    pub added: bool,
}

/// The closed value set for a form key, or `None` for free text.
fn field_options(key: &str) -> Option<Vec<String>> {
    let spec = lib_fields::spec_for_key(key)?;
    (!spec.options.is_empty()).then(|| spec.options.clone())
}

impl LibState {
    /// The form's rows: every recipe key in recipe order, then the added
    /// settings. Removed rows STAY listed — a row that vanished would read as
    /// never having existed, and the point of removal is that it is visible
    /// and reversible.
    pub fn config_rows(&self) -> Vec<ConfigRow> {
        let Some(recipe) = self.config_recipe() else {
            return Vec::new();
        };
        let mut rows: Vec<ConfigRow> = recipe
            .defaults
            .iter()
            .map(|(key, value)| {
                let edited = self.overrides.get(key);
                ConfigRow {
                    key: key.clone(),
                    value: edited.unwrap_or(value).clone(),
                    changed: edited.is_some(),
                    removed: self.removed.contains(key),
                    added: false,
                }
            })
            .collect();
        rows.extend(
            self.overrides
                .iter()
                .filter(|(key, _)| !recipe.defaults.contains_key(*key))
                .map(|(key, value)| ConfigRow {
                    key: key.clone(),
                    value: value.clone(),
                    changed: true,
                    removed: false,
                    added: true,
                }),
        );
        rows
    }

    /// Enter on the current row: restore it if removed, open the picker for
    /// a closed-set field, the text editor otherwise.
    pub fn open_value_editor(&mut self) -> Outcome {
        let Some(row) = self.config_rows().into_iter().nth(self.row) else {
            return Outcome::None;
        };
        if row.removed {
            // Enter means "I want this row back in play"; making the user
            // find a second key first is a step for nothing.
            return self.restore_key(&row.key);
        }
        match field_options(&row.key) {
            Some(options) => {
                let selected = options.iter().position(|o| *o == row.value).unwrap_or(0);
                self.modal = Some(ConfigModal::Options {
                    key: row.key,
                    options,
                    selected,
                });
            }
            None => {
                // Seed the buffer with the current value: editing a setting
                // is usually adjusting it, not retyping it.
                self.edit_buffer = row.value;
                self.editing = true;
            }
        }
        Outcome::None
    }

    /// `a`: list what can still be added — every serve flag, minus the ones
    /// already on the form (matched through the flag they render to, so a
    /// recipe's `max_model_len` also blocks `max_seq_len`, its rename).
    pub fn open_add_modal(&mut self) -> Outcome {
        if self.config_recipe().is_none() {
            return Outcome::None;
        }
        let present: Vec<String> = self
            .config_rows()
            .iter()
            .filter_map(|r| crate::recipe::schema::flag_for(&r.key))
            .collect();
        let fields: Vec<&'static FieldSpec> = lib_fields::serve_fields()
            .iter()
            .filter(|s| !present.contains(&s.flag))
            .collect();
        if fields.is_empty() {
            return Outcome::Toast {
                text: "every serve flag is already on the form".into(),
                error: false,
            };
        }
        self.modal = Some(ConfigModal::Add {
            fields,
            selected: 0,
            help_scroll: 0,
        });
        Outcome::None
    }

    /// `x`: remove the setting under the cursor, or bring it back.
    pub fn toggle_removed(&mut self) -> Outcome {
        let Some(row) = self.config_rows().into_iter().nth(self.row) else {
            return Outcome::None;
        };
        if row.added {
            // An added setting has no recipe value to grey out against;
            // removing it is simply un-adding it.
            self.overrides.remove(&row.key);
            self.error = None;
            self.row = self.row.min(self.config_rows().len().saturating_sub(1));
            return Outcome::Toast {
                text: format!("{} removed", row.key),
                error: false,
            };
        }
        if row.removed {
            return self.restore_key(&row.key);
        }
        // Validate the removal against the WHOLE config before applying it:
        // flags interact, and dropping one can strand another (the same
        // reason `commit_edit` validates whole-recipe).
        let Some(recipe) = self.config_recipe().cloned() else {
            return Outcome::None;
        };
        let mut overrides = self.overrides.clone();
        overrides.remove(&row.key);
        let mut removed = self.removed.clone();
        removed.insert(row.key.clone());
        match recipe.serve_args_edited(&overrides, &removed) {
            Ok(_) => {
                self.overrides = overrides;
                self.removed = removed;
                self.error = None;
                // Name what actually happens next, in the toast as well as on
                // the row: "removed" without the consequence reads as "off".
                let text = match lib_fields::spec_for_key(&row.key).and_then(|s| s.default.clone())
                {
                    Some(d) => format!("{} removed — server default {d} applies", row.key),
                    None => format!("{} removed — the flag is not passed", row.key),
                };
                Outcome::Toast { text, error: false }
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                Outcome::None
            }
        }
    }

    fn restore_key(&mut self, key: &str) -> Outcome {
        let Some(recipe) = self.config_recipe().cloned() else {
            return Outcome::None;
        };
        let mut removed = self.removed.clone();
        removed.remove(key);
        // Restoring returns to the recipe's own value; validated anyway,
        // because an override on ANOTHER row may only be legal without this
        // flag (the combinations are the validator's whole subject).
        match recipe.serve_args_edited(&self.overrides, &removed) {
            Ok(_) => {
                self.removed = removed;
                self.error = None;
                Outcome::Toast {
                    text: format!("{key} restored"),
                    error: false,
                }
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                Outcome::None
            }
        }
    }

    /// Commit the edit buffer, validating the whole recipe.
    ///
    /// Validation is of the WHOLE config, not the one field: flags interact
    /// (`--ep-size` against `--world-size`, KV dtype against high-precision
    /// layers), so a per-field check would accept combinations that cannot
    /// serve.
    pub fn commit_edit(&mut self) {
        self.editing = false;
        let raw = self.edit_buffer.trim().to_string();
        if let Some(key) = self.pending_add.take() {
            if raw.is_empty() {
                // The add is abandoned rather than half-made: no override, no
                // row, only the reason.
                self.error = Some(format!("{key} was not added — no value given"));
                return;
            }
            if self.try_set(&key, &raw) {
                self.select_row(&key);
            }
            return;
        }
        let Some(row) = self.config_rows().into_iter().nth(self.row) else {
            return;
        };
        if raw.is_empty() {
            self.error = Some(format!("{} must not be empty", row.key));
            return;
        }
        // Keep the rejected value out of the form: an invalid override that
        // stays visible reads as accepted. `try_set` already does.
        self.try_set(&row.key, &raw);
    }

    /// Leave editing with nothing committed. A half-typed ADD evaporates
    /// entirely — its key exists nowhere but `pending_add`.
    pub fn cancel_edit(&mut self) {
        self.editing = false;
        self.edit_buffer.clear();
        self.pending_add = None;
    }

    /// Validate `overrides ∪ {key: raw}` and commit it if the server would
    /// accept it. On failure the form keeps its previous state and `error`
    /// says why.
    pub(super) fn try_set(&mut self, key: &str, raw: &str) -> bool {
        let Some(recipe) = self.config_recipe().cloned() else {
            return false;
        };
        let mut candidate = self.overrides.clone();
        candidate.insert(key.to_string(), raw.to_string());
        match recipe.serve_args_edited(&candidate, &self.removed) {
            Ok(_) => {
                self.overrides = candidate;
                self.error = None;
                true
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                false
            }
        }
    }

    /// Put the cursor on `key`'s row — an added setting lands at the end of
    /// the form, and leaving the cursor where it was would put the follow-up
    /// Enter on some unrelated row.
    pub(super) fn select_row(&mut self, key: &str) {
        if let Some(i) = self.config_rows().iter().position(|r| r.key == key) {
            self.row = i;
        }
    }

    /// Drop every edit, removal, half-typed addition and borrowed value; back
    /// to the recipe. `borrowed` clears WITH the values it described — a
    /// provenance line outliving the values would warn about settings that
    /// are no longer on the form.
    pub fn reset_overrides(&mut self) {
        self.overrides.clear();
        self.removed.clear();
        self.pending_add = None;
        self.borrowed = None;
        self.error = None;
    }

    /// The argv this form would launch, for the preview line. Derived from
    /// the LIVE edit state — additions included, removals absent — because a
    /// preview built from anything else is a preview of a different launch.
    pub fn preview_argv(&self) -> Option<Vec<String>> {
        self.config_recipe()?
            .argv_edited(&self.overrides, &self.removed)
            .ok()
    }
}

#[cfg(test)]
#[path = "lib_config_tests.rs"]
mod tests;
