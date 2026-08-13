// SPDX-License-Identifier: AGPL-3.0-only

//! Borrowing parameters from another recipe, from inside the Config form.
//!
//! Some models have no recipe; others have one whose settings the user does
//! not want. `b` lists the loadable donors — the same Atlas single-node set,
//! family matches first, that `lib_start` offers as starting points, through
//! the same `ranked_donors`, because two copies of that filter is how one of
//! them ends up offering a vLLM donor — and applies the chosen recipe's
//! `defaults:` over the form.
//!
//! Two rules shape everything here:
//!
//! * **Nothing is overwritten unseen.** Enter on a donor opens a preview of
//!   exactly the rows that would change, current value beside incoming, and
//!   only a second Enter commits. A blanket apply would silently discard the
//!   user's own edits; a pick-per-field flow would be sixteen prompts for one
//!   decision. The preview IS the transaction: what it lists is what
//!   `apply_borrow` applies, from the same computed set, so the two cannot
//!   disagree.
//! * **A borrow is not a measurement.** The donor was measured on ITS model;
//!   applied to this one, the values are copies without their evidence. The
//!   form records the provenance in `LibState::borrowed` and says so beside
//!   any existing `starting_point` marker — neither replaces the other,
//!   because "synthesized card" and "borrowed values" are two different
//!   claims and erasing either would present a guess as a measurement.

use std::collections::BTreeMap;

use crate::recipe::{Recipe, schema};
use crate::tui::lib_keys::Outcome;
use crate::tui::lib_modal::ConfigModal;
use crate::tui::lib_state::{LibState, problem_line};

/// One row of the preview: what applying the donor does to `key`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowChange {
    /// The FORM's spelling of the key. A donor's `max_model_len` lands on an
    /// existing `max_seq_len` row when both render to the same flag — matched
    /// through `schema::flag_for`, exactly as the add-picker de-duplicates —
    /// because inserting the donor's spelling beside the recipe's would put
    /// the same flag on the launch line twice.
    pub key: String,
    /// What the form shows now: the effective value, or a word ("removed",
    /// "not set") when there is none. Words, not colours — the preview must
    /// read under NO_COLOR.
    pub from: String,
    pub to: String,
}

/// The rows applying `donor` would change, in donor order. Unchanged keys are
/// left out: a preview padded with no-ops buries the one row that matters.
fn changes_from(
    recipe: &Recipe,
    overrides: &BTreeMap<String, String>,
    removed: &std::collections::BTreeSet<String>,
    donor: &Recipe,
) -> Vec<BorrowChange> {
    donor
        .defaults
        .iter()
        .filter_map(|(donor_key, to)| {
            let key = form_key(recipe, overrides, donor_key);
            if removed.contains(&key) {
                return Some(BorrowChange {
                    key,
                    from: "removed".into(),
                    to: to.clone(),
                });
            }
            let current = overrides.get(&key).or_else(|| recipe.defaults.get(&key));
            match current {
                Some(v) if v == to => None,
                Some(v) => Some(BorrowChange {
                    key,
                    from: v.clone(),
                    to: to.clone(),
                }),
                None => Some(BorrowChange {
                    key,
                    from: "not set".into(),
                    to: to.clone(),
                }),
            }
        })
        .collect()
}

/// The form key a donor key lands on: an existing row whose flag matches, or
/// the donor's own spelling when the form has no such row.
fn form_key(recipe: &Recipe, overrides: &BTreeMap<String, String>, donor_key: &str) -> String {
    let Some(flag) = schema::flag_for(donor_key) else {
        return donor_key.to_string();
    };
    recipe
        .defaults
        .keys()
        .chain(overrides.keys())
        .find(|k| schema::flag_for(k).as_deref() == Some(flag.as_str()))
        .cloned()
        .unwrap_or_else(|| donor_key.to_string())
}

impl LibState {
    /// `b` on the Config form: list the recipes whose parameters can be
    /// applied over it.
    pub fn open_borrow_modal(&mut self) -> Outcome {
        let Some(recipe) = self.config_recipe() else {
            return Outcome::None;
        };
        let current_id = recipe.id.clone();
        let Some(entry) = self.current() else {
            return Outcome::None;
        };
        let model_type = entry
            .local
            .as_ref()
            .map(|l| l.model_type.as_str())
            .unwrap_or_default();
        let donors: Vec<Recipe> =
            super::lib_start::ranked_donors(&self.index.recipes, &entry.model, model_type)
                .into_iter()
                // The form's own recipe is not a donor: taking a recipe's values over
                // itself is what `d` already does, honestly named.
                .filter(|d| d.id != current_id)
                .cloned()
                .collect();
        if donors.is_empty() {
            return Outcome::Toast {
                text: "no other recipe to borrow from".into(),
                error: true,
            };
        }
        self.modal = Some(ConfigModal::Borrow {
            donors,
            selected: 0,
        });
        Outcome::None
    }

    /// Enter on a donor: preview what applying it would change. Nothing is
    /// written here — the preview's second Enter is the commit.
    pub(super) fn pick_donor(&mut self, donors: Vec<Recipe>, selected: usize) -> Outcome {
        let (Some(donor), Some(recipe)) = (donors.get(selected), self.config_recipe()) else {
            return Outcome::None;
        };
        let changes = changes_from(recipe, &self.overrides, &self.removed, donor);
        if changes.is_empty() {
            // Nothing to preview and nothing to apply; saying so beats an
            // empty box the user has to interpret.
            return Outcome::Toast {
                text: format!("the form already matches {}", donor.id),
                error: false,
            };
        }
        self.modal = Some(ConfigModal::Preview {
            donors,
            donor: selected,
            changes,
            scroll: 0,
        });
        Outcome::None
    }

    /// Enter on the preview: apply exactly the changes it showed.
    ///
    /// Validated as a WHOLE config before anything is committed, the same
    /// rule as `commit_edit`: donor flags interact with the rows they land
    /// beside, and a donor measured on another model can name a combination
    /// this recipe's model cannot serve. On failure the form is untouched.
    pub(super) fn apply_borrow(&mut self, donor: &Recipe, changes: &[BorrowChange]) -> Outcome {
        let Some(recipe) = self.config_recipe().cloned() else {
            return Outcome::None;
        };
        let mut overrides = self.overrides.clone();
        let mut removed = self.removed.clone();
        for change in changes {
            // Borrowing re-pins a removed row: the donor names a value for
            // it, and "removed" plus an override is a state the form forbids.
            removed.remove(&change.key);
            if recipe.defaults.get(&change.key) == Some(&change.to) {
                overrides.remove(&change.key);
            } else {
                overrides.insert(change.key.clone(), change.to.clone());
            }
        }
        match recipe.serve_args_edited(&overrides, &removed) {
            Ok(_) => {
                self.overrides = overrides;
                self.removed = removed;
                // Provenance: the donor AND the model its values were
                // measured on, because "borrowed from X" without the model is
                // half the warning.
                self.borrowed = Some(format!("{} (measured on {})", donor.id, donor.model));
                self.error = None;
                Outcome::Toast {
                    text: format!(
                        "{} settings borrowed from {} — d restores the recipe",
                        changes.len(),
                        donor.id
                    ),
                    error: false,
                }
            }
            Err(e) => {
                self.error = Some(problem_line(&format!("{e:#}")));
                Outcome::None
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_borrow_tests.rs"]
mod tests;
