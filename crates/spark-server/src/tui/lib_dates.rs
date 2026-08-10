// SPDX-License-Identifier: AGPL-3.0-only

//! When a recipe was last updated, and where that answer comes from.
//!
//! Two sources, in order. `metadata.updated` in the recipe file is
//! authoritative — it is what the author wrote. For a recipe that carries none,
//! which today is all of them, GitHub's commit history for that file is the
//! fallback.
//!
//! ## Why the fallback is lazy
//!
//! Dating a recipe costs one `api.github.com` call, and that API allows **60
//! per hour unauthenticated**. Dating the whole index would spend 25 of them
//! per refresh, so two refreshes would exhaust the budget the Library itself
//! depends on. So exactly one recipe is dated at a time, only the one whose
//! details are on screen, only if it has no date of its own, and only once per
//! session — including when the lookup fails, so an offline box does not spend
//! a thread per frame rediscovering that it is offline.
//!
//! ## Threading
//!
//! The same contract as `recipe::fetch`, which this calls into: the render
//! thread never polls a future, only `try_recv`s. The work happens on a plain
//! `std::thread`.

use super::lib_state::{LibState, View};

use crate::recipe::fetch;

impl LibState {
    /// The `updated` row's value for one recipe: its date, a skeleton while
    /// that date is being looked up, or empty — which both detail panes render
    /// as no row at all. Shared rather than written twice, so a skeleton cannot
    /// appear in one pane and not the other.
    ///
    /// Three sources, in precedence order, and the order matters:
    ///
    /// 1. `metadata.updated` from the recipe file — what the author wrote.
    /// 2. A date fetched from GitHub this session, held in `fetched_dates`.
    /// 3. A skeleton, while that fetch is in flight.
    ///
    /// The fetched date is kept in its own map rather than written back into
    /// `index.recipes`, because **the index is replaceable**: a background
    /// refresh finishing later does `self.index = index` (`LibState::poll`) and
    /// silently discarded a date written into the old one. That was invisible
    /// in tests — every unit test and the end-to-end network test passed —
    /// and only showed up running the real dashboard, where the refresh lands
    /// seconds after the date does. A map the refresh does not own cannot be
    /// clobbered by it.
    pub fn date_text(&self, recipe: &crate::recipe::Recipe) -> String {
        if !recipe.updated.is_empty() {
            return recipe.updated.clone();
        }
        if let Some(d) = self.fetched_dates.get(&recipe.id) {
            return d.clone();
        }
        if self.dating.as_deref() == Some(recipe.id.as_str()) {
            // A placeholder of the width a date will occupy, so the row does
            // not jump when the real value lands.
            return "░░░░░░░░░░".to_string();
        }
        String::new()
    }

    /// The recipe whose details are on screen right now, if any.
    ///
    /// Which one that is depends on the pane: the list's detail pane describes
    /// the row's `primary()` recipe, while Cards and Config describe the
    /// selected card. Dating anything else would spend a rate-limited request
    /// on a recipe nobody is looking at.
    pub fn visible_recipe_id(&self) -> Option<String> {
        match self.view {
            View::List => self
                .current()
                .and_then(|e| e.primary())
                .map(|r| r.id.clone()),
            View::Cards | View::Config => self.selected_card().map(|r| r.id.clone()),
        }
    }

    /// Ask GitHub when the selected recipe was last changed, if it has no date.
    ///
    /// Called from the render tick, so it must be cheap and idempotent: it does
    /// nothing unless the recipe genuinely lacks a `metadata.updated`, nothing
    /// is already in flight, and this id has not been asked about before.
    pub fn want_date_for(&mut self, id: &str) {
        if self.pending_date.is_some() || self.dated.contains(id) {
            return;
        }
        // A recipe that states its own date is authoritative — never override
        // the file with a commit timestamp, which dates the last edit to the
        // file rather than the recipe.
        let needs = self
            .index
            .recipes
            .iter()
            .any(|r| r.id == id && r.updated.is_empty());
        if !needs {
            return;
        }
        tracing::debug!("dating recipe {id} from GitHub commit history");
        self.dated.insert(id.to_string());
        self.dating = Some(id.to_string());
        self.pending_date = Some(fetch::updated_in_background(id));
    }

    /// Collect a finished date lookup. Returns true when something changed.
    ///
    /// Stores into `fetched_dates`, which no index refresh owns — see
    /// [`LibState::date_text`] for why writing it into the recipe was wrong.
    /// That also means no `rebuild` is needed: the render reads the map, not a
    /// cloned field on the joined row.
    pub fn poll_date(&mut self) -> bool {
        let Some(rx) = &self.pending_date else {
            return false;
        };
        match rx.try_recv() {
            Ok((id, date)) => {
                self.pending_date = None;
                self.dating = None;
                // The id is the one the worker echoed back, never the current
                // selection: by now the user may have moved on, and applying a
                // date to whatever is selected then would be silently wrong.
                if let Some(d) = date.filter(|d| !d.is_empty()) {
                    self.fetched_dates.insert(id, d);
                    return true;
                }
                false
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            // The thread died without sending. Stop the skeleton; the id stays
            // in `dated`, so this is not retried on the next frame.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_date = None;
                self.dating = None;
                false
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_dates_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_dates_more_tests.rs"]
mod more_tests;
