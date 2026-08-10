// SPDX-License-Identifier: AGPL-3.0-only

//! Library state: the joined catalogue, the recipe fetch, and the edit form.
//!
//! Nothing here blocks. The fetch runs on its own `std::thread` and is polled
//! with `try_recv` on the normal tick, so a 20-second GitHub round trip never
//! costs a frame — see `recipe::fetch` for the threading contract.

use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;

use crate::recipe::fetch::{self, Index};
use crate::tui::data::catalogue::{self, Entry};
use crate::tui::data::library::LibraryEntry;

/// Which pane of the Library is showing.
///
/// `List → Cards → Config` — the model, then which recipe for it, then that
/// recipe's settings. A model with ONE recipe still passes through Cards: the
/// card is where the recipe's measured rationale is readable, and the list row
/// has no room for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum View {
    /// The joined list, one row per model.
    #[default]
    List,
    /// The selected model's recipes, as cards.
    Cards,
    /// One recipe's settings, editable before launch.
    Config,
}

#[derive(Default)]
pub struct LibState {
    pub view: View,
    pub index: Index,
    pub rows: Vec<Entry>,
    pub selected: usize,
    pub filter: String,
    pub filter_editing: bool,
    /// Which recipe card is selected, within the current model's row.
    pub card: usize,
    /// Set while a fetch is in flight, for the title spinner.
    pub fetching: bool,
    pending: Option<Receiver<Index>>,
    /// Cancels the in-flight index refresh. Set beside `pending` and cleared
    /// with it, so the two cannot disagree about whether one is running.
    fetch_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// An in-flight scan of the local HF cache.
    pub(super) pending_scan: Option<Receiver<Vec<LibraryEntry>>>,
    /// The local cache needs re-scanning. Set by the reducer, which has no
    /// access to `App`; the event loop drains it into `App::library_dirty`.
    pub mark_dirty: bool,
    /// The recipe id whose date is being looked up, and the channel it will
    /// arrive on. `Some` drives the skeleton placeholder in the detail pane.
    ///
    /// One at a time, and only for the recipe on screen: dating the whole index
    /// costs one rate-limited GitHub call per recipe against a 60/hour limit.
    pub dating: Option<String>,
    pub(super) pending_date: Option<Receiver<(String, Option<String>)>>,
    /// Recipe ids already looked up, so a redraw cannot re-ask. Holds ids whose
    /// lookup FAILED too — an offline box must not spend a thread per frame
    /// rediscovering that it is offline.
    pub(super) dated: std::collections::HashSet<String>,
    /// Dates fetched from GitHub this session, by recipe id.
    ///
    /// Deliberately NOT written back into `index.recipes`: a background refresh
    /// replaces `index` wholesale, which silently discarded them.
    pub(super) fetched_dates: std::collections::HashMap<String, String>,
    /// The loader thread's outcome, if a launch is in flight.
    ///
    /// `launch` returns when the thread is SPAWNED, not when the swap
    /// succeeds, so without this a later failure was a log line and nothing
    /// else — the dashboard kept showing a load that had already given up.
    launch_result: Option<Receiver<String>>,
    /// The store root, so a refresh knows where the cache lives.
    pub(super) root: Option<std::path::PathBuf>,
    /// There is no recipe store to attach, and there will not be one.
    ///
    /// `ArtifactStore::discover()` fails only on process-lifetime facts —
    /// neither `ATLAS_HOME` nor `HOME` set, or `ATLAS_HOME` empty — so a retry
    /// cannot succeed. Without this the tick saw `!attached()` and tried again
    /// ten times a second, warning each time; see `events_rules::tick_work`.
    pub(super) recipes_unavailable: bool,

    // --- config form ---
    /// Edited values, keyed as the recipe keys them. Only touched keys appear,
    /// so an untouched setting always follows the recipe rather than a stale copy.
    pub overrides: BTreeMap<String, String>,
    pub row: usize,
    pub editing: bool,
    pub edit_buffer: String,
    /// Why the current form will not launch, if it will not.
    pub error: Option<String>,
}

impl LibState {
    /// Point the Library at a store and show whatever is already cached.
    ///
    /// The cache read is synchronous and local, so the list is populated on the
    /// first frame; the network only ever improves it.
    pub fn attach(&mut self, root: std::path::PathBuf, local: &[LibraryEntry]) {
        self.index = fetch::cached(&root);
        self.root = Some(root);
        self.rebuild(local);
    }

    /// Start a background refresh. Cheap and idempotent: a second call while
    /// one is in flight is ignored rather than queued.
    pub fn refresh(&mut self) {
        if self.fetching {
            return;
        }
        let Some(root) = &self.root else {
            return;
        };
        let (rx, cancel) = fetch::refresh_in_background(root);
        self.pending = Some(rx);
        self.fetch_cancel = Some(cancel);
        self.fetching = true;
    }

    /// Stop an in-flight refresh, if there is one.
    ///
    /// Called on the way out of the dashboard. Without it, quitting during a
    /// refresh left up to eight workers fetching files for a Library nobody was
    /// going to see, and the process could not finish exiting until the slowest
    /// of them hit the 20 s timeout.
    pub fn cancel_refresh(&self) {
        if let Some(c) = &self.fetch_cancel {
            c.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Poll the fetch. Called once per tick; returns true if the list changed.
    pub fn poll(&mut self, local: &[LibraryEntry]) -> bool {
        let Some(rx) = &self.pending else {
            return false;
        };
        match rx.try_recv() {
            Ok(index) => {
                self.index = index;
                self.pending = None;
                self.fetch_cancel = None;
                self.fetching = false;
                self.rebuild(local);
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            // The fetcher thread died without sending. Stop showing a spinner
            // forever; the cache is still on screen.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending = None;
                self.fetch_cancel = None;
                self.fetching = false;
                false
            }
        }
    }

    /// Re-join, keeping the selection on the MODEL the user had, not the index.
    ///
    /// A background fetch lands on whatever screen the user is on, and the join
    /// re-sorts (runnable rows first, then by id). Clamping an index is not
    /// enough: row 2 before the refresh and row 2 after it can be different
    /// models, so the cards, the settings form, and what `s` launches all
    /// change silently underneath. Anchoring to the model id makes a refresh
    /// invisible when the model survives, and explicit when it does not.
    pub fn rebuild(&mut self, local: &[LibraryEntry]) {
        let anchor = self.current().map(|e| e.model.clone());
        let card_anchor = self.selected_card().map(|r| r.id.clone());
        self.rows = catalogue::join(&self.index.recipes, local);

        if let Some(model) = anchor {
            match self.visible().iter().position(|e| e.model == model) {
                Some(i) => self.selected = i,
                // The model the user was reading about is no longer listed.
                // Staying in its cards would show another model's recipes under
                // its heading, so step back to the list rather than lie.
                None => self.view = View::List,
            }
        }
        // Same argument one level down: a model can keep its row and lose a
        // recipe. If the recipe the user was editing is gone, its edits are
        // edits to nothing — carrying them onto whichever recipe inherits the
        // index would silently launch a config the user never typed.
        if let Some(id) = card_anchor {
            match self.cards().iter().position(|r| r.id == id) {
                Some(i) => self.card = i,
                None => {
                    self.overrides.clear();
                    self.editing = false;
                    if self.view == View::Config {
                        self.view = View::Cards;
                    }
                }
            }
        }
        self.clamp();
    }

    fn clamp(&mut self) {
        let n = self.visible().len();
        self.selected = self.selected.min(n.saturating_sub(1));
        // `card` and `row` index into lists that a refresh can shorten.
        let cards = self.cards().len();
        self.card = self.card.min(cards.saturating_sub(1));
        let rows = self.config_rows().len();
        self.row = self.row.min(rows.saturating_sub(1));
    }

    /// Take the loader's failure, if it reported one.
    pub fn poll_launch(&mut self) -> Option<String> {
        let rx = self.launch_result.as_ref()?;
        match rx.try_recv() {
            Ok(msg) => {
                self.launch_result = None;
                Some(msg)
            }
            // Empty: still loading. Disconnected without a message: the thread
            // finished successfully and dropped the sender.
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.launch_result = None;
                None
            }
        }
    }

    /// Rows passing the filter.
    pub fn visible(&self) -> Vec<&Entry> {
        self.rows
            .iter()
            .filter(|r| r.matches(&self.filter))
            .collect()
    }

    pub fn current(&self) -> Option<&Entry> {
        self.visible().get(self.selected).copied()
    }

    pub fn move_selection(&mut self, delta: isize) {
        let n = self.visible().len();
        if n == 0 {
            return;
        }
        let next = self.selected as isize + delta;
        self.selected = next.clamp(0, n as isize - 1) as usize;
    }

    /// Open the card view for the selected model.
    ///
    /// Refuses rather than opening an empty pane: a row with no recipe has no
    /// cards to show, and a blank pane would imply otherwise. That is the
    /// unchanged behaviour for local-only checkpoints.
    pub fn open_cards(&mut self) -> Result<(), String> {
        let Some(entry) = self.current() else {
            return Err("nothing selected".into());
        };
        if !entry.has_recipe() {
            return Err(format!(
                "{} has no recipe — serve it with explicit flags instead",
                entry.model
            ));
        }
        self.card = 0;
        self.view = View::Cards;
        Ok(())
    }

    /// The recipes of the selected model, in id order.
    pub fn cards(&self) -> &[crate::recipe::Recipe] {
        match self.current() {
            Some(entry) => &entry.recipes,
            None => &[],
        }
    }

    /// The card the cursor is on.
    pub fn selected_card(&self) -> Option<&crate::recipe::Recipe> {
        self.cards().get(self.card)
    }

    /// Open the config form for the selected card.
    pub fn open_config(&mut self) -> Result<(), String> {
        let Some(recipe) = self.selected_card() else {
            return Err("nothing selected".into());
        };
        if !recipe.is_atlas() {
            return Err(format!(
                "{} is a {} recipe and cannot be configured here",
                recipe.id,
                recipe.runtime.as_deref().unwrap_or("non-atlas")
            ));
        }
        self.overrides.clear();
        self.error = None;
        self.row = 0;
        self.editing = false;
        self.view = View::Config;
        Ok(())
    }

    /// The recipe backing the config form, if it is open on one.
    pub fn config_recipe(&self) -> Option<&crate::recipe::Recipe> {
        self.selected_card()
    }

    /// The form's rows: every recipe key, in recipe order, with the effective
    /// value (override if edited, recipe value otherwise).
    pub fn config_rows(&self) -> Vec<(String, String, bool)> {
        let Some(recipe) = self.config_recipe() else {
            return Vec::new();
        };
        recipe
            .defaults
            .iter()
            .map(|(key, value)| {
                let edited = self.overrides.get(key);
                (
                    key.clone(),
                    edited.unwrap_or(value).clone(),
                    edited.is_some(),
                )
            })
            .collect()
    }

    /// Commit the edit buffer into the overrides, validating the whole recipe.
    ///
    /// Validation is of the WHOLE config, not the one field: flags interact
    /// (`--ep-size` against `--world-size`, KV dtype against high-precision
    /// layers), so a per-field check would accept combinations that cannot serve.
    pub fn commit_edit(&mut self) {
        let rows = self.config_rows();
        let Some((key, _, _)) = rows.get(self.row) else {
            return;
        };
        let key = key.clone();
        let raw = self.edit_buffer.trim().to_string();
        self.editing = false;
        if raw.is_empty() {
            self.error = Some(format!("{key} must not be empty"));
            return;
        }
        let Some(recipe) = self.config_recipe().cloned() else {
            return;
        };
        let mut candidate = self.overrides.clone();
        candidate.insert(key.clone(), raw);
        match recipe.serve_args(&candidate) {
            Ok(_) => {
                self.overrides = candidate;
                self.error = None;
            }
            // Keep the rejected value out of the form: an invalid override that
            // stays visible reads as accepted.
            Err(e) => self.error = Some(problem_line(&format!("{e:#}"))),
        }
    }

    /// Drop every edit and return to the recipe's own values.
    pub fn reset_overrides(&mut self) {
        self.overrides.clear();
        self.error = None;
    }

    /// Start the selected recipe, replacing whatever is running.
    ///
    /// Runs on a worker thread: a load is minutes long and the dashboard must
    /// stay responsive — the render loop is what draws the progress the user is
    /// waiting on. The load emits the same `progress::phase` events a first
    /// boot does, so Main's checklist renders it with no extra plumbing, which
    /// is why "start a model" and "swap a model" look identical to the user.
    pub fn launch(
        &mut self,
        host: std::sync::Arc<crate::main_modules::model_host::ModelHost>,
    ) -> Result<(), String> {
        let recipe = self.config_recipe().ok_or("no recipe selected")?;
        // Refuse a launch whose weights are not on disk, HERE, rather than
        // spawning a swap that will fail in `resolve_model_dir` seconds later.
        // The old path tore down whatever was serving, reset the checklist,
        // sent the user to Main and only then said it could not find the
        // model — with nothing to do about it. The weights are downloadable
        // now, so the honest answer is to say so before anything is torn down.
        if !self.selected_has_weights() {
            return Err(format!(
                "{} is not downloaded — press Esc, then d to download it",
                recipe.model
            ));
        }
        // Build and validate argv HERE, on the UI thread, so a bad recipe is a
        // toast rather than a torn-down model: `swap` re-validates, but by then
        // the user has already been told the run started.
        let args = recipe
            .serve_args(&self.overrides)
            .map_err(|e| problem_line(&format!("{e:#}")))?;
        let previous = host.current().map(|s| s.model_name.clone());
        // The loader reports back. `launch` returns as soon as the thread is
        // SPAWNED, so a swap that fails afterwards — no compiled kernels for
        // that model, a shutdown, or the load itself — used to be visible only
        // as a log line, while the dashboard sat on a reset 0/12 checklist and
        // a LOADING pill for a load that had already given up.
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        self.launch_result = Some(rx);
        std::thread::Builder::new()
            .name("atlas-swap".into())
            .spawn(move || {
                if let Err(e) = crate::main_modules::model_swap::swap(&host, args) {
                    // The host reports the failure honestly (503, /health
                    // "loading"); this line is what says WHY in the log ring.
                    tracing::error!(
                        "swap failed{}: {e:#}",
                        previous
                            .as_deref()
                            .map(|p| format!(" (was serving {p})"))
                            .unwrap_or_default()
                    );
                    // A disconnected receiver means the dashboard moved on.
                    let _ = tx.send(format!("{e:#}"));
                }
            })
            .map_err(|e| format!("could not start the loader thread: {e}"))?;
        Ok(())
    }

    /// Are the selected model's weights actually on disk and complete?
    ///
    /// `has_weights` means "the resolver would load this" — a real shard AND a
    /// published `refs/main` — so a half-finished download reads as false,
    /// which is what makes the refusal above correct rather than merely
    /// cautious.
    pub fn selected_has_weights(&self) -> bool {
        self.current().is_some_and(|e| e.has_weights())
    }

    /// The argv this form would launch, for the confirm line.
    pub fn preview_argv(&self) -> Option<Vec<String>> {
        self.config_recipe()?.argv(&self.overrides).ok()
    }
}

/// Reduce a validation report to the one line a form field can show.
///
/// The validator's report is a header, then `  [1] <what>` / `why:` / `fix:`.
/// Taking the FIRST line yields only "Atlas CLI: 1 invalid flag combination",
/// which tells the reader nothing they did not already know — the actionable
/// part is `what`, and `fix` when it fits. clap's own errors have no `[1]`
/// block, so those fall back to their first `error:` line.
pub(crate) fn problem_line(s: &str) -> String {
    let lines: Vec<&str> = s.lines().map(str::trim).collect();
    let what = lines
        .iter()
        .find_map(|l| l.strip_prefix("[").and_then(|r| r.split_once("] ")))
        .map(|(_, what)| what);
    let fix = lines
        .iter()
        .find_map(|l| l.strip_prefix("fix: "))
        .filter(|f| !f.is_empty());
    match (what, fix) {
        (Some(what), Some(fix)) => format!("{what} — {fix}"),
        (Some(what), None) => what.to_string(),
        (None, _) => lines
            .iter()
            .find(|l| !l.is_empty() && !l.starts_with("Atlas CLI:"))
            .unwrap_or(&"invalid")
            .to_string(),
    }
}

#[cfg(test)]
#[path = "lib_state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_state_more_tests.rs"]
mod more_tests;

#[cfg(test)]
#[path = "lib_state_recipe_tests.rs"]
mod recipe_tests;

#[cfg(test)]
#[path = "lib_state_list_tests.rs"]
mod list_tests;
