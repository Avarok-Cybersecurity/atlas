// SPDX-License-Identifier: AGPL-3.0-only

//! Library actions that need more than the Library's own state.
//!
//! The Library reducer (`lib_keys`) is deliberately pure: it has no cache
//! directory, no server handle and no way to spawn work, so it returns an
//! `Outcome` and something that does have those performs it. That something is
//! `App`, and these are the three it performs — download, cancel, and the
//! freshness check.
//!
//! Split out of `app.rs` when it reached the 500-LoC cap; they belong together
//! because they share the cache-root lookup and the "which model is selected"
//! question, and getting either wrong sends a multi-gigabyte request at the
//! wrong repository.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::{App, MainSub};
use super::section::Section;

impl App {
    /// The HF cache root these actions operate on.
    pub(super) fn cache_root(&self) -> Option<std::path::PathBuf> {
        crate::model_resolver::resolve_cache_root(self.args.cache_dir.as_deref()).ok()
    }

    /// Which model the Library is pointing at, for a download or a check.
    pub(super) fn selected_model(&self) -> Option<String> {
        self.lib.current().map(|e| e.model.clone())
    }

    /// Download, resume, or update the selected model.
    pub(super) fn download_selected_model(&mut self) {
        let Some(model) = self.selected_model() else {
            self.toast("no model selected".to_string(), true);
            return;
        };
        let Some(root) = self.cache_root() else {
            self.toast(
                "no HuggingFace cache directory to download into".to_string(),
                true,
            );
            return;
        };
        // A second download is refused by `DownloadState::start` with a
        // message, which is correct as the backstop but a dead end as UX: it
        // names the running job and leaves the user to work out that `x` on
        // ANOTHER row is the way forward. Ask instead.
        if let Some(running) = self.download.job.as_ref().map(|j| j.repo.clone())
            && running != model
        {
            self.download_switch = Some((running, model));
            return;
        }
        let (text, error) = self.download.start(&model, root);
        self.toast(text, error);
    }

    /// Answer the "one download at a time" question.
    ///
    /// Only an affirmative goes through; every other key keeps the running
    /// download. Same rule and the same reason as `answer_quit_prompt`: the
    /// reflex a user learns on one confirmation has to be safe on the other,
    /// so a fat-finger can never throw away a transfer in progress.
    ///
    /// `x` because the help overlay already teaches it as "stop the running
    /// download"; `y` because the quit prompt already taught the affirmative.
    pub(super) fn answer_download_switch(&mut self, key: KeyEvent) -> bool {
        let Some((running, wanted)) = self.download_switch.take() else {
            return false;
        };
        if matches!(
            key.code,
            KeyCode::Char('x') | KeyCode::Char('y') | KeyCode::Char('Y')
        ) {
            // `cancel` returns None when there was nothing to stop — the job
            // can settle between the question being asked and answered.
            if let Some((text, error)) = self.download.cancel() {
                self.toast(text, error);
            }
            // Queue rather than start: the slot is not free until the worker
            // acknowledges, and starting now would break the one-job rule.
            self.pending_start = Some(wanted.clone());
            self.toast(format!("{wanted} starts when {running} settles"), false);
        }
        true
    }

    /// Start a queued download once the slot is actually free.
    ///
    /// Called from the event loop after `pump`. Covers every way A can end —
    /// cancelled, abandoned, failed, or finished on its own — because the
    /// user's expressed intent was B either way.
    pub(super) fn start_pending_download(&mut self) {
        if self.download.job.is_some() {
            return;
        }
        let Some(model) = self.pending_start.take() else {
            return;
        };
        let Some(root) = self.cache_root() else {
            self.toast(
                "no HuggingFace cache directory to download into".to_string(),
                true,
            );
            return;
        };
        let (text, error) = self.download.start(&model, root);
        self.toast(text, error);
    }

    /// Ask the Hub whether the selected model is behind.
    pub(super) fn check_selected_model(&mut self) {
        let Some(model) = self.selected_model() else {
            self.toast("no model selected".to_string(), true);
            return;
        };
        let Some(root) = self.cache_root() else {
            self.toast("no HuggingFace cache directory".to_string(), true);
            return;
        };
        let (text, error) = self.download.check(&model, root);
        self.toast(text, error);
    }

    /// First entry into the Library: point it at the recipe store and start the
    /// one GitHub fetch it is allowed.
    ///
    /// Lives here rather than in the tick because it needs `App::library` — the
    /// locally-scanned weights the recipe index is joined against.
    ///
    /// The failure path is a DEAD END, and says so by setting
    /// `recipes_unavailable`. `discover()` reads `ATLAS_HOME`/`HOME` and
    /// nothing else, so a second call in the same process answers the same way;
    /// the tick used to retry it at 10 Hz on `!attached()` alone, warning into
    /// the log ring every 100 ms. The local scan still renders — that is what
    /// the `rebuild` is for.
    pub(super) fn attach_recipes(&mut self) {
        match atlas_plugin::ArtifactStore::discover() {
            Ok(store) => {
                self.lib.attach(store.root().to_path_buf(), &self.library);
                self.lib.refresh();
            }
            Err(e) => {
                tracing::warn!("recipes unavailable: {e:#}");
                self.lib.recipes_unavailable = true;
                self.lib.rebuild(&self.library);
            }
        }
    }

    /// Start the configured recipe and follow it to Main.
    ///
    /// The jump is the point: a load is what the operator wants to watch, and
    /// Main is where the 12-phase checklist already lives. Resetting the
    /// progress model first is what makes the SECOND load render as a load —
    /// without it every phase is still `Done` from the first one.
    pub(super) fn launch_selected_recipe(&mut self) {
        let Some(host) = self.host.clone() else {
            self.toast("no server attached to this dashboard".to_string(), true);
            return;
        };
        match self.lib.launch(host) {
            Ok(()) => {
                self.progress.reset();
                // A load is now genuinely in flight, so the checklist is
                // tracking something again and the pill may say LOADING.
                self.awaiting_model = false;
                self.repaint = true;
                self.section = Section::Main;
                self.main_sub = MainSub::Overview;
            }
            Err(e) => self.toast(e, true),
        }
    }
}

#[cfg(test)]
#[path = "app_library_tests.rs"]
mod tests;
