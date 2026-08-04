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
