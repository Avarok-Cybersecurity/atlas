// SPDX-License-Identifier: AGPL-3.0-only

//! Scanning the local HuggingFace cache for the Library.
//!
//! Split from `lib_state.rs` at the 500-LoC cap. It is a unit on its own: the
//! scan is the only part of the Library that reads the DISK rather than the
//! recipe index, and it is the part a finished download has to re-run.
//!
//! It runs off the render thread through `tui::worker` — it stats every blob
//! directory, which on a cache holding a few dozen multi-gigabyte checkpoints
//! is not something to do between frames.

use super::lib_state::LibState;
use crate::tui::data::library::LibraryEntry;

impl LibState {
    /// Has the recipe store been attached yet?
    ///
    /// Distinguishes "first entry into the Library" from "rescan": the local
    /// scan may run many times, but attaching — and the GitHub fetch it kicks
    /// off — happens once.
    pub fn attached(&self) -> bool {
        self.root.is_some()
    }

    /// Has attaching already been tried and found impossible?
    pub fn recipes_unavailable(&self) -> bool {
        self.recipes_unavailable
    }

    /// Is a background scan running?
    ///
    /// A scan reports the cache as it was when it STARTED, so anything that
    /// dirties the cache while one is in flight needs a LATER scan, not this
    /// one — which is why the caller asks before clearing its dirty flag.
    pub fn scan_in_flight(&self) -> bool {
        self.pending_scan.is_some()
    }

    /// Start a background scan of the local HF cache.
    ///
    /// Idempotent: a second call while one is in flight is ignored, so a
    /// dirty flag set repeatedly cannot spawn a thread per frame.
    pub fn start_scan(&mut self, cache_dir: Option<&std::path::Path>) {
        if self.pending_scan.is_some() {
            return;
        }
        self.pending_scan = Some(crate::tui::data::library::scan_in_background(cache_dir));
    }

    /// Collect a finished scan, if one has landed.
    ///
    /// Returns the new entries for the caller to store. During the scan the
    /// previous list keeps rendering, so there is no empty frame and no
    /// flicker — the list simply becomes more correct.
    pub fn poll_scan(&mut self) -> Option<Vec<LibraryEntry>> {
        let rx = self.pending_scan.as_ref()?;
        match rx.try_recv() {
            Ok(found) => {
                self.pending_scan = None;
                Some(found)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            // The scanner thread died. Keep the list that is on screen.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending_scan = None;
                None
            }
        }
    }
}

#[cfg(test)]
#[path = "lib_scan_tests.rs"]
mod tests;
