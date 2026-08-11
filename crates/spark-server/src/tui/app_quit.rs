// SPDX-License-Identifier: AGPL-3.0-only

//! What `q` costs, and when it costs a second press.
//!
//! Split from `app.rs` at the 500-LoC cap, and a coherent unit on its own:
//! these three functions are the whole of the guard, and the one thing worth
//! keeping in view together is that `q` is not "close the window".
//!
//! ★ **`q` DRAINS AND STOPS THE SERVER**, exactly like Ctrl+C — it sets
//! `should_quit`, the event loop breaks, and the loop then calls
//! `shutdown::request`. It was bound globally, on every screen, with no
//! confirmation, so one stray keypress while reading logs ended a four-hour
//! benchmark with no way back. The key map already said so honestly; saying so
//! is not the same as not doing it.
//!
//! The confirmation is deliberately conditional. A prompt on an idle dashboard
//! costs a keystroke and protects nothing, and a prompt that is usually
//! pointless is one the user learns to dismiss without reading — which is
//! precisely the reflex that would carry them through the one that mattered.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::App;

impl App {
    /// What `q` would throw away, named for the prompt — `None` when the
    /// dashboard is idle and quitting costs nothing.
    ///
    /// All three are work the user started and cannot resume: a benchmark run,
    /// a partially-fetched checkpoint, a reply mid-stream. A download the user
    /// has ALREADY asked to stop is not in flight — it is on its way out, and
    /// asking about it would be asking about work they just cancelled.
    pub fn work_in_flight(&self) -> Option<&'static str> {
        if self.bench.is_running() {
            return Some("a benchmark is running");
        }
        if self.download.job.as_ref().is_some_and(|j| !j.cancelling) {
            return Some("a model download is in progress");
        }
        if self.chat.streaming {
            return Some("a chat reply is still streaming");
        }
        None
    }

    /// `q` pressed with no text field claiming it.
    pub(super) fn on_quit_key(&mut self) {
        if self.work_in_flight().is_some() {
            self.confirm_quit = true;
        } else {
            self.should_quit = true;
        }
    }

    /// Answer an open prompt. Returns `true` when the key was consumed by it,
    /// which is always — a prompt that let some keys through to the section
    /// underneath would make dismissing it jump the user somewhere as a side
    /// effect.
    ///
    /// Only an affirmative goes through. Every other key cancels, because the
    /// safe reading of an ambiguous keystroke is "do not stop the server".
    pub(super) fn answer_quit_prompt(&mut self, key: KeyEvent) -> bool {
        self.confirm_quit = false;
        if matches!(
            key.code,
            KeyCode::Char('q') | KeyCode::Char('y') | KeyCode::Char('Y')
        ) {
            self.should_quit = true;
        }
        true
    }
}
