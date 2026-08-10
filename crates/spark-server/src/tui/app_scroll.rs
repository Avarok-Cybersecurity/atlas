// SPDX-License-Identifier: AGPL-3.0-only

//! Where the mouse wheel goes, per section — and how far.
//!
//! # The ceilings
//!
//! `App::{log,kernel,chat}_scroll_max` are written by the RENDERER each frame
//! and read here. The limit is `content_lines - viewport_height`, and only the
//! renderer knows either: the viewport is a layout result, and the content is
//! filtered and wrapped on the way to the screen. They are `Cell` because
//! `draw` takes `&App`.
//!
//! Without them the offset grew without bound past the first or last line —
//! the pane had long since gone blank, and coming back took as many turns of
//! the wheel as had been spent going up.
//!
//! Split from `app.rs` at the 500-LoC cap, and a coherent unit on its own: it
//! is the one place that knows what each section considers "scrolling". Two
//! rules hold across all of them and are easier to keep in one file —
//! a section routes the wheel to whatever its own KEYS already move, so wheel
//! and keyboard can never disagree about position; and a section with nothing
//! to scroll does nothing rather than guessing.

use super::app::{App, MainSub, TermSub};
use super::section::Section;

impl App {
    /// Scroll the current view by `rows` (positive = further down/later).
    ///
    /// The wheel used to work only on the Main log pane, which meant it looked
    /// broken everywhere else — a mouse that does nothing in five of six
    /// sections reads as no mouse support at all. Each section routes to
    /// whatever it already scrolls with the keyboard, so the wheel and `j/k`
    /// can never disagree about position.
    pub fn scroll(&mut self, rows: i32) {
        match self.section {
            Section::Main => match self.main_sub {
                // The log pane counts BACKWARDS from the newest line, so a
                // wheel-up (negative rows) has to increase the offset.
                MainSub::Overview => self.scroll_log(rows),
                MainSub::Kernels => {
                    let max = self.kernel_scroll_max.get() as i32;
                    self.kernel_scroll =
                        (self.kernel_scroll as i32 + rows).clamp(0, max.max(0)) as usize;
                }
            },
            // Lists move their SELECTION rather than a viewport: that is what
            // the arrow keys do here, and a wheel that scrolled the view
            // without moving the cursor would leave the two out of step.
            Section::Library => self.lib.move_selection(rows as isize),
            Section::Benchmarks => {
                let n = atlas_plugin::registry::all().len();
                if n > 0 {
                    let cur = self.bench.selected as i32;
                    let next = (cur + rows).clamp(0, n as i32 - 1);
                    self.bench.select(next as usize);
                }
            }
            Section::Terminal => match self.term_sub {
                TermSub::Ops => self.scroll_log(rows),
                TermSub::Chat => {
                    self.chat.scroll_by(-rows);
                    // Same ceiling argument as the log pane.
                    let max = self.chat_scroll_max.get();
                    if self.chat.scroll.is_some_and(|n| n > max) {
                        self.chat.scroll = if max == 0 { None } else { Some(max) };
                    }
                }
            },
            // Nothing scrollable: these panes are gauges, not documents.
            Section::Stats | Section::Network => {}
        }
    }

    /// The log pane, which both Main/Overview and Terminal/Ops share.
    ///
    /// The offset counts BACKWARDS from the newest line, so wheel-up (negative
    /// `rows`) increases it — up to the oldest line the pane can show, and no
    /// further.
    fn scroll_log(&mut self, rows: i32) {
        let max = self.log_scroll_max.get() as i32;
        let next = (self.log_scroll.unwrap_or(0) as i32 - rows).clamp(0, max.max(0));
        self.log_scroll = if next <= 0 { None } else { Some(next as usize) };
    }
}

#[cfg(test)]
#[path = "app_scroll_tests.rs"]
mod tests;
