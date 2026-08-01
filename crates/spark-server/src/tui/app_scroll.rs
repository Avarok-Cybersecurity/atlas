// SPDX-License-Identifier: AGPL-3.0-only

//! Where the mouse wheel goes, per section.
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
                MainSub::Overview => {
                    let cur = self.log_scroll.unwrap_or(0) as i32;
                    let next = cur - rows;
                    self.log_scroll = if next <= 0 { None } else { Some(next as usize) };
                }
                MainSub::Kernels => {
                    self.kernel_scroll = (self.kernel_scroll as i32 + rows).max(0) as usize;
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
                TermSub::Ops => {
                    let cur = self.log_scroll.unwrap_or(0) as i32;
                    let next = cur - rows;
                    self.log_scroll = if next <= 0 { None } else { Some(next as usize) };
                }
                TermSub::Chat => self.chat.scroll_by(-rows),
            },
            // Nothing scrollable: these panes are gauges, not documents.
            Section::Stats | Section::Network => {}
        }
    }
}
