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

use crossterm::event::{KeyCode, KeyEvent};

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
                // The Ops pane's OWN offset. This used to route to
                // `scroll_log` — the MAIN tab's log offset, which Ops does
                // not render: the wheel visibly did nothing, and Main was
                // later found scrolled.
                TermSub::Ops => {
                    let max = self.ops.scroll_max.get() as i32;
                    self.ops.scroll_up =
                        (self.ops.scroll_up as i32 - rows).clamp(0, max.max(0)) as usize;
                }
                TermSub::Chat => self.chat_scroll(-rows),
            },
            // The report preview is the one scrollable surface in Help; the
            // rest of the section is short static prose.
            Section::Help => self.help.scroll_preview(rows),
            // Nothing scrollable: these panes are gauges, not documents.
            Section::Stats | Section::Network => {}
        }
    }

    /// Keys while the help modal is open.
    ///
    /// j/k and friends move the key list — at the 80x24 floor the table is
    /// taller than the modal, and a list that cannot move is a list whose
    /// tail cannot be read (see `render/overlay.rs::draw_help`). Anything
    /// else closes the modal, which is the contract `?` already taught; the
    /// scroll keys are consumed rather than passed through, for the same
    /// reason the quit prompt consumes everything — a key that dismisses AND
    /// acts navigates somewhere the user did not ask for.
    pub(super) fn on_help_overlay_key(&mut self, key: KeyEvent) {
        let max = self.help_scroll_max.get();
        match key.code {
            KeyCode::Down | KeyCode::Char('j') => {
                self.help_scroll = (self.help_scroll + 1).min(max);
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.help_scroll = self.help_scroll.saturating_sub(1);
            }
            KeyCode::Char('g') | KeyCode::Home => self.help_scroll = 0,
            KeyCode::Char('G') | KeyCode::End => self.help_scroll = max,
            _ => {
                self.help_open = false;
                // The next open starts at the top: a remembered offset would
                // reopen onto a random middle of the list.
                self.help_scroll = 0;
            }
        }
    }

    /// Ops keys when the output pane, not the input line, has focus.
    ///
    /// Routed through [`App::scroll`], the same entry point the wheel uses —
    /// the log pane's lesson about second copies of "what scrolling means
    /// here" applies unchanged.
    pub(super) fn on_ops_content_key(&mut self, key: KeyEvent) {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => self.scroll(-1),
            KeyCode::Down | KeyCode::Char('j') => self.scroll(1),
            KeyCode::PageUp => self.scroll(-10),
            KeyCode::PageDown => self.scroll(10),
            KeyCode::Char('g') | KeyCode::Home => {
                self.ops.scroll_up = self.ops.scroll_max.get();
            }
            KeyCode::Char('G') | KeyCode::End => self.ops.scroll_up = 0,
            _ => {}
        }
    }

    /// Chat transcript scroll with the ceiling applied — the one entry point
    /// the wheel and BOTH keyboard paths share (positive rows = back toward
    /// older turns). The keyboard used to call `scroll_by` bare, which banked
    /// presses past the oldest row that a reader then paid back one dead key
    /// at a time — the exact defect the ceilings were introduced to end.
    pub(super) fn chat_scroll(&mut self, rows: i32) {
        self.chat.scroll_by(rows);
        self.clamp_chat_scroll();
    }

    /// Pull a banked offset back inside the renderer-published ceiling.
    /// Separate from [`Self::chat_scroll`] because `ChatState::on_content_key`
    /// moves the offset itself and only the `App` can see the ceiling.
    pub(super) fn clamp_chat_scroll(&mut self) {
        let max = self.chat_scroll_max.get();
        if self.chat.scroll.is_some_and(|n| n > max) {
            self.chat.scroll = if max == 0 { None } else { Some(max) };
        }
    }

    /// Jump the chat transcript to its oldest row — the `g` half of the
    /// advertised "g / G" pair (`G`/End, the follow half, lives in
    /// `ChatState`). Here because it needs the renderer-published ceiling,
    /// which `ChatState` cannot see.
    pub(super) fn chat_jump_top(&mut self) {
        let max = self.chat_scroll_max.get();
        self.chat.scroll = (max > 0).then_some(max);
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
