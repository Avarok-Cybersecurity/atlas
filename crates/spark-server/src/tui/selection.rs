// SPDX-License-Identifier: AGPL-3.0-only

//! Dragging a selection across the dashboard, and turning it into text.
//!
//! # Why the dashboard has to implement this at all
//!
//! `TerminalGuard` enables mouse capture, which is what makes clicking the
//! sidebar and scrolling work — and it also takes mouse selection AWAY from the
//! terminal, because the terminal now forwards drags to us instead of painting
//! its own highlight. So a dashboard with mouse capture and no selection of its
//! own is strictly worse at copying text than one with no mouse support at all.
//! (Most terminals still offer their native selection if you hold Shift; that
//! is the escape hatch, not the plan.)
//!
//! # Linear, not rectangular
//!
//! Selection runs in reading order — start cell to end cell, wrapping through
//! whole lines — because that is what every terminal does and therefore what
//! the muscle memory expects. The cost is that a drag across a pane border
//! picks up the border glyphs of the pane beside it; [`extract`] trims trailing
//! whitespace per line, which handles the common case of selecting one row.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

/// An in-progress or finished drag, in terminal cell coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    /// Where the button went down. Fixed for the life of the drag.
    pub anchor: (u16, u16),
    /// Where the pointer is now.
    pub cursor: (u16, u16),
}

impl Selection {
    pub fn new(at: (u16, u16)) -> Self {
        Self {
            anchor: at,
            cursor: at,
        }
    }

    /// `(start, end)` in reading order, regardless of drag direction.
    ///
    /// Dragging up or leftwards is as normal as dragging down, and every
    /// consumer wants the ordered pair.
    pub fn ordered(&self) -> ((u16, u16), (u16, u16)) {
        let (ax, ay) = self.anchor;
        let (cx, cy) = self.cursor;
        if (ay, ax) <= (cy, cx) {
            (self.anchor, self.cursor)
        } else {
            (self.cursor, self.anchor)
        }
    }

    /// Is this cell inside the selection?
    pub fn contains(&self, x: u16, y: u16) -> bool {
        let ((sx, sy), (ex, ey)) = self.ordered();
        if y < sy || y > ey {
            return false;
        }
        // Single line: a plain column range. Otherwise the first line runs to
        // the right edge and the last from the left, as in any editor.
        if sy == ey {
            return x >= sx && x <= ex;
        }
        if y == sy {
            return x >= sx;
        }
        if y == ey {
            return x <= ex;
        }
        true
    }

    /// Has the pointer actually moved? A click is not a selection.
    ///
    /// Without this, every click on the dashboard would copy one character and
    /// raise a toast — which is how a helpful feature becomes an irritation.
    pub fn is_drag(&self) -> bool {
        self.anchor != self.cursor
    }
}

/// The text under a selection, read back out of the rendered frame.
///
/// Trailing whitespace is trimmed per line and fully blank lines are kept, so
/// selecting a row of a table yields that row rather than the row plus forty
/// columns of padding. Returns an empty string when nothing is covered.
pub fn extract(buf: &Buffer, area: Rect, sel: &Selection) -> String {
    let mut lines: Vec<String> = Vec::new();
    let ((_, sy), (_, ey)) = sel.ordered();
    for y in sy..=ey {
        if y < area.y || y >= area.y.saturating_add(area.height) {
            continue;
        }
        let mut line = String::new();
        for x in area.x..area.x.saturating_add(area.width) {
            if !sel.contains(x, y) {
                continue;
            }
            line.push_str(buf[(x, y)].symbol());
        }
        lines.push(line.trim_end().to_string());
    }
    // Trailing blank lines are an artefact of dragging past the content, not
    // something anyone means to copy.
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    lines.join("\n")
}

#[cfg(test)]
#[path = "selection_tests.rs"]
mod tests;
