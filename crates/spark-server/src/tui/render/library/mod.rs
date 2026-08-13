// SPDX-License-Identifier: AGPL-3.0-only

//! The Library section: browse recipes joined with local weights, then
//! configure one.

pub mod cards;
pub mod config;
pub mod list;
mod list_detail;
mod modal;

use ratatui::Frame;
use ratatui::layout::Rect;

use crate::tui::app::App;
use crate::tui::lib_state::View;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    match app.lib.view {
        View::List => list::draw(f, app, area),
        View::Cards => cards::draw(f, app, area),
        View::Config => config::draw(f, app, area),
    }
    // On top of the pane, under the app-level overlays (toasts, help, the
    // confirmations) — a picker is part of the form, not a question that
    // outranks the section.
    modal::draw(f, app, area);
}

#[cfg(test)]
#[path = "library_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "start_tests.rs"]
mod start_tests;
