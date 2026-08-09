// SPDX-License-Identifier: AGPL-3.0-only

//! The Library section: browse recipes joined with local weights, then
//! configure one.

pub mod cards;
pub mod config;
pub mod list;

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
}

#[cfg(test)]
#[path = "library_tests.rs"]
mod tests;
