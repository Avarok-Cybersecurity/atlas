// SPDX-License-Identifier: AGPL-3.0-only

//! Plain data carried by [`super::app::App`]: the toast record and the Ops
//! REPL state. Moved out of `app.rs` at the 500-LoC cap; they travel
//! together because both are dumb state — no method here reads any other
//! part of the `App`, so the move is a pure piecewise copy.

use std::time::Instant;

pub struct Toast {
    pub text: String,
    pub error: bool,
    pub at: Instant,
}

/// Ops REPL state.
#[derive(Default)]
pub struct OpsState {
    pub input: String,
    pub history: Vec<String>,
    pub history_pos: Option<usize>,
    pub output: Vec<String>,
    /// Rows scrolled up from the newest output line; 0 = follow. Declared
    /// with the pane and left unread for a release — the scrollback was
    /// planned and never wired, while the footer said "↑/↓ scroll".
    pub scroll_up: usize,
    /// Scroll ceiling, published by the renderer each frame — the
    /// `log_scroll_max` contract (see `app_scroll`), held here rather than
    /// on `App` because everything that reads it already holds `ops`.
    pub scroll_max: std::cell::Cell<usize>,
}
