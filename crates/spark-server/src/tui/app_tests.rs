// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the [`super`] input reducer.

use super::*;

/// The ⇥ order must contain the subsection rows, in the order the sidebar draws
/// them. This is the regression: the traversal list was top-level-only, so
/// Main ▸ Kernels and Terminal ▸ Chat could not be reached with Tab at all.
#[test]
fn nav_rows_include_subsections_in_sidebar_order() {
    let labels: Vec<String> = App::nav_rows()
        .iter()
        .map(|(s, i)| match s.subs().get(*i) {
            Some(sub) => format!("{}/{}", s.label(), sub),
            None => s.label().to_string(),
        })
        .collect();
    assert_eq!(
        labels,
        [
            "Main/Overview",
            "Main/Kernels",
            "Stats",
            "Network",
            "Library",
            "Terminal/Ops",
            "Terminal/Chat",
        ]
    );
}

/// A section without subsections must still contribute exactly one stop, or ⇥
/// would silently skip it.
#[test]
fn every_section_is_reachable() {
    let rows = App::nav_rows();
    for s in Section::ALL {
        assert!(
            rows.iter().any(|(r, _)| *r == s),
            "{} unreachable",
            s.label()
        );
    }
}

/// Chat scrollback contract, mirroring the Main log pane: `None` follows the tip,
/// and scrolling back down to (or past) the bottom restores follow rather than
/// parking at `Some(0)`, which would freeze the view one row off the live tip.
#[test]
fn chat_scroll_returns_to_follow_at_the_bottom() {
    let mut c = crate::tui::chat::ChatState::default();
    assert_eq!(c.scroll, None, "starts following");
    c.scroll_by(3);
    assert_eq!(c.scroll, Some(3));
    c.scroll_by(-1);
    assert_eq!(c.scroll, Some(2));
    c.scroll_by(-5); // overshoot past the bottom
    assert_eq!(c.scroll, None, "overshoot resumes follow");
    c.scroll_by(10);
    c.follow();
    assert_eq!(c.scroll, None);
}
