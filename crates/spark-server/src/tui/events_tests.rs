// SPDX-License-Identifier: AGPL-3.0-only

//! The mouse half of the input layer: what a click, a drag and a wheel turn do
//! to the app, and which sidebar row a click actually landed on.

use super::*;
use crossterm::event::{KeyModifiers, MouseEvent};
use ratatui::layout::Size;

/// A terminal wide and tall enough for the full sidebar (18 cols) and the tall
/// header (3 rows) — the geometry the row arithmetic below assumes, and which
/// `render::Chrome` now decides for the renderer and this handler alike. The
/// renderer's half of the invariant is
/// `render::chrome_tests::the_frame_is_laid_out_where_the_chrome_says_it_is`.
const WIDE: Size = Size {
    width: 120,
    height: 40,
};
/// Narrow enough that the sidebar collapses to icons and the header to one row.
const NARROW: Size = Size {
    width: 80,
    height: 20,
};

fn app() -> App {
    App::new(clap::Parser::parse_from(["spark", "org/m"]))
}

fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

fn click(a: &mut App, column: u16, row: u16, size: Size) -> MouseOutcome {
    on_mouse(
        a,
        mouse(MouseEventKind::Down(MouseButton::Left), column, row),
        Some(size),
    )
}

/// Where the sidebar cursor ended up, spelled as the sidebar draws it.
fn at(a: &App) -> String {
    match a.section.subs().get(a.sub_index(a.section)) {
        Some(sub) => format!("{}/{sub}", a.section.label()),
        None => a.section.label().to_string(),
    }
}

#[test]
fn clicking_a_sidebar_section_row_selects_that_section() {
    // Rows under the tall header, with Main active and drawing two subsections:
    // Main, ├Overview, └Kernels, Stats, Network, Library, Benchmarks, Terminal.
    for (row, expected) in [
        (6, "Stats"),
        (7, "Network"),
        (8, "Library"),
        (9, "Benchmarks/Suite"),
        (10, "Terminal/Ops"),
    ] {
        let mut a = app();
        click(&mut a, 2, row, WIDE);
        assert_eq!(at(&a), expected, "row {row}");
    }
}

#[test]
fn clicking_the_section_already_shown_cycles_its_subsections() {
    // Same contract as pressing its number key twice.
    let mut a = app();
    click(&mut a, 2, 3, WIDE);
    assert_eq!(at(&a), "Main/Kernels");
    click(&mut a, 2, 3, WIDE);
    assert_eq!(at(&a), "Main/Overview");
}

#[test]
fn clicking_a_subsection_row_selects_that_subsection() {
    // The regression: the offset for the rows BELOW the subsections was applied
    // but the subsection rows themselves were left mapping to whatever section
    // index they sat at — clicking "Overview" landed on Stats.
    let mut a = app();
    click(&mut a, 2, 5, WIDE);
    assert_eq!(at(&a), "Main/Kernels", "└ Kernels is the third row");
    click(&mut a, 2, 4, WIDE);
    assert_eq!(at(&a), "Main/Overview", "and ├ Overview the second");
}

#[test]
fn the_rows_below_shift_with_whichever_section_is_expanded() {
    // The subsections move with the selection, so the arithmetic has to be
    // re-derived from the ACTIVE section rather than assumed about Main.
    let mut a = app();
    click(&mut a, 2, 10, WIDE); // Terminal, now the expanded one
    assert_eq!(at(&a), "Terminal/Ops");
    // Its rows are now the last two, and every section above is unshifted.
    click(&mut a, 2, 10, WIDE);
    assert_eq!(at(&a), "Terminal/Chat", "└ Chat");
    click(&mut a, 2, 9, WIDE);
    assert_eq!(at(&a), "Terminal/Ops", "├ Ops");
    click(&mut a, 2, 6, WIDE);
    assert_eq!(at(&a), "Library");
}

#[test]
fn a_narrow_sidebar_draws_no_subsections_and_offsets_nothing() {
    // Icons only under a one-row header, so row N is section N.
    for (row, expected) in [
        (2, "Stats"),
        (3, "Network"),
        (4, "Library"),
        (5, "Benchmarks/Suite"),
        (6, "Terminal/Ops"),
    ] {
        let mut a = app();
        click(&mut a, 1, row, NARROW);
        assert_eq!(at(&a), expected, "row {row}");
    }
}

#[test]
fn a_click_past_the_last_sidebar_row_selects_nothing() {
    let mut a = app();
    click(&mut a, 2, 30, WIDE);
    assert_eq!(at(&a), "Main/Overview", "unchanged");
}

#[test]
fn a_click_on_the_header_navigates_nowhere() {
    // The sidebar starts below the header, so a click up there is ordinary
    // content — selectable text, not a section.
    let mut a = app();
    click(&mut a, 2, 1, WIDE);
    assert_eq!(at(&a), "Main/Overview");
    assert_eq!(a.selection.expect("armed").anchor, (2, 1));
}

#[test]
fn a_sidebar_click_is_navigation_not_the_start_of_a_drag() {
    let mut a = app();
    a.selection = Some(crate::tui::selection::Selection::new((60, 9)));
    click(&mut a, 2, 8, WIDE);
    assert!(a.selection.is_none());
}

#[test]
fn a_click_in_the_content_area_arms_a_selection() {
    let mut a = app();
    click(&mut a, 40, 9, WIDE);
    let sel = a.selection.expect("armed");
    assert_eq!(sel.anchor, (40, 9));
    assert!(!sel.is_drag(), "nothing is copied until it moves");
}

#[test]
fn a_drag_tracks_the_pointer_and_copies_on_release() {
    let mut a = app();
    click(&mut a, 40, 9, WIDE);
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Drag(MouseButton::Left), 52, 11),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::None, "nothing to copy mid-drag");
    assert_eq!(a.selection.expect("still armed").cursor, (52, 11));

    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Up(MouseButton::Left), 52, 11),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::CopySelection);
    assert!(
        a.selection.is_some(),
        "the selection outlives the release: the text is read out of the frame"
    );
}

#[test]
fn a_click_that_never_moved_copies_nothing() {
    // Otherwise every click on the dashboard copies one character and toasts.
    let mut a = app();
    click(&mut a, 40, 9, WIDE);
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Up(MouseButton::Left), 40, 9),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::None);
    assert!(a.selection.is_none(), "and clears rather than lingering");
}

#[test]
fn a_release_with_no_button_down_is_harmless() {
    let mut a = app();
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Up(MouseButton::Left), 40, 9),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::None);
    assert!(a.selection.is_none());
}

#[test]
fn a_drag_with_nothing_armed_does_not_invent_a_selection() {
    let mut a = app();
    on_mouse(
        &mut a,
        mouse(MouseEventKind::Drag(MouseButton::Left), 40, 9),
        Some(WIDE),
    );
    assert!(a.selection.is_none());
}

#[test]
fn the_wheel_scrolls_the_active_pane_and_drops_the_highlight() {
    // Scrolling moves the content out from under the highlight, so its cells no
    // longer hold the text that was chosen.
    let mut a = app();
    a.log_scroll_max.set(50);
    a.selection = Some(crate::tui::selection::Selection::new((40, 9)));
    on_mouse(&mut a, mouse(MouseEventKind::ScrollUp, 40, 9), Some(WIDE));
    assert_eq!(a.log_scroll, Some(3), "three rows a notch");
    assert!(a.selection.is_none());
    on_mouse(&mut a, mouse(MouseEventKind::ScrollDown, 40, 9), Some(WIDE));
    assert_eq!(a.log_scroll, None);
}

#[test]
fn a_mouse_event_before_the_terminal_size_is_known_is_ignored() {
    // `terminal.size()` can fail; guessing the geometry would map the click to
    // an arbitrary section.
    let mut a = app();
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Down(MouseButton::Left), 2, 8),
        None,
    );
    assert_eq!(out, MouseOutcome::None);
    assert_eq!(at(&a), "Main/Overview");
    assert!(a.selection.is_none());
}

#[test]
fn buttons_other_than_the_left_one_are_left_to_the_terminal() {
    let mut a = app();
    for kind in [
        MouseEventKind::Down(MouseButton::Right),
        MouseEventKind::Down(MouseButton::Middle),
        MouseEventKind::Up(MouseButton::Right),
        MouseEventKind::Moved,
    ] {
        let out = on_mouse(&mut a, mouse(kind, 40, 9), Some(WIDE));
        assert_eq!(out, MouseOutcome::None, "{kind:?}");
        assert!(a.selection.is_none(), "{kind:?}");
    }
}
