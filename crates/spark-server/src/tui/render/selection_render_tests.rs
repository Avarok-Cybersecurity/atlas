// SPDX-License-Identifier: AGPL-3.0-only

//! What a mouse drag draws.
//!
//! Split from `render_tests.rs` at the LoC cap. The distinction that matters
//! here: the highlight is a STYLE change, so it must be invisible to any
//! assertion about the rendered symbols — if a drag ever altered the text, the
//! copy would no longer match what the user saw.

use super::*;
// The fixtures live with the main render tests; sharing them keeps one
// definition of "a rendered App" rather than two that can drift.
use super::tests::{app, render};

#[test]
fn a_drag_highlights_what_it_covers_and_a_click_highlights_nothing() {
    use crate::tui::selection::Selection;
    let mut a = app();
    a.section = Section::Main;

    // A click (no movement) must not paint anything — otherwise every click
    // flashes a one-cell highlight.
    a.selection = Some(Selection::new((10, 5)));
    let clean = render(&a, 120, 40);

    // A real drag reverses cells, which changes styling but never the SYMBOLS.
    a.selection = Some(Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });
    let dragged = render(&a, 120, 40);
    assert_eq!(
        clean, dragged,
        "the highlight is a style change; it must not alter the text"
    );
}

#[test]
fn the_selection_highlight_survives_hostile_geometry() {
    // A drag that ends past the bottom of a shrinking terminal must not index
    // outside the buffer.
    use crate::tui::selection::Selection;
    let mut a = app();
    a.selection = Some(Selection {
        anchor: (0, 0),
        cursor: (250, 250),
    });
    for (w, h) in [(1u16, 1u16), (40, 12), (200, 50)] {
        let _ = render(&a, w, h);
    }
}

#[test]
fn the_highlight_actually_sets_reverse_video_on_the_covered_cells() {
    // The sibling test asserts the SYMBOLS are unchanged, which passes even if
    // the highlight is never drawn at all. This one asserts the styling really
    // is applied — without it, "selection renders" was untested.
    use crate::tui::selection::Selection;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    let mut a = app();
    a.section = Section::Main;
    a.selection = Some(Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });

    let mut t = Terminal::new(TestBackend::new(120, 40)).expect("backend");
    t.draw(|f| draw(f, &a)).expect("draw");
    let buf = t.backend().buffer();

    let rev = |x: u16, y: u16| buf[(x, y)].modifier.contains(Modifier::REVERSED);
    assert!(rev(10, 5), "the first covered cell is highlighted");
    assert!(rev(20, 5), "and the middle");
    assert!(rev(30, 5), "and the last");
    assert!(!rev(9, 5), "but not the cell before it");
    assert!(!rev(31, 5), "nor the cell after");
    assert!(!rev(20, 4), "nor another row");
}

#[test]
fn the_buffer_to_copy_from_is_the_completed_frame_not_the_current_one() {
    // The bug this pins cost a live debugging session. Copying inline from
    // `Terminal::current_buffer_mut()` looks obviously right and is wrong:
    // ratatui swaps its two buffers after each draw and RESETS the one that
    // becomes current, so between frames `current_buffer_mut()` is blank. The
    // text only exists in the `CompletedFrame` that `draw()` returns, which is
    // why the copy is deferred to just after the draw instead of being done in
    // the mouse handler.
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let a = app();
    let mut t = Terminal::new(TestBackend::new(120, 40)).expect("backend");

    let rendered: String = {
        let frame = t.draw(|f| draw(f, &a)).expect("draw");
        frame.buffer.content().iter().map(|c| c.symbol()).collect()
    };
    assert!(
        rendered.contains("Benchmarks"),
        "the completed frame holds the rendered text:\n{rendered}"
    );

    let current: String = t
        .current_buffer_mut()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert_ne!(
        current.trim(),
        rendered.trim(),
        "current_buffer_mut() is NOT the frame just drawn — extracting from it \
         copies a blank screen"
    );
}

#[test]
fn a_selection_does_not_survive_a_keystroke() {
    // Reported: select text, switch screen, and the highlight follows you —
    // painting reversed cells over unrelated content, because the selection is
    // stored in SCREEN CELLS and means something different on every screen.
    use crate::tui::selection::Selection;
    use crossterm::event::{KeyCode, KeyEvent};

    let mut a = app();
    a.section = Section::Main;
    a.selection = Some(Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });
    assert!(a.selection.is_some());

    // Any key at all, including the section-switch keys that triggered the
    // report.
    a.on_key(KeyEvent::from(KeyCode::Char('4')));
    assert!(
        a.selection.is_none(),
        "a keystroke must end the selection, not carry it to the next screen"
    );
}

#[test]
fn a_cleared_selection_paints_nothing() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    let mut a = app();
    a.section = Section::Main;
    a.selection = None;

    let mut t = Terminal::new(TestBackend::new(120, 40)).expect("backend");
    t.draw(|f| draw(f, &a)).expect("draw");
    let buf = t.backend().buffer();
    let reversed = buf
        .content()
        .iter()
        .filter(|c| c.modifier.contains(Modifier::REVERSED))
        .count();
    // The dashboard uses REVERSED for its own chips and badges, so this cannot
    // assert zero — only that the 21-cell selection band is not among them.
    let before = reversed;

    a.selection = Some(crate::tui::selection::Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });
    let mut t2 = Terminal::new(TestBackend::new(120, 40)).expect("backend");
    t2.draw(|f| draw(f, &a)).expect("draw");
    let after = t2
        .backend()
        .buffer()
        .content()
        .iter()
        .filter(|c| c.modifier.contains(Modifier::REVERSED))
        .count();
    assert_eq!(
        after,
        before + 21,
        "a live selection adds exactly its own cells; a cleared one adds none"
    );
}
