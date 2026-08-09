// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;

fn buf_of(lines: &[&str]) -> (Buffer, Rect) {
    let w = lines.iter().map(|l| l.chars().count()).max().unwrap_or(0) as u16;
    let area = Rect::new(0, 0, w, lines.len() as u16);
    let mut b = Buffer::empty(area);
    for (y, l) in lines.iter().enumerate() {
        for (x, ch) in l.chars().enumerate() {
            b[(x as u16, y as u16)].set_symbol(&ch.to_string());
        }
    }
    (b, area)
}

#[test]
fn a_click_without_movement_is_not_a_selection() {
    // Otherwise every click on the dashboard copies one character and toasts.
    let s = Selection::new((5, 5));
    assert!(!s.is_drag());
    assert!(
        Selection {
            anchor: (5, 5),
            cursor: (6, 5)
        }
        .is_drag()
    );
}

#[test]
fn dragging_backwards_selects_the_same_text_as_dragging_forwards() {
    let fwd = Selection {
        anchor: (2, 0),
        cursor: (6, 0),
    };
    let back = Selection {
        anchor: (6, 0),
        cursor: (2, 0),
    };
    assert_eq!(fwd.ordered(), back.ordered());
    for x in 0..9u16 {
        assert_eq!(fwd.contains(x, 0), back.contains(x, 0), "col {x}");
    }
}

#[test]
fn a_single_line_selection_is_a_column_range() {
    let s = Selection {
        anchor: (2, 3),
        cursor: (5, 3),
    };
    assert!(!s.contains(1, 3));
    assert!(s.contains(2, 3) && s.contains(5, 3));
    assert!(!s.contains(6, 3));
    assert!(!s.contains(3, 2), "other rows are untouched");
}

#[test]
fn a_multi_line_selection_wraps_like_a_text_editor() {
    // First line to the right edge, middle lines whole, last line from the left
    // — the behaviour every terminal and editor has.
    let s = Selection {
        anchor: (4, 1),
        cursor: (2, 3),
    };
    assert!(!s.contains(3, 1));
    assert!(
        s.contains(4, 1) && s.contains(99, 1),
        "first line runs right"
    );
    assert!(
        s.contains(0, 2) && s.contains(99, 2),
        "middle line is whole"
    );
    assert!(
        s.contains(0, 3) && s.contains(2, 3),
        "last line starts left"
    );
    assert!(!s.contains(3, 3));
}

#[test]
fn extract_returns_the_covered_text() {
    let (b, area) = buf_of(&["hello world", "second line"]);
    let s = Selection {
        anchor: (0, 0),
        cursor: (4, 0),
    };
    assert_eq!(extract(&b, area, &s), "hello");
}

#[test]
fn extract_trims_the_padding_that_makes_a_table_row_unusable() {
    // Selecting one row of a pane otherwise yields the text plus however many
    // columns of blanks the pane is wide.
    let (b, area) = buf_of(&["id: model-a          ", "id: model-b          "]);
    let s = Selection {
        anchor: (0, 0),
        cursor: (20, 0),
    };
    assert_eq!(extract(&b, area, &s), "id: model-a");
}

#[test]
fn extract_joins_lines_with_newlines_and_drops_trailing_blanks() {
    let (b, area) = buf_of(&["one", "two", "   ", "   "]);
    let s = Selection {
        anchor: (0, 0),
        cursor: (2, 3),
    };
    assert_eq!(extract(&b, area, &s), "one\ntwo");
}

#[test]
fn extract_keeps_a_blank_line_between_content() {
    // A gap inside the selection is real content; only trailing blanks are an
    // artefact of dragging past the end.
    let (b, area) = buf_of(&["one", "   ", "two"]);
    let s = Selection {
        anchor: (0, 0),
        cursor: (2, 2),
    };
    assert_eq!(extract(&b, area, &s), "one\n\ntwo");
}

#[test]
fn extract_outside_the_area_is_empty_not_a_panic() {
    let (b, area) = buf_of(&["abc"]);
    let s = Selection {
        anchor: (0, 90),
        cursor: (2, 99),
    };
    assert_eq!(extract(&b, area, &s), "");
}

#[test]
fn extract_preserves_unicode_cells() {
    let (b, area) = buf_of(&["✓ Qwen3.6 — ▓░"]);
    let s = Selection {
        anchor: (0, 0),
        cursor: (13, 0),
    };
    assert_eq!(extract(&b, area, &s), "✓ Qwen3.6 — ▓░");
}
