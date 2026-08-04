// SPDX-License-Identifier: AGPL-3.0-only

//! Log-line wrapping for the Main log pane.
//!
//! Split from `main_tab.rs` only to stay under the repository's per-file cap.

use super::*;

fn logline(msg: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw("12:35:04 ".to_string()),
        Span::raw("INFO  ".to_string()),
        Span::raw("spark_model    ".to_string()),
        Span::raw(msg.to_string()),
    ])
}

fn text(line: &Line<'_>) -> String {
    line.spans.iter().map(|s| s.content.as_ref()).collect()
}

#[test]
fn a_line_that_fits_is_left_alone() {
    let rows = wrap_line(logline("short"), 120);
    assert_eq!(rows.len(), 1);
    assert!(text(&rows[0]).ends_with("short"));
}

#[test]
fn a_long_line_wraps_instead_of_losing_its_tail() {
    // The reported bug: the pane cut at the panel edge and the rest of the
    // message was simply gone.
    let msg = "SSM snapshot pool: Marconi 16 slots (2424 MB), decode-rollback 8 slots x 8 seqs (9696 MB), 48 layers";
    let rows = wrap_line(logline(msg), 80);
    assert!(rows.len() > 1, "it wrapped");
    // Join on the WORDS, not the rendered rows: a wrap boundary puts the
    // continuation indent between "48" and "layers", so searching the
    // concatenated rows for the literal phrase would fail on correct
    // output. (It did.)
    let joined = rows.iter().map(|r| text(r)).collect::<Vec<_>>().join(" ");
    let words: Vec<&str> = joined.split_whitespace().collect();
    assert!(
        words.windows(2).any(|w| w == ["48", "layers"]),
        "the tail survives: {joined}"
    );
    for r in &rows {
        assert!(
            text(r).chars().count() <= 80,
            "no row exceeds the width: {:?}",
            text(r)
        );
    }
}

#[test]
fn continuations_are_indented_under_the_message() {
    let rows = wrap_line(logline(&"word ".repeat(40)), 60);
    assert!(rows.len() > 1);
    // The prefix is 9 + 6 + 15 = 30 characters wide.
    assert!(
        text(&rows[1]).starts_with(&" ".repeat(30)),
        "continuation lines up under the message, not under the timestamp"
    );
}

#[test]
fn a_zero_width_pane_does_not_panic_or_loop() {
    let rows = wrap_line(logline("anything"), 0);
    assert_eq!(rows.len(), 1);
}
