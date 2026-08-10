// SPDX-License-Identifier: AGPL-3.0-only

//! The frame chrome: sidebar, footer, toasts, the key map, and the two text
//! primitives every pane wraps with.
//!
//! These are drawn on EVERY frame regardless of section, so a defect here is a
//! defect everywhere — and they are also where the width arithmetic lives, so
//! they are what a 20-column terminal breaks first.

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Size;

use super::harness::{has, screen};
use super::tests::app;
use super::{Chrome, gradient_bar, live_model_name, overlay, wrap};
use crate::tui::app::{Focus, Section};
use crate::tui::theme;

/// Bottom row of an `h`-tall frame — where `draw_footer` always lands.
fn footer(rows: &[String]) -> &str {
    rows.last().map(String::as_str).unwrap_or("")
}

#[test]
fn the_sidebar_spells_out_its_sections_only_when_there_is_room_for_the_words() {
    // 96 columns is the switch between the 18-cell labelled rail and the
    // 4-cell icon rail; below it the labels would be cut mid-word.
    let a = app();
    let wide = screen(&a, 96, 40);
    assert!(has(&wide, "Benchmarks"), "{wide:#?}");
    assert!(has(&wide, "Library"));

    let narrow = screen(&a, 95, 40);
    assert!(
        !has(&narrow, "Benchmarks"),
        "the icon rail has no room for a label:\n{narrow:#?}"
    );
    assert!(
        narrow.iter().any(|r| r.contains('▰')),
        "but the icon is still there:\n{narrow:#?}"
    );
}

#[test]
fn the_active_section_lists_its_subsections_beneath_it() {
    let mut a = app();
    a.section = Section::Main;
    let rows = screen(&a, 120, 40);
    assert!(has(&rows, "Overview"), "{rows:#?}");
    assert!(has(&rows, "Kernels"));
    assert!(
        !has(&rows, "Suite"),
        "an inactive section keeps its subsections folded away:\n{rows:#?}"
    );
}

#[test]
fn the_footer_names_keys_that_do_something_in_the_section_it_is_showing() {
    for (section, needle) in [
        (Section::Main, "f filter"),
        (Section::Stats, "cycle"),
        (Section::Network, "node"),
        (Section::Library, "search"),
        (Section::Benchmarks, "configure"),
        (Section::Terminal, "Ops↔Chat"),
    ] {
        let mut a = app();
        a.section = section;
        let rows = screen(&a, 160, 40);
        assert!(
            footer(&rows).contains(needle),
            "{} footer is missing {needle:?}: {:?}",
            section.label(),
            footer(&rows)
        );
    }
}

#[test]
fn the_mode_pill_says_who_owns_the_keyboard() {
    let mut a = app();
    assert!(footer(&screen(&a, 120, 40)).contains("NORMAL"));

    a.focus = Focus::Input;
    assert!(footer(&screen(&a, 120, 40)).contains("INPUT"));

    // Help wins over focus: while the overlay is up, the keys it lists are the
    // ones that work.
    a.help_open = true;
    assert!(footer(&screen(&a, 120, 40)).contains("HELP"));
}

#[test]
fn the_help_overlay_shows_every_key_it_documents_including_the_last() {
    // The defect: the modal was a hardcoded 18 rows tall, and everything past
    // the sixteenth entry sat below its bottom border.
    let mut a = app();
    a.help_open = true;
    let rows = screen(&a, 160, 48);
    for (key, _) in overlay::KEYS {
        // The padded key column, so a one-character key like `/` cannot be
        // satisfied by an unrelated slash elsewhere on the frame.
        let column = format!("  {key:<16}");
        assert!(
            has(&rows, &column),
            "the {key:?} row is missing:\n{rows:#?}"
        );
    }
    assert!(
        has(&rows, "this help"),
        "the LAST entry is the one that fell below the border:\n{rows:#?}"
    );
}

#[test]
fn the_help_overlay_shrinks_rather_than_drawing_outside_a_short_frame() {
    let mut a = app();
    a.help_open = true;
    for (w, h) in [(20u16, 6u16), (40, 8), (60, 12), (1, 1)] {
        let rows = screen(&a, w, h);
        assert_eq!(rows.len(), h as usize, "{w}x{h} drew a partial frame");
    }
}

#[test]
fn a_toast_is_boxed_and_ellipsised_rather_than_spilling_past_its_border() {
    let mut a = app();
    a.toast(
        "recipe qwen3.6/flagship launched against nvidia/Qwen3.6-27B-NVFP4 on port 8123",
        false,
    );
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "╭"), "the box is drawn:\n{rows:#?}");
    assert!(
        has(&rows, "…"),
        "and the message is cut at the border, not past it:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "port 8123"),
        "the tail is inside the ellipsis, not on the pane behind it:\n{rows:#?}"
    );
}

#[test]
fn a_failure_toast_is_bordered_differently_from_a_success_one() {
    // The border colour is the whole signal — the text alone does not say
    // whether something worked. Compared across every rounded corner on the
    // frame, because the panels underneath draw corners of their own.
    let corners = |error: bool| -> Vec<String> {
        let mut a = app();
        a.toast("something happened", error);
        let mut t = Terminal::new(TestBackend::new(160, 48)).expect("backend");
        t.draw(|f| super::draw(f, &a)).expect("draw");
        let buf = t.backend().buffer();
        let mut styles: Vec<String> = (0..160u16)
            .flat_map(|x| (0..48u16).map(move |y| (x, y)))
            .filter(|(x, y)| buf[(*x, *y)].symbol() == "╭")
            .map(|(x, y)| format!("{:?}", buf[(x, y)].style()))
            .collect();
        styles.sort();
        styles
    };
    assert_ne!(corners(false), corners(true));
}

#[test]
fn a_toast_is_dropped_rather_than_clipped_when_the_pane_cannot_hold_a_box() {
    let mut a = app();
    a.toast("saved", false);
    // Eight columns leaves the content pane two cells wide — narrower than a
    // border plus one character of message.
    let rows = screen(&a, 8, 12);
    assert!(
        !has(&rows, "saved"),
        "a box that cannot be drawn honestly is skipped, not clipped:\n{rows:#?}"
    );
}

#[test]
fn only_the_three_newest_toasts_are_on_screen() {
    let mut a = app();
    for i in 0..5 {
        a.toast(format!("toast-{i}"), false);
    }
    let rows = screen(&a, 160, 48);
    for i in 2..5 {
        assert!(has(&rows, &format!("toast-{i}")), "{rows:#?}");
    }
    assert!(
        !has(&rows, "toast-1"),
        "the fourth-newest is off screen:\n{rows:#?}"
    );
}

#[test]
fn the_toast_text_is_cut_exactly_at_the_width_it_was_given() {
    assert_eq!(overlay::truncate_toast("anything", 0), "");
    assert_eq!(overlay::truncate_toast("exact", 5), "exact");
    assert_eq!(overlay::truncate_toast("exactly", 5), "exac…");
    // Counted in CHARS: a byte count would cut a multi-byte glyph in half and
    // put a replacement character on screen.
    assert_eq!(overlay::truncate_toast("日本語のテキスト", 4), "日本語…");
    assert_eq!(overlay::truncate_toast("ok", 1), "…");
}

#[test]
fn the_gradient_bar_is_exactly_as_wide_as_it_was_asked_to_be() {
    for width in [1u16, 2, 12, 40] {
        for frac in [0.0, 0.25, 0.5, 0.999, 1.0] {
            let line = gradient_bar(frac, width);
            assert_eq!(
                line.spans.len(),
                width as usize,
                "{frac} at width {width} drew {} cells",
                line.spans.len()
            );
        }
    }
    // Zero width would be an invisible bar, so it is floored at one cell.
    assert_eq!(gradient_bar(0.5, 0).spans.len(), 1);
}

#[test]
fn the_gradient_bar_fills_in_proportion_and_clamps_outside_zero_to_one() {
    let glyphs = |frac: f64| -> String {
        gradient_bar(frac, 10)
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect()
    };
    assert_eq!(glyphs(0.0), "░".repeat(10));
    assert_eq!(glyphs(1.0), "█".repeat(10));
    // The leading edge is the half-block, so a partial bar reads as moving.
    assert_eq!(glyphs(0.5), format!("{}▓{}", "█".repeat(4), "░".repeat(5)));
    assert_eq!(glyphs(-3.0), glyphs(0.0), "a negative fraction clamps");
    assert_eq!(glyphs(7.0), glyphs(1.0), "and so does one over one");
}

#[test]
fn wrapping_to_a_zero_width_pane_yields_nothing_rather_than_looping() {
    assert!(wrap("some text", 0, theme::text()).is_empty());
}

#[test]
fn a_token_wider_than_the_pane_is_split_rather_than_clipped_at_the_edge() {
    // `Paragraph` does not wrap what these panes hand it, so an unbreakable
    // 500-character URL left whole loses everything past the panel border.
    let url = format!("https://example.invalid/{}", "a".repeat(480));
    let text = format!("see {url} for details");
    let lines = wrap(&text, 40, theme::text());
    for line in &lines {
        let w: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
        assert!(w <= 40, "a {w}-column row in a 40-column pane");
    }
    let joined: String = lines
        .iter()
        .flat_map(|l| l.spans.iter().map(|s| s.content.to_string()))
        .collect();
    assert!(joined.contains(&url), "no characters were dropped");
    assert!(joined.ends_with("for details"), "and the tail survived");
}

#[test]
fn wrapping_breaks_between_words_and_keeps_them_whole() {
    let lines = wrap("alpha beta gamma delta", 12, theme::text());
    let rows: Vec<String> = lines
        .iter()
        .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
        .collect();
    assert_eq!(rows, vec!["alpha beta", "gamma delta"]);
}

#[test]
fn the_live_model_name_prefers_what_is_serving_over_what_was_asked_for() {
    let a = app();
    assert_eq!(live_model_name(&a), "nvidia/Qwen3.6-27B-NVFP4");

    let mut none = app();
    none.args.model = None;
    none.args.model_name = None;
    assert_eq!(
        live_model_name(&none),
        "",
        "an empty name, never a placeholder that reads like a model id"
    );
}

#[test]
fn every_section_draws_a_whole_frame_at_hostile_geometries() {
    // Layout maths underflows at the extremes, and a panic here takes the
    // server's foreground down with the dashboard.
    for section in Section::ALL {
        for (w, h) in [
            (1u16, 1u16),
            (2, 2),
            (20, 4),
            (20, 8),
            (20, 40),
            (40, 3),
            (200, 3),
        ] {
            let mut a = app();
            a.section = section;
            let rows = screen(&a, w, h);
            assert_eq!(
                rows.len(),
                h as usize,
                "{} at {w}x{h} drew a partial frame",
                section.label()
            );
        }
    }
}

/// The prompt has to say WHAT is in flight. "Are you sure?" tells the user
/// nothing they did not already know, and the thing they are weighing is the
/// hours already spent against getting their prompt back.
#[test]
fn the_quit_prompt_names_the_work_it_is_about_to_destroy() {
    let mut a = app();
    a.chat.streaming = true;
    a.confirm_quit = true;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "STOP THE SERVER?"), "{rows:#?}");
    assert!(has(&rows, "a chat reply is still streaming"), "{rows:#?}");
    assert!(has(&rows, "quit anyway"), "{rows:#?}");
    assert!(has(&rows, "stay"), "{rows:#?}");
}

/// It outranks the help modal — a question the user must answer beats a
/// reference they were browsing — and it must not draw outside a small frame,
/// the same failure the key map had.
#[test]
fn the_quit_prompt_covers_the_help_overlay_and_fits_any_frame() {
    let mut a = app();
    a.chat.streaming = true;
    a.confirm_quit = true;
    a.help_open = true;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "STOP THE SERVER?"), "{rows:#?}");

    for (w, h) in [(20u16, 6u16), (40, 8), (60, 12), (1, 1)] {
        let rows = screen(&a, w, h);
        assert_eq!(rows.len(), h as usize, "{w}x{h} drew a partial frame");
    }
}

/// Nothing in flight means nothing to warn about: the modal declines to draw
/// rather than asking about work that does not exist.
#[test]
fn the_quit_prompt_declines_when_the_dashboard_is_idle() {
    let mut a = app();
    a.confirm_quit = true;
    assert!(a.work_in_flight().is_none());
    assert!(!has(&screen(&a, 160, 48), "STOP THE SERVER?"));
}

/// ★ The mouse handler used to hold its own copy of both sidebar breakpoints,
/// in another file, with nothing tying them to what is actually drawn. Every
/// number below was stated twice; had either copy drifted, a click would have
/// selected the section above or below the one under the pointer, silently.
///
/// This is the renderer's half of the invariant — that the frame really is laid
/// out where [`Chrome`] says. `events_tests` owns the other half: that a click
/// at `chrome.header_h + n` selects the section drawn on row `n`.
#[test]
fn the_frame_is_laid_out_where_the_chrome_says_it_is() {
    let a = app();
    // Both breakpoints, from both sides: a tall/wide frame, and one column and
    // one row short of each.
    for (w, h) in [(96u16, 40u16), (95, 40), (120, 28), (120, 27)] {
        let chrome = Chrome::of(Size {
            width: w,
            height: h,
        });
        let rows = screen(&a, w, h);
        let first_sidebar_row = rows
            .iter()
            .position(|r| r.contains(Section::Main.icon()))
            .unwrap_or_else(|| panic!("{w}x{h} drew no sidebar:\n{rows:#?}"));
        assert_eq!(
            first_sidebar_row, chrome.header_h as usize,
            "{w}x{h}: the sidebar starts where the header ends"
        );
        // The labelled rail and the icon rail differ by exactly the label.
        assert_eq!(
            has(&rows, Section::Benchmarks.label()),
            chrome.full_sidebar(),
            "{w}x{h}: labels iff the wide rail:\n{rows:#?}"
        );
    }
}
