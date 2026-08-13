// SPDX-License-Identifier: AGPL-3.0-only

//! What the Config pickers and the removed rows actually put on screen.
//!
//! Every assertion here is against GLYPHS — `▌`, `✓`, `✗`, the words
//! "removed" and "server default" — because the harness captures symbols,
//! not styles, which is exactly the NO_COLOR condition: whatever these tests
//! can see is what a colourless terminal still shows. A state that only a
//! hue distinguishes would be invisible here, and failing to assert it is
//! the point.

use crossterm::event::{KeyCode, KeyEvent};

use super::super::tests::{lib, local, recipe};
use crate::tui::app::App;
use crate::tui::render::harness::{has, screen};

fn form() -> App {
    let r = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let model = r.model.clone();
    let mut a = lib(vec![r], vec![local(&model, true)]);
    a.lib.open_cards().expect("cards");
    a.lib.open_config().expect("form");
    a
}

fn press(a: &mut App, code: KeyCode) {
    a.lib.on_key(KeyEvent::from(code));
}

fn select_row(a: &mut App, key: &str) {
    a.lib.row = a
        .lib
        .config_rows()
        .iter()
        .position(|r| r.key == key)
        .unwrap_or_else(|| panic!("{key} is not on the form"));
}

#[test]
fn the_options_picker_names_the_flag_and_marks_the_current_value() {
    let mut a = form();
    select_row(&mut a, "kv_cache_dtype");
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "KV-CACHE-DTYPE"), "titled by flag:\n{rows:#?}");
    // The recipe pins bf16; the ✓ is the mark that survives NO_COLOR.
    assert!(has(&rows, "✓ bf16"), "current value marked:\n{rows:#?}");
    assert!(has(&rows, "▌"), "the cursor bar is a glyph, not a hue");
}

#[test]
fn a_sixteen_row_option_list_scrolls_rather_than_clips() {
    let mut a = form();
    select_row(&mut a, "kv_cache_dtype");
    press(&mut a, KeyCode::Enter);
    press(&mut a, KeyCode::Char('G'));
    // 30 rows is plenty of frame; the constrained dimension is the pane, so
    // force the squeeze with a short terminal instead.
    let rows = screen(&a, 100, 14);
    assert!(
        has(&rows, "fp8k_turbo2v"),
        "the cursor's row is inside the window at the bottom:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "✓ bf16"),
        "the top of the list scrolled out:\n{rows:#?}"
    );
    assert!(
        has(&rows, "16/16"),
        "the clipped list says where you are:\n{rows:#?}"
    );
}

#[test]
fn the_add_picker_lists_flags_with_their_help() {
    let mut a = form();
    press(&mut a, KeyCode::Char('a'));
    let rows = screen(&a, 200, 48);
    assert!(has(&rows, "ADD A SETTING"), "{rows:#?}");
    // A flag the recipe does not pin, with the first line of its clap help
    // beside it — the same words `--help` would give.
    assert!(has(&rows, "block_size"), "{rows:#?}");
    assert!(has(&rows, "KV cache block size"), "{rows:#?}");
}

#[test]
fn a_removed_row_reads_removed_in_words_not_in_colour() {
    let mut a = form();
    select_row(&mut a, "scheduling_policy");
    press(&mut a, KeyCode::Char('x'));
    let rows = screen(&a, 200, 50);
    // The row stays, with the ✗ gutter glyph and the consequence in words:
    // the server default is what actually applies now.
    assert!(has(&rows, "✗"), "the gutter mark:\n{rows:#?}");
    assert!(
        has(&rows, "removed — server default fifo"),
        "the value column names what the server will do:\n{rows:#?}"
    );
    // And the preview command is the proof: the flag is genuinely not passed.
    assert!(
        !has(&rows, "--scheduling-policy"),
        "a removed flag must not survive into the launch command:\n{rows:#?}"
    );
}

#[test]
fn an_added_row_is_marked_and_reaches_the_launch_command() {
    let mut a = form();
    a.lib.overrides.insert("block_size".into(), "32".into());
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "+ block_size"), "the + gutter mark:\n{rows:#?}");
    assert!(has(&rows, "--block-size 32"), "{rows:#?}");
}

#[test]
fn removals_count_toward_the_changed_tally_in_the_title() {
    let mut a = form();
    select_row(&mut a, "scheduling_policy");
    press(&mut a, KeyCode::Char('x'));
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "1 changed"),
        "a removal is a change to the launch:\n{rows:#?}"
    );
}

#[test]
fn the_footer_teaches_the_picker_keys_while_one_is_open() {
    let a = form();
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "a add"), "add is discoverable:\n{rows:#?}");
    assert!(has(&rows, "x remove"), "remove is discoverable:\n{rows:#?}");
    let mut a = form();
    select_row(&mut a, "kv_cache_dtype");
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "⏎ select · Esc cancel"),
        "the footer answers for the option picker:\n{rows:#?}"
    );
    // The add picker has a second scroll surface, and its Enter ADDS: the
    // generic picker hint would be wrong on both counts.
    let mut a = form();
    press(&mut a, KeyCode::Char('a'));
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "J/K scroll help · ⏎ add"),
        "the footer answers for the add picker's own keys:\n{rows:#?}"
    );
}

#[test]
fn the_pickers_render_at_every_size_without_panicking() {
    // The same underflow sweep the whole dashboard gets: a Rect computed past
    // the frame takes the server's foreground down with it.
    for (w, h) in [(160u16, 48u16), (100, 30), (80, 24), (40, 12), (12, 4)] {
        for open in ["options", "add"] {
            let mut a = form();
            if open == "options" {
                select_row(&mut a, "kv_cache_dtype");
                press(&mut a, KeyCode::Enter);
            } else {
                press(&mut a, KeyCode::Char('a'));
            }
            let out = screen(&a, w, h);
            assert!(!out.is_empty(), "{open} at {w}x{h} drew nothing");
        }
    }
}

/// The flagship's form with the 27B recipe in the index as a donor.
fn form_with_donor() -> App {
    let flagship = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let donor = recipe("qwen3.6-27b-nvfp4");
    let model = flagship.model.clone();
    let mut a = lib(vec![flagship, donor], vec![local(&model, true)]);
    a.lib.open_cards().expect("cards");
    a.lib.open_config().expect("form");
    a
}

#[test]
fn the_borrow_picker_names_each_donors_measured_model() {
    let mut a = form_with_donor();
    press(&mut a, KeyCode::Char('b'));
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "BORROW PARAMETERS FROM"), "{rows:#?}");
    assert!(has(&rows, "qwen3.6/qwen3.6-27b-nvfp4"), "{rows:#?}");
    // The provenance is on the row, not discovered after applying: the point
    // of this picker is that the values belong to another checkpoint.
    assert!(
        has(&rows, "measured on nvidia/Qwen3.6-27B-NVFP4"),
        "{rows:#?}"
    );
}

#[test]
fn the_preview_shows_old_to_new_and_says_it_is_not_a_measurement() {
    let mut a = form_with_donor();
    press(&mut a, KeyCode::Char('b'));
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 200, 50);
    // The `→` glyph and the column order are the NO_COLOR signal; the hues
    // only reinforce them.
    assert!(has(&rows, "65536 → 32768"), "old beside new:\n{rows:#?}");
    assert!(
        has(&rows, "not set → qwen3_coder"),
        "an added key says it was absent:\n{rows:#?}"
    );
    assert!(
        has(&rows, "not measured on this model"),
        "the honesty header:\n{rows:#?}"
    );
    assert!(
        has(&rows, "keep their values"),
        "what the borrow does NOT touch is stated:\n{rows:#?}"
    );
    // The footer answers for the preview's own Enter, which applies rather
    // than selects.
    assert!(has(&rows, "⏎ apply changes"), "{rows:#?}");
}

#[test]
fn an_applied_borrow_marks_the_form_with_its_provenance() {
    let mut a = form_with_donor();
    press(&mut a, KeyCode::Char('b'));
    press(&mut a, KeyCode::Enter);
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "borrowed — values from qwen3.6/qwen3.6-27b-nvfp4"),
        "the form says where its values came from:\n{rows:#?}"
    );
    assert!(
        has(&rows, "not a measurement for this model"),
        "and that they are copies, in words that survive NO_COLOR:\n{rows:#?}"
    );
    // The changed rows carry the same green-dot gutter any edit gets.
    assert!(has(&rows, "• max_model_len"), "{rows:#?}");
}

#[test]
fn the_borrow_surfaces_render_at_every_size_without_panicking() {
    for (w, h) in [(160u16, 48u16), (100, 30), (80, 24), (40, 12), (12, 4)] {
        for depth in [1u8, 2] {
            let mut a = form_with_donor();
            press(&mut a, KeyCode::Char('b'));
            if depth == 2 {
                press(&mut a, KeyCode::Enter);
            }
            let out = screen(&a, w, h);
            assert!(
                !out.is_empty(),
                "borrow depth {depth} at {w}x{h} drew nothing"
            );
        }
    }
}

/// The add picker, cursor parked on `check_kernels` — whose clap help runs to
/// four paragraphs, the case the side panel exists for.
fn add_picker_on_check_kernels(w: u16, h: u16) -> (App, Vec<String>) {
    let mut a = form();
    press(&mut a, KeyCode::Char('a'));
    let Some(crate::tui::lib_modal::ConfigModal::Add { fields, .. }) = &a.lib.modal else {
        panic!("add picker open");
    };
    let target = fields
        .iter()
        .position(|f| f.key == "check_kernels")
        .expect("check_kernels is addable");
    for _ in 0..target {
        press(&mut a, KeyCode::Char('j'));
    }
    let rows = screen(&a, w, h);
    (a, rows)
}

#[test]
fn the_side_panel_shows_the_help_beyond_the_first_line() {
    let (_, rows) = add_picker_on_check_kernels(200, 50);
    assert!(
        has(&rows, "CHECK_KERNELS"),
        "titled by the flag:\n{rows:#?}"
    );
    // "CLAMPED" is in the third paragraph — text the one-line row never
    // carried anywhere before the panel.
    assert!(
        has(&rows, "CLAMPED"),
        "a later paragraph is readable:\n{rows:#?}"
    );
}

#[test]
fn the_panel_follows_the_cursor_and_shift_j_scrolls_it() {
    // A SHORT terminal, so the help genuinely overflows the panel — at 50
    // rows the whole text fits and there is nothing to scroll.
    let (mut a, rows) = add_picker_on_check_kernels(200, 24);
    // A needle from the help's first paragraph that the LIST row's truncated
    // help line cannot contain — the panel is the only place it can appear.
    let panel_only = "reporting any";
    assert!(has(&rows, panel_only), "top of the help:\n{rows:#?}");
    // The panel names its own scroll keys where the scrolling happens.
    assert!(has(&rows, "J/K 1/"), "position and binding:\n{rows:#?}");
    for _ in 0..8 {
        press(&mut a, KeyCode::Char('J'));
    }
    let rows = screen(&a, 200, 24);
    assert!(
        !has(&rows, panel_only),
        "the opening lines scrolled away:\n{rows:#?}"
    );
    assert!(has(&rows, "J/K 9/"), "{rows:#?}");
    // Moving the cursor swaps the panel's contents for the new row's help.
    press(&mut a, KeyCode::Char('j'));
    let rows = screen(&a, 200, 24);
    assert!(
        !has(&rows, "CHECK_KERNELS"),
        "the panel answers for the highlighted row only:\n{rows:#?}"
    );
}

#[test]
fn a_narrow_terminal_drops_the_panel_whole_and_keeps_the_list() {
    // 80 columns cannot hold 50 of list beside 34 of panel: the stated
    // fallback is the single list the picker always was.
    let (_, rows) = add_picker_on_check_kernels(80, 30);
    assert!(
        has(&rows, "ADD A SETTING"),
        "the picker survives:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "CHECK_KERNELS"),
        "no panel sliver at 80 columns:\n{rows:#?}"
    );
    // The clipped row announces the cut instead of posing as complete.
    assert!(has(&rows, "…"), "ellipsis on clipped rows:\n{rows:#?}");
}
