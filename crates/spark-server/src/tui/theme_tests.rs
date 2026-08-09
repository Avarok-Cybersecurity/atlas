// SPDX-License-Identifier: AGPL-3.0-only

//! Colour-depth resolution and the signals that must survive losing colour.
//!
//! `depth_of` takes the two variables as values precisely so these can be
//! tested without `set_var` — the environment is process-global, and
//! `logo_tests.rs` already pins `COLORTERM` for the whole binary.

use super::*;

#[test]
fn a_preference_outranks_a_capability() {
    // The terminal can do 24-bit and says so; the user has still said no.
    assert_eq!(
        depth_of(Some("1"), Some("truecolor")),
        Depth::None,
        "COLORTERM describes the terminal, NO_COLOR describes the user"
    );
}

/// The standard failure: treating `NO_COLOR` as a boolean. It is a presence
/// test — `NO_COLOR=0` means no colour, and only an empty value is ignored.
#[test]
fn no_color_is_presence_not_truth() {
    for set_to in ["1", "0", "false", "no", "yes", "anything at all"] {
        assert_eq!(
            depth_of(Some(set_to), None),
            Depth::None,
            "NO_COLOR={set_to}"
        );
    }
    assert_eq!(
        depth_of(Some(""), Some("truecolor")),
        Depth::True,
        "an empty value is explicitly not set"
    );
    assert_eq!(depth_of(None, Some("truecolor")), Depth::True);
}

#[test]
fn colorterm_still_picks_the_fallback_when_colour_is_allowed() {
    assert_eq!(depth_of(None, Some("24bit")), Depth::True);
    assert_eq!(depth_of(None, Some("truecolor")), Depth::True);
    assert_eq!(depth_of(None, None), Depth::Ansi256);
    assert_eq!(depth_of(None, Some("8bit")), Depth::Ansi256);
}

/// Every signal the palette carries in hue alone has to survive `NO_COLOR`, or
/// the flag does not produce a usable UI — it produces a blank one.
///
/// This asserts the shape rather than the live depth, because the live depth
/// is whatever the harness's environment says. `selected()` and `border(true)`
/// are the two that would otherwise vanish entirely: a list you cannot see
/// your position in, and a pane you cannot tell has focus.
#[test]
fn the_signals_that_are_only_colour_get_a_modifier_instead() {
    let colourless = |s: Style| s.fg.is_none() && s.bg.is_none();

    // Under no colour, both fall back to a modifier and carry no hue.
    let sel = Style::default().add_modifier(Modifier::REVERSED);
    assert!(colourless(sel) && sel.add_modifier.contains(Modifier::REVERSED));

    // And with colour they are the surface styles they always were.
    assert_eq!(
        depth_of(None, Some("truecolor")),
        Depth::True,
        "the coloured branch is the one the palette constants describe"
    );

    // The live functions must agree with whichever branch this process is in.
    match depth() {
        Depth::None => {
            assert!(colourless(selected()), "{:?}", selected());
            assert!(selected().add_modifier.contains(Modifier::REVERSED));
            assert!(border(true).add_modifier.contains(Modifier::BOLD));
            assert!(warn().add_modifier.contains(Modifier::BOLD));
            assert_eq!(gradient_at(0.5), Color::Reset);
            assert_eq!(glow(3), Color::Reset);
            assert_eq!(TEXT.color(), Color::Reset);
        }
        _ => {
            assert_eq!(selected().bg, Some(BG_SELECTION.color()));
            assert_eq!(border(true).fg, Some(CYAN.color()));
            assert_ne!(TEXT.color(), Color::Reset);
        }
    }
}
