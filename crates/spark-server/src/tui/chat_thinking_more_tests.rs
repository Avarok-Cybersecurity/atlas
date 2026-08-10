// SPDX-License-Identifier: AGPL-3.0-only

//! The narrow chip, the view labels, and the two clocks.
//!
//! A second test file rather than a longer first one: these pin what the pane
//! DRAWS, and the widths in particular are claims about an 80-column terminal
//! that nothing else in the tree keeps honest.

use super::*;

#[test]
fn the_narrow_chip_drops_the_observation_rather_than_the_state() {
    // At 80 columns the header has no room for a suffix. What must survive is
    // WHICH state is active — a chip that truncated to "think" would leave the
    // toggle unreadable, and the suffix is only ever a nicety.
    assert_eq!(ThinkingRequest::Auto.chip(None, false), "think auto");
    assert_eq!(ThinkingRequest::Auto.chip(Some(true), false), "think auto");
    assert_eq!(ThinkingRequest::Off.chip(Some(false), false), "think off");
    assert_eq!(ThinkingRequest::On.chip(None, false), "think on");
    for r in [
        ThinkingRequest::Auto,
        ThinkingRequest::Off,
        ThinkingRequest::On,
    ] {
        let chip = r.chip(Some(false), false);
        assert!(
            chip.len() <= 10,
            "narrow chip must fit a tight header: {chip}"
        );
    }
}

#[test]
fn every_view_has_a_label_and_no_two_share_one() {
    let labels: Vec<&str> = [
        ThinkingView::Collapsed,
        ThinkingView::Expanded,
        ThinkingView::Hidden,
    ]
    .iter()
    .map(|v| v.label())
    .collect();
    assert_eq!(labels, vec!["collapsed", "expanded", "hidden"]);
}

#[test]
fn the_request_cycle_reaches_off_in_one_press() {
    // "Stop thinking at me" is the reason anyone reaches for the key, so it
    // sits one press from the default; On is two.
    assert_eq!(ThinkingRequest::default(), ThinkingRequest::Auto);
    assert_eq!(ThinkingRequest::Auto.next(), ThinkingRequest::Off);
    assert_eq!(ThinkingRequest::Off.next(), ThinkingRequest::On);
    assert_eq!(ThinkingRequest::On.next(), ThinkingRequest::Auto);
}

#[test]
fn the_view_cycle_shows_the_trace_before_it_hides_it() {
    assert_eq!(ThinkingView::default(), ThinkingView::Collapsed);
    assert_eq!(ThinkingView::Collapsed.next(), ThinkingView::Expanded);
    assert_eq!(ThinkingView::Expanded.next(), ThinkingView::Hidden);
    assert_eq!(ThinkingView::Hidden.next(), ThinkingView::Collapsed);
}

#[test]
fn sealing_a_clock_that_never_started_leaves_it_unmeasured() {
    // A model that answered without thinking still runs through `seal` on its
    // first token; inventing a 0 ms span there would caption every plain
    // reply with a thinking summary it never earned.
    let mut r = Reasoning::default();
    r.seal();
    assert_eq!(r.think_ms, None);
    assert_eq!(r.seconds(), None);
    assert!(r.is_empty());
}

#[test]
fn the_first_seal_wins_and_a_later_one_cannot_move_it() {
    // Reasoning that keeps arriving after the answer began is normal; a second
    // seal would restart the summary under a reply already being read.
    let mut r = Reasoning::default();
    r.begin();
    std::thread::sleep(std::time::Duration::from_millis(5));
    r.seal();
    let first = r.think_ms.expect("sealed");
    std::thread::sleep(std::time::Duration::from_millis(5));
    r.seal();
    assert_eq!(r.think_ms, Some(first));
}

#[test]
fn a_trace_is_only_empty_until_text_arrives() {
    let mut r = Reasoning::default();
    assert!(r.is_empty());
    // Whitespace is text: the model emits leading newlines and a trace that
    // reported itself empty would be dropped from the pane entirely.
    r.text.push('\n');
    assert!(!r.is_empty());
}

#[test]
fn the_live_timer_runs_until_the_measurement_arrives() {
    let mut r = Reasoning::default();
    r.begin();
    let live = r.seconds().expect("a live elapsed");
    assert!(live < 1.0, "just started, not {live}");
    r.think_ms = Some(18_200.0);
    assert_eq!(r.seconds(), Some(18.2), "the measurement wins");
}

#[test]
fn durations_switch_units_at_the_boundaries_not_near_them() {
    assert_eq!(dur(0.0), "0ms");
    assert_eq!(dur(0.9994), "999ms");
    assert_eq!(dur(1.0), "1.0s");
    assert_eq!(dur(59.94), "59.9s");
    assert_eq!(dur(60.0), "1m 00s");
    assert_eq!(dur(3599.0), "59m 59s");
    assert_eq!(dur(3600.0), "60m 00s");
}
