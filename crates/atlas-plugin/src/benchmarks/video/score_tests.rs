// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for color extraction and the verdict.

use super::*;

const PAL: &[&str] = &["red", "green", "blue", "yellow"];

// ── reading colors out of a reply ───────────────────────────────────────

#[test]
fn a_bare_comma_list_reads_in_order() {
    assert_eq!(
        colors_in_order("red, green, blue, yellow", PAL),
        vec!["red", "green", "blue", "yellow"]
    );
}

#[test]
fn case_and_surrounding_prose_do_not_matter() {
    let reply = "The video shows Red first, then Green, followed by Blue and finally Yellow.";
    assert!(order_matches(
        reply,
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

/// ★ A model that recaps ("...the red one was first") must not be scored as
/// having named red twice. The question asked for a sequence, and first
/// mentions are the sequence.
#[test]
fn a_recap_does_not_duplicate_a_color() {
    let reply = "red, green, blue, yellow — and the red one came first.";
    assert_eq!(
        colors_in_order(reply, PAL),
        vec!["red", "green", "blue", "yellow"]
    );
    assert!(order_matches(
        reply,
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

/// The assertion the benchmark exists for: the reversed clip must NOT match
/// the forward answer.
#[test]
fn the_reversed_sequence_is_a_different_answer() {
    let fwd = "red, green, blue, yellow";
    assert!(order_matches(fwd, &["red", "green", "blue", "yellow"], PAL));
    assert!(
        !order_matches(fwd, &["yellow", "blue", "green", "red"], PAL),
        "forward and reversed must not both match — that would make the pair worthless"
    );
}

/// The exact wording the splice defect produced. It named no palette color,
/// so it must read as NOT SEEN rather than as some partial credit.
#[test]
fn the_grey_field_answer_names_no_colors() {
    let reply = "gray, gray, gray, gray, gray, gray";
    assert!(colors_in_order(reply, PAL).is_empty());
    assert!(!order_matches(
        reply,
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

#[test]
fn color_names_inside_other_words_are_not_evidence() {
    assert!(
        colors_in_order("hundred evergreen blueprints yellowish", PAL).is_empty(),
        "substring hits do not show that the model named a color"
    );
}

#[test]
fn a_partial_sequence_does_not_match() {
    assert!(!order_matches(
        "red, green",
        &["red", "green", "blue", "yellow"],
        PAL
    ));
}

// ── the verdict ──────────────────────────────────────────────────────────

fn ok_order() -> OrderCell {
    OrderCell::Match {
        clip: "c",
        seen: "red, green".into(),
    }
}
fn bad_order() -> OrderCell {
    OrderCell::WrongOrder {
        clip: "c",
        want: "a".into(),
        got: "b".into(),
    }
}
fn ok_count() -> CountCell {
    CountCell::Match {
        id: "x",
        detail: String::new(),
    }
}

#[test]
fn all_legs_passing_with_a_held_control_is_a_pass() {
    assert_eq!(verdict(&[ok_order()], &[ok_count()], true), Verdict::Pass);
}

#[test]
fn any_failing_leg_fails_the_run() {
    assert_eq!(verdict(&[bad_order()], &[ok_count()], true), Verdict::Fail);
    assert_eq!(
        verdict(
            &[OrderCell::NotSeen {
                clip: "c",
                reply: "no idea".into(),
            }],
            &[ok_count()],
            true
        ),
        Verdict::Fail
    );
    assert_eq!(
        verdict(
            &[ok_order()],
            &[CountCell::Mismatch {
                id: "x",
                detail: "wrong geometry".into(),
            }],
            true
        ),
        Verdict::Fail
    );
}

/// ★ Every leg green and the control ALSO green is not a pass. This is the
/// state a server that stopped splicing embeddings would reach if the model's
/// priors happened to be right.
#[test]
fn a_broken_control_makes_the_run_vacuous_not_green() {
    assert_eq!(
        verdict(&[ok_order()], &[ok_count()], false),
        Verdict::Vacuous
    );
}

/// A run with no decoder measures nothing. That must be distinguishable from
/// success — a skipped suite reading as PASS is how a capability silently
/// stops being tested.
#[test]
fn skipping_everything_is_inconclusive_rather_than_pass() {
    let skipped = vec![OrderCell::Skipped {
        clip: "c",
        why: "no ffmpeg".into(),
    }];
    let counts = vec![CountCell::Skipped {
        id: "x",
        why: "no ffmpeg".into(),
    }];
    assert_eq!(verdict(&skipped, &counts, true), Verdict::Inconclusive);
    assert_eq!(asserted(&skipped, &counts), 0);
}

/// A partially-skipped run still judges what it measured.
#[test]
fn a_partial_skip_still_judges_the_rest() {
    let order = vec![
        ok_order(),
        OrderCell::Skipped {
            clip: "c",
            why: "no ffmpeg".into(),
        },
    ];
    assert_eq!(asserted(&order, &[]), 1);
    assert_eq!(verdict(&order, &[], true), Verdict::Pass);
}

#[test]
fn a_leg_that_errored_counts_as_asserted_and_fails() {
    let order = vec![OrderCell::Error {
        clip: "c",
        msg: "request reset".into(),
    }];
    let counts = vec![CountCell::Error {
        id: "x",
        msg: "decode failed".into(),
    }];
    assert_eq!(asserted(&order, &counts), 2);
    assert_eq!(passed(&order, &counts), 0);
    assert_eq!(verdict(&order, &counts, true), Verdict::Fail);
}

#[test]
fn every_verdict_has_an_exact_operator_label() {
    assert_eq!(
        [
            Verdict::Pass,
            Verdict::Fail,
            Verdict::Vacuous,
            Verdict::Inconclusive,
        ]
        .map(|verdict| verdict.to_string()),
        ["PASS", "FAIL", "VACUOUS", "INCONCLUSIVE"]
    );
}
