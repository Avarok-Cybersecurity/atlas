// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn m(f: &'static str, t: usize) -> GeomCell {
    GeomCell::Match {
        fixture: f,
        tokens: t,
    }
}
fn mm(f: &'static str, want: usize, got: usize) -> GeomCell {
    GeomCell::Mismatch {
        fixture: f,
        want,
        got,
    }
}
fn um(f: &'static str) -> GeomCell {
    GeomCell::Unmeasured {
        fixture: f,
        why: "over encoder capacity".into(),
    }
}
fn pass(id: &'static str) -> ProbeCell {
    ProbeCell::Pass { id }
}
fn fail(id: &'static str) -> ProbeCell {
    ProbeCell::Fail {
        id,
        reply: "nope".into(),
    }
}

#[test]
fn everything_green_with_a_held_control_passes() {
    assert_eq!(verdict(&[m("a", 49)], &[pass("p")], true), Verdict::Pass);
}

#[test]
fn a_failed_control_is_vacuous_even_when_all_green() {
    // THE case. An all-green run whose control also answered is the exact
    // shape of a server replying from language priors with vision detached.
    // Reporting PASS here is the failure mode this benchmark exists to stop.
    assert_eq!(
        verdict(&[m("a", 49)], &[pass("p")], false),
        Verdict::Vacuous,
        "all-green plus a broken control must NOT read as PASS"
    );
}

#[test]
fn geometry_mismatch_fails_on_its_own() {
    // Token counts cannot be faked by priors, so geometry stands alone —
    // it fails the run even with every capability probe green.
    assert_eq!(
        verdict(&[mm("a", 49, 196)], &[pass("p")], true),
        Verdict::Fail
    );
}

#[test]
fn a_failed_probe_fails_the_run() {
    assert_eq!(verdict(&[m("a", 49)], &[fail("p")], true), Verdict::Fail);
}

#[test]
fn a_run_that_asserted_nothing_is_visible_as_such() {
    // All-Unmeasured is green under `verdict`, which is correct but useless.
    // `asserted_cells` is what stops that being silent.
    let cells = [um("a"), um("b")];
    assert_eq!(verdict(&cells, &[], true), Verdict::Pass);
    assert_eq!(
        asserted_cells(&cells),
        0,
        "a reader must be able to see it asserted nothing"
    );
    assert_eq!(
        asserted_cells(&[
            m("a", 1),
            um("b"),
            mm("c", 1, 2),
            GeomCell::Error {
                fixture: "d",
                msg: "decode failed".into(),
            },
        ]),
        2,
        "only measured matches and mismatches contribute to the geometry denominator"
    );
}

#[test]
fn matching_is_case_insensitive_and_honours_want_none() {
    assert!(reply_matches("The colour is RED.", &["red"], &[]));
    assert!(!reply_matches("It is blue.", &["red"], &[]));
    assert!(
        !reply_matches("Maybe red, maybe blue.", &["red"], &["blue"]),
        "want_none must veto a reply that hedges across every option"
    );
    assert!(!reply_matches("The label reads 12800.", &["1280"], &[]));
    assert!(!reply_matches("The shape is squared.", &["square"], &[]));
    assert!(reply_matches("It is square.", &["square"], &[]));
}

#[test]
fn an_empty_want_all_is_satisfied_but_want_none_still_bites() {
    // The control's shape: nothing required, one thing forbidden.
    assert!(reply_matches("I cannot see an image.", &[], &["1280"]));
    assert!(!reply_matches("The label reads 1280x720.", &[], &["1280"]));
}

#[test]
fn errors_count_as_failures_not_as_absences() {
    // A transport error must not be quietly skipped — that would turn an
    // unreachable endpoint into a green run.
    let g = [GeomCell::Error {
        fixture: "a",
        msg: "connection reset".into(),
    }];
    assert_eq!(verdict(&g, &[], true), Verdict::Fail);
    let p = [ProbeCell::Error {
        id: "p",
        msg: "timeout".into(),
    }];
    assert_eq!(verdict(&[], &p, true), Verdict::Fail);
}

#[test]
fn runtime_integrity_and_concurrency_are_part_of_the_final_verdict() {
    assert_eq!(
        with_runtime_checks(Verdict::Pass, false, true),
        Verdict::Pass,
        "clean runtime legs must preserve a pass"
    );
    assert_eq!(
        with_runtime_checks(Verdict::Pass, true, true),
        Verdict::Fail,
        "an integrity failure must demote a pass"
    );
    assert_eq!(
        with_runtime_checks(Verdict::Pass, false, false),
        Verdict::Fail,
        "an incomplete or inconsistent concurrency sweep must demote a pass"
    );
    assert_eq!(
        with_runtime_checks(Verdict::Vacuous, true, false),
        Verdict::Vacuous,
        "a broken control remains the more precise diagnosis"
    );
}

#[test]
fn every_vision_verdict_has_an_exact_operator_label() {
    assert_eq!(
        [Verdict::Pass, Verdict::Fail, Verdict::Vacuous].map(|verdict| verdict.to_string()),
        ["PASS", "FAIL", "VACUOUS"]
    );
}
