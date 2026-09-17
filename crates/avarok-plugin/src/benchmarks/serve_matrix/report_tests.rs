// SPDX-License-Identifier: AGPL-3.0-only

//! What the table shows — in particular, that it shows the rounds that did NOT
//! run. A matrix that renders only the survivors reads as complete.

use super::*;
use crate::benchmarks::serve_matrix::host::{Absence, ServeCandidate};
use crate::benchmarks::serve_matrix::score::{Signal, Signals};
use crate::result::VerdictKind;

fn signals() -> Signals {
    Signals {
        identity: Signal::Pass,
        coherence_pass: 2,
        coherence_total: 2,
        codegen: Signal::Pass,
        tool_call: Signal::NotApplicable("no parser".into()),
        long_ctx: Signal::Pass,
        tps: Some(41.5),
    }
}

#[test]
fn every_candidate_gets_a_row_including_the_ones_that_were_skipped() {
    let plan = Plan::build(
        &[
            ServeCandidate::ready("a", "nvfp4"),
            ServeCandidate::absent("b", "fp8", Absence::NoWeights),
        ],
        "",
    );
    let results = vec![RoundResult {
        label: "a · nvfp4".into(),
        outcome: Outcome::Probed(Box::new(signals())),
        baseline_tps: None,
    }];
    let t = table(&plan, &results);
    assert_eq!(t.rows.len(), 2);
    assert_eq!(
        t.rows[0]
            .iter()
            .map(|cell| cell.text.as_str())
            .collect::<Vec<_>>(),
        [
            "a · nvfp4",
            "PASS",
            "2/2",
            "PASS",
            "N/A",
            "PASS",
            "41.5",
            "PASS",
        ]
    );
    assert_eq!(
        t.rows[0].iter().map(|cell| cell.style).collect::<Vec<_>>(),
        [
            crate::result::CellStyle::Neutral,
            crate::result::CellStyle::Good,
            crate::result::CellStyle::Good,
            crate::result::CellStyle::Good,
            crate::result::CellStyle::Warn,
            crate::result::CellStyle::Good,
            crate::result::CellStyle::Accent,
            crate::result::CellStyle::Good,
        ]
    );
    let skipped = &t.rows[1];
    assert_eq!(
        skipped
            .iter()
            .map(|cell| cell.text.as_str())
            .collect::<Vec<_>>(),
        [
            "b · fp8",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "SKIP · weights not fully downloaded",
        ]
    );
    assert_eq!(
        skipped.iter().map(|cell| cell.style).collect::<Vec<_>>(),
        [
            crate::result::CellStyle::Neutral,
            crate::result::CellStyle::Dim,
            crate::result::CellStyle::Dim,
            crate::result::CellStyle::Dim,
            crate::result::CellStyle::Dim,
            crate::result::CellStyle::Dim,
            crate::result::CellStyle::Dim,
            crate::result::CellStyle::Dim,
        ]
    );
    // The LAST cell, not a magic index: the verdict is the last column and
    // adding a signal column must not silently move what this asserts on.
    let verdict = skipped.last().expect("a verdict cell");
    assert!(
        verdict.text.contains(Absence::NoWeights.reason()),
        "{:?}",
        verdict.text
    );
    assert_eq!(skipped.len(), t.columns.len());
}

#[test]
fn a_planned_round_with_no_result_renders_as_a_failure_not_a_blank() {
    let plan = Plan::build(&[ServeCandidate::ready("a", "")], "");
    let t = table(&plan, &[]);
    let verdict = t.rows[0].last().expect("a verdict cell");
    assert_eq!(verdict.text, "FAIL · no result — did not run");
    assert_eq!(verdict.style, crate::result::CellStyle::Bad);
}

#[test]
fn the_verdict_is_info_rather_than_pass_when_nothing_was_measured() {
    let plan = Plan::build(&[], "");
    let v = verdict(&super::super::score::tally(&plan, &[]), &plan);
    assert_eq!(v.kind, VerdictKind::Info);
    assert!(v.reason.contains("nothing was measured"));
}

#[test]
fn a_round_line_attributes_a_failure_to_the_signal_that_caused_it() {
    let mut s = signals();
    s.codegen = Signal::Fail("no indented body".into());
    let r = RoundResult {
        label: "org/m · nvfp4".into(),
        outcome: Outcome::Probed(Box::new(s)),
        baseline_tps: None,
    };
    let line = round_line(&r);
    assert!(line.contains("org/m · nvfp4"), "{line}");
    assert!(line.contains("codegen FAIL"), "{line}");
    assert!(line.contains("FAIL codegen"), "{line}");
    assert!(line.contains("no baseline"), "{line}");
    assert!(line.contains("no indented body"), "{line}");
}

#[test]
fn a_boot_failure_line_says_so_in_words() {
    let r = RoundResult {
        label: "org/m".into(),
        outcome: Outcome::BootFailed("CUDA out of memory".into()),
        baseline_tps: None,
    };
    assert_eq!(round_line(&r), "org/m: DID NOT BOOT — CUDA out of memory");
}
