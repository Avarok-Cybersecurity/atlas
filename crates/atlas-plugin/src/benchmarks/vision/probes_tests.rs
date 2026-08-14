// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side. These check the probe DEFINITIONS are coherent; whether the
//! model answers them correctly is what the run measures.

use super::*;
use crate::benchmarks::vision::provision::FIXTURES;

#[test]
fn every_referenced_fixture_exists() {
    // A typo in a filename would otherwise surface as a mid-run file-not-found
    // on a GPU box, long after the cheap moment to catch it.
    let known: Vec<&str> = FIXTURES.iter().map(|(n, _, _, _)| *n).collect();
    for p in PROBES.iter().chain(std::iter::once(&CONTROL)) {
        for img in p.images {
            assert!(
                known.contains(img),
                "{}: fixture {img:?} is not in the provisioned set {known:?}",
                p.id
            );
        }
    }
}

#[test]
fn expectations_are_lowercase_and_not_self_contradictory() {
    // Scoring lowercases the reply, so an uppercase expectation can never
    // match — it would fail silently and forever.
    for p in PROBES.iter().chain(std::iter::once(&CONTROL)) {
        for w in p.want_all.iter().chain(p.want_none.iter()) {
            assert_eq!(*w, w.to_lowercase(), "{}: {w:?} must be lowercase", p.id);
        }
        for w in p.want_all {
            assert!(
                !p.want_none.contains(w),
                "{}: {w:?} is in both want_all and want_none",
                p.id
            );
        }
    }
}

#[test]
fn every_probe_asserts_something() {
    // A probe with neither want_all nor want_none passes unconditionally and
    // is worse than no probe, because it inflates the pass count.
    for p in PROBES {
        assert!(
            !p.want_all.is_empty() || !p.want_none.is_empty(),
            "{}: asserts nothing",
            p.id
        );
        assert!(
            !p.images.is_empty(),
            "{}: capability probe with no image",
            p.id
        );
    }
}

#[test]
fn the_control_sends_no_image_and_guards_a_real_probe() {
    // Both halves matter. No image, or it is not a control. And it must guard
    // a token that a REAL probe depends on, or it guards nothing.
    assert!(CONTROL.images.is_empty(), "the control must send no image");
    assert!(!CONTROL.want_none.is_empty(), "the control asserts nothing");

    let guarded: Vec<&str> = CONTROL.want_none.to_vec();
    let protects = PROBES
        .iter()
        .any(|p| p.want_all.iter().any(|w| guarded.contains(w)));
    assert!(
        protects,
        "the control guards {guarded:?}, which no probe actually depends on — \
         so a vacuous capability leg would still report PASS"
    );
}

#[test]
fn probe_ids_are_unique_and_filename_safe() {
    let mut seen = std::collections::BTreeSet::new();
    for p in PROBES.iter().chain(std::iter::once(&CONTROL)) {
        assert!(seen.insert(p.id), "duplicate probe id {}", p.id);
        assert!(
            p.id.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{} is not filename-safe",
            p.id
        );
    }
}
