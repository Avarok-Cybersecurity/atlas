// SPDX-License-Identifier: AGPL-3.0-only

//! Host-side. These check the probe DEFINITIONS are coherent; whether the
//! model answers them correctly is what the run measures.

use super::*;
use crate::benchmarks::vision::provision::FIXTURES;
use crate::benchmarks::vision::score::reply_matches;

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
            assert!(
                !w.is_empty(),
                "{}: empty expectation matches every reply",
                p.id
            );
            assert_eq!(*w, w.trim(), "{}: {w:?} has invisible padding", p.id);
            assert_eq!(*w, w.to_lowercase(), "{}: {w:?} must be lowercase", p.id);
        }
        for required in p.want_all {
            for forbidden in p.want_none {
                assert!(
                    !required.contains(forbidden),
                    "{}: required {required:?} necessarily contains forbidden {forbidden:?}",
                    p.id
                );
            }
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
        assert!(!p.id.is_empty(), "probe id must not be empty");
        assert!(seen.insert(p.id), "duplicate probe id {}", p.id);
        assert!(
            p.id.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{} is not filename-safe",
            p.id
        );
    }
}

#[test]
fn probe_definitions_match_the_committed_measurement() {
    let got: Vec<_> = PROBES
        .iter()
        .chain(std::iter::once(&CONTROL))
        .map(|p| (p.id, p.images, p.prompt, p.want_all, p.want_none))
        .collect();
    assert_eq!(
        got,
        vec![
            (
                "sees-an-image",
                &["01_square_224.png"][..],
                "Describe what you see in this image in one short sentence.",
                &[][..],
                &["cannot see", "no image", "unable to see", "don't see"][..],
            ),
            (
                "reads-the-size-label",
                &["07_hd_1280x720.png"][..],
                "This image has a size label drawn on it. Read the label exactly.",
                &["1280"][..],
                &["cannot see", "no image"][..],
            ),
            (
                "multi-image-order",
                &["01_square_224.png", "08_portrait_480x854.png"][..],
                "You are shown two images. Is the FIRST one square or portrait? Answer with one word.",
                &["square"][..],
                &["portrait"][..],
            ),
            (
                "control-no-image",
                &[][..],
                "This image has a size label drawn on it. Read the label exactly.",
                &[][..],
                &["1280"][..],
            ),
        ]
    );
}

#[test]
fn the_concurrency_probe_requires_image_specific_evidence() {
    let probe = concurrency_probe();
    assert_eq!(probe.images, &["07_hd_1280x720.png"]);
    assert!(
        !reply_matches("OK", probe.want_all, probe.want_none),
        "a generic non-empty reply does not show that concurrent vision worked"
    );
    assert!(reply_matches(
        "The label reads 1280 x 720.",
        probe.want_all,
        probe.want_none
    ));
}
