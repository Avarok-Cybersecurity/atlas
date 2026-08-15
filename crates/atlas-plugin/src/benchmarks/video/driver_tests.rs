// SPDX-License-Identifier: AGPL-3.0-only

//! Descriptor-level tests. The legs themselves need a served model; what is
//! checkable here is that the benchmark is well-formed and honestly
//! registered.

use super::*;

#[test]
fn the_descriptor_is_well_formed() {
    assert_eq!(DESCRIPTOR.id, "video-fidelity");
    assert!(!DESCRIPTOR.name.is_empty());
    assert!(!DESCRIPTOR.summary.is_empty());
    assert!(
        DESCRIPTOR.detail.len() > 200,
        "the detail pane is where an operator learns what the legs mean"
    );
    // Read back through the REGISTRY rather than the local const: asserting
    // on the const directly folds to a constant and clippy rejects it, and
    // this is the stronger claim anyway — it checks the descriptor the runner
    // will actually see. The flag gates a confirmation prompt, so flipping it
    // silently would change how the benchmark is launched.
    let registered = crate::registry::find("video-fidelity").expect("registered");
    assert!(
        !registered.needs_confirmation,
        "it sends chat requests; it runs no shell and does not replace the served model"
    );
}

/// ★ Registered AND required, but declared on exactly one target.
///
/// The per-model constraint lives in BENCH.toml, not here: gate coverage is
/// path-based and has no model dimension, so a `gate = "video-fidelity"` entry
/// exists only where video is validated. A target without one has nothing to
/// run and nothing to satisfy — which is what makes requiring it safe for the
/// text-only and untested targets alike.
#[test]
fn it_is_registered_and_required() {
    assert!(
        crate::registry::find("video-fidelity").is_some(),
        "must be runnable from the registry"
    );
    assert!(
        crate::gate::REQUIRED_GATES.contains(&"video-fidelity"),
        "promoted 2026-08-14 with a measured reference run"
    );
    // Required and excused are mutually exclusive; the coverage test enforces
    // that globally, and this pins the side this benchmark is on.
    assert!(
        !crate::gate::coverage::NOT_REQUIRED
            .iter()
            .any(|(id, _)| *id == "video-fidelity"),
        "must not be excused as well as required"
    );
}

/// Every color a fixture shows has to be one the scorer looks for, or that
/// leg could never pass no matter how well the engine worked.
#[test]
fn the_palette_covers_every_fixture_color() {
    for c in crate::benchmarks::video::provision::CLIPS {
        for color in c.colors {
            assert!(
                PALETTE.contains(color),
                "{color} appears in {} but not in the scorer's palette",
                c.name
            );
        }
    }
}
