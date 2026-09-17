// SPDX-License-Identifier: AGPL-3.0-only

//! Descriptor-level tests. The legs themselves need a served model; what is
//! checkable here is that the benchmark is well-formed and honestly
//! registered.

use super::*;
use sha2::Digest;

#[test]
fn the_registered_descriptor_matches_the_measurement() {
    let registered = crate::registry::find("video-fidelity").expect("registered");
    assert_eq!(
        (
            registered.id,
            registered.name,
            registered.summary,
            registered.duration_hint,
            registered.updated,
            registered.needs_confirmation,
            registered.intended_for.is_none(),
            registered.threshold_params,
            registered.sensitivity,
        ),
        (
            "video-fidelity",
            "Video Fidelity",
            "Video fidelity: temporal-order reading of a color sequence, group-count geometry, \
             MP4/GIF backend parity, and a no-video control.",
            "~1-2 min",
            "2026-08-24",
            false,
            true,
            &[][..],
            crate::hardware::Sensitivity::Correctness,
        )
    );
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(registered.detail)),
        "0b7193e1765ef69a0697c9fa1ed79dff988148faca4d6c0a9f33d36a00fd7c5a"
    );
}

/// Every color a fixture shows has to be one the scorer looks for, or that
/// leg could never pass no matter how well the engine worked.
#[test]
fn the_palette_covers_every_fixture_color() {
    assert_eq!(PALETTE, &["red", "green", "blue", "yellow"]);
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

/// ★ A decoder-unavailable transport error must come back as a SKIP.
///
/// The heterogeneous-concurrency leg is the one place a cell arrives through
/// the modality-agnostic `media_integrity` helper, which reports any transport
/// failure as `Error`. On a serve without `--video-allow-ffmpeg` that single
/// leg FAILED the whole run while the other twelve skipped (observed
/// 2026-08-15 under `--pull-request-gate`), contradicting the descriptor's
/// "skipped, never failed" contract. The remap is exercised with the server's
/// real refusal wording, which `is_decoder_unavailable` matches on.
#[test]
fn a_decoder_unavailable_error_from_the_shared_helper_becomes_a_skip() {
    use crate::benchmarks::media_integrity::Cell;
    let refused = "HTTP 400: Video decode error: this container needs ffmpeg to decode and \
                   subprocess decoding is disabled; pass --video-allow-ffmpeg to enable it";
    let got = skip_if_decoder_unavailable(Cell::Error {
        id: "heterogeneous-concurrency",
        msg: refused.to_string(),
    });
    assert_eq!(
        got,
        Cell::Skipped {
            id: "heterogeneous-concurrency",
            why: refused.to_string(),
        }
    );

    // Any other error stays an error — a timeout or a 500 is a real failure,
    // and reclassifying it would let a broken serve read as a deployment
    // choice.
    let other = Cell::Error {
        id: "heterogeneous-concurrency",
        msg: "request timed out after 300 s".to_string(),
    };
    assert_eq!(skip_if_decoder_unavailable(other.clone()), other);

    // Non-error cells pass through untouched.
    let pass = Cell::Pass {
        id: "heterogeneous-concurrency",
        detail: "2 different requests in flight, each got its own answer".to_string(),
    };
    assert_eq!(skip_if_decoder_unavailable(pass.clone()), pass);
}
