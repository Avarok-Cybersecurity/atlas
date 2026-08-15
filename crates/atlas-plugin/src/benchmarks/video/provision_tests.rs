// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the fixture set itself.

use super::*;

/// The pairs ARE the assertions, so the set has to actually contain them.
/// A fixture set that quietly lost its reversed clip would leave a benchmark
/// that still runs and still passes while testing nothing about order.
#[test]
fn the_fixture_set_contains_every_pair_the_benchmark_needs() {
    let fwd = clip("01_colors_fwd.mp4").expect("forward mp4");
    let rev = clip("02_colors_rev.mp4").expect("reversed mp4");
    let gif = clip("03_colors_fwd.gif").expect("gif");
    let half = clip("04_colors_half.mp4").expect("half-length mp4");

    // Order pair: same colors, opposite order.
    let mut a = fwd.colors.to_vec();
    a.sort_unstable();
    let mut b = rev.colors.to_vec();
    b.sort_unstable();
    assert_eq!(a, b, "the pair must show the SAME colors");
    assert_eq!(
        rev.colors,
        fwd.colors.iter().rev().copied().collect::<Vec<_>>(),
        "exactly reversed, so no partial match can satisfy both"
    );

    // Parity pair: same content, different container.
    assert_eq!(gif.colors, fwd.colors);
    assert_ne!(gif.mime, fwd.mime);
    assert!(!gif.needs_ffmpeg, "the gif is the no-dependency path");
    assert!(fwd.needs_ffmpeg, "the mp4 is the subprocess path");

    // Ratio pair: exactly 2:1 in duration.
    assert_eq!(fwd.seconds, half.seconds * 2);
}

/// Committed assets, so they must stay small enough to embed comfortably.
#[test]
fn every_clip_is_small_enough_to_embed() {
    for c in CLIPS {
        assert!(!c.bytes.is_empty(), "{} is empty", c.name);
        assert!(
            c.bytes.len() < 64 * 1024,
            "{} is {} bytes — too large to carry in the binary",
            c.name,
            c.bytes.len()
        );
    }
}

/// Magic bytes, not the file extension: the decoder dispatches on CONTENT, so
/// a fixture whose bytes disagree with its name would send a leg down the
/// wrong backend and quietly change what is being tested.
#[test]
fn each_clip_really_is_the_container_it_claims() {
    for c in CLIPS {
        match c.mime {
            "image/gif" => assert!(
                c.bytes.starts_with(b"GIF87a") || c.bytes.starts_with(b"GIF89a"),
                "{} claims gif but has no GIF signature",
                c.name
            ),
            "video/mp4" => assert_eq!(
                &c.bytes[4..8],
                b"ftyp",
                "{} claims mp4 but has no ftyp box",
                c.name
            ),
            other => panic!("{} has an unhandled mime {other}", c.name),
        }
    }
}

#[test]
fn the_stamp_is_deterministic_and_named() {
    let a = stamp_value();
    assert!(a.starts_with("video-fixtures-v1-"));
    assert_eq!(a, stamp_value());
}

#[test]
fn an_unknown_name_resolves_to_nothing() {
    assert!(clip("nope.mp4").is_none());
}
