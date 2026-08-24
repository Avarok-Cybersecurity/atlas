// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the fixture set itself.

use super::*;
use crate::artifacts::ArtifactStore;
use sha2::{Digest, Sha256};

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-video-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temporary video root");
    dir
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The pairs ARE the assertions, so the set has to actually contain them.
/// A fixture set that quietly lost its reversed clip would leave a benchmark
/// that still runs and still passes while testing nothing about order.
#[test]
fn the_fixture_set_contains_every_pair_the_benchmark_needs() {
    let fwd = clip("01_colors_fwd.mp4").expect("forward mp4");
    let rev = clip("02_colors_rev.mp4").expect("reversed mp4");
    let gif = clip("03_colors_fwd.gif").expect("gif");
    let unit = clip("05_colors_unit.mp4").expect("unit mp4");
    let half = clip("04_colors_half.mp4").expect("half-length mp4");

    let identity: Vec<_> = CLIPS
        .iter()
        .map(|c| {
            (
                c.name,
                c.colors,
                c.seconds,
                c.needs_ffmpeg,
                c.mime,
                c.bytes.len(),
                sha256(c.bytes),
            )
        })
        .collect();
    assert_eq!(
        identity,
        vec![
            (
                "01_colors_fwd.mp4",
                &["red", "green", "blue", "yellow"][..],
                4,
                true,
                "video/mp4",
                2595,
                "28fb8b7efb379ce8af446b0d4a43b46fe87863e005947fb63e96a13880254379".into()
            ),
            (
                "02_colors_rev.mp4",
                &["yellow", "blue", "green", "red"][..],
                4,
                true,
                "video/mp4",
                2633,
                "93104b31a2ec8743759a5e819c086a56550d0f9b6f3f44bc1a62c29acdd100f5".into()
            ),
            (
                "03_colors_fwd.gif",
                &["red", "green", "blue", "yellow"][..],
                4,
                false,
                "image/gif",
                1785,
                "e84e76c18417e8c5f1f2d97b480c2e7ea4535768218667c406feb04b87e82c93".into()
            ),
            (
                "05_colors_unit.mp4",
                &["red"][..],
                1,
                true,
                "video/mp4",
                1821,
                "f04702a412db2d1a8e66f786e7161270d2dc281677bf3a26b14efefee21596f9".into()
            ),
            (
                "04_colors_half.mp4",
                &["red", "green"][..],
                2,
                true,
                "video/mp4",
                2069,
                "3aafec621fdbd55330184b87fe70da2630f5620b88605917fd0c17486876df4e".into()
            ),
        ]
    );

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

    // Ratio ladder: exact 1:2:4 durations and matching color prefixes.
    assert_eq!((unit.seconds, half.seconds, fwd.seconds), (1, 2, 4));
    assert_eq!(unit.colors, &fwd.colors[..1]);
    assert_eq!(half.colors, &fwd.colors[..2]);
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
    assert_eq!(a, "video-fixtures-v1-a3d8f9d4cd8e9ed5");
    assert_eq!(a, stamp_value());
    assert_ne!(
        crate::benchmarks::content_stamp("video-fixtures-v1", [("clip", b"one".as_slice())]),
        crate::benchmarks::content_stamp("video-fixtures-v1", [("clip", b"two".as_slice())]),
        "fixture bytes must affect the provisioning stamp"
    );
    assert_ne!(
        crate::benchmarks::content_stamp("video-fixtures-v1", [("first", b"same".as_slice())]),
        crate::benchmarks::content_stamp("video-fixtures-v1", [("second", b"same".as_slice())]),
        "fixture names must affect the provisioning stamp"
    );
}

#[test]
fn an_unknown_name_resolves_to_nothing() {
    assert!(clip("nope.mp4").is_none());
}

#[test]
fn provision_writes_every_exact_fixture_then_stamps() {
    let root = tmp("writes");
    let store = ArtifactStore::with_root(&root);
    let dir = provision(&store).expect("provision video fixtures");
    for clip in CLIPS {
        assert_eq!(
            std::fs::read(dir.join(clip.name)).expect("read provisioned clip"),
            clip.bytes,
            "{} was not provisioned exactly",
            clip.name
        );
    }
    assert_eq!(
        std::fs::read_to_string(dir.join(".provisioned")).expect("read stamp"),
        stamp_value()
    );
    std::fs::remove_dir_all(root).expect("remove temporary video root");
}

#[test]
fn a_current_stamp_makes_reprovision_a_no_op() {
    let root = tmp("current");
    let store = ArtifactStore::with_root(&root);
    let dir = provision(&store).expect("first provision");
    let victim = dir.join(CLIPS[0].name);
    let original_permissions = std::fs::metadata(&victim)
        .expect("fixture metadata")
        .permissions();
    let mut readonly_permissions = original_permissions.clone();
    readonly_permissions.set_readonly(true);
    std::fs::set_permissions(&victim, readonly_permissions).expect("make fixture read-only");
    provision(&store).expect("current stamp should avoid every fixture write");
    std::fs::set_permissions(&victim, original_permissions).expect("restore fixture permissions");
    std::fs::remove_dir_all(root).expect("remove temporary video root");
}

#[test]
fn a_stale_stamp_rewrites_corrupted_fixture_bytes() {
    let root = tmp("stale");
    let store = ArtifactStore::with_root(&root);
    let dir = provision(&store).expect("first provision");
    let victim = dir.join(CLIPS[0].name);
    std::fs::write(&victim, b"corrupt").expect("corrupt fixture");
    std::fs::write(dir.join(".provisioned"), "stale").expect("stale stamp");
    provision(&store).expect("reprovision stale fixtures");
    assert_eq!(
        std::fs::read(victim).expect("read repaired fixture"),
        CLIPS[0].bytes
    );
    std::fs::remove_dir_all(root).expect("remove temporary video root");
}

#[test]
fn a_failed_fixture_write_does_not_commit_the_stamp() {
    let root = tmp("partial");
    let store = ArtifactStore::with_root(&root);
    let dir = store
        .plugin_dir(PLUGIN_ID)
        .expect("video artifact directory");
    std::fs::create_dir(dir.join(CLIPS[1].name)).expect("block the second fixture path");
    assert!(
        provision(&store).is_err(),
        "the obstructed fixture must fail"
    );
    assert!(
        !dir.join(".provisioned").exists(),
        "a partial fixture set must never be marked current"
    );
    std::fs::remove_dir_all(root).expect("remove temporary video root");
}
