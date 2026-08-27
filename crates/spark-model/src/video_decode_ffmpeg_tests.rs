// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the subprocess decode backend.
//!
//! The ones that need a real decoder generate their own MP4 with ffmpeg and
//! skip when it is absent, so the suite stays green on a machine without it —
//! the same condition the backend itself has to survive.

use super::*;

fn have_ffmpeg() -> bool {
    matches!(probe(&enabled()), Availability::Ready(_))
}

fn enabled() -> FfmpegPolicy {
    FfmpegPolicy {
        enabled: true,
        ..Default::default()
    }
}

/// Encode a short clip and hand back the container bytes.
fn make_mp4(seconds: u32, fps: u32, size: &str, codec: &str) -> Option<Vec<u8>> {
    // Unique per CALL, not per parameter set. Keying the directory on
    // (pid, seconds, fps, codec) collided between tests that differ only in
    // frame size, and since each call ends by removing its directory, one
    // test deleted another's clip mid-decode — two failures that looked like
    // encoder problems and were not.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "atlas-vid-{}-{n}-{}-{}-{size}-{codec}",
        std::process::id(),
        seconds,
        fps
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("clip.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size={size}:rate={fps}:duration={seconds}"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out).ok();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

// ── the default posture ──────────────────────────────────────────────────

#[test]
fn the_default_policy_runs_nothing() {
    let p = FfmpegPolicy::default();
    assert!(!p.enabled, "subprocess decoding must be opt-in");
    let err = decode_frames(b"whatever", 2.0, &p).unwrap_err().to_string();
    assert!(err.contains("--video-allow-ffmpeg"), "{err}");
}

/// Disabled must refuse WITHOUT spawning anything, even when the binary is
/// nonsense — proof the gate is checked before the process is built.
#[test]
fn disabled_refuses_before_touching_the_binary() {
    let p = FfmpegPolicy {
        enabled: false,
        binary: "/nonexistent/definitely-not-a-decoder".to_string(),
        ..Default::default()
    };
    let err = decode_frames(b"x", 2.0, &p).unwrap_err().to_string();
    assert!(err.contains("disabled"), "{err}");
}

// ── availability reporting ───────────────────────────────────────────────

#[test]
fn probing_a_missing_binary_reports_it_by_name() {
    let p = FfmpegPolicy {
        enabled: true,
        binary: "/nonexistent/definitely-not-a-decoder".to_string(),
        ..Default::default()
    };
    match probe(&p) {
        Availability::Missing(why) => {
            assert!(why.contains("definitely-not-a-decoder"), "{why}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

#[test]
fn probing_while_disabled_says_disabled_rather_than_missing() {
    assert_eq!(probe(&FfmpegPolicy::default()), Availability::Disabled);
}

/// A misconfigured deployment must fail with the BINARY named, not with
/// something that reads like a corrupt video — the two have completely
/// different fixes.
#[test]
fn a_missing_binary_fails_the_decode_by_name() {
    let p = FfmpegPolicy {
        enabled: true,
        binary: "/nonexistent/definitely-not-a-decoder".to_string(),
        ..Default::default()
    };
    let err = format!("{:#}", decode_frames(b"x", 2.0, &p).unwrap_err());
    assert!(err.contains("definitely-not-a-decoder"), "{err}");
    assert!(err.contains("ffmpeg installed"), "no remedy offered: {err}");
}

// ── real decoding ────────────────────────────────────────────────────────

#[test]
fn an_h264_mp4_decodes_to_frames_at_the_requested_rate() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "320x240", "libx264").expect("encode");
    let frames = decode_frames(&mp4, 2.0, &enabled()).expect("decode");
    // 3 s at 2 fps. ffmpeg's fps filter can emit an extra boundary frame, so
    // the assertion is a band rather than an exact count — pinning it exactly
    // would make the test hostage to a filter-timing detail.
    assert!(
        (5..=7).contains(&frames.len()),
        "expected ~6 frames, got {}",
        frames.len()
    );
    assert_eq!(frames[0].dimensions(), (320, 240));
}

#[test]
fn the_requested_rate_actually_changes_the_frame_count() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(4, 10, "160x120", "libx264").expect("encode");
    let slow = decode_frames(&mp4, 1.0, &enabled()).expect("1 fps");
    let fast = decode_frames(&mp4, 4.0, &enabled()).expect("4 fps");
    assert!(
        fast.len() > slow.len(),
        "4 fps gave {} frames, 1 fps gave {}",
        fast.len(),
        slow.len()
    );
}

/// Frames must be distinct. A backend that returned the same frame N times
/// would pass every count and dimension check while feeding the model a
/// still — the video equivalent of the duplicated-frame bug the grouping
/// tests guard against.
#[test]
fn decoded_frames_are_not_all_identical() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "160x120", "libx264").expect("encode");
    let frames = decode_frames(&mp4, 2.0, &enabled()).expect("decode");
    assert!(frames.len() >= 2);
    assert_ne!(
        frames[0].as_raw(),
        frames[frames.len() - 1].as_raw(),
        "first and last frame are byte-identical — the clip did not advance"
    );
}

#[test]
fn the_frame_cap_is_honoured() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(5, 10, "160x120", "libx264").expect("encode");
    let p = FfmpegPolicy {
        max_frames: 3,
        ..enabled()
    };
    let frames = decode_frames(&mp4, 10.0, &p).expect("decode");
    assert_eq!(frames.len(), 3);
}

#[test]
fn an_output_cap_smaller_than_the_clip_is_refused() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "320x240", "libx264").expect("encode");
    let p = FfmpegPolicy {
        max_output_bytes: 1024,
        ..enabled()
    };
    let err = format!("{:#}", decode_frames(&mp4, 2.0, &p).unwrap_err());
    assert!(err.contains("cap"), "{err}");
}

/// Garbage must produce the decoder's complaint, not a panic or a hang.
#[test]
fn a_non_video_payload_fails_cleanly() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let err = format!(
        "{:#}",
        decode_frames(b"this is not a video at all", 2.0, &enabled()).unwrap_err()
    );
    assert!(err.contains("decoder failed"), "{err}");
}

#[cfg(unix)]
#[test]
fn a_hanging_decoder_is_killed_at_timeout() {
    use std::os::unix::fs::PermissionsExt;

    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("atlas-hanging-ffmpeg-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("hanging-ffmpeg");
    std::fs::write(&binary, "#!/bin/sh\nexec sleep 10\n").unwrap();
    let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&binary, permissions).unwrap();

    let policy = FfmpegPolicy {
        enabled: true,
        binary: binary.display().to_string(),
        timeout_secs: 1,
        ..Default::default()
    };
    let started = std::time::Instant::now();
    let err = decode_frames(b"input", 2.0, &policy)
        .unwrap_err()
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(err.contains("decoding exceeded 1s"), "{err}");
    assert!(started.elapsed() < std::time::Duration::from_secs(3));
}

/// H.265 in the same container — the case a linked H.264-only decoder could
/// not have served, and the reason this backend exists.
#[test]
fn an_hevc_clip_decodes_too_when_the_encoder_is_available() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let Some(mp4) = make_mp4(2, 10, "160x120", "libx265") else {
        eprintln!("skipping: no libx265 in this ffmpeg build");
        return;
    };
    let frames = decode_frames(&mp4, 2.0, &enabled()).expect("decode hevc");
    assert!(!frames.is_empty());
    assert_eq!(frames[0].dimensions(), (160, 120));
}

// ── stream splitting ─────────────────────────────────────────────────────

#[test]
fn an_empty_stream_splits_to_nothing() {
    assert!(
        split_png_stream(b"")
            .expect("empty is not an error")
            .is_empty()
    );
}

#[test]
fn a_stream_of_two_pngs_splits_into_two_frames() {
    let mut buf = Vec::new();
    for pixel in [[1, 2, 3], [4, 5, 6]] {
        let img = image::RgbImage::from_pixel(4, 3, image::Rgb(pixel));
        let mut one = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut one), image::ImageFormat::Png)
            .expect("encode");
        buf.extend_from_slice(&one);
    }
    let frames = split_png_stream(&buf).expect("split");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].dimensions(), (4, 3));
    assert_eq!(frames[0].get_pixel(0, 0).0, [1, 2, 3]);
    assert_eq!(frames[1].get_pixel(0, 0).0, [4, 5, 6]);
}
