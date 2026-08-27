// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for video decode, frame sampling and temporal grouping.
//!
//! The sampling arithmetic is where a video's token count comes from, so it is
//! tested as arithmetic — directly, at the boundaries — rather than only
//! through a decoded fixture.

use super::*;
use crate::video_decode_ffmpeg::FfmpegPolicy;

/// The default: no subprocess. Every GIF case must pass without it, which is
/// the guarantee that the pure-Rust path is not quietly depending on ffmpeg.
fn no_ffmpeg() -> FfmpegPolicy {
    FfmpegPolicy::default()
}

fn cfg() -> VisionConfig {
    VisionConfig {
        depth: 2,
        hidden_size: 32,
        num_heads: 2,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 64,
        out_hidden_size: 32,
        deepstack_visual_indexes: vec![],
        image_pad_token_id: 248_056,
        video_pad_token_id: 248_057,
        max_pixels: None,
    }
}

// ── sampling ─────────────────────────────────────────────────────────────

/// The count must be a whole number of temporal groups whenever one full
/// group is achievable — a partial group has no representation in the patch
/// tensor, so an odd count at tp=2 is a latent shape bug.
///
/// The one exception is a clip with FEWER frames than a single group. There,
/// sampling reports honestly what it has (see
/// `one_frame_cannot_make_a_group_at_tp2`) and `preprocess_video` is what
/// refuses it. Rounding 1 frame down to 0 here would turn a describable error
/// — "this clip is too short" — into an empty selection that the caller has
/// to reverse-engineer. Writing this invariant without the exception is what
/// caught the disagreement between the two tests.
#[test]
fn the_frame_count_is_always_a_whole_number_of_groups() {
    const TP: usize = 2;
    for n in 1..64usize {
        for fps in [1.0f32, 2.0, 5.0, 7.5, 30.0] {
            let got = sample_indices(n, 30.0, fps, 4, 768, TP);
            if n < TP {
                assert_eq!(got.len(), n, "n={n}: too short for a group, report as-is");
                continue;
            }
            assert_eq!(
                got.len() % TP,
                0,
                "n={n} fps={fps} produced {} frames, not a whole number of groups",
                got.len()
            );
            assert!(got.len() >= TP, "n={n} fps={fps} dropped below one group");
            assert!(got.len() <= n, "n={n} fps={fps} sampled more than it had");
        }
    }
}

#[test]
fn indices_are_in_range_and_non_decreasing() {
    let got = sample_indices(100, 30.0, 2.0, 4, 768, 2);
    assert!(!got.is_empty());
    for w in got.windows(2) {
        assert!(w[0] <= w[1], "indices went backwards: {got:?}");
    }
    assert!(*got.last().unwrap() < 100);
}

/// A 30 fps, 100-frame clip is 3.33 s; at 2 fps that is ~7 frames, rounded
/// down to 6 for three whole groups.
#[test]
fn a_clip_is_sampled_to_the_requested_rate() {
    let got = sample_indices(100, 30.0, 2.0, 4, 768, 2);
    assert_eq!(got.len(), 6, "got {got:?}");
    // Spread across the whole clip, not clustered at the start.
    assert_eq!(got[0], 0);
    assert_eq!(*got.last().unwrap(), 99);
}

/// Never upsample. Asking 30 fps of a clip that only has 4 frames must return
/// those 4, not 4 repeated — repeats inflate the token count with no new
/// information, and the model would be charged for frames that do not exist.
#[test]
fn a_short_clip_is_never_padded_up_to_the_minimum() {
    let got = sample_indices(4, 2.0, 30.0, 16, 768, 2);
    assert_eq!(got.len(), 4);
    assert_eq!(got, vec![0, 1, 2, 3]);
}

#[test]
fn the_max_frames_ceiling_is_honoured() {
    let got = sample_indices(10_000, 30.0, 30.0, 4, 768, 2);
    assert_eq!(got.len(), 768);
}

#[test]
fn an_inverted_frame_band_uses_the_ceiling() {
    let got = sample_indices(100, 30.0, 30.0, 16, 4, 2);
    assert_eq!(got.len(), 4);
    assert_eq!(got[0], 0);
    assert_eq!(*got.last().unwrap(), 99);
}

/// A degenerate rate must not divide by zero or return nothing; it falls back
/// to the checkpoint default rather than failing the request.
#[test]
fn zero_and_nonfinite_rates_fall_back_rather_than_exploding() {
    for (native, target, valid_native, valid_target) in [
        (0.0f32, 1.0f32, DEFAULT_FPS, 1.0),
        (30.0, 0.0, 30.0, DEFAULT_FPS),
        (f32::NAN, 1.0, DEFAULT_FPS, 1.0),
        (30.0, f32::INFINITY, 30.0, DEFAULT_FPS),
    ] {
        let got = sample_indices(100, native, target, 4, 768, 2);
        let expected = sample_indices(100, valid_native, valid_target, 4, 768, 2);
        assert_eq!(got, expected, "native={native} target={target}");
    }
}

#[test]
fn an_empty_clip_samples_to_nothing() {
    assert!(sample_indices(0, 30.0, 2.0, 4, 768, 2).is_empty());
}

/// A single frame cannot fill a tp=2 group. Sampling reports what it has and
/// `preprocess_video` is what refuses it — checked here so the two stay
/// consistent.
#[test]
fn one_frame_cannot_make_a_group_at_tp2() {
    let got = sample_indices(1, 30.0, 2.0, 4, 768, 2);
    assert_eq!(got.len(), 1, "sampling reports the single frame it has");
    assert!(got.len() < 2, "and it is short of one tp=2 group");
}

// ── decode + grouping, against a real generated GIF ──────────────────────

/// Build an animated GIF of `n` solid-color frames at `size`x`size`.
fn make_gif(n: u16, size: u16) -> String {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, RgbaImage};
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut enc = GifEncoder::new(&mut buf);
        for i in 0..n {
            // Distinct color per frame so a frame-ordering bug is visible in
            // the pixel data rather than invisible.
            let v = (20 + (i as u32 * 37) % 200) as u8;
            let img = RgbaImage::from_pixel(
                size as u32,
                size as u32,
                // `255 - v` with v <= 219, not `200 - v`: v exceeds 200 from
                // frame 5 on and the subtraction underflowed in debug builds.
                image::Rgba([v, 40, 255 - v, 255]),
            );
            let mut f = Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1));
            f = Frame::from_parts(f.into_buffer(), 0, 0, Delay::from_numer_denom_ms(100, 1));
            enc.encode_frame(f).expect("encode frame");
        }
    }
    use base64::Engine as _;
    format!(
        "data:image/gif;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&buf)
    )
}

#[test]
fn an_animated_gif_decodes_to_its_frames() {
    let uri = make_gif(6, 64);
    let (frames, fps) = decode_frames(&uri, 10.0, &no_ffmpeg()).expect("decode");
    assert_eq!(frames.len(), 6);
    // 100 ms per frame → 10 fps.
    assert!((fps - 10.0).abs() < 0.5, "fps was {fps}");
}

#[test]
fn gif_magic_wins_over_a_mislabelled_mime() {
    let uri = make_gif(4, 64).replacen("data:image/gif", "data:video/mp4", 1);
    let (frames, fps) = decode_frames(&uri, 10.0, &no_ffmpeg()).expect("decode by magic");
    assert_eq!(frames.len(), 4);
    assert!((fps - 10.0).abs() < 0.5, "fps was {fps}");
}

/// The shape contract the encoder depends on: one buffer per temporal group,
/// each exactly `grid_h * grid_w * (C * tp * patch²)` long.
#[test]
fn grouping_produces_correctly_shaped_buffers() {
    let uri = make_gif(8, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    assert_eq!(v.grid_h, 4, "64px / 16 = 4 patches");
    assert_eq!(v.grid_w, 4);
    assert_eq!(v.grid_t, 4, "8 frames at tp=2 = 4 groups");
    assert_eq!(v.groups.len(), 4);
    let expect = v.grid_h * v.grid_w * 3 * 2 * 16 * 16;
    for (i, g) in v.groups.iter().enumerate() {
        assert_eq!(g.len(), expect, "group {i} is the wrong length");
    }
}

/// Token accounting, which is what the prompt-side pad run must match.
#[test]
fn pad_count_is_groups_times_the_merged_plane() {
    let uri = make_gif(8, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    // 4 groups × (4/2 × 4/2) = 4 × 4 = 16
    assert_eq!(v.pad_count(2), 16);
    assert_eq!(v.pad_count(1), 4 * 16);
}

/// Distinct frames must land in distinct temporal slots. If the grouping ever
/// duplicated one frame across the tp axis — the way a STILL deliberately
/// does — a video would silently become a slideshow of doubled stills, and
/// every shape check above would still pass.
#[test]
fn the_two_frames_of_a_group_are_actually_different() {
    let uri = make_gif(4, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    let ps = 16usize;
    let g = &v.groups[0];
    // Offsets of t=0 and t=1 within channel 0 of patch 0.
    let a = g[0];
    let b = g[ps * ps];
    assert_ne!(
        a, b,
        "t=0 and t=1 hold identical pixels — the frames were duplicated, not paired"
    );
}

/// Consecutive groups must differ too: group 1 must not be a copy of group 0.
#[test]
fn consecutive_groups_hold_different_frames() {
    let uri = make_gif(4, 64);
    let v = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("preprocess");
    assert_eq!(v.groups.len(), 2);
    assert_ne!(v.groups[0][0], v.groups[1][0], "group 1 repeated group 0");
}

/// The area bound applies to video exactly as it does to stills — a video is
/// far more expensive per item, so a bound that silently did not apply here
/// would be the worse omission.
#[test]
fn the_area_bound_shrinks_the_grid() {
    let uri = make_gif(4, 256);
    let big = preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).expect("unbounded");
    let small = preprocess_video(&uri, &cfg(), Some(64 * 64), 10.0, &no_ffmpeg()).expect("bounded");
    assert!(
        small.grid_h < big.grid_h,
        "bound {}x{} did not shrink {}x{}",
        small.grid_h,
        small.grid_w,
        big.grid_h,
        big.grid_w
    );
    assert_eq!(
        small.grid_t, big.grid_t,
        "the bound is spatial, not temporal"
    );
}

// ── refusals ─────────────────────────────────────────────────────────────

/// An mp4 must be named, not misparsed. The error tells the operator what to
/// do, because "not a decodable animation" leaves them guessing at whether
/// the file is corrupt.
#[test]
fn an_mp4_is_refused_by_name_with_a_conversion_hint() {
    use base64::Engine as _;
    let uri = format!(
        "data:video/mp4;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(b"\x00\x00\x00\x20ftypmp42")
    );
    let err = format!("{:#}", decode_frames(&uri, 10.0, &no_ffmpeg()).unwrap_err());
    assert!(err.contains("mp4"), "{err}");
    assert!(err.contains("ffmpeg"), "no conversion hint: {err}");
}

#[test]
fn a_single_frame_gif_is_refused_rather_than_treated_as_a_still() {
    let uri = make_gif(1, 64);
    let err = format!(
        "{:#}",
        preprocess_video(&uri, &cfg(), None, 10.0, &no_ffmpeg()).unwrap_err()
    );
    assert!(err.contains("temporal group"), "{err}");
}

#[test]
fn garbage_is_an_error_not_a_panic() {
    assert!(decode_frames("data:video/gif;base64,bm90LWEtZ2lm", 10.0, &no_ffmpeg()).is_err());
    assert!(decode_frames("!!! not base64 !!!", 10.0, &no_ffmpeg()).is_err());
}

#[test]
fn invalid_geometry_is_refused_before_dividing_by_it() {
    let uri = make_gif(4, 64);
    let mut c = cfg();
    c.temporal_patch_size = 0;
    assert!(preprocess_video(&uri, &c, None, 10.0, &no_ffmpeg()).is_err());
    let mut c = cfg();
    c.patch_size = 0;
    assert!(preprocess_video(&uri, &c, None, 10.0, &no_ffmpeg()).is_err());
}
