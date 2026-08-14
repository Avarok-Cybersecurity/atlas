// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `vision_preprocess.rs`.
//!
//! A sibling file rather than an inline `mod tests`: adding the video path's
//! shared helpers carried the parent past the 500-line cap, and it is not on
//! the allow-list.

use super::*;

#[test]
fn test_target_size_no_upscale() {
    // Small image: grid_unit=32, no upscale needed.
    let (h, w) = target_size_with_max_pixels(100, 150, 32, None);
    assert!(h <= 1280 && w <= 1280);
    assert_eq!(h % 32, 0);
    assert_eq!(w % 32, 0);
}

#[test]
fn test_target_size_downscale() {
    // Large image: should be downscaled.
    let (h, w) = target_size_with_max_pixels(2000, 3000, 32, None);
    assert!(h.max(w) <= 1280);
    assert_eq!(h % 32, 0);
    assert_eq!(w % 32, 0);
}

#[test]
fn test_target_size_max_pixels() {
    let (h, w) = target_size_with_max_pixels(1254, 1254, 32, Some(512 * 512));
    assert_eq!((h, w), (512, 512));
}

#[test]
fn test_image_pad_count_2x2_merge() {
    // Standard Qwen3-VL: 2×2 spatial merger folds a patch block
    // into one embedding token.
    assert_eq!(image_pad_count(64, 64, 2), 32 * 32);
    assert_eq!(image_pad_count(40, 80, 2), 20 * 40);
}

#[test]
fn test_image_pad_count_no_merge() {
    // spatial_merge_size=1 → identity (each patch → one token).
    assert_eq!(image_pad_count(64, 64, 1), 64 * 64);
    assert_eq!(image_pad_count(8, 12, 1), 96);
}

#[test]
fn test_image_pad_count_zero_sms_clamps_to_one() {
    // sms=0 is invalid; clamps to 1 so we never divide by zero.
    assert_eq!(image_pad_count(64, 64, 0), 64 * 64);
}

/// A `vision_config` for the shipped Qwen3-VL geometry.
fn ok_cfg() -> VisionConfig {
    VisionConfig {
        depth: 27,
        hidden_size: 1152,
        num_heads: 16,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 4304,
        out_hidden_size: 2048,
        deepstack_visual_indexes: vec![8, 16, 24],
        image_pad_token_id: 151_655,
        video_pad_token_id: 151_656,
        // These tests drive `preprocess_image_with_max_pixels` directly
        // with an explicit bound, so the config-carried one is not the
        // subject here.
        max_pixels: None,
    }
}

/// A 1×1 PNG, the smallest decodable input.
const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// `parse_vision_config` reports a MISSING key as 0, so a checkpoint whose
/// `vision_config` omits `patch_size` reaches here as a zero divisor. The
/// old code divided `th / ps` at line 98 and panicked — from an HTTP
/// request body, i.e. a remote crash of the request task.
#[test]
fn zero_patch_size_is_an_error_not_a_divide_by_zero_panic() {
    let mut vcfg = ok_cfg();
    vcfg.patch_size = 0;
    let err = preprocess_image(TINY_PNG_B64, &vcfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("patch_size"), "{err}");
}

/// `grid_unit = patch_size * spatial_merge_size` is the other divisor, and
/// a zero here made it 0.0 in the f32 scale computation → a 0×0 target.
#[test]
fn zero_spatial_merge_size_is_an_error() {
    let mut vcfg = ok_cfg();
    vcfg.spatial_merge_size = 0;
    let err = preprocess_image(TINY_PNG_B64, &vcfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("spatial_merge_size"), "{err}");
}

/// `temporal_patch_size = 0` collapses `patch_dim` to 0, yielding an empty
/// pixel buffer that the encoder would then have to catch.
#[test]
fn zero_temporal_patch_size_is_an_error() {
    let mut vcfg = ok_cfg();
    vcfg.temporal_patch_size = 0;
    let err = preprocess_image(TINY_PNG_B64, &vcfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("temporal_patch_size"), "{err}");
}

/// The geometry check must not have broken the working path: a valid
/// config still decodes and produces `num_patches × patch_dim` floats.
#[test]
fn valid_config_still_preprocesses() {
    let vcfg = ok_cfg();
    let (pixels, gh, gw) = preprocess_image(TINY_PNG_B64, &vcfg).unwrap();
    let patch_dim = 3 * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size;
    assert_eq!(pixels.len(), gh * gw * patch_dim);
    assert!(gh > 0 && gw > 0);
}

/// Garbage that is not an image must be a clean error, not a panic.
#[test]
fn undecodable_input_is_an_error() {
    assert!(preprocess_image("bm90LWFuLWltYWdl", &ok_cfg()).is_err());
    assert!(preprocess_image("!!! not base64 !!!", &ok_cfg()).is_err());
}

/// A PNG whose IHDR declares 65535×65535 (~12.9 GB of RGB) but whose file
/// is 70 bytes. The dimension limit rejects it from the header, before any
/// pixel buffer is reserved.
#[test]
fn decode_bomb_dimensions_are_rejected_before_allocation() {
    let mut png: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr: Vec<u8> = b"IHDR".to_vec();
    ihdr.extend_from_slice(&65535u32.to_be_bytes()); // width
    ihdr.extend_from_slice(&65535u32.to_be_bytes()); // height
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
    let crc = {
        // CRC-32 over the chunk type + data, as PNG requires.
        let mut c: u32 = 0xFFFF_FFFF;
        for &b in &ihdr {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    (c >> 1) ^ 0xEDB8_8320
                } else {
                    c >> 1
                };
            }
        }
        !c
    };
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(&ihdr);
    png.extend_from_slice(&crc.to_be_bytes());

    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png);
    // The fixture must exceed the limit under test, or it proves nothing.
    const { assert!(65535 > DECODE_MAX_SIDE) };
    // Assert on the LIMIT error specifically, not just "some error": the
    // fixture is also a truncated PNG, so a decoder without limits would
    // still fail — just later, after reserving the buffer. Only this exact
    // message proves the header check fired first.
    let err = format!("{:#}", preprocess_image(&b64, &ok_cfg()).unwrap_err());
    assert!(err.contains("Image size exceeds limit"), "{err}");
}

#[test]
fn test_image_pad_count_non_divisible_floors() {
    // Integer division truncates: 65/2 = 32 (not 33).
    assert_eq!(image_pad_count(65, 64, 2), 32 * 32);
}

// ── area-bound sizing (Gap 1, 2026-08-14) ────────────────────────────
// grid_unit 32 = patch_size 16 x spatial_merge_size 2, the Qwen3.8 shape.
const GU: u32 = 32;

/// Merged vision tokens the LM will see for a (h, w) target.
fn merged_tokens(h: u32, w: u32) -> u32 {
    (h / 16) * (w / 16) / 4
}

#[test]
fn no_declared_bound_keeps_the_1280_fallback() {
    // Non-regression: a checkpoint shipping no preprocessor_config.json
    // must behave exactly as it did before this change.
    let (h, w) = target_size_with_max_pixels(1344, 1344, GU, None);
    assert_eq!((h, w), (1280, 1280));
    assert_eq!(merged_tokens(h, w), 1600, "the pre-change measured value");
}

#[test]
fn declared_area_bound_raises_above_the_fallback() {
    // THE FIX. Qwen3.8-27B declares size.longest_edge = 16777216 (4096^2).
    // Before, .min() against the 1280 clamp meant this could never exceed
    // 1280 on the long side: the model's own bound could only ever lower.
    let (h, w) = target_size_with_max_pixels(2048, 2048, GU, Some(16_777_216));
    assert_eq!((h, w), (2048, 2048), "2048^2 sits inside a 4096^2 bound");
    assert!(
        h > 1280,
        "a declared bound must be able to RAISE past the fallback clamp"
    );
}

#[test]
fn declared_area_bound_still_downscales_when_exceeded() {
    // 8192^2 is 4x over the bound and must come back to the BOUND, not to
    // the old 1280 clamp.
    let (h, w) = target_size_with_max_pixels(8192, 8192, GU, Some(16_777_216));
    assert!(
        (h as u64) * (w as u64) <= 16_777_216,
        "area {} exceeds the declared bound",
        (h as u64) * (w as u64)
    );
    assert!(h > 1280, "downscaled to the bound, not to the old clamp");
}

#[test]
fn operator_override_can_lower_below_the_fallback() {
    let (h, w) = target_size_with_max_pixels(1344, 1344, GU, Some(256 * 256));
    assert!((h as u64) * (w as u64) <= 256 * 256);
    assert!(h < 1280);
}

#[test]
fn absolute_ceiling_bounds_a_pathological_aspect_ratio() {
    // max_pixels is an AREA, so a 1xN strip satisfies any area bound at an
    // unbounded long side. ABS_MAX_DIM is the guard.
    let (_h, w) = target_size_with_max_pixels(32, 1_000_000, GU, Some(16_777_216));
    assert!(
        w <= ABS_MAX_DIM,
        "long side {w} escaped the absolute ceiling"
    );
}

#[test]
fn never_upscales() {
    // A tiny image under a huge bound stays tiny. Atlas diverges from HF
    // here (HF honours shortest_edge/min_pixels by scaling UP); that is
    // deliberate and out of scope, but it must not drift silently.
    let (h, w) = target_size_with_max_pixels(64, 64, GU, Some(16_777_216));
    assert_eq!((h, w), (64, 64));
}

#[test]
fn sides_are_always_grid_multiples() {
    for (oh, ow, mp) in [
        (1080u32, 1920u32, None),
        (1080, 1920, Some(16_777_216usize)),
        (450, 300, None),
        (33, 33, Some(65_536)),
    ] {
        let (h, w) = target_size_with_max_pixels(oh, ow, GU, mp);
        assert_eq!(h % GU, 0, "h={h} not a multiple of {GU}");
        assert_eq!(w % GU, 0, "w={w} not a multiple of {GU}");
        assert!(h >= GU && w >= GU, "degenerate target {h}x{w}");
    }
}

#[test]
fn aspect_ratio_is_preserved_within_grid_rounding() {
    let (h, w) = target_size_with_max_pixels(1080, 1920, GU, Some(16_777_216));
    let want = 1920.0f32 / 1080.0;
    let got = w as f32 / h as f32;
    assert!(
        (got - want).abs() < 0.05,
        "aspect drifted: {got} vs {want} ({h}x{w})"
    );
}

// ── EXIF orientation ─────────────────────────────────────────────────────

/// The committed benchmark fixtures, reused so the unit test and the
/// benchmark cannot disagree about what the bytes contain: identical pixels
/// (red top half, blue bottom half), one tagged `Orientation = 6`, one
/// untagged.
const EXIF_ROT90: &[u8] = include_bytes!("../../../tests/fixtures/images/15_exif_rot90_224.jpg");
const EXIF_NONE: &[u8] = include_bytes!("../../../tests/fixtures/images/16_exif_none_224.jpg");

fn as_data_uri(bytes: &[u8]) -> String {
    use base64::Engine as _;
    format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// Mean red channel of a quadrant, as a fraction of the image.
fn region_red(img: &image::RgbImage, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (xs, ys) = ((x0 * w) as u32, (y0 * h) as u32);
    let (xe, ye) = ((x1 * w) as u32, (y1 * h) as u32);
    let mut sum = 0f32;
    let mut n = 0f32;
    for y in ys..ye {
        for x in xs..xe {
            sum += img.get_pixel(x, y)[0] as f32;
            n += 1.0;
        }
    }
    sum / n.max(1.0)
}

/// The untagged control: nothing to apply, so red stays on TOP. This is the
/// non-regression half — every PNG and every untagged JPEG must decode
/// exactly as it did before orientation handling existed.
#[test]
fn an_untagged_image_is_not_rotated() {
    let img = decode_image(&as_data_uri(EXIF_NONE))
        .expect("decode")
        .to_rgb8();
    let top = region_red(&img, 0.1, 0.05, 0.9, 0.35);
    let bottom = region_red(&img, 0.1, 0.65, 0.9, 0.95);
    assert!(
        top > 200.0 && bottom < 60.0,
        "expected red on top, blue below; got top={top:.0} bottom={bottom:.0}"
    );
}

/// ★ The tagged image MUST come out rotated. `Orientation = 6` means "rotate
/// 90° clockwise to display", which moves the stored top edge to the right
/// side — so the red half ends up on the RIGHT.
///
/// Before 2026-08-14 this assertion would have failed: the tag was ignored and
/// red stayed on top, meaning every rotated phone photo reached the model a
/// quarter turn from how its owner saw it.
#[test]
fn an_exif_tagged_image_is_rotated_upright() {
    let img = decode_image(&as_data_uri(EXIF_ROT90))
        .expect("decode")
        .to_rgb8();
    let left = region_red(&img, 0.05, 0.1, 0.35, 0.9);
    let right = region_red(&img, 0.65, 0.1, 0.95, 0.9);
    assert!(
        right > 200.0 && left < 60.0,
        "Orientation=6 should put the red half on the RIGHT; got left={left:.0} right={right:.0}"
    );
}

/// Rotation happens BEFORE the grid snap, so a portrait photo tagged as
/// landscape is measured at its displayed shape rather than its stored one.
/// A 90° turn swaps the axes; on a square fixture the token count is
/// unchanged, which is what makes this safe to apply universally.
#[test]
fn rotation_does_not_change_the_token_count_of_a_square() {
    let (_, gh_a, gw_a) = preprocess_image(&as_data_uri(EXIF_ROT90), &ok_cfg()).expect("tagged");
    let (_, gh_b, gw_b) = preprocess_image(&as_data_uri(EXIF_NONE), &ok_cfg()).expect("untagged");
    assert_eq!((gh_a, gw_a), (gh_b, gw_b));
}
