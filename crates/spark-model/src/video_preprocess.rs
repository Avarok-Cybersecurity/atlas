// SPDX-License-Identifier: AGPL-3.0-only

//! Video → patch tensor, the temporal sibling of [`crate::vision_preprocess`].
//!
//! # What a video is, to this encoder
//!
//! Qwen3-VL's ViT has NO temporal attention. Frames fuse inside a patch: the
//! flattened patch dimension is `C × temporal_patch_size × patch² `, so a
//! patch already spans `tp` frames' worth of pixels. A still image fills that
//! axis by REPLICATING itself `tp` times (see `preprocess_image`) — the axis
//! was always there, and a still is the degenerate case of a video.
//!
//! So a video of `n` frames becomes `grid_t = n / tp` TEMPORAL GROUPS, each
//! group a full `grid_h × grid_w` patch plane built from `tp` consecutive
//! frames. Each group is shaped exactly like a preprocessed still, which is
//! why the encoder needs no change at all: the groups ride the existing
//! per-image path and only the bookkeeping downstream knows they belong to one
//! item.
//!
//! What DOES differ is position. An image holds MRoPE's T coordinate constant
//! across its whole pad run; a video advances T once per group. That is the
//! reason `grid_t` is carried rather than groups being flattened into
//! independent images, and it is why videos get their own pad token.
//!
//! # Container support
//!
//! Two backends, chosen by MAGIC BYTES rather than the declared MIME:
//!
//! - **GIF** decodes in-process, pure Rust, always available, no dependency.
//! - **Everything else** (MP4/MOV, WebM/Matroska, AVI — H.264, H.265, VP9,
//!   AV1) goes to ffmpeg as a subprocess, which is opt-in.
//!
//! Sniffing the bytes rather than trusting the label means a client that
//! sends an mp4 as `video/gif`, or as `application/octet-stream`, still gets
//! the right decoder. See `video_decode_ffmpeg` for why a subprocess rather
//! than a linked decoder, and issue #515.

use anyhow::{Context, Result, ensure};
use atlas_core::config::VisionConfig;
use image::RgbImage;

use crate::vision_preprocess::{MEAN, STD, decode_data_uri_bytes, target_size_for};

/// Frames per second to sample at, when the caller has no better idea.
/// Matches the `fps: 2` every Qwen3-VL `video_processor` block declares.
pub const DEFAULT_FPS: f32 = 2.0;

/// Sampling floor and ceiling, also from the checkpoints' own video processor
/// (`min_frames: 4`, `max_frames: 768`). The floor matters more than it looks:
/// with `temporal_patch_size = 2`, fewer than 2 frames cannot fill a single
/// temporal group, and a 1-frame "video" would silently become a still.
pub const DEFAULT_MIN_FRAMES: usize = 4;
pub const DEFAULT_MAX_FRAMES: usize = 768;

/// A decoded, ready-to-encode video.
pub struct PreprocessedVideo {
    /// One entry per temporal group, each shaped exactly like a preprocessed
    /// still: `[grid_h * grid_w, C * tp * patch * patch]`.
    pub groups: Vec<Vec<f32>>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Summarised rather than derived: the payload is megabytes of f32 and a
/// derived `Debug` would dump all of it into any test failure or log line
/// that happens to format one.
impl std::fmt::Debug for PreprocessedVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreprocessedVideo {{ grid_t: {}, grid_h: {}, grid_w: {}, groups: {} x {} f32 }}",
            self.grid_t,
            self.grid_h,
            self.grid_w,
            self.groups.len(),
            self.groups.first().map_or(0, Vec::len)
        )
    }
}

impl PreprocessedVideo {
    /// Merged tokens this video contributes: one per `merge × merge` block of
    /// patches, per temporal group.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.grid_t * (self.grid_h / sms) * (self.grid_w / sms)
    }
}

/// Pick which frame indices to keep so the clip plays at `fps`.
///
/// `native_fps` is what the container says it runs at. Sampling is by
/// NEAREST-INDEX over a uniform grid rather than by dropping every Nth frame:
/// the latter quantises badly when the ratio is not an integer (a 30fps clip
/// sampled at 2fps by "keep every 15th" is fine, at 2.5fps it is not).
///
/// The result is clamped into `[min_frames, max_frames]` and then to a
/// multiple of `temporal_patch_size`, because a partial group cannot be
/// encoded. Returns indices into the decoded frame list.
pub fn sample_indices(
    n_frames: usize,
    native_fps: f32,
    target_fps: f32,
    min_frames: usize,
    max_frames: usize,
    temporal_patch_size: usize,
) -> Vec<usize> {
    if n_frames == 0 {
        return Vec::new();
    }
    let tp = temporal_patch_size.max(1);
    let native_fps = if native_fps.is_finite() && native_fps > 0.0 {
        native_fps
    } else {
        DEFAULT_FPS
    };
    let target_fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        DEFAULT_FPS
    };

    let duration = n_frames as f32 / native_fps;
    let wanted = (duration * target_fps).round().max(1.0) as usize;

    // Clamp to the checkpoint's band, but never ask for more frames than
    // exist — upsampling a short clip by repeating frames would inflate the
    // token count with no new information.
    let max_frames = max_frames.max(1);
    let min_frames = min_frames.max(1).min(max_frames);
    let wanted = wanted.clamp(min_frames, max_frames).min(n_frames);

    // Round DOWN to a whole number of temporal groups; a partial group has no
    // representation. Never below one group, or there is nothing to encode.
    let wanted = (wanted / tp).max(1) * tp;
    let wanted = wanted.min((n_frames / tp).max(1) * tp).min(n_frames);

    if wanted >= n_frames {
        return (0..n_frames).collect();
    }
    // Uniform positions across the clip, nearest index, deduplicated in order.
    let mut out = Vec::with_capacity(wanted);
    for i in 0..wanted {
        let pos = if wanted == 1 {
            0.0
        } else {
            (i as f32) * ((n_frames - 1) as f32) / ((wanted - 1) as f32)
        };
        out.push((pos.round() as usize).min(n_frames - 1));
    }
    out
}

/// Decode every frame of a container, choosing a backend by what the bytes
/// actually are.
///
/// Returns the frames and the rate they represent. GIF is decoded in-process
/// (pure Rust, no dependency) and reports the container's own average rate,
/// so the caller still has to sample it. ffmpeg resamples during decode, so
/// its frames are ALREADY at `target_fps` and it reports that — which makes
/// the caller's sampling step a no-op rather than a second, lossy resample.
///
/// Dispatch is on MAGIC BYTES, not the declared MIME. A client that labels an
/// mp4 `video/gif`, or sends `application/octet-stream`, still gets the right
/// decoder; and a GIF mislabelled as mp4 does not needlessly spawn a process.
pub fn decode_frames(
    data_uri: &str,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<(Vec<RgbImage>, f32)> {
    let (mime, bytes) = decode_data_uri_bytes(data_uri)?;
    ensure!(!bytes.is_empty(), "the video payload is empty");

    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return decode_gif(&bytes);
    }

    // Everything else goes to the subprocess backend. If it is disabled the
    // error names the flag AND the container, so the operator is not left
    // guessing which of the two problems they have.
    let kind = sniff_container(&bytes, &mime);
    crate::video_decode_ffmpeg::decode_frames(&bytes, target_fps, ffmpeg)
        .with_context(|| format!("decoding {kind}"))
        .map(|f| (f, target_fps))
}

/// Best-effort container name for error messages. Cosmetic only — nothing
/// branches on it — so an unrecognized blob is described as such rather than
/// guessed at.
fn sniff_container(bytes: &[u8], mime: &str) -> String {
    let by_magic = if bytes.len() > 12 && &bytes[4..8] == b"ftyp" {
        Some("an MP4/MOV container")
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("a Matroska/WebM container")
    } else if bytes.starts_with(b"RIFF") {
        Some("an AVI container")
    } else {
        None
    };
    match (by_magic, mime.is_empty()) {
        (Some(k), _) => k.to_string(),
        (None, false) => format!("a {mime} payload"),
        (None, true) => "an unrecognized container".to_string(),
    }
}

/// In-process GIF decode. The rate is derived from the per-frame delays the
/// format stores; a GIF may declare 0 delay ("as fast as possible"), which is
/// treated as the default rather than divided by.
fn decode_gif(bytes: &[u8]) -> Result<(Vec<RgbImage>, f32)> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;
    let decoder =
        GifDecoder::new(std::io::Cursor::new(bytes.to_vec())).context("not a decodable GIF")?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .context("failed to decode animation frames")?;
    ensure!(!frames.is_empty(), "the container decoded to zero frames");

    let total_ms: f64 = frames
        .iter()
        .map(|f| {
            let (num, den) = f.delay().numer_denom_ms();
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        })
        .sum();
    let fps = if total_ms > 0.0 {
        (frames.len() as f64 * 1000.0 / total_ms) as f32
    } else {
        DEFAULT_FPS
    };

    let rgb: Vec<RgbImage> = frames
        .into_iter()
        .map(|f| image::DynamicImage::ImageRgba8(f.into_buffer()).to_rgb8())
        .collect();
    Ok((rgb, fps))
}

/// Full pipeline: a base64 `data:` URI holding an animated container becomes
/// temporal groups of patches.
pub fn preprocess_video(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<PreprocessedVideo> {
    ensure!(
        vcfg.patch_size > 0 && vcfg.spatial_merge_size > 0 && vcfg.temporal_patch_size > 0,
        "vision_config geometry is invalid (patch/merge/temporal size is 0)"
    );
    let (frames, native_fps) = decode_frames(data_uri, target_fps, ffmpeg)?;
    let tp = vcfg.temporal_patch_size;

    let keep = sample_indices(
        frames.len(),
        native_fps,
        target_fps,
        DEFAULT_MIN_FRAMES,
        DEFAULT_MAX_FRAMES,
        tp,
    );
    ensure!(!keep.is_empty(), "frame sampling selected no frames");

    // A clip shorter than one temporal group cannot be encoded as video.
    // Saying so beats silently padding it into a still, which would report a
    // plausible token count for something the model never saw as motion.
    ensure!(
        keep.len() >= tp,
        "video has {} usable frame(s) but temporal_patch_size is {tp}; a clip must carry at \
         least one full temporal group",
        keep.len()
    );

    // Geometry is decided ONCE, from the first kept frame, and applied to all
    // of them. Per-frame sizing would be a correctness bug rather than a
    // refinement: the groups are concatenated into one pad run whose token
    // count assumes a single grid.
    let first = &frames[keep[0]];
    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (th, tw) = target_size_for(first.height(), first.width(), grid_unit, max_pixels);

    let ps = vcfg.patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let grid_t = keep.len() / tp;
    let patch_dim = 3 * tp * ps * ps;
    let plane = grid_h * grid_w;

    let mut groups = Vec::with_capacity(grid_t);
    for g in 0..grid_t {
        // Resize this group's `tp` frames once each, up front: the patch loop
        // reads every pixel of every frame, so resizing inside it would redo
        // the work `patch²` times.
        let resized: Vec<RgbImage> = (0..tp)
            .map(|k| {
                let f = &frames[keep[g * tp + k]];
                image::imageops::resize(f, tw, th, image::imageops::FilterType::CatmullRom)
            })
            .collect();

        let mut pixels = vec![0.0f32; plane * patch_dim];
        for ph in 0..grid_h {
            for pw in 0..grid_w {
                let patch_idx = ph * grid_w + pw;
                for c in 0..3usize {
                    for (t, frame) in resized.iter().enumerate() {
                        for py in 0..ps {
                            for px in 0..ps {
                                let raw = frame
                                    .get_pixel((pw * ps + px) as u32, (ph * ps + py) as u32)[c]
                                    as f32
                                    / 255.0;
                                let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                                pixels[patch_idx * patch_dim + off] = (raw - MEAN[c]) / STD[c];
                            }
                        }
                    }
                }
            }
        }
        groups.push(pixels);
    }

    Ok(PreprocessedVideo {
        groups,
        grid_t,
        grid_h,
        grid_w,
    })
}

#[cfg(test)]
#[path = "video_preprocess_tests.rs"]
mod tests;
