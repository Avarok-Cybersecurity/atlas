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
/// How to spend a fixed row budget when a clip asks for more than it.
///
/// The budget is a CONSERVED PRODUCT — `groups * frame_area <= rows * 1024` —
/// so every choice is a point on one hyperbola and there is no policy that
/// wins for every clip. A security camera wants every second at any
/// resolution; a UI recording wants legible text and can drop frames. The
/// server cannot know which, so the caller picks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VideoFitPolicy {
    /// Keep every temporal group, shrink each frame. Full temporal coverage,
    /// possibly unreadable detail. Matches the HF Qwen3-VL video processor,
    /// whose declared budget is a TOTAL `t*h*w` rather than a per-frame cap.
    #[default]
    Coverage,
    /// Keep the frame size, drop groups. Legible frames, motion missed
    /// between samples.
    Detail,
    /// Split the deficit geometrically: an 8x overshoot becomes ~2.83x on
    /// each axis.
    Balanced,
}

/// The row budget a clip must fit inside, and how to spend it.
#[derive(Debug, Clone, Copy)]
pub struct VideoBudget {
    /// Merged rows this clip may occupy. The encoder's `out_rows`, or a
    /// smaller share when a request carries several media items.
    pub rows: usize,
    /// Per-frame area ceiling already in force (`--vision-max-pixels`).
    pub area_max: Option<usize>,
    pub policy: VideoFitPolicy,
}

/// What the fit decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoFit {
    pub n_groups: usize,
    /// Per-frame pixel budget the resize must respect.
    pub area_per_group: usize,
    /// True when the clip already fitted and nothing was changed.
    pub unchanged: bool,
}

/// Merged rows one frame of `area` pixels costs.
///
/// A merged row is `spatial_merge^2` patches of `patch^2` pixels — 2*2 * 16*16
/// = 1024 px at this checkpoint's geometry. Exact for areas that are whole
/// multiples of the grid unit, which `target_size_for` guarantees.
fn rows_for_area(area: usize, patch: usize, merge: usize) -> usize {
    let per_row = patch * patch * merge * merge;
    if per_row == 0 {
        return usize::MAX;
    }
    area.div_ceil(per_row)
}

/// Choose `(groups, per-frame area)` so the clip fits its row budget.
///
/// ★ A NO-OP WHEN THE CLIP ALREADY FITS. That is the property the whole gate
/// suite depends on: every fixture in vision-fidelity and video-fidelity fits,
/// so their token counts must not move by one.
///
/// 🪤 The floor is ONE grid cell (`patch*merge` squared). Below that a frame
/// has no representable size, so a budget too small for even one cell per
/// group forces groups down instead — refusing is the caller's job, not the
/// fit's.
pub fn fit_video(
    native_area: usize,
    n_groups_req: usize,
    budget: VideoBudget,
    patch: usize,
    merge: usize,
) -> VideoFit {
    let cell = patch * merge;
    let floor_area = cell * cell;
    let area_req = budget.area_max.map_or(native_area, |m| native_area.min(m));
    let rows_req = rows_for_area(area_req, patch, merge);
    let cost = n_groups_req.saturating_mul(rows_req);

    if n_groups_req == 0 || budget.rows == 0 || cost <= budget.rows {
        return VideoFit {
            n_groups: n_groups_req,
            area_per_group: area_req,
            unchanged: true,
        };
    }

    let per_row = patch * patch * merge * merge;
    let area_for_groups = |n: usize| -> usize {
        if n == 0 {
            return floor_area;
        }
        ((budget.rows / n) * per_row).max(floor_area)
    };

    let (n, a) = match budget.policy {
        VideoFitPolicy::Coverage => {
            if n_groups_req > budget.rows {
                // Not even one cell per group: coverage is impossible, keep as
                // many groups as there are rows.
                (budget.rows.max(1), floor_area)
            } else {
                (n_groups_req, area_for_groups(n_groups_req))
            }
        }
        VideoFitPolicy::Detail => {
            let n = (budget.rows / rows_req.max(1)).max(1);
            (n, area_req)
        }
        VideoFitPolicy::Balanced => {
            // k = overshoot; take sqrt(k) off the group count and recompute the
            // area from the groups actually chosen, so integer rounding is
            // absorbed and the product bound holds by construction rather than
            // by a second check.
            let k = (cost as f64) / (budget.rows as f64);
            let n = (((n_groups_req as f64) / k.sqrt()).floor() as usize).max(1);
            let n = n.min(budget.rows.max(1));
            (n, area_for_groups(n))
        }
    };
    VideoFit {
        n_groups: n,
        area_per_group: a,
        unchanged: false,
    }
}

pub fn preprocess_video(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
    // Row budget to fit inside. `None` keeps the previous behaviour byte for
    // byte: no fit is computed and no geometry moves. Every existing caller
    // passes None until the HTTP layer knows the encoder's capacity.
    budget: Option<VideoBudget>,
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

    // ── Fit the clip to its row budget ──
    //
    // This is the ONLY point where both halves of the trade are known: `keep`
    // has fixed the group count and `first` carries the native frame size. A
    // policy upstream of the decode would have to guess one of them; one
    // downstream would be looking at rows already committed.
    //
    // `fit_video` is a no-op when the clip already fits, which is what keeps
    // every gate fixture's token count byte-identical.
    let mut keep = keep;
    let mut eff_max_pixels = max_pixels;
    if let Some(b) = budget {
        let tp_groups = keep.len() / tp;
        let native_area = (first.height() as usize) * (first.width() as usize);
        let fit = fit_video(
            native_area,
            tp_groups,
            b,
            vcfg.patch_size,
            vcfg.spatial_merge_size,
        );
        if !fit.unchanged {
            // Drop whole GROUPS, never part of one: a partial group would be
            // padded into a frame the model never saw.
            let want_frames = (fit.n_groups * tp).min(keep.len());
            if want_frames < keep.len() {
                // Re-sample ACROSS the clip rather than truncating to a prefix
                // — the same mistake `-frames:v` makes in the ffmpeg path.
                let stride = keep.len() as f64 / want_frames as f64;
                let spanned: Vec<usize> = (0..want_frames)
                    .map(|i| keep[((i as f64) * stride) as usize])
                    .collect();
                keep = spanned;
            }
            eff_max_pixels = Some(match eff_max_pixels {
                Some(m) => m.min(fit.area_per_group),
                None => fit.area_per_group,
            });
            tracing::info!(
                policy = ?b.policy,
                budget_rows = b.rows,
                groups_before = tp_groups,
                groups_after = keep.len() / tp,
                native_area,
                area_after = fit.area_per_group,
                "video does not fit its row budget: refitted (a silent rescale is \
                 the trap --vision-max-pixels already fell into, so this says so)"
            );
        }
    }
    let keep = keep;

    // 🚩 THE GRID ROUNDS UP, SO THE FIT'S AREA IS NOT THE FINAL COST.
    //
    // `fit_video` budgets an AREA, but `target_size_for` then snaps the frame
    // to whole grid units and rounds UP. 720 px is 22.5 grid units at this
    // checkpoint's 32-px unit, so a 1280x720 frame becomes 1280x736 and costs
    // 920 merged rows where the area implies 900 — and 145 groups of that
    // overshoot the budget by 2,328 rows. Budgeting the area alone therefore
    // lands just OVER on exactly the sizes people actually send.
    //
    // So the rounding is MEASURED, not predicted: size the frame, cost it with
    // the real grid, shrink, repeat. Each pass scales by the overshoot it just
    // measured, so it converges in a pass or two, and every exit is a frame
    // that has been costed rather than estimated.
    let (mut th, mut tw) =
        target_size_for(first.height(), first.width(), grid_unit, eff_max_pixels);
    if let Some(b) = budget {
        let groups_now = (keep.len() / tp).max(1);
        let sms2 = vcfg.spatial_merge_size * vcfg.spatial_merge_size;
        let cell = (grid_unit as usize) * (grid_unit as usize);
        for _ in 0..8 {
            let rows = ((th as usize / vcfg.patch_size) * (tw as usize / vcfg.patch_size) / sms2)
                * groups_now;
            if rows <= b.rows {
                break;
            }
            let area = (th as usize) * (tw as usize);
            let shrunk =
                (((area as f64) * (b.rows as f64) / (rows as f64)).floor() as usize).max(cell);
            let next = eff_max_pixels.map_or(shrunk, |m| m.min(shrunk));
            // No progress is possible once the cap stops moving or the grid
            // snaps back to the same frame: stop rather than spin. The caller
            // still refuses an over-budget request, so this fails closed.
            if eff_max_pixels == Some(next) {
                break;
            }
            eff_max_pixels = Some(next);
            let (nh, nw) =
                target_size_for(first.height(), first.width(), grid_unit, eff_max_pixels);
            if nh == th && nw == tw {
                break;
            }
            tracing::debug!(
                rows,
                budget_rows = b.rows,
                from = format!("{th}x{tw}"),
                to = format!("{nh}x{nw}"),
                "grid rounding pushed the fitted frame over budget: tightening"
            );
            th = nh;
            tw = nw;
        }
    }
    let (th, tw) = (th, tw);

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

#[cfg(test)]
mod fit_tests {
    use super::*;

    // This checkpoint's geometry: 16-px patches, 2x2 merge -> 1024 px/row.
    const PATCH: usize = 16;
    const MERGE: usize = 2;
    const PER_ROW: usize = PATCH * PATCH * MERGE * MERGE; // 1024
    const CELL: usize = PATCH * MERGE; // 32 -> floor area 1024

    fn budget(rows: usize, policy: VideoFitPolicy) -> VideoBudget {
        VideoBudget {
            rows,
            area_max: None,
            policy,
        }
    }

    /// The property every gate depends on: a clip that already fits is not
    /// touched. If this breaks, all 14 vision cells and all 14 video legs move.
    #[test]
    fn a_clip_that_fits_is_left_exactly_alone() {
        // 4 groups of 224x224 = 49 rows each = 196 rows, far under 16384.
        let f = fit_video(
            224 * 224,
            4,
            budget(16384, VideoFitPolicy::Coverage),
            PATCH,
            MERGE,
        );
        assert!(f.unchanged, "a fitting clip must report unchanged");
        assert_eq!(f.n_groups, 4);
        assert_eq!(f.area_per_group, 224 * 224);
        // ...and under every policy, not just the default.
        for p in [VideoFitPolicy::Detail, VideoFitPolicy::Balanced] {
            let f = fit_video(224 * 224, 4, budget(16384, p), PATCH, MERGE);
            assert!(f.unchanged, "{p:?} must also leave a fitting clip alone");
            assert_eq!(f.n_groups, 4);
        }
    }

    /// Every policy must land inside the budget. This is the bound the encoder
    /// would otherwise enforce by refusing the request.
    #[test]
    fn every_policy_lands_inside_the_row_budget() {
        let rows = 16384;
        // The observed failure: 384 groups of ~592x592 = 134.8 M px, 8.04x over.
        let native = 592 * 592;
        for p in [
            VideoFitPolicy::Coverage,
            VideoFitPolicy::Detail,
            VideoFitPolicy::Balanced,
        ] {
            let f = fit_video(native, 384, budget(rows, p), PATCH, MERGE);
            assert!(!f.unchanged, "{p:?}: an 8x overshoot must be fitted");
            let cost = f.n_groups * f.area_per_group.div_ceil(PER_ROW);
            assert!(
                cost <= rows,
                "{p:?}: fitted to {} groups x {} px = {cost} rows, over budget {rows}",
                f.n_groups,
                f.area_per_group
            );
            assert!(f.n_groups >= 1, "{p:?}: must keep at least one group");
            assert!(
                f.area_per_group >= CELL * CELL,
                "{p:?}: must not go below one grid cell"
            );
        }
    }

    /// COVERAGE keeps time, DETAIL keeps pixels. The whole point of having
    /// both is that they differ on the same clip.
    #[test]
    fn coverage_keeps_groups_and_detail_keeps_area() {
        let rows = 16384;
        let native = 592 * 592;
        let cov = fit_video(
            native,
            384,
            budget(rows, VideoFitPolicy::Coverage),
            PATCH,
            MERGE,
        );
        let det = fit_video(
            native,
            384,
            budget(rows, VideoFitPolicy::Detail),
            PATCH,
            MERGE,
        );
        assert_eq!(cov.n_groups, 384, "coverage keeps every group");
        assert!(cov.area_per_group < native, "coverage pays in pixels");
        assert_eq!(
            det.area_per_group, native,
            "detail keeps the native frame size"
        );
        assert!(det.n_groups < 384, "detail pays in groups");
        // BALANCED sits between them on both axes.
        let bal = fit_video(
            native,
            384,
            budget(rows, VideoFitPolicy::Balanced),
            PATCH,
            MERGE,
        );
        assert!(bal.n_groups > det.n_groups && bal.n_groups < cov.n_groups);
        assert!(bal.area_per_group > cov.area_per_group && bal.area_per_group < det.area_per_group);
    }

    /// The degenerate end: more groups than rows. Coverage cannot give every
    /// group even one cell, so it must give up groups rather than emit a
    /// zero-area frame.
    #[test]
    fn more_groups_than_rows_still_produces_a_representable_frame() {
        let f = fit_video(
            592 * 592,
            5000,
            budget(100, VideoFitPolicy::Coverage),
            PATCH,
            MERGE,
        );
        assert_eq!(
            f.area_per_group,
            CELL * CELL,
            "must fall back to one grid cell"
        );
        assert!(f.n_groups <= 100 && f.n_groups >= 1);
        let cost = f.n_groups * f.area_per_group.div_ceil(PER_ROW);
        assert!(
            cost <= 100,
            "even the degenerate case must fit: {cost} > 100"
        );
    }

    /// A frame so large that one group alone busts the budget: DETAIL cannot
    /// keep the native size, but must still return something encodable.
    #[test]
    fn a_single_oversized_group_is_clamped_not_zeroed() {
        let f = fit_video(
            4096 * 4096,
            1,
            budget(16, VideoFitPolicy::Detail),
            PATCH,
            MERGE,
        );
        assert!(f.n_groups >= 1, "never zero groups");
        assert!(f.area_per_group >= CELL * CELL);
    }

    /// `area_max` (--vision-max-pixels) is a ceiling the fit must respect, and
    /// it must never UPSCALE a frame that is already smaller.
    #[test]
    fn area_max_caps_but_never_upscales() {
        let b = VideoBudget {
            rows: 16384,
            area_max: Some(224 * 224),
            policy: VideoFitPolicy::Coverage,
        };
        let big = fit_video(4096 * 4096, 2, b, PATCH, MERGE);
        assert!(big.area_per_group <= 224 * 224, "area_max must cap");
        let small = fit_video(64 * 64, 2, b, PATCH, MERGE);
        assert_eq!(
            small.area_per_group,
            64 * 64,
            "a small frame must not be upscaled"
        );
    }
}
