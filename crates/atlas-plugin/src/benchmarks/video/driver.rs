// SPDX-License-Identifier: AGPL-3.0-only

//! The state machine: one leg per `next()`.
//!
//! Order mirrors the image benchmark. Geometry first (it establishes that the
//! clip was sampled and grouped at all), then the ordered-color readings,
//! then parity and mixed media, and the CONTROL last — so a vacuous run is
//! discovered after the evidence it invalidates has been collected and can be
//! shown beside the verdict.

use crate::hardware::Sensitivity;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::benchmarks::one_line;
use crate::http;
use crate::metadata::PluginMetadata;
use crate::params::{ParamKind, ParamSpec, ParamValue, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::{BenchmarkResult, LogLine, RunStatus, Stat, Verdict as RunVerdict};

use super::concurrency::{LEVELS, LevelResult, run_level};
use super::geometry::{check_proportional, tokens_per_group};
use super::provision::{CLIPS, Clip, clip, provision};
use super::request;
use super::score::{
    CountCell, OrderCell, Verdict as VideoVerdict, asserted, order_matches, passed, verdict,
};

const SUMMARY: &str = "Video fidelity: temporal-order reading of a color sequence, group-count \
                       geometry, MP4/GIF backend parity, and a no-video control.";

pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "video-fidelity",
    name: "Video Fidelity",
    summary: SUMMARY,
    detail: "Seven check groups over clips of solid colors, one per second. ORDER sends the same \
             sequence forwards and REVERSED and requires the answer to reverse with it — the \
             only assertion that separates 'the frames arrived in order' from 'something \
             arrived', and the one that caught the splice defect where video pad tokens \
             received no encoder rows at all and the model calmly described a gray field while \
             every token count was perfect. GEOMETRY asserts that a clip of twice the duration \
             costs twice the temporal groups, stated as a RATIO so it holds at any \
             --video-fps the server was started with. PARITY requires an MP4 and an identical \
             GIF to produce the same geometry, one through ffmpeg and one through the \
             in-process decoder. MIXED sends an image and a video together, exercising the \
             ordering contract between collection, template markers and pad expansion. \
             INTEGRITY varies media order, request history, and opposite clips in flight. \
             CONCURRENCY requires correct replies and the same prompt-token geometry at \
             C=1, C=2, and C=4. A \
             no-video CONTROL runs last: if it describes a clip it never received, the run is \
             VACUOUS rather than PASS. Legs needing a decoder the server lacks are SKIPPED, \
             never failed — that is a deployment choice.",
    duration_hint: "~1-2 min",
    updated: "2026-08-24",
    needs_confirmation: false,
    intended_for: None,
    threshold_params: &[],
    // Video fidelity does not change with clock.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(VideoFidelity::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Geometry,
    Order,
    Parity,
    Mixed,
    Integrity,
    Concurrency,
    Control,
    Score,
    Done,
}

#[derive(Default)]
pub struct VideoFidelity {
    handle: Option<PluginHandle>,
    phase: Phase,
    started: Option<Instant>,
    order: Vec<OrderCell>,
    counts: Vec<CountCell>,
    control_held: bool,
    /// Tokens reported for the full-length MP4, reused by the parity leg so
    /// it costs one request rather than two.
    full_tokens: Option<usize>,
    /// Merged tokens per temporal group, from the clip's known geometry.
    plane: usize,
    cursor: usize,
    conc_results: Vec<LevelResult>,
    integrity: Vec<crate::benchmarks::media_integrity::Cell>,
    max_tokens: usize,
    request_timeout_s: u64,
}

/// Every color any fixture uses. The scorer looks only for these, so a reply
/// mentioning "the screen" or "background" cannot accidentally count.
const PALETTE: &[&str] = &["red", "green", "blue", "yellow"];

/// Reclassify a transport-level `Error` cell as `Skipped` when the message is
/// the server refusing a container it has no decoder for.
///
/// Every other leg makes the `is_decoder_unavailable` call at its own error
/// site; the heterogeneous-concurrency leg gets its cell from the shared
/// `media_integrity` helper, which is modality-agnostic and cannot know about
/// video decoders — so the call is made here, on the way back. Without it a
/// serve without `--video-allow-ffmpeg` FAILED the whole run at this one leg
/// while the other twelve skipped, contradicting the descriptor's "skipped,
/// never failed" contract (observed 2026-08-15 under `--pull-request-gate`).
fn skip_if_decoder_unavailable(
    cell: crate::benchmarks::media_integrity::Cell,
) -> crate::benchmarks::media_integrity::Cell {
    use crate::benchmarks::media_integrity::Cell;
    match cell {
        Cell::Error { id, msg } if request::is_decoder_unavailable(&msg) => {
            Cell::Skipped { id, why: msg }
        }
        other => other,
    }
}

impl VideoFidelity {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_s)
    }

    fn frame(&self, phase: &str, log: Vec<LogLine>) -> BenchmarkResult {
        let mut r = BenchmarkResult::running(phase, self.elapsed());
        r.progress = Some((
            (self.order.len() + self.counts.len()) as u64,
            (CLIPS.len() + 2) as u64,
        ));
        r.log = log;
        r
    }

    /// Ask about one clip and score the ordered colors it reports.
    async fn read_order(&self, c: &'static Clip) -> OrderCell {
        let h = match self.handle() {
            Ok(h) => h,
            Err(e) => {
                return OrderCell::Error {
                    clip: c.name,
                    msg: one_line(format!("{e:#}")),
                };
            }
        };
        let body = request::video_body(
            &h.target().model,
            c.mime,
            c.bytes,
            request::ORDER_PROMPT,
            self.max_tokens,
        );
        match http::chat_stream(h.target(), &body, self.timeout()).await {
            Ok(out) => {
                let reply = out.text.trim().to_string();
                if order_matches(&reply, c.colors, PALETTE) {
                    OrderCell::Match {
                        clip: c.name,
                        seen: one_line(reply),
                    }
                } else {
                    let got = super::score::colors_in_order(&reply, PALETTE);
                    if got.is_empty() {
                        OrderCell::NotSeen {
                            clip: c.name,
                            reply: one_line(reply),
                        }
                    } else {
                        OrderCell::WrongOrder {
                            clip: c.name,
                            want: c.colors.join(", "),
                            got: got.join(", "),
                        }
                    }
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                if request::is_decoder_unavailable(&msg) {
                    OrderCell::Skipped {
                        clip: c.name,
                        why: one_line(msg),
                    }
                } else {
                    OrderCell::Error {
                        clip: c.name,
                        msg: one_line(msg),
                    }
                }
            }
        }
    }

    /// Prompt tokens for one clip, or a reason it could not be measured.
    async fn tokens_for(&self, c: &'static Clip) -> std::result::Result<usize, (bool, String)> {
        let h = self.handle().map_err(|e| (false, format!("{e:#}")))?;
        let body = request::video_body(&h.target().model, c.mime, c.bytes, "Reply with OK.", 8);
        match http::chat_stream(h.target(), &body, self.timeout()).await {
            Ok(out) => Ok(out.prompt_tokens),
            Err(e) => {
                let msg = format!("{e:#}");
                let skip = request::is_decoder_unavailable(&msg);
                Err((skip, one_line(msg)))
            }
        }
    }
}

impl Plugin for VideoFidelity {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    async fn load(&mut self, handle: PluginHandle) -> Result<()> {
        // Materialize the clips before anything needs them, so a provisioning
        // failure is reported where the user can act on it rather than
        // mid-run.
        provision(handle.artifacts()).context("provisioning video fixtures")?;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Benchmark for VideoFidelity {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "max_tokens",
                "Max tokens per reply",
                "The color list is short by design, so this only needs to be generous \
                 enough that a reply is not truncated mid-sequence. Keep it well above the \
                 model's thinking budget if you re-enable thinking, or a reasoning block \
                 consumes the whole budget and returns empty content that reads as a video \
                 failure and is not one. The same trap has a thinking-OFF form, which is \
                 why the default is 320 and not the 120 it was: a cell that sends TWO \
                 media items can provoke a preamble the single-item cells never see, and \
                 a budget that truncates the preamble truncates the answer with it.",
                ParamKind::Int { min: 16, max: 2048 },
                // 320, from measurement rather than taste. On qwen3.8-27B-NVFP4 the
                // video-before-image reply needs 153 completion tokens and returned
                // `finish_reason=length` at the old 120, so the 4th color could fall
                // outside the budget and the cell FAILED as "wanted [red, green, blue,
                // yellow], got [red, green, blue]" — which reads as a model fidelity
                // limit and is not one. The identical video-ONLY control answers in 8
                // tokens: with a second media item present the model stops obeying the
                // prompt's "Only color names" and writes a preamble first. Whether the
                // list survived came down to where that preamble happened to end, so
                // the cell passed on one serve config and failed on another (2026-08-15;
                // a one-variable A/B also cleared the non_thinking preset's
                // presence_penalty=1.5 as the cause — pp=1.5 and pp=0 answer alike).
                // Raising the cap cannot slow the well-behaved cells: every one of them
                // stops at EOS far below even the old ceiling.
                ParamValue::Int(320),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Per-request timeout (s)",
                "Video prefill is heavier than an image's: a clip costs its whole patch \
                 grid once per temporal group, so a long clip at a high --video-fps is \
                 several images' worth of work.",
                ParamKind::Int { min: 30, max: 3600 },
                ParamValue::Int(300),
            ),
        ]
    }

    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        self.max_tokens = values.int("max_tokens")? as usize;
        self.request_timeout_s = values.int("request_timeout_s")? as u64;
        if self.max_tokens == 0 {
            bail!("max_tokens must be positive");
        }
        // Every fixture is 224x224, so one figure covers them all.
        self.plane = tokens_per_group(224, 224, 16, 2) as usize;
        self.started = Some(Instant::now());
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        match self.phase {
            // ── Geometry: proportionality at 1x / 2x / 4x ────────────────
            Phase::Geometry => {
                self.phase = Phase::Order;
                let unit = clip("05_colors_unit.mp4").context("fixture 05 missing")?;
                let half = clip("04_colors_half.mp4").context("fixture 04 missing")?;
                let full = clip("01_colors_fwd.mp4").context("fixture 01 missing")?;

                let mut totals = Vec::new();
                for c in [unit, half, full] {
                    match self.tokens_for(c).await {
                        Ok(n) => totals.push(n),
                        Err((skip, why)) => {
                            let cell = if skip {
                                CountCell::Skipped {
                                    id: "group-proportionality",
                                    why: why.clone(),
                                }
                            } else {
                                CountCell::Error {
                                    id: "group-proportionality",
                                    msg: why.clone(),
                                }
                            };
                            self.counts.push(cell);
                            return Ok(self.frame(
                                "geometry",
                                vec![LogLine::info(format!("group-proportionality: {why}"))],
                            ));
                        }
                    }
                }
                let (t1, t2, t4) = (totals[0], totals[1], totals[2]);
                // Reused by the parity leg, so the 4s clip is measured once.
                self.full_tokens = Some(t4);

                let cell = match check_proportional(t1, t2, t4, self.plane) {
                    Some(r) => CountCell::Match {
                        id: "group-proportionality",
                        detail: format!(
                            "1s={t1}, 2s={t2}, 4s={t4} tok -> {}/{}/{} groups at {} tok/group \
                             (template overhead {})",
                            r.unit_groups,
                            r.unit_groups * 2,
                            r.unit_groups * 4,
                            self.plane,
                            r.overhead
                        ),
                    },
                    None => CountCell::Mismatch {
                        id: "group-proportionality",
                        detail: format!(
                            "1s={t1}, 2s={t2}, 4s={t4} tok: groups do not scale with duration \
                             ((t4-t2) is {} where 2*(t2-t1) is {}), so sampling or temporal \
                             grouping is wrong",
                            t4.saturating_sub(t2),
                            2 * t2.saturating_sub(t1)
                        ),
                    },
                };
                let line = match &cell {
                    CountCell::Match { detail, .. } => {
                        LogLine::info(format!("group-proportionality: {detail}"))
                    }
                    CountCell::Mismatch { detail, .. } => {
                        LogLine::warn(format!("group-proportionality: {detail}"))
                    }
                    _ => LogLine::info("group-proportionality".to_string()),
                };
                self.counts.push(cell);
                Ok(self.frame("geometry", vec![line]))
            }

            // ── Order: forwards, then the same sequence reversed ─────────
            Phase::Order => {
                let ordered: Vec<&'static Clip> =
                    CLIPS.iter().filter(|c| c.colors.len() == 4).collect();
                let c = ordered[self.cursor];
                let cell = self.read_order(c).await;
                let line = match &cell {
                    OrderCell::Match { seen, .. } => LogLine::info(format!("{}: {seen}", c.name)),
                    OrderCell::WrongOrder { want, got, .. } => {
                        LogLine::warn(format!("{}: wanted [{want}], got [{got}]", c.name))
                    }
                    OrderCell::NotSeen { reply, .. } => {
                        LogLine::warn(format!("{}: no colors named — {reply}", c.name))
                    }
                    OrderCell::Skipped { why, .. } => {
                        LogLine::info(format!("{}: skipped — {why}", c.name))
                    }
                    OrderCell::Error { msg, .. } => LogLine::warn(format!("{}: {msg}", c.name)),
                };
                self.order.push(cell);
                self.cursor += 1;
                if self.cursor >= ordered.len() {
                    self.cursor = 0;
                    self.phase = Phase::Parity;
                }
                Ok(self.frame("order", vec![line]))
            }

            // ── Parity: the MP4 and the GIF must agree on geometry ───────
            Phase::Parity => {
                self.phase = Phase::Mixed;
                let gif = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
                let cell = match (self.full_tokens, self.tokens_for(gif).await) {
                    (Some(mp4), Ok(g)) if mp4 == g => CountCell::Match {
                        id: "backend-parity",
                        detail: format!("mp4 and gif both {g} prompt tokens"),
                    },
                    (Some(mp4), Ok(g)) => CountCell::Mismatch {
                        id: "backend-parity",
                        detail: format!(
                            "identical content decoded to different geometry: mp4 {mp4} tokens, \
                             gif {g}"
                        ),
                    },
                    (None, Ok(_)) => CountCell::Skipped {
                        id: "backend-parity",
                        why: "the mp4 side was not measured".to_string(),
                    },
                    (_, Err((skip, why))) => {
                        if skip {
                            CountCell::Skipped {
                                id: "backend-parity",
                                why,
                            }
                        } else {
                            CountCell::Error {
                                id: "backend-parity",
                                msg: why,
                            }
                        }
                    }
                };
                let line = match &cell {
                    CountCell::Match { detail, .. } => {
                        LogLine::info(format!("backend-parity: {detail}"))
                    }
                    CountCell::Mismatch { detail, .. } => {
                        LogLine::warn(format!("backend-parity: {detail}"))
                    }
                    CountCell::Skipped { why, .. } => {
                        LogLine::info(format!("backend-parity: skipped — {why}"))
                    }
                    CountCell::Error { msg, .. } => LogLine::warn(format!("backend-parity: {msg}")),
                };
                self.counts.push(cell);
                Ok(self.frame("parity", vec![line]))
            }

            // ── Mixed: an image and a video in one request ───────────────
            Phase::Mixed => {
                self.phase = Phase::Integrity;
                let h = self.handle()?;
                let vid = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
                // The image comes from the vision benchmark's own ladder
                // rather than a second copy of the same bytes.
                // GRAYSCALE on purpose: the question asks for the VIDEO's
                // colors, so a colored still lets a model answer from the
                // wrong item and still look right. With a gray one, any
                // palette color in the reply can only have come from the clip.
                let png = crate::benchmarks::vision::provision::FIXTURES
                    .iter()
                    .find(|(n, _, _, _)| *n == "13_gray_224.jpg")
                    .map(|(_, b, _, _)| *b)
                    .context("grayscale fixture missing")?;
                let body = request::mixed_body(
                    &h.target().model,
                    png,
                    vid.mime,
                    vid.bytes,
                    "You were given one image and one video. Reply with only the colors in the \
                     VIDEO, in order, separated by commas.",
                    self.max_tokens,
                );
                let cell = match http::chat_stream(h.target(), &body, self.timeout()).await {
                    Ok(out) => {
                        let reply = out.text.trim();
                        if order_matches(reply, vid.colors, PALETTE) {
                            CountCell::Match {
                                id: "mixed-media",
                                detail: format!(
                                    "image + video in one request, {} prompt tokens, video read \
                                     correctly",
                                    out.prompt_tokens
                                ),
                            }
                        } else {
                            CountCell::Mismatch {
                                id: "mixed-media",
                                detail: format!(
                                    "image + video together: wanted [{}], got [{}] — the \
                                     ordering contract between collection, markers and pad \
                                     expansion is off",
                                    vid.colors.join(", "),
                                    super::score::colors_in_order(reply, PALETTE).join(", ")
                                ),
                            }
                        }
                    }
                    Err(e) => {
                        let msg = one_line(format!("{e:#}"));
                        if request::is_decoder_unavailable(&msg) {
                            CountCell::Skipped {
                                id: "mixed-media",
                                why: msg,
                            }
                        } else {
                            CountCell::Error {
                                id: "mixed-media",
                                msg,
                            }
                        }
                    }
                };
                let line = match &cell {
                    CountCell::Match { detail, .. } => {
                        LogLine::info(format!("mixed-media: {detail}"))
                    }
                    CountCell::Mismatch { detail, .. } => {
                        LogLine::warn(format!("mixed-media: {detail}"))
                    }
                    CountCell::Skipped { why, .. } => {
                        LogLine::info(format!("mixed-media: skipped — {why}"))
                    }
                    CountCell::Error { msg, .. } => LogLine::warn(format!("mixed-media: {msg}")),
                };
                self.counts.push(cell);
                Ok(self.frame("mixed", vec![line]))
            }

            // ── Integrity: the shapes that need MORE THAN ONE video item ─
            //
            // The shared media_integrity legs (run by the image benchmark)
            // already cover the machinery both modalities share. What they
            // cannot reach is grid bookkeeping across SEVERAL vision items
            // when at least one has t_len > 1 — a video occupies grid_t
            // encoder rows behind a single pad run, so any place that walks
            // rows and items in lockstep is only exercised here.
            Phase::Integrity => {
                use crate::benchmarks::media_integrity as mi;
                let h = self.handle()?;
                let model = h.target().model.clone();
                let tmo = self.timeout();
                let fwd = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
                let rev = clip("02_colors_rev.mp4").context("fixture 02 missing")?;

                let cell = match self.cursor {
                    // 1. TWO VIDEOS IN ONE REQUEST. Two items, each t_len > 1,
                    //    so the second video's rows start at an offset that
                    //    only a correct per-item walk produces. Asking about
                    //    the SECOND is the discriminating question: an
                    //    off-by-one hands back the first's colors.
                    0 => {
                        let body = serde_json::json!({
                            "model": model, "stream": true, "temperature": 0.0,
                            "max_tokens": self.max_tokens,
                            "chat_template_kwargs": {"enable_thinking": false},
                            "messages": [{"role": "user", "content": [
                                {"type": "video_url", "video_url": {"url":
                                    request::data_uri(fwd.mime, fwd.bytes)}},
                                {"type": "video_url", "video_url": {"url":
                                    request::data_uri(rev.mime, rev.bytes)}},
                                {"type": "text", "text":
                                    "Two videos were provided. List the colors of the SECOND \
                                     video in the order they appear, separated by commas. \
                                     Answer with only the color names."},
                            ]}],
                        });
                        match http::chat_stream(h.target(), &body, tmo).await {
                            Ok(o) if order_matches(o.text.trim(), rev.colors, PALETTE) => {
                                mi::Cell::Pass {
                                    id: "two-videos",
                                    detail: format!(
                                        "second of two clips read correctly ({} prompt tokens)",
                                        o.prompt_tokens
                                    ),
                                }
                            }
                            Ok(o) => {
                                let got = super::score::colors_in_order(o.text.trim(), PALETTE);
                                let first_instead = got == fwd.colors.to_vec();
                                mi::Cell::Fail {
                                    id: "two-videos",
                                    detail: if first_instead {
                                        "asked for the SECOND clip and got the FIRST — the \
                                         per-item row offset is wrong across two videos"
                                            .to_string()
                                    } else {
                                        format!(
                                            "wanted [{}], got [{}]",
                                            rev.colors.join(", "),
                                            got.join(", ")
                                        )
                                    },
                                }
                            }
                            Err(e) => {
                                let msg = one_line(format!("{e:#}"));
                                if request::is_decoder_unavailable(&msg) {
                                    mi::Cell::Skipped {
                                        id: "two-videos",
                                        why: msg,
                                    }
                                } else {
                                    mi::Cell::Error {
                                        id: "two-videos",
                                        msg,
                                    }
                                }
                            }
                        }
                    }
                    // 2. VIDEO BEFORE IMAGE. The Mixed leg sends image-then-
                    //    video; this is the same contract in the other order,
                    //    where the image's pad run follows a multi-group item
                    //    rather than preceding it.
                    1 => {
                        // GRAYSCALE on purpose: the question asks for the VIDEO's
                        // colors, so a colored still lets a model answer from the
                        // wrong item and still look right. With a gray one, any
                        // palette color in the reply can only have come from the clip.
                        let png = crate::benchmarks::vision::provision::FIXTURES
                            .iter()
                            .find(|(n, _, _, _)| *n == "13_gray_224.jpg")
                            .map(|(_, b, _, _)| *b)
                            .context("grayscale fixture missing")?;
                        let body = serde_json::json!({
                            "model": model, "stream": true, "temperature": 0.0,
                            "max_tokens": self.max_tokens,
                            "chat_template_kwargs": {"enable_thinking": false},
                            "messages": [{"role": "user", "content": [
                                {"type": "video_url", "video_url": {"url":
                                    request::data_uri(fwd.mime, fwd.bytes)}},
                                {"type": "image_url", "image_url": {"url":
                                    request::data_uri("image/jpeg", png)}},
                                {"type": "text", "text":
                                    "A video came first, then a still image. List the colors of \
                                     the VIDEO in order, separated by commas. Only color names."},
                            ]}],
                        });
                        match http::chat_stream(h.target(), &body, tmo).await {
                            Ok(o) if order_matches(o.text.trim(), fwd.colors, PALETTE) => {
                                mi::Cell::Pass {
                                    id: "video-before-image",
                                    detail: format!(
                                        "video read correctly when it precedes the image ({} \
                                         prompt tokens)",
                                        o.prompt_tokens
                                    ),
                                }
                            }
                            Ok(o) => mi::Cell::Fail {
                                id: "video-before-image",
                                detail: format!(
                                    "wanted [{}], got [{}] — the ordering contract holds one way \
                                     round but not the other",
                                    fwd.colors.join(", "),
                                    super::score::colors_in_order(o.text.trim(), PALETTE)
                                        .join(", ")
                                ),
                            },
                            Err(e) => {
                                let msg = one_line(format!("{e:#}"));
                                if request::is_decoder_unavailable(&msg) {
                                    mi::Cell::Skipped {
                                        id: "video-before-image",
                                        why: msg,
                                    }
                                } else {
                                    mi::Cell::Error {
                                        id: "video-before-image",
                                        msg,
                                    }
                                }
                            }
                        }
                    }
                    // 3. TWO DIFFERENT CLIPS CONCURRENTLY. The C=1/2/4 sweep
                    //    sends identical requests, which cannot detect a
                    //    mis-sliced per-request offset. Two clips with
                    //    OPPOSITE color orders in flight together can: a
                    //    crossed offset returns the other one's sequence, and
                    //    the reversal makes that unmistakable.
                    2 => {
                        let f = fwd.colors.to_vec();
                        let r = rev.colors.to_vec();
                        let subjects: Vec<mi::Subject> = vec![
                            (
                                request::video_body(
                                    &model,
                                    fwd.mime,
                                    fwd.bytes,
                                    request::ORDER_PROMPT,
                                    self.max_tokens,
                                ),
                                Box::new(move |t: &str| order_matches(t, &f, PALETTE)),
                                "forward".to_string(),
                            ),
                            (
                                request::video_body(
                                    &model,
                                    rev.mime,
                                    rev.bytes,
                                    request::ORDER_PROMPT,
                                    self.max_tokens,
                                ),
                                Box::new(move |t: &str| order_matches(t, &r, PALETTE)),
                                "reversed".to_string(),
                            ),
                        ];
                        skip_if_decoder_unavailable(
                            mi::heterogeneous_concurrency(h, subjects, tmo).await,
                        )
                    }
                    // 4. VIDEO IN AN EARLIER TURN.
                    _ => {
                        let body = serde_json::json!({
                            "model": model, "stream": true, "temperature": 0.0,
                            "max_tokens": self.max_tokens,
                            "chat_template_kwargs": {"enable_thinking": false},
                            "messages": [
                                {"role": "user", "content": [
                                    {"type": "video_url", "video_url": {"url":
                                        request::data_uri(fwd.mime, fwd.bytes)}},
                                    {"type": "text", "text": "Here is a clip."}]},
                                {"role": "assistant", "content": "Understood."},
                                {"role": "user", "content":
                                    "List the colors of the video I sent earlier, in order, \
                                     separated by commas. Only color names."},
                            ],
                        });
                        let want = fwd.colors.to_vec();
                        mi::media_in_history(
                            h,
                            body,
                            &move |t: &str| order_matches(t, &want, PALETTE),
                            "video-in-history",
                            tmo,
                        )
                        .await
                    }
                };
                // A pass and a skip both read as info; only a real failure
                // warns. (Written as one condition rather than two identical
                // arms, which clippy rightly objects to.)
                let line = if cell.passed() || matches!(cell, mi::Cell::Skipped { .. }) {
                    LogLine::info(cell.line())
                } else {
                    LogLine::warn(cell.line())
                };
                // Integrity cells join the same tally the other legs use, so a
                // failure here fails the run rather than being informational.
                self.counts.push(if cell.passed() {
                    CountCell::Match {
                        id: cell.id(),
                        detail: String::new(),
                    }
                } else if matches!(cell, mi::Cell::Skipped { .. }) {
                    CountCell::Skipped {
                        id: cell.id(),
                        why: String::new(),
                    }
                } else {
                    CountCell::Mismatch {
                        id: cell.id(),
                        detail: cell.line(),
                    }
                });
                self.integrity.push(cell);
                self.cursor += 1;
                if self.cursor >= 4 {
                    self.cursor = 0;
                    self.phase = Phase::Concurrency;
                }
                Ok(self.frame("integrity", vec![line]))
            }

            // ── Concurrency: C = 1, 2, 4 of the same request in flight ───
            //
            // Survival AND correctness. The vision path shares a packed
            // encoder output buffer and a grid vector across concurrent
            // requests, so a base-offset mistake hands one request another's
            // embeddings — a plausible answer to the wrong question, with no
            // error anywhere. Timing is recorded, not asserted: prefill
            // serializes today so wall time grows with C, and turning that
            // into a threshold would make a future scheduler improvement fail
            // the run.
            Phase::Concurrency => {
                let level = LEVELS[self.cursor];
                let c = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
                let want: Vec<&str> = c.colors.to_vec();
                let body = request::video_body(
                    &self.handle()?.target().model,
                    c.mime,
                    c.bytes,
                    request::ORDER_PROMPT,
                    self.max_tokens,
                );
                let is_correct = |reply: &str| order_matches(reply, &want, PALETTE);
                let r = run_level(self.handle()?, &body, level, self.timeout(), &is_correct).await;

                let baseline_prompt_tokens = self
                    .conc_results
                    .first()
                    .and_then(|baseline| baseline.prompt_tokens);
                let clean = baseline_prompt_tokens
                    .map_or_else(|| r.ok(), |baseline| r.ok_against(baseline));
                let geometry = r.geometry_detail(baseline_prompt_tokens);
                let line = if clean {
                    LogLine::info(format!(
                        "C={level}: {}/{} correct, {geometry}, {} ms",
                        r.correct, r.conc, r.wall_ms,
                    ))
                } else {
                    LogLine::warn(format!(
                        "C={level}: {}/{} returned, {}/{} CORRECT, {geometry}, {} ms{}",
                        r.returned,
                        r.conc,
                        r.correct,
                        r.conc,
                        r.wall_ms,
                        if r.errors.is_empty() {
                            String::new()
                        } else {
                            format!(" — {}", r.errors.join("; "))
                        }
                    ))
                };
                let cell = if r.errors.iter().any(|e| request::is_decoder_unavailable(e)) {
                    CountCell::Skipped {
                        id: "concurrency",
                        why: format!("C={level}: no video decoder"),
                    }
                } else if clean {
                    CountCell::Match {
                        id: "concurrency",
                        detail: format!("C={level} clean in {} ms", r.wall_ms),
                    }
                } else {
                    CountCell::Mismatch {
                        id: "concurrency",
                        detail: format!("C={level}: {}/{} correct, {geometry}", r.correct, r.conc),
                    }
                };
                self.counts.push(cell);
                self.conc_results.push(r);
                self.cursor += 1;
                if self.cursor >= LEVELS.len() {
                    self.cursor = 0;
                    self.phase = Phase::Control;
                }
                Ok(self.frame("concurrency", vec![line]))
            }

            // ── Control: the same question, no video ─────────────────────
            Phase::Control => {
                self.phase = Phase::Score;
                let h = self.handle()?;
                let body = request::text_only_body(
                    &h.target().model,
                    request::ORDER_PROMPT,
                    self.max_tokens,
                );
                let line = match http::chat_stream(h.target(), &body, self.timeout()).await {
                    Ok(out) => {
                        let reply = out.text.trim();
                        // It holds unless it produces a full four-color
                        // sequence — which would mean the earlier readings
                        // could have come from the prompt rather than pixels.
                        let looks_seen =
                            super::score::colors_in_order(reply, PALETTE).len() >= PALETTE.len();
                        self.control_held = !looks_seen;
                        if self.control_held {
                            LogLine::info(format!(
                                "control: no video, no full sequence — readings stand ({})",
                                one_line(reply.chars().take(60).collect::<String>())
                            ))
                        } else {
                            LogLine::warn(format!(
                                "control: named every color with NO video attached — the \
                                 readings are not evidence ({})",
                                one_line(reply.chars().take(60).collect::<String>())
                            ))
                        }
                    }
                    Err(e) => {
                        // A control that could not run cannot vouch for
                        // anything, so it does not get to pass by default.
                        self.control_held = false;
                        LogLine::warn(format!("control failed: {}", one_line(format!("{e:#}"))))
                    }
                };
                Ok(self.frame("control", vec![line]))
            }

            Phase::Score => {
                self.phase = Phase::Done;
                let v = verdict(&self.order, &self.counts, self.control_held);
                let asserted_n = asserted(&self.order, &self.counts);
                let passed_n = passed(&self.order, &self.counts);
                let skipped = self
                    .order
                    .iter()
                    .filter(|c| matches!(c, OrderCell::Skipped { .. }))
                    .count()
                    + self
                        .counts
                        .iter()
                        .filter(|c| matches!(c, CountCell::Skipped { .. }))
                        .count();

                let mut r = BenchmarkResult::running("score", self.elapsed());
                r.status = if v == VideoVerdict::Pass {
                    RunStatus::Completed
                } else {
                    RunStatus::Failed
                };
                r.summary = vec![
                    Stat::new("verdict", v.to_string(), ""),
                    Stat::new("legs", format!("{passed_n}/{asserted_n}"), "passed"),
                    Stat::new("skipped", skipped.to_string(), "legs"),
                ];
                r.metrics.insert("legs_passed".into(), passed_n as f64);
                r.metrics.insert("legs_asserted".into(), asserted_n as f64);
                r.metrics.insert("legs_skipped".into(), skipped as f64);
                r.metrics
                    .insert("control_held".into(), self.control_held as u8 as f64);
                for lr in &self.conc_results {
                    r.metrics
                        .insert(format!("conc_{}_wall_ms", lr.conc), lr.wall_ms as f64);
                    r.metrics
                        .insert(format!("conc_{}_correct", lr.conc), lr.correct as f64);
                }
                r.verdict = Some(match v {
                    VideoVerdict::Pass => RunVerdict::pass(format!(
                        "{passed_n}/{asserted_n} legs passed, control held, {skipped} skipped"
                    )),
                    VideoVerdict::Fail => RunVerdict::fail(format!(
                        "{passed_n}/{asserted_n} legs passed, {skipped} skipped"
                    )),
                    VideoVerdict::Vacuous => RunVerdict::fail(
                        "VACUOUS: the no-video control named the whole color sequence, so the \
                         readings are not evidence"
                            .to_string(),
                    ),
                    VideoVerdict::Inconclusive => RunVerdict::fail(
                        "INCONCLUSIVE: every leg was skipped — no video decoder is available, \
                         so nothing was measured"
                            .to_string(),
                    ),
                });
                Ok(r)
            }

            Phase::Done => {
                let mut r = BenchmarkResult::running("done", self.elapsed());
                r.status = RunStatus::Completed;
                Ok(r)
            }
        }
    }
}

#[cfg(test)]
#[path = "driver_tests.rs"]
mod driver_tests;
