// SPDX-License-Identifier: AGPL-3.0-only

//! The state machine: one leg per `next()`.
//!
//! Order matters. Calibration first (it measures the template overhead every
//! later assertion subtracts), then geometry, then the capability probes, then
//! the control LAST — so a vacuous run is discovered after the evidence it
//! invalidates has been collected and can be shown alongside the verdict.

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

use super::geometry::expected_vision_tokens;
use super::probes::{CONTROL, PROBES, Probe, concurrency_probe};
use super::provision::{FIXTURES, provision};
use super::request;
use super::score::{
    GeomCell, ProbeCell, Verdict as VisionVerdict, asserted_cells, reply_matches, verdict,
    with_runtime_checks,
};

const SUMMARY: &str = "Vision fidelity: exact vision-token geometry across a resolution ladder, \
                       plus capability probes with a no-image control.";

pub const METADATA: PluginMetadata = PluginMetadata::atlas(SUMMARY);

pub const DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "vision-fidelity",
    name: "Vision Fidelity",
    summary: SUMMARY,
    detail: "Two legs. GEOMETRY sends a ladder of committed fixtures (224² through 1280×720, \
             square, wide and portrait, deliberately mixing grid-exact sizes with ones that \
             must snap) and asserts the EXACT vision-token count from usage.prompt_tokens \
             against patch/merge arithmetic — the observable that moves when preprocessing \
             changes, and the one a capability check cannot see. CAPABILITY asks unambiguous \
             questions about those images. A no-image CONTROL runs last: if it answers as \
             though it saw a picture, the capability leg proved nothing and the run reports \
             VACUOUS rather than PASS. Images above the server's encoder capacity report \
             UNMEASURED, never FAIL — that is a deployment setting, not a defect.",
    duration_hint: "~1-2 min",
    updated: "2026-08-14",
    needs_confirmation: false,
    // Vision correctness is a property of the ENGINE plus the checkpoint's own
    // declared geometry, not of one model, so any vision-capable checkpoint is
    // a valid subject. A model without a vision tower fails at the first
    // request with a clear error rather than being silently skipped.
    intended_for: None,
    threshold_params: &[],
    // Image fidelity does not change with clock.
    sensitivity: Sensitivity::Correctness,
    ctor: || Box::new(VisionFidelity::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Calibrate,
    Geometry,
    Probes,
    Integrity,
    Concurrency,
    Control,
    Score,
    Done,
}

#[derive(Default)]
pub struct VisionFidelity {
    handle: Option<PluginHandle>,
    phase: Phase,
    started: Option<Instant>,
    /// Chat-template cost in tokens, measured in `Calibrate`. Every geometry
    /// assertion subtracts it, so it is measured rather than assumed — it is a
    /// property of the checkpoint's template and moves when the template does.
    overhead: Option<usize>,
    /// Encoder capacity in patches, inferred from the first over-capacity
    /// rejection. `None` until something is rejected.
    geom: Vec<GeomCell>,
    probes: Vec<ProbeCell>,
    control_held: bool,
    cursor: usize,
    conc_results: Vec<crate::benchmarks::video::concurrency::LevelResult>,
    integrity: Vec<crate::benchmarks::media_integrity::Cell>,
    max_tokens: usize,
    request_timeout_s: u64,
}

impl VisionFidelity {
    fn handle(&self) -> Result<&PluginHandle> {
        self.handle.as_ref().context("benchmark was not loaded")
    }

    fn elapsed(&self) -> Duration {
        self.started.map(|s| s.elapsed()).unwrap_or_default()
    }

    fn timeout(&self) -> Duration {
        Duration::from_secs(self.request_timeout_s)
    }

    fn fixture(&self, name: &str) -> Result<&'static [u8]> {
        FIXTURES
            .iter()
            .find(|(n, _, _, _)| *n == name)
            .map(|(_, b, _, _)| *b)
            .with_context(|| format!("fixture {name} is not in the provisioned set"))
    }

    fn frame(&self, phase: &str, log: Vec<LogLine>) -> BenchmarkResult {
        let mut r = BenchmarkResult::running(phase, self.elapsed());
        r.progress = Some((
            (self.geom.len() + self.probes.len()) as u64,
            (FIXTURES.len() + PROBES.len()) as u64,
        ));
        r.log = log;
        r
    }

    /// Send one probe and score it.
    async fn run_probe(&self, p: &Probe) -> ProbeCell {
        let handle = match self.handle() {
            Ok(h) => h,
            Err(e) => {
                return ProbeCell::Error {
                    id: p.id,
                    msg: one_line(format!("{e:#}")),
                };
            }
        };
        let images: Vec<&[u8]> = match p
            .images
            .iter()
            .map(|n| self.fixture(n))
            .collect::<Result<Vec<_>>>()
        {
            Ok(v) => v,
            Err(e) => {
                return ProbeCell::Error {
                    id: p.id,
                    msg: one_line(format!("{e:#}")),
                };
            }
        };
        let body = request::body(&handle.target().model, &images, p.prompt, self.max_tokens);
        match http::chat_stream(handle.target(), &body, self.timeout()).await {
            Ok(o) => {
                if reply_matches(&o.text, p.want_all, p.want_none) {
                    ProbeCell::Pass { id: p.id }
                } else {
                    ProbeCell::Fail {
                        id: p.id,
                        reply: one_line(&o.text),
                    }
                }
            }
            Err(e) => ProbeCell::Error {
                id: p.id,
                msg: one_line(format!("{e:#}")),
            },
        }
    }
}

impl Plugin for VisionFidelity {
    fn metadata(&self) -> &'static PluginMetadata {
        &METADATA
    }

    async fn load(&mut self, handle: PluginHandle) -> Result<()> {
        // Materialise the fixtures before anything needs them, so a
        // provisioning failure is reported where the user can act on it
        // rather than mid-run.
        provision(handle.artifacts()).context("provisioning vision fixtures")?;
        self.handle = Some(handle);
        Ok(())
    }
}

impl Benchmark for VisionFidelity {
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        &DESCRIPTOR
    }

    fn parameters(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new(
                "max_tokens",
                "Max tokens per reply",
                "Probe replies are short by design; the geometry leg needs almost none. \
                 Keep this well above the model's thinking budget if you disable the \
                 thinking-off default, or a reasoning block will consume the whole budget \
                 and return empty content that reads as a vision failure.",
                ParamKind::Int { min: 16, max: 2048 },
                ParamValue::Int(128),
            ),
            ParamSpec::new(
                "request_timeout_s",
                "Per-request timeout (s)",
                "A large fixture at a high area bound can take a while to prefill.",
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
        Ok(())
    }

    async fn next(&mut self) -> Result<BenchmarkResult> {
        let handle = self.handle()?.clone();
        handle.check_cancelled()?;
        if self.started.is_none() {
            self.started = Some(Instant::now());
        }

        match self.phase {
            // ── Calibrate ────────────────────────────────────────────────
            // Measure the chat template's own token cost using a fixture whose
            // vision-token count is known. Hard-coding it would silently drift
            // the moment a checkpoint changed its template — which #513 just
            // did for this very family.
            Phase::Calibrate => {
                let (name, bytes, w, h) = FIXTURES[0];
                let body = request::body(&handle.target().model, &[bytes], "Colour?", 8);
                let out = http::chat_stream(handle.target(), &body, self.timeout())
                    .await
                    .context("calibration request failed — is this a vision-capable model?")?;
                let want = expected_vision_tokens(w, h, 16, 2) as usize;
                let overhead = out.prompt_tokens.checked_sub(want).with_context(|| {
                    format!(
                        "calibration: {name} reported {} prompt tokens but its {want} vision \
                         tokens alone exceed that — the served geometry is not patch 16 / \
                         merge 2, so this benchmark's arithmetic does not apply",
                        out.prompt_tokens
                    )
                })?;
                self.overhead = Some(overhead);
                self.phase = Phase::Geometry;
                Ok(self.frame(
                    "calibrate",
                    vec![LogLine::info(format!(
                        "template overhead {overhead} tokens (from {name}: {} total − {want} vision)",
                        out.prompt_tokens
                    ))],
                ))
            }

            // ── Geometry ─────────────────────────────────────────────────
            Phase::Geometry => {
                let (name, bytes, w, h) = FIXTURES[self.cursor];
                let overhead = self.overhead.context("geometry ran before calibration")?;
                let want = expected_vision_tokens(w, h, 16, 2) as usize;
                let body = request::body(&handle.target().model, &[bytes], "Colour?", 8);
                let cell = match http::chat_stream(handle.target(), &body, self.timeout()).await {
                    Ok(o) => match request::vision_tokens(o.prompt_tokens, overhead) {
                        Ok(got) if got == want => GeomCell::Match {
                            fixture: name,
                            tokens: got,
                        },
                        Ok(got) => GeomCell::Mismatch {
                            fixture: name,
                            want,
                            got,
                        },
                        Err(e) => GeomCell::Error {
                            fixture: name,
                            msg: one_line(format!("{e:#}")),
                        },
                    },
                    // An image past the server's encoder capacity is a
                    // deployment setting, not a defect: UNMEASURED, not FAIL.
                    // The engine now says so by name rather than failing an
                    // H2D copy, which is what makes this distinguishable.
                    Err(e) if format!("{e:#}").contains("this encoder holds") => {
                        GeomCell::Unmeasured {
                            fixture: name,
                            why: one_line(format!("{e:#}")),
                        }
                    }
                    Err(e) => GeomCell::Error {
                        fixture: name,
                        msg: one_line(format!("{e:#}")),
                    },
                };
                let line = match &cell {
                    GeomCell::Match { tokens, .. } => {
                        LogLine::info(format!("{name}: {tokens} tokens"))
                    }
                    GeomCell::Mismatch { want, got, .. } => {
                        LogLine::warn(format!("{name}: expected {want}, got {got}"))
                    }
                    GeomCell::Unmeasured { .. } => {
                        LogLine::info(format!("{name}: over encoder capacity — unmeasured"))
                    }
                    GeomCell::Error { msg, .. } => LogLine::warn(format!("{name}: {msg}")),
                };
                self.geom.push(cell);
                self.cursor += 1;
                if self.cursor >= FIXTURES.len() {
                    self.cursor = 0;
                    self.phase = Phase::Probes;
                }
                Ok(self.frame("geometry", vec![line]))
            }

            // ── Capability probes ────────────────────────────────────────
            Phase::Probes => {
                let p = &PROBES[self.cursor];
                let cell = self.run_probe(p).await;
                let line = match &cell {
                    ProbeCell::Pass { id } => LogLine::info(format!("{id}: pass")),
                    ProbeCell::Fail { id, reply } => LogLine::warn(format!("{id}: {reply}")),
                    ProbeCell::Error { id, msg } => LogLine::warn(format!("{id}: {msg}")),
                };
                self.probes.push(cell);
                self.cursor += 1;
                if self.cursor >= PROBES.len() {
                    self.cursor = 0;
                    self.phase = Phase::Integrity;
                }
                Ok(self.frame("probes", vec![line]))
            }

            // ── Control, LAST ────────────────────────────────────────────
            // ── Integrity: the paths a single well-formed request misses ─
            //
            // Five checks, each aimed at machinery whose failure mode is a
            // fluent answer to the WRONG input rather than an error. Run one
            // per `next()` so the pane shows progress and a hang is
            // attributable.
            Phase::Integrity => {
                use crate::benchmarks::media_integrity as mi;
                let h = self.handle()?;
                let model = h.target().model.clone();
                let tmo = self.timeout();
                // The flat red/blue split, used as the SECOND image of the cache
                // probe: its dominant color cannot be confused with the gradient
                // rungs. It lives in EXIF_PAIR, not the geometry ladder, so it is
                // looked up there — `self.fixture` searches FIXTURES and would
                // silently return None, turning the probe into a permanent skip.
                let red = super::provision::EXIF_PAIR
                    .iter()
                    .find(|(n, _)| *n == "16_exif_none_224.jpg")
                    .map(|(_, b)| *b);
                let cell = match self.cursor {
                    // 1. Different requests in flight, each scored against its
                    //    own input. The identical-request sweep in the next
                    //    phase cannot see a mis-sliced per-request offset.
                    0 => {
                        let (_, a, _, _) = FIXTURES[0]; // 224 square
                        let (_, b, _, _) = FIXTURES[6]; // 1280x720 HD
                        let (_, c, _, _) = FIXTURES[1]; // 336 square
                        let q = "Reply with exactly one word: the number of white \
                                 rectangles you can see, spelled out.";
                        let subjects: Vec<mi::Subject> = vec![
                            (
                                mi::image_request(&model, "image/png", a, q, 24),
                                Box::new(|r: &str| !r.trim().is_empty()),
                                "224".to_string(),
                            ),
                            (
                                mi::image_request(&model, "image/png", b, q, 24),
                                Box::new(|r: &str| !r.trim().is_empty()),
                                "1280x720".to_string(),
                            ),
                            (
                                mi::image_request(&model, "image/png", c, q, 24),
                                Box::new(|r: &str| !r.trim().is_empty()),
                                "336".to_string(),
                            ),
                            // A TEXT-ONLY request in the same batch: it owns
                            // zero grids, which is the offset arithmetic's
                            // most awkward case.
                            (
                                serde_json::json!({
                                    "model": model, "stream": true, "temperature": 0.0,
                                    "max_tokens": 8,
                                    "chat_template_kwargs": {"enable_thinking": false},
                                    "messages": [{"role": "user",
                                        "content": "Reply with exactly: BANANA"}],
                                }),
                                Box::new(|r: &str| r.to_uppercase().contains("BANANA")),
                                "text-only".to_string(),
                            ),
                        ];
                        mi::heterogeneous_concurrency(h, subjects, tmo).await
                    }
                    // 2. Same prompt text, different image, back to back.
                    1 => {
                        let (_, first, _, _) = FIXTURES[0];
                        let Some(second) = red else {
                            return Ok(self.frame(
                                "integrity",
                                vec![LogLine::warn(
                                    "prefix-cache-isolation: fixture missing".to_string(),
                                )],
                            ));
                        };
                        let q = "What is the dominant colour in this image? One word.";
                        // The second image is a flat red/blue split; the first
                        // is the gradient ladder rung. "red" or "blue" can only
                        // come from the second.
                        mi::cache_leak(
                            h,
                            mi::image_request(&model, "image/png", first, q, 16),
                            mi::image_request(&model, "image/jpeg", second, q, 16),
                            &|r: &str| {
                                let l = r.to_lowercase();
                                l.contains("red") || l.contains("blue")
                            },
                            &|r: &str| {
                                let l = r.to_lowercase();
                                l.contains("purple") || l.contains("gradient")
                            },
                            tmo,
                        )
                        .await
                    }
                    // 3. A prompt long enough to leave the single-chunk path.
                    2 => {
                        let (_, img, _, _) = FIXTURES[0];
                        let q = "Reply with exactly one word: YES if this image contains a \
                                 white rectangle, NO otherwise.";
                        // ~6k tokens of filler, comfortably past any sane
                        // per-request chunk budget, and inert: it says nothing
                        // that could change the answer.
                        let filler = "The quick brown fox jumps over the lazy dog. ".repeat(700);
                        let long_q = format!("{filler}\n\n{q}");
                        mi::long_prompt_path(
                            h,
                            mi::image_request(&model, "image/png", img, q, 16),
                            mi::image_request(&model, "image/png", img, &long_q, 16),
                            &|r: &str| r.to_uppercase().contains("YES"),
                            tmo,
                        )
                        .await
                    }
                    // 4. Media in an earlier turn and in a tool result.
                    3 => {
                        let (_, img, _, _) = FIXTURES[0];
                        use base64::Engine;
                        let mut uri = String::from("data:image/png;base64,");
                        base64::engine::general_purpose::STANDARD.encode_string(img, &mut uri);
                        let body = serde_json::json!({
                            "model": model, "stream": true, "temperature": 0.0,
                            "max_tokens": 24,
                            "chat_template_kwargs": {"enable_thinking": false},
                            "messages": [
                                {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": uri}},
                                    {"type": "text", "text": "Here is a screenshot."}]},
                                {"role": "assistant", "content": "Understood."},
                                {"role": "user", "content":
                                    "Reply with exactly one word: YES if the image I sent \
                                     earlier contains a white rectangle, NO otherwise."},
                            ],
                        });
                        mi::media_in_history(
                            h,
                            body,
                            &|r: &str| r.to_uppercase().contains("YES"),
                            "media-in-history",
                            tmo,
                        )
                        .await
                    }
                    // 5. Streamed vs not: two different response paths.
                    4 => {
                        let (_, img, _, _) = FIXTURES[0];
                        mi::stream_parity(
                            h,
                            mi::image_request(
                                &model,
                                "image/png",
                                img,
                                "Reply with exactly one word: YES if this image contains a \
                                 white rectangle, NO otherwise.",
                                16,
                            ),
                            &|r: &str| r.to_uppercase().contains("YES"),
                            tmo,
                        )
                        .await
                    }
                    // 6. n>1, where only choice 0 carries the pixels.
                    5 => {
                        let (_, img, _, _) = FIXTURES[0];
                        mi::multi_choice(
                            h,
                            mi::image_request(
                                &model,
                                "image/png",
                                img,
                                "Reply with exactly one word: YES if this image contains a \
                                 white rectangle, NO otherwise.",
                                16,
                            ),
                            2,
                            &|r: &str| r.to_uppercase().contains("YES"),
                            tmo,
                        )
                        .await
                    }
                    // 7. The image on a TOOL RESULT — the other call site of
                    //    collect_message_images, and the case #165 exists for.
                    //    An agent sees a screenshot come back from a tool and
                    //    is asked about it afterwards.
                    6 => {
                        let (_, img, _, _) = FIXTURES[0];
                        use base64::Engine;
                        let mut uri = String::from("data:image/png;base64,");
                        base64::engine::general_purpose::STANDARD.encode_string(img, &mut uri);
                        let body = serde_json::json!({
                            "model": model, "stream": true, "temperature": 0.0,
                            "max_tokens": 16,
                            "chat_template_kwargs": {"enable_thinking": false},
                            "messages": [
                                {"role": "user", "content": "Take a screenshot."},
                                {"role": "assistant", "content": "",
                                 "tool_calls": [{"id": "c1", "type": "function",
                                   "function": {"name": "screenshot", "arguments": "{}"}}]},
                                {"role": "tool", "tool_call_id": "c1", "content": [
                                    {"type": "image_url", "image_url": {"url": uri}}]},
                                {"role": "user", "content":
                                    "Reply with exactly one word: YES if the screenshot \
                                     contains a white rectangle, NO otherwise."},
                            ],
                        });
                        mi::media_in_history(
                            h,
                            body,
                            &|r: &str| r.to_uppercase().contains("YES"),
                            "image-on-tool-result",
                            tmo,
                        )
                        .await
                    }
                    // 8. The Responses API — a separate parse path that
                    //    nothing else in either benchmark drives. Compared
                    //    against ITSELF on two images, not against
                    //    chat-completions: the two surfaces render different
                    //    template branches, so a cross-surface token
                    //    comparison cannot tell an envelope difference from a
                    //    vision one.
                    7 => {
                        let (_, small, sw, sh) = FIXTURES[0]; // 224 -> 49
                        let (_, large, lw, lh) = FIXTURES[1]; // 336 -> 121
                        let delta = (expected_vision_tokens(lw, lh, 16, 2)
                            - expected_vision_tokens(sw, sh, 16, 2))
                            as usize;
                        let q = "Reply with exactly one word: YES if this image contains a \
                                 white rectangle, NO otherwise.";
                        mi::responses_parity(
                            h,
                            mi::responses_image_request(&model, "image/png", small, q),
                            mi::responses_image_request(&model, "image/png", large, q),
                            delta,
                            &|r: &str| r.to_uppercase().contains("YES"),
                            tmo,
                        )
                        .await
                    }
                    // 9. Thinking ON — the configuration these checkpoints
                    //    ship in, and the one every other leg turns off.
                    8 => {
                        let (_, small, sw, sh) = FIXTURES[0]; // 224 -> 49
                        let (_, large, lw, lh) = FIXTURES[1]; // 336 -> 121
                        let delta = (expected_vision_tokens(lw, lh, 16, 2)
                            - expected_vision_tokens(sw, sh, 16, 2))
                            as usize;
                        let q = "Reply with exactly one word: YES if this image contains a \
                                 white rectangle, NO otherwise.";
                        mi::thinking_parity(
                            h,
                            mi::thinking_image_request(&model, "image/png", small, q),
                            mi::thinking_image_request(&model, "image/png", large, q),
                            delta,
                            &|r: &str| r.to_uppercase().contains("YES"),
                            tmo,
                        )
                        .await
                    }
                    // 10. EXIF orientation, PINNED rather than judged.
                    _ => {
                        let pair: Vec<(&str, &[u8])> = super::provision::EXIF_PAIR.to_vec();
                        let q = "The image is split into two halves of solid colour. Is the \
                                 RED half on the top, bottom, left, or right? One word.";
                        let mut answers = Vec::new();
                        for (name, bytes) in &pair {
                            let body = mi::image_request(&model, "image/jpeg", bytes, q, 12);
                            match http::chat_stream(h.target(), &body, tmo).await {
                                Ok(o) => answers.push((*name, o.text.trim().to_lowercase())),
                                Err(e) => answers.push((
                                    *name,
                                    format!("error: {}", one_line(format!("{e:#}"))),
                                )),
                            }
                        }
                        // Orientation = 6 means "rotate 90 CW to display",
                        // which carries the stored TOP edge to the RIGHT. So
                        // the tagged image must read RIGHT and the untagged
                        // one TOP — they must DIFFER, and differ in that
                        // specific way. Requiring them merely to differ would
                        // pass on any random rotation.
                        let tagged = answers.first().map(|a| a.1.clone()).unwrap_or_default();
                        let untagged = answers.get(1).map(|a| a.1.clone()).unwrap_or_default();
                        if tagged.contains("right") && untagged.contains("top") {
                            mi::Cell::Pass {
                                id: "exif-orientation",
                                detail: format!(
                                    "tagged -> \"{tagged}\", untagged -> \"{untagged}\": EXIF \
                                     orientation is APPLIED, so a rotated photo reaches the \
                                     model the way its owner sees it"
                                ),
                            }
                        } else if tagged == untagged {
                            mi::Cell::Fail {
                                id: "exif-orientation",
                                detail: format!(
                                    "both answered \"{tagged}\" — the EXIF tag is being IGNORED \
                                     again. Every rotated phone photo is reaching the model a \
                                     quarter turn from how the user saw it, and nothing errors"
                                ),
                            }
                        } else {
                            mi::Cell::Fail {
                                id: "exif-orientation",
                                detail: format!(
                                    "unexpected orientation: tagged -> \"{tagged}\", untagged -> \
                                     \"{untagged}\". Orientation=6 should put the red half on \
                                     the RIGHT and the untagged one on TOP"
                                ),
                            }
                        }
                    }
                };
                let line = if cell.passed() {
                    LogLine::info(cell.line())
                } else {
                    LogLine::warn(cell.line())
                };
                self.integrity.push(cell);
                self.cursor += 1;
                if self.cursor >= 10 {
                    self.cursor = 0;
                    self.phase = Phase::Concurrency;
                }
                Ok(self.frame("integrity", vec![line]))
            }

            // ── Concurrency: C = 1, 2, 4 of the same image request ───────
            //
            // Correctness under concurrency, not throughput. Several requests
            // share one packed ViT output buffer and one grid vector, indexed
            // by per-request base offsets, and a mistake there hands one
            // request another's embeddings — a fluent answer about the wrong
            // picture, with nothing logged. Identical requests must also agree
            // on prompt_tokens; two different counts mean the shared buffers
            // were sliced wrongly. Wall time is recorded, not asserted:
            // vision prefill serializes today.
            Phase::Concurrency => {
                use crate::benchmarks::video::concurrency::{LEVELS, run_level};
                let level = LEVELS[self.cursor];
                let probe = concurrency_probe();
                let png = self.fixture(probe.images[0])?;
                // `self.max_tokens`, not a literal. The 16 here was calibrated
                // for the stimulus this leg used to send — a 224px square and
                // "Reply with the single word OK", where any non-empty reply
                // passed. This leg now sends the 1280x720 fixture and demands
                // the label "1280" appear, and a model asked to read a label
                // exactly spends its first tokens saying so. At temperature 0
                // the reply is cut off by `finish_reason=length` before the
                // number arrives, every time, on every box — which reads as a
                // vision failure and is not one. The capability phase already
                // uses `self.max_tokens` for the same probe and passes; this is
                // now the same number in both places.
                let body = request::body(
                    &self.handle()?.target().model,
                    &[png],
                    probe.prompt,
                    self.max_tokens,
                );
                let is_correct =
                    |reply: &str| reply_matches(reply, probe.want_all, probe.want_none);
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
                        "C={level}: {}/{} returned, {geometry}, {} ms",
                        r.returned, r.conc, r.wall_ms,
                    ))
                } else {
                    LogLine::warn(format!(
                        "C={level}: {}/{} returned, {geometry}, {} ms{}",
                        r.returned,
                        r.conc,
                        r.wall_ms,
                        if r.errors.is_empty() {
                            String::new()
                        } else {
                            format!(" — {}", r.errors.join("; "))
                        }
                    ))
                };
                self.conc_results.push(r);
                self.cursor += 1;
                if self.cursor >= LEVELS.len() {
                    self.cursor = 0;
                    self.phase = Phase::Control;
                }
                Ok(self.frame("concurrency", vec![line]))
            }

            Phase::Control => {
                let cell = self.run_probe(&CONTROL).await;
                self.control_held = matches!(cell, ProbeCell::Pass { .. });
                let line = if self.control_held {
                    LogLine::info("control: no image, no answer — capability results stand")
                } else {
                    LogLine::warn(
                        "control: answered as though it saw an image — capability results are \
                         VACUOUS, the server may not be splicing vision embeddings at all",
                    )
                };
                self.phase = Phase::Score;
                Ok(self.frame("control", vec![line]))
            }

            // ── Score ────────────────────────────────────────────────────
            Phase::Score => {
                self.phase = Phase::Done;
                // An integrity failure must FAIL THE RUN. These legs were
                // added because they catch wrong answers that geometry and
                // capability cannot see; leaving them as metrics only would
                // mean the benchmark reports PASS while telling you, in its
                // own log, that a request got another request's picture.
                let integ_failed = self.integrity.iter().any(|c| c.measured() && !c.passed());
                let concurrency_clean =
                    crate::benchmarks::video::concurrency::sweep_ok(&self.conc_results);
                let v = with_runtime_checks(
                    verdict(&self.geom, &self.probes, self.control_held),
                    integ_failed,
                    concurrency_clean,
                );
                let asserted = asserted_cells(&self.geom);
                let passed = self
                    .probes
                    .iter()
                    .filter(|c| matches!(c, ProbeCell::Pass { .. }))
                    .count();

                let mut r = BenchmarkResult::running("score", self.elapsed());
                r.status = if v == VisionVerdict::Pass {
                    RunStatus::Completed
                } else {
                    RunStatus::Failed
                };
                r.summary = vec![
                    Stat::new("verdict", v.to_string(), ""),
                    Stat::new(
                        "geometry",
                        format!("{asserted}/{}", self.geom.len()),
                        "asserted",
                    ),
                    Stat::new(
                        "probes",
                        format!("{passed}/{}", self.probes.len()),
                        "passed",
                    ),
                ];
                r.metrics
                    .insert("geometry_asserted".into(), asserted as f64);
                // ASSERTED counts Match AND Mismatch — it answers "did the rung
                // get measured", which is the guard against an encoder-capacity
                // regression turning every cell UNMEASURED and reading as a
                // pass. It is NOT a correctness count: a run where every cell
                // reported the wrong number still scores full marks on it. The
                // gate needs a threshold that moves when the ANSWER is wrong,
                // so matched is emitted separately and is the one BENCH.toml
                // bounds. Both are kept: they fail on different defects.
                let matched = self
                    .geom
                    .iter()
                    .filter(|c| matches!(c, GeomCell::Match { .. }))
                    .count();
                r.metrics.insert("geometry_matched".into(), matched as f64);
                r.metrics
                    .insert("geometry_cells".into(), self.geom.len() as f64);
                r.metrics.insert("probes_passed".into(), passed as f64);
                r.metrics
                    .insert("probes_total".into(), self.probes.len() as f64);
                r.metrics
                    .insert("control_held".into(), self.control_held as u8 as f64);
                // Recorded so a run shows how vision behaves under a little
                // load; NOT thresholded — see the Concurrency phase.
                let baseline_prompt_tokens = self
                    .conc_results
                    .first()
                    .and_then(|baseline| baseline.prompt_tokens);
                let conc_clean = self
                    .conc_results
                    .iter()
                    .filter(|r| {
                        baseline_prompt_tokens.is_some_and(|baseline| r.ok_against(baseline))
                    })
                    .count();
                r.metrics
                    .insert("concurrency_levels_clean".into(), conc_clean as f64);
                let integ_passed = self.integrity.iter().filter(|c| c.passed()).count();
                let integ_measured = self.integrity.iter().filter(|c| c.measured()).count();
                r.metrics
                    .insert("integrity_passed".into(), integ_passed as f64);
                r.metrics
                    .insert("integrity_measured".into(), integ_measured as f64);
                for lr in &self.conc_results {
                    r.metrics
                        .insert(format!("conc_{}_wall_ms", lr.conc), lr.wall_ms as f64);
                }
                r.verdict = Some(match v {
                    VisionVerdict::Pass => RunVerdict::pass(format!(
                        "{asserted} geometry cells matched, {passed}/{} probes, control held",
                        self.probes.len()
                    )),
                    VisionVerdict::Fail => RunVerdict::fail(format!(
                        "{}/{} geometry cells asserted, {passed}/{} probes passed",
                        asserted,
                        self.geom.len(),
                        self.probes.len()
                    )),
                    VisionVerdict::Vacuous => RunVerdict::fail(
                        "VACUOUS: the no-image control answered as though it saw one, so the \
                         capability probes are not evidence"
                            .to_string(),
                    ),
                });
                r.log = vec![match v {
                    VisionVerdict::Pass => LogLine::info(format!(
                        "PASS — {asserted} geometry cells asserted, {passed}/{} probes",
                        self.probes.len()
                    )),
                    VisionVerdict::Fail => LogLine::warn("FAIL — see the cells above"),
                    VisionVerdict::Vacuous => LogLine::warn(
                        "VACUOUS — the no-image control answered, so the capability probes are \
                         not evidence. Geometry results above are still valid.",
                    ),
                }];
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
