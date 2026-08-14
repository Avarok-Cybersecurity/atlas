// SPDX-License-Identifier: AGPL-3.0-only

//! The state machine: one leg per `next()`.
//!
//! Order matters. Calibration first (it measures the template overhead every
//! later assertion subtracts), then geometry, then the capability probes, then
//! the control LAST — so a vacuous run is discovered after the evidence it
//! invalidates has been collected and can be shown alongside the verdict.

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
use super::probes::{CONTROL, PROBES, Probe};
use super::provision::{FIXTURES, provision};
use super::request;
use super::score::{
    GeomCell, ProbeCell, Verdict as VisionVerdict, asserted_cells, reply_matches, verdict,
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
    ctor: || Box::new(VisionFidelity::default()),
};

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    #[default]
    Calibrate,
    Geometry,
    Probes,
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
                    self.phase = Phase::Control;
                }
                Ok(self.frame("probes", vec![line]))
            }

            // ── Control, LAST ────────────────────────────────────────────
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
                let v = verdict(&self.geom, &self.probes, self.control_held);
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
