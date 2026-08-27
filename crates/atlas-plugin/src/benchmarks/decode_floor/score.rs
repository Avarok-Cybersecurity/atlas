// SPDX-License-Identifier: AGPL-3.0-only

//! The pure half of the decode-floor gate: the pins (constants), the reduced
//! per-run observation, and the evaluation/verdict functions. Exact piecewise
//! copy out of the driver (file-size cap) — no endpoint, no I/O, so every
//! verdict path is provable in unit tests.

use crate::benchmarks::stats;
use crate::http;
use crate::result::Verdict;

/// Timed runs. PINNED — the median-of-3 is the metric's definition, and a
/// different run count is a different benchmark.
pub(crate) const RUNS: usize = 3;
/// Output budget. PINNED at the measured basis (MinHeap 1500).
pub(crate) const MAX_TOKENS: usize = 1500;
/// Vacuity floor on every run's `completion_tokens`.
///
/// ★ 750, not 1200 — recalibrated to the gate's own instrument at promotion
/// (2026-08-15). The 12-run calibration (temp 0 / seed 0, this fixture)
/// completes at a DETERMINISTIC 915 tokens of the 1500 budget: the model's
/// natural stop for the MinHeap task, identical every run. The original 1200
/// floor predated that measurement and would have made the calibrated
/// instrument INCONCLUSIVE by construction — a gate that fails its own
/// reference behaviour deterministically gates nothing. 750 keeps the pin's
/// purpose (a 49-token burst can never read as a decode measurement) while
/// sitting safely under the instrument's natural 915 with margin for small
/// completion-length drift.
pub(crate) const MIN_OUTPUT_TOKENS: usize = 750;
/// Vacuity floor on the derived tokens-per-decode-step.
pub(crate) const MIN_ACCEPT_LEN: f64 = 1.5;

/// The committed code prompt. MinHeap-class on purpose: a structured,
/// code-shaped generation whose accept behaviour is the documented middle of
/// the road (~2–2.5 per verify), unlike counting prompts which accept near
/// ceiling and flatter the rate. Owned here — benchmark drivers must not
/// import each other, so nothing is borrowed from quick-speed's fixtures.
pub(crate) const MINHEAP_PROMPT: &str = "Implement a complete, production-quality MinHeap class in Python. Include the methods \
     insert, extract_min, peek, heapify (bottom-up from an arbitrary list), decrease_key, \
     delete_at_index, merge (with another MinHeap), __len__ and __iter__. Every method needs a \
     full docstring with time-complexity analysis. Then write a comprehensive pytest test \
     suite covering the empty heap, a single element, duplicate keys, and long interleaved \
     insert/extract sequences. Finish with a line-by-line explanation of the sift_up and \
     sift_down invariants. Be exhaustive and do not stop early.";

/// One timed run, reduced to what the pins and the metric need.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct RunObs {
    pub completion_tokens: usize,
    pub server_tps: Option<f64>,
    /// `None` = the server reported no details object (no instrumentation);
    /// `Some(0)` = instrumented but nothing accepted. The pins treat the two
    /// differently only in the message — both are INCONCLUSIVE.
    pub accepted_prediction_tokens: Option<usize>,
    pub e2e_ms: f64,
}

impl RunObs {
    pub(crate) fn from_outcome(o: &http::ChatOutcome) -> Self {
        Self {
            completion_tokens: o.completion_tokens,
            server_tps: o.server_tps,
            accepted_prediction_tokens: o.accepted_prediction_tokens,
            e2e_ms: o.e2e_ms,
        }
    }

    /// Emitted tokens per decode step, `1 + accepted/steps` — the honest
    /// accept-depth lower bound derivable from the wire (see module docs).
    /// `None` when it cannot be derived (no accept field, or a corrupt
    /// `accepted >= completion` which would divide by zero or go negative).
    pub(crate) fn accept_len(&self) -> Option<f64> {
        let accepted = self.accepted_prediction_tokens?;
        (accepted < self.completion_tokens && self.completion_tokens > 0)
            .then(|| self.completion_tokens as f64 / (self.completion_tokens - accepted) as f64)
    }
}

/// What three runs add up to. Pure — the whole verdict is unit-testable
/// without an endpoint.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Evaluation {
    /// A vacuity pin failed; the message names which one and why. Never PASS.
    Inconclusive(String),
    Measured {
        /// MEDIAN server decode tok/s across the runs — THE metric.
        median_decode_tok_s: f64,
        /// Minimum `completion_tokens` across runs, so the BENCH.toml
        /// `output_tokens >= 750` bound means "every run", not "on average".
        min_output_tokens: usize,
        /// Mean of the per-run derived accept lengths.
        accept_len_mean: f64,
    },
}

/// The run verdict for an evaluation. Pure, like `evaluate`.
///
/// Vacuity stays INCONCLUSIVE (a failing verdict) regardless of `min_tok_s` —
/// a run that measured nothing must never PASS. A Measured run self-verdicts
/// against `min_tok_s` when it is set (> 0, gate-filled from BENCH.toml), and
/// stays info when it is not: a standalone run has no committed floor to be
/// judged against.
///
/// ★ Deliberately STRICTER than gate scoring: this compares the raw
/// median >= min, while `gate::scoring` allows value + noise >= min. A
/// sub-noise dip fails the run verdict even though scoring would have passed
/// it — safe conservatism (it can only re-run a healthy build, never
/// green-light a regression).
pub(crate) fn verdict_for(eval: &Evaluation, min_tok_s: f64) -> Verdict {
    match eval {
        Evaluation::Inconclusive(why) => Verdict::fail(format!("INCONCLUSIVE: {why}")),
        Evaluation::Measured {
            median_decode_tok_s,
            accept_len_mean,
            ..
        } => {
            let basis = format!(
                "median decode {median_decode_tok_s:.1} tok/s over {RUNS} pinned runs \
                 (accept_len_mean {accept_len_mean:.2})"
            );
            if min_tok_s <= 0.0 {
                Verdict::info(format!(
                    "{basis} — judged against the BENCH.toml floor under --pull-request-gate"
                ))
            } else if *median_decode_tok_s >= min_tok_s {
                Verdict::pass(format!("{basis} — clears the {min_tok_s:.1} tok/s floor"))
            } else {
                Verdict::fail(format!(
                    "BELOW THE DECODE FLOOR — {basis} vs the {min_tok_s:.1} tok/s floor"
                ))
            }
        }
    }
}

pub(crate) fn evaluate(samples: &[RunObs]) -> Evaluation {
    if samples.len() != RUNS {
        return Evaluation::Inconclusive(format!(
            "{} run(s) completed, the pinned count is {RUNS}",
            samples.len()
        ));
    }
    for (i, s) in samples.iter().enumerate() {
        if s.completion_tokens < MIN_OUTPUT_TOKENS {
            return Evaluation::Inconclusive(format!(
                "run {} emitted {} tokens, below the {MIN_OUTPUT_TOKENS}-token vacuity floor \
                 (of the {MAX_TOKENS} budget) — too short a decode to measure a floor on",
                i + 1,
                s.completion_tokens
            ));
        }
        match s.server_tps {
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported no server decode rate (usage.\"response_token/s\") — without \
                     the server's own clock there is no defensible per-token number",
                    i + 1
                ));
            }
            Some(rate) if !rate.is_finite() || rate <= 0.0 => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported server decode rate {rate}, which is not a finite positive \
                     per-token measurement",
                    i + 1
                ));
            }
            Some(_) => {}
        }
        match s.accepted_prediction_tokens {
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {} reported no usage.completion_tokens_details.\
                     accepted_prediction_tokens — this gate depends on the accept-stats \
                     instrumentation (the commit wiring real MTP accept counts into usage); \
                     serve a binary that has it",
                    i + 1
                ));
            }
            Some(0) => {
                return Evaluation::Inconclusive(format!(
                    "run {} accepted 0 draft tokens — either the serve is not speculating or \
                     the accept-stats instrumentation is not live; a serial-floor number must \
                     not be recorded as the decode floor",
                    i + 1
                ));
            }
            Some(_) => {}
        }
    }
    let mut accept_lens = Vec::with_capacity(samples.len());
    for (i, s) in samples.iter().enumerate() {
        match s.accept_len() {
            Some(l) => accept_lens.push(l),
            None => {
                return Evaluation::Inconclusive(format!(
                    "run {}: accepted ({}) >= completion_tokens ({}) — corrupt accounting, \
                     nothing derivable",
                    i + 1,
                    s.accepted_prediction_tokens.unwrap_or(0),
                    s.completion_tokens
                ));
            }
        }
    }
    let accept_len_mean = accept_lens.iter().sum::<f64>() / accept_lens.len() as f64;
    if accept_len_mean < MIN_ACCEPT_LEN {
        return Evaluation::Inconclusive(format!(
            "accept_len_mean {accept_len_mean:.2} < {MIN_ACCEPT_LEN} — speculation is not \
             engaged at gate depth, so this run measures the serial floor, not the engine"
        ));
    }
    let tps: Vec<f64> = samples.iter().filter_map(|s| s.server_tps).collect();
    // stats::median, NOT stats::percentile(_, 50): the nearest-rank p50 of
    // three samples is the maximum, and the floor must not ride the best run.
    let median = stats::median(&tps).unwrap_or(0.0);
    Evaluation::Measured {
        median_decode_tok_s: median,
        min_output_tokens: samples
            .iter()
            .map(|s| s.completion_tokens)
            .min()
            .unwrap_or(0),
        accept_len_mean,
    }
}
