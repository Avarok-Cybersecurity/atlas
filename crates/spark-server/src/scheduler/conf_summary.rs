// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the per-request logprob confidence summary (master plan
//! L05, "B1") and the three `/metrics` series it feeds.
//!
//! After the first K generated tokens of a request (default 32,
//! `AVAROK_CONF_SUMMARY_K`) the summary reports
//!
//! * `mean_top1`: the mean probability of the model's top-1 token,
//! * `min_margin`: the smallest gap `p(top-1) - p(top-2)` (probability space,
//!   0 to 1), and
//! * `low_frac`: the fraction of tokens whose top-1 probability was below 0.5.
//!
//! All of it is pure arithmetic over the log-softmax top-k lists the scheduler
//! already produces for a request that asked for logprobs
//! (`spark_model::traits::logprob_of`); nothing here touches sampling.
//!
//! # How the numbers reach the summary
//!
//! The HTTP layer decides per request (see [`plan`]). When the summary is
//! wanted and the client did not ask for logprobs, it hands the scheduler
//! `top_logprobs = Some(`[`INTERNAL_TOP_LOGPROBS`]`)`: a value outside the
//! client range (0 to 20), meaning "extract top-k only for the summary". The
//! scheduler gives the extraction back after K logprobs
//! ([`release_internal_logprobs`], one hook line in each of the two token
//! funnels), so the host-logits path the extraction forces lasts K tokens, not
//! the whole generation. A client that did ask for logprobs keeps its own
//! extraction for the whole generation and the summary reads the first K.
//!
//! With the `conf_summary` row off and no `atlas_confidence` on the request,
//! none of this runs: [`plan`] returns the client's `top_logprobs` untouched
//! and the only per-token cost is one `Option<u8>` compare in
//! [`release_internal_logprobs`].
//!
//! Caveat for anyone using this as an escalation signal: confidence alone is a
//! weak trigger (B0's data decides); this module only measures it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::TokenLogprobs;

/// Environment variable holding K, the number of leading generated tokens the
/// summary covers.
pub(crate) const ENV_K: &str = "AVAROK_CONF_SUMMARY_K";
/// K when [`ENV_K`] is unset.
pub(crate) const DEFAULT_K: u32 = 32;
/// Largest accepted K (a summary over more tokens than this is not a "first
/// tokens" signal any more, and the scheduler would keep the host-logits path
/// that long).
pub(crate) const MAX_K: u32 = 4096;
/// A token counts toward `low_frac` when its top-1 probability is below this.
pub(crate) const LOW_TOP1: f32 = 0.5;
/// The `top_logprobs` value that means "extraction for the summary only".
/// Client values are clamped to 20 at the API edge
/// (`openai::to_ir::resolve_top_logprobs`), so 21 can never be a client's.
pub(crate) const INTERNAL_TOP_LOGPROBS: u8 = 21;

/// Parse [`ENV_K`]: `None` is the default, anything outside `1..=MAX_K` is an
/// error naming the variable.
pub(crate) fn parse_k(raw: Option<&str>) -> Result<u32, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_K);
    };
    let v: u32 = raw
        .trim()
        .parse()
        .map_err(|_| format!("{ENV_K}={raw:?} is not an integer in 1..={MAX_K}"))?;
    if (1..=MAX_K).contains(&v) {
        Ok(v)
    } else {
        Err(format!("{ENV_K}={v} is outside 1..={MAX_K}"))
    }
}

/// Startup check for [`ENV_K`] (called from `port_apply::validate`, so a typo
/// is a startup error and never a silent fallback).
pub(crate) fn validate_env() -> anyhow::Result<()> {
    parse_k(std::env::var(ENV_K).ok().as_deref()).map_err(anyhow::Error::msg)?;
    Ok(())
}

/// K for this process: [`ENV_K`] read once. An invalid value was already
/// rejected by [`validate_env`]; the default here only covers a path that
/// skipped validation (a unit test).
pub(crate) fn conf_k() -> u32 {
    static K: OnceLock<u32> = OnceLock::new();
    *K.get_or_init(|| parse_k(std::env::var(ENV_K).ok().as_deref()).unwrap_or(DEFAULT_K))
}

/// The `conf_summary` row (`AVAROK_CONF_SUMMARY`): collect the summary for
/// every request and feed the `atlas_conf_*` series.
pub(crate) fn row_on() -> bool {
    spark_model::layers::ops::port_rows::conf_summary::row()
}

/// What the HTTP layer does for one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConfPlan {
    /// Collect the summary for this request (row on, or the request asked).
    pub collect: bool,
    /// Put the summary in `usage.atlas_confidence` (the request asked).
    pub report: bool,
    /// The `top_logprobs` to hand the scheduler.
    pub sched_top_logprobs: Option<u8>,
    /// What the client asked for; its logprobs are trimmed to this.
    pub client_top_logprobs: Option<u8>,
}

/// Decide the plan. `row_on` is the `conf_summary` row, `asked` the request's
/// `atlas_confidence`, `client_top_logprobs` the client's own logprob request
/// (already clamped to 0 to 20).
///
/// Nothing wanted: the scheduler gets the client's value unchanged. Wanted:
/// the scheduler gets at least the top-2 list the margin needs, and when the
/// client asked for none it gets [`INTERNAL_TOP_LOGPROBS`] so the extraction
/// is released after K tokens.
pub(crate) fn plan(row_on: bool, asked: bool, client_top_logprobs: Option<u8>) -> ConfPlan {
    let collect = row_on || asked;
    let sched_top_logprobs = if !collect {
        client_top_logprobs
    } else {
        match client_top_logprobs {
            None => Some(INTERNAL_TOP_LOGPROBS),
            Some(k) => Some(k.max(2)),
        }
    };
    ConfPlan {
        collect,
        report: asked,
        sched_top_logprobs,
        client_top_logprobs,
    }
}

/// Give the summary-only logprob extraction back once `logprobs_seen`
/// (`ActiveSeq::logprobs_data.len()`) reaches K. A client's own
/// `top_logprobs` is never touched.
#[inline]
pub(crate) fn release_internal_logprobs(top_logprobs: &mut Option<u8>, logprobs_seen: usize) {
    if *top_logprobs == Some(INTERNAL_TOP_LOGPROBS) && logprobs_seen >= conf_k() as usize {
        *top_logprobs = None;
    }
}

/// The finished summary of one request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfSummary {
    /// Tokens the summary covers (at most K; fewer for a short answer).
    pub tokens: u32,
    /// Mean top-1 probability.
    pub mean_top1: f32,
    /// Smallest `p(top-1) - p(top-2)` seen, in `[0, 1]`.
    pub min_margin: f32,
    /// Fraction of tokens with top-1 probability below [`LOW_TOP1`].
    pub low_frac: f32,
}

/// Running accumulator for one request; stops counting at its limit.
#[derive(Debug, Clone, Copy)]
pub struct ConfAcc {
    limit: u32,
    n: u32,
    sum_top1: f64,
    min_margin: f32,
    low: u32,
}

impl ConfAcc {
    /// An accumulator that counts at most `limit` tokens.
    pub fn new(limit: u32) -> Self {
        Self {
            limit,
            n: 0,
            sum_top1: 0.0,
            min_margin: f32::INFINITY,
            low: 0,
        }
    }

    /// True once `limit` tokens are counted.
    pub fn is_full(&self) -> bool {
        self.n >= self.limit
    }

    /// Count one position from its top-k list (`(token, logprob)`, sorted
    /// descending, as `logprob_of` returns it). Returns whether it counted: a
    /// full accumulator, an empty list and a NaN logprob do not.
    pub fn observe_top(&mut self, top: &[(u32, f32)]) -> bool {
        if self.is_full() {
            return false;
        }
        let Some(&(_, lp1)) = top.first() else {
            return false;
        };
        if lp1.is_nan() {
            return false;
        }
        let p1 = f64::from(lp1).exp();
        let p2 = top.get(1).map_or(0.0, |&(_, lp)| f64::from(lp).exp());
        let margin = (p1 - p2).max(0.0) as f32;
        self.sum_top1 += p1;
        if (p1 as f32) < LOW_TOP1 {
            self.low += 1;
        }
        self.min_margin = self.min_margin.min(margin);
        self.n += 1;
        true
    }

    /// [`Self::observe_top`] over a scheduler logprob record.
    pub fn observe_lp(&mut self, lp: &TokenLogprobs) -> bool {
        self.observe_top(&lp.top)
    }

    /// Count one position from raw logits (CPU oracle path: the same
    /// `logprob_of` the scheduler uses, top-2).
    pub fn observe_logits(&mut self, logits: &[f32]) -> bool {
        let (_, top) = spark_model::traits::logprob_of(logits, 0, 2);
        self.observe_top(&top)
    }

    /// The summary so far; `None` before the first counted token.
    pub fn summary(&self) -> Option<ConfSummary> {
        if self.n == 0 {
            return None;
        }
        let n = f64::from(self.n);
        Some(ConfSummary {
            tokens: self.n,
            mean_top1: (self.sum_top1 / n) as f32,
            min_margin: self.min_margin,
            low_frac: (f64::from(self.low) / n) as f32,
        })
    }
}

/// The summary over the first `limit` logprob records of a finished blocking
/// response.
pub(crate) fn summarize(lps: &[TokenLogprobs], limit: u32) -> Option<ConfSummary> {
    let mut acc = ConfAcc::new(limit);
    for lp in lps {
        if !acc.observe_lp(lp) && acc.is_full() {
            break;
        }
    }
    acc.summary()
}

/// Feed one streamed token event. A record whose token is not the streamed
/// token is the scheduler repeating its last record for a token that had none,
/// so it is skipped.
pub(crate) fn observe_stream(acc: Option<&mut ConfAcc>, tok: u32, lp: &TokenLogprobs) {
    if let Some(a) = acc
        && lp.token_id == tok
    {
        a.observe_lp(lp);
    }
}

static SUMMARIES: AtomicU64 = AtomicU64::new(0);
static TOP1_MICROS: AtomicU64 = AtomicU64::new(0);
static MARGIN_MICROS: AtomicU64 = AtomicU64::new(0);

fn micros(x: f32) -> u64 {
    (f64::from(x.max(0.0)) * 1_000_000.0).round() as u64
}

/// Observe one finished summary into the `atlas_conf_*` series.
pub(crate) fn record(s: &ConfSummary) {
    SUMMARIES.fetch_add(1, Ordering::Relaxed);
    TOP1_MICROS.fetch_add(micros(s.mean_top1), Ordering::Relaxed);
    MARGIN_MICROS.fetch_add(micros(s.min_margin), Ordering::Relaxed);
}

/// Append the `atlas_conf_*` series (zeros while nothing is recorded, so the
/// names exist on every target).
pub(crate) fn render_prometheus(out: &mut String) {
    use crate::api::port_metrics::{counter, hist_sum_count};
    let n = SUMMARIES.load(Ordering::Relaxed);
    let sum = |m: &AtomicU64| m.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    hist_sum_count(
        out,
        "atlas_conf_mean_top1",
        "Mean top-1 token probability over the first K generated tokens of a request.",
        sum(&TOP1_MICROS),
        n,
    );
    hist_sum_count(
        out,
        "atlas_conf_min_margin",
        "Smallest top-1 minus top-2 probability over the first K generated tokens of a request.",
        sum(&MARGIN_MICROS),
        n,
    );
    counter(
        out,
        "atlas_conf_summaries_total",
        "Confidence summaries recorded.",
        n,
    );
}

#[cfg(test)]
#[path = "conf_summary_tests.rs"]
mod tests;
