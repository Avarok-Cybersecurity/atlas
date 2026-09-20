// SPDX-License-Identifier: AGPL-3.0-only

//! Intra-response arrival-gap accumulator — the jitter instrument.
//!
//! Two different jitters exist and only one diagnoses a stall. The spread
//! of ITL ACROSS requests is already derivable from a cell's p50/p99. The
//! variation of gaps WITHIN one response is what shows a mid-generation
//! stall — a thermal park, a scheduler hiccup — and that is what this
//! captures.
//!
//! # What a gap is
//!
//! An ARRIVAL is one socket read that carried at least one token delta; a
//! gap is the time between consecutive arrivals of one response. Not
//! token-to-token: under speculative decoding a verify step lands several
//! tokens in one flush, so token-to-token gaps are bimodal (≈0 within a
//! burst, a step time between bursts) by construction, and their spread
//! would measure the drafter, not the engine's smoothness. Per-arrival
//! gaps are the engine's step cadence as seen by the client, which is the
//! quantity a stall perturbs.
//!
//! # Design: online, bounded
//!
//! Count, mean and variance are kept online (Welford) and never retain a
//! sample; the maximum is a running max. Percentiles need the samples, so
//! gaps (as `f32`, 4 bytes) are retained up to [`RETAIN_CAP`] per response
//! and then dropped while the online statistics keep counting; the stats
//! report how many gaps the percentiles cover so a truncated response is
//! auditable. Worst case at C=128 with the 8192-token output ceiling:
//! 128 × 8192 × 4 B = 4 MiB, allocated incrementally; at the gate's
//! OSL 320 it is 160 KiB. Per arrival the work is one `Instant::now()`
//! (~32 ns, taken once per socket read — the client used to take one per
//! token-carrying chunk, so per token this is a net saving), four flops
//! and one `Vec::push` — the push measured at 8.3 ns/gap (standalone
//! `rustc -O` micro-benchmark, 5 M gaps, 2026-09-20) against the ~µs
//! `serde_json` parse of the chunk it rides on.
//!
//! # What is emitted
//!
//! Primitives — the distribution summary — so it can be re-aggregated: a
//! cell pools its requests' gaps ([`GapSample::merge`]) and reports the
//! pooled distribution. The dimensionless headline, **`stability`** =
//! `(p99 − p50) / p50` ([`GapStats::stability`]), and the coefficient of
//! variation are DERIVED from the stored primitives.
//!
//! ★ `stability` is a DISPERSION measure: **lower is better** — a smaller
//! value means smoother token delivery, 0 means every arrival gap was the
//! median. The owner chose the name and the direction (2026-09-20); the
//! stored value is the honest primitive, never an inverted 0–100 score,
//! because a score baked into a record cannot be un-inverted once other
//! records sit beside it. A presentation layer may relabel it.

use std::collections::BTreeMap;

use crate::benchmarks::stats;

/// Gaps retained per response for the percentiles. The largest output
/// budget any driver parameter admits (concurrency `osl` max 8192), so a
/// gate response is never truncated; longer responses keep counting online
/// and report the truncation through [`GapStats::retained`].
pub const RETAIN_CAP: usize = 8192;

/// Online accumulator of one response's arrival gaps (or a pool of them).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GapSample {
    count: u64,
    mean: f64,
    m2: f64,
    max: f64,
    retained: Vec<f32>,
}

impl GapSample {
    /// Record one gap, in milliseconds.
    pub fn push(&mut self, gap_ms: f64) {
        self.count += 1;
        let delta = gap_ms - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (gap_ms - self.mean);
        if gap_ms > self.max {
            self.max = gap_ms;
        }
        if self.retained.len() < RETAIN_CAP {
            self.retained.push(gap_ms as f32);
        }
    }

    /// Pool another sample into this one (Chan's parallel update for the
    /// moments; the retained gaps concatenate, so a pool of `k` responses
    /// holds at most `k × RETAIN_CAP` gaps).
    pub fn merge(&mut self, other: &GapSample) {
        if other.count == 0 {
            return;
        }
        let na = self.count as f64;
        let nb = other.count as f64;
        let n = na + nb;
        let delta = other.mean - self.mean;
        self.mean += delta * nb / n;
        self.m2 += other.m2 + delta * delta * na * nb / n;
        self.count += other.count;
        self.max = self.max.max(other.max);
        self.retained.extend_from_slice(&other.retained);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// The distribution summary, or `None` for a response with no gap (fewer
    /// than two arrivals) — there is nothing to summarise, and a zero would
    /// read as "perfectly smooth".
    pub fn stats(&self) -> Option<GapStats> {
        if self.count == 0 {
            return None;
        }
        let retained: Vec<f64> = self.retained.iter().map(|g| f64::from(*g)).collect();
        let pct = |p| stats::percentile(&retained, p).unwrap_or(f64::NAN);
        Some(GapStats {
            count: self.count,
            retained: self.retained.len(),
            mean_ms: self.mean,
            // Population deviation: the gaps ARE the population of this
            // response, not a sample of a larger one.
            stddev_ms: (self.m2 / self.count as f64).sqrt(),
            max_ms: self.max,
            p50_ms: pct(50),
            p90_ms: pct(90),
            p99_ms: pct(99),
        })
    }
}

/// One response's (or one pool's) arrival-gap distribution, in ms.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GapStats {
    /// Gaps observed.
    pub count: u64,
    /// Gaps the percentiles were computed over (`< count` only when a
    /// response ran past [`RETAIN_CAP`]).
    pub retained: usize,
    pub mean_ms: f64,
    pub stddev_ms: f64,
    /// A stall is a tail event; the mean hides it and this does not.
    pub max_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
}

impl GapStats {
    /// `stability = (p99 − p50) / p50`: the arrival-gap tail spread relative
    /// to its median. Dimensionless, so it compares across models and rungs
    /// whose absolute step times differ by an order of magnitude.
    ///
    /// **Lower is better.** This is a dispersion measure — a smaller value
    /// means smoother token delivery; 0 means no tail at all. `None` when
    /// p50 is not a positive finite number.
    pub fn stability(&self) -> Option<f64> {
        (self.p50_ms.is_finite() && self.p50_ms > 0.0 && self.p99_ms.is_finite())
            .then(|| (self.p99_ms - self.p50_ms) / self.p50_ms)
    }

    /// Coefficient of variation, `σ / mean`. Also a dispersion measure:
    /// lower is better (smoother).
    pub fn cv(&self) -> Option<f64> {
        (self.mean_ms > 0.0).then(|| self.stddev_ms / self.mean_ms)
    }

    /// The record keys, under `prefix` (e.g. `"c8_"` or `""`). The primitives
    /// always; the two derived ratios only when defined.
    pub fn metrics(&self, prefix: &str, m: &mut BTreeMap<String, f64>) {
        let put = |m: &mut BTreeMap<String, f64>, k: &str, v: f64| {
            m.insert(format!("{prefix}{k}"), v);
        };
        put(m, "arrival_gap_count", self.count as f64);
        put(m, "arrival_gap_retained", self.retained as f64);
        put(m, "arrival_gap_mean_ms", self.mean_ms);
        put(m, "arrival_gap_stddev_ms", self.stddev_ms);
        put(m, "arrival_gap_max_ms", self.max_ms);
        put(m, "arrival_gap_p50_ms", self.p50_ms);
        put(m, "arrival_gap_p90_ms", self.p90_ms);
        put(m, "arrival_gap_p99_ms", self.p99_ms);
        // `stability`: LOWER IS BETTER (dispersion; see `Self::stability`).
        if let Some(s) = self.stability() {
            put(m, "stability", s);
        }
        if let Some(cv) = self.cv() {
            put(m, "arrival_gap_cv", cv);
        }
    }
}

/// Inter-Token Latency per AIPerf: `(request_latency − TTFT) /
/// (output_tokens − 1)`, where the numerator is the decode window — first
/// token → FINAL response chunk, not the last content token. The one
/// definition both clocks use: the client passes its own first-arrival →
/// stream-end window, the server-clock caller passes the endpoint's
/// `usage.decode_time_ms`.
///
/// Undefined below two output tokens (nothing to divide by) and for a
/// non-positive window; `None`, never 0 — a zero would read as an
/// infinitely fast decode.
pub fn itl_ms(decode_window_ms: f64, output_tokens: usize) -> Option<f64> {
    (output_tokens >= 2 && decode_window_ms.is_finite() && decode_window_ms > 0.0)
        .then(|| decode_window_ms / (output_tokens - 1) as f64)
}

#[cfg(test)]
#[path = "gaps_tests.rs"]
mod tests;
