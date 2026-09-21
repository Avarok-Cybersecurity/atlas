// SPDX-License-Identifier: AGPL-3.0-only

//! Prometheus metrics for Atlas Spark.

use lazy_static::lazy_static;
use prometheus::{
    HistogramVec, IntCounter, IntCounterVec, IntGauge, register_histogram_vec,
    register_int_counter, register_int_counter_vec, register_int_gauge,
};

lazy_static! {
    pub static ref REQUESTS_TOTAL: IntCounter =
        register_int_counter!("atlas_requests_total", "Total requests processed").unwrap();
    pub static ref REQUESTS_ACTIVE: IntGauge =
        register_int_gauge!("atlas_requests_active", "Currently active requests").unwrap();
    /// Time to first token, LABELLED BY MODEL.
    ///
    /// A label rather than a reset. The counters in this file are process
    /// totals and are correct across a hot-swap — "requests this process
    /// handled" does not become false when the model changes, and resetting a
    /// Prometheus counter breaks `rate()`, which assumes monotonicity.
    ///
    /// A latency histogram is different: pooling two models' distributions
    /// makes every quantile a statement about neither of them. The standard
    /// answer is to separate by label, which also keeps the pre-swap data
    /// rather than discarding it — `sum by (le)` aggregates back to the old
    /// single-series view for anyone who wants it.
    pub static ref TTFT_SECONDS: HistogramVec = register_histogram_vec!(
        "atlas_time_to_first_token_seconds",
        "Time to first token",
        &["model"],
        vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    .unwrap();
    pub static ref GENERATION_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_generation_tokens_total", "Total tokens generated").unwrap();
    /// Tokens counted AS THEY ARE DECODED, not at request completion.
    ///
    /// `GENERATION_TOKENS_TOTAL` is incremented once per request, from the
    /// final usage block. That is correct for a total and useless for a RATE:
    /// differentiating it at 1 Hz reads 0 tok/s for the whole generation and
    /// then one spike at the end, which is exactly how a live dashboard
    /// renders as "nothing, then a burst". This one advances per token so the
    /// derivative is the real decode rate.
    ///
    /// It counts tokens the engine DECODED, which is the honest quantity for a
    /// throughput view: a token suppressed by the tool-call sanitizer still
    /// cost a decode step. So this can run slightly ahead of
    /// `GENERATION_TOKENS_TOTAL`, and the two are not interchangeable.
    pub static ref DECODED_TOKENS_TOTAL: IntCounter =
        register_int_counter!(
            "atlas_decoded_tokens_total",
            "Tokens decoded, counted as they are produced (rate-friendly)"
        ).unwrap();
    // ── HTTP byte accounting (Atlas TUI Server Stats) ──
    //
    // Request side counts body bytes as received by the byte-count
    // middleware; response side counts bytes actually written through the
    // wrapped body (streaming/SSE included, where Content-Length lies).
    pub static ref HTTP_BYTES_IN: IntCounter =
        register_int_counter!("atlas_http_bytes_in_total", "Total HTTP request body bytes")
            .unwrap();
    pub static ref HTTP_BYTES_OUT: IntCounter =
        register_int_counter!("atlas_http_bytes_out_total", "Total HTTP response body bytes")
            .unwrap();
    pub static ref PROMPT_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_prompt_tokens_total", "Total prompt tokens processed")
            .unwrap();

    // ── Loop-detector telemetry (P5.2, 2026-04-25) ──
    //
    // Track the verdict distribution emitted by `loop_detector::detect`
    // so we can tune thresholds against production traffic instead of
    // single dump fixtures. Labels:
    //   - verdict ∈ {none, hint, suppress}
    //   - channel ∈ {text, tools, combined, n/a (None verdict)}
    //   - spinning ∈ {0, 1} — was Layer-2 spinning detection also active
    pub static ref LOOP_DETECTOR_VERDICTS: IntCounterVec =
        register_int_counter_vec!(
            "atlas_loop_detector_verdicts_total",
            "Loop detector verdicts emitted, by verdict + channel + spinning flag",
            &["verdict", "channel", "spinning"]
        ).unwrap();

    // ── Speculative-decode telemetry (A.2 EASD scaffolding) ──
    //
    // Per-K acceptance counters. Enables measuring baseline accept
    // rates across MTP K-paths so we can decide whether EASD
    // activation (per-step D2H of verify logits + entropy gating,
    // arXiv:2512.23765) is worth its cost. EASD itself is gated
    // behind future activation once these baselines are measured.
    pub static ref SPEC_DECODE_VERIFY: IntCounterVec =
        register_int_counter_vec!(
            "atlas_spec_decode_verify_total",
            "MTP draft verify outcomes by K and result",
            &["k", "outcome"]
        ).unwrap();

    // ── Tool-call telemetry ──
    //
    // Total successful tool calls emitted by the API layer (sum across
    // streaming + blocking). Paired with the "Tool call: name(args)"
    // info log so operators can both grep logs and graph rates.
    // Unlabeled (no `name` label) — high-cardinality tool names would
    // blow up Prometheus cardinality.
    pub static ref TOOL_CALLS_TOTAL: IntCounter =
        register_int_counter!(
            "atlas_tool_calls_total",
            "Total successful tool calls emitted by the server"
        ).unwrap();
}

/// RAII guard for the `atlas_requests_active` gauge: increments on construction,
/// decrements exactly once on drop.
///
/// Replaces a hand-balanced `inc()` + seven scattered `dec()` calls. That shape
/// leaked: any terminal path that forgot to decrement — or, critically, a
/// handler future DROPPED because the client disconnected (axum drops the future
/// on disconnect, so no `dec()` in the body ever runs) — pinned the gauge
/// forever. Orphans then accumulate monotonically and can exhaust the scheduler's
/// admission accounting while `/health` still reports ready.
/// See Avarok-Cybersecurity/atlas#368.
///
/// For streaming the guard is moved into `StreamCtx`, which the SSE `flat_map`
/// closure owns — so it also drops when the client hangs up mid-stream.
pub struct ActiveRequestGuard(());

impl ActiveRequestGuard {
    pub fn new() -> Self {
        REQUESTS_ACTIVE.inc();
        Self(())
    }
}

impl Default for ActiveRequestGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ActiveRequestGuard {
    fn drop(&mut self) {
        REQUESTS_ACTIVE.dec();
    }
}
