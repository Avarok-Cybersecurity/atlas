// SPDX-License-Identifier: AGPL-3.0-only

//! Prometheus metrics for Atlas Spark.

use lazy_static::lazy_static;
use prometheus::{
    Histogram, IntCounter, IntCounterVec, IntGauge, register_histogram, register_int_counter,
    register_int_counter_vec, register_int_gauge,
};

lazy_static! {
    pub static ref REQUESTS_TOTAL: IntCounter =
        register_int_counter!("atlas_requests_total", "Total requests processed").unwrap();
    pub static ref REQUESTS_ACTIVE: IntGauge =
        register_int_gauge!("atlas_requests_active", "Currently active requests").unwrap();
    pub static ref TTFT_SECONDS: Histogram = register_histogram!(
        "atlas_time_to_first_token_seconds",
        "Time to first token",
        vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
    )
    .unwrap();
    pub static ref GENERATION_TOKENS_TOTAL: IntCounter =
        register_int_counter!("atlas_generation_tokens_total", "Total tokens generated").unwrap();
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
