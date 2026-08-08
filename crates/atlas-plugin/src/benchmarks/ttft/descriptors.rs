// SPDX-License-Identifier: AGPL-3.0-only

//! The two TTFT gate descriptors: warm (cached prefix) and cold (uncached
//! prefill). Both compare against a STORED SAME-BOX baseline — TTFT is
//! box-local by nature, so a different machine must re-record rather than
//! reuse.

use super::{Mode, TtftGate};
use crate::benchmark::BenchmarkDescriptor;
use crate::metadata::PluginMetadata;

const WARM_SUMMARY: &str = "Cached-prefix TTFT vs the stored same-box baseline";
const COLD_SUMMARY: &str = "Uncached prefill TTFT vs the stored same-box baseline";
pub const WARM_METADATA: PluginMetadata = PluginMetadata::atlas(WARM_SUMMARY);
pub const COLD_METADATA: PluginMetadata = PluginMetadata::atlas(COLD_SUMMARY);

pub const WARM_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-warm-gate",
    name: "Warm TTFT Regression Gate",
    summary: WARM_SUMMARY,
    detail: "Measures time-to-first-token on the WARM path: each sample repeats a bit-identical \
             prompt so the prefix cache hits. Gates at median ≤3% and p90 ≤5% against a baseline \
             recorded on this box — the guard that catches an optimization silently falling back \
             to a slow path while the correctness gates stay green.",
    duration_hint: "~3–6 min",
    updated: "2026-07-31",
    needs_confirmation: false,
    // A TTFT gate compares against a baseline recorded on the SAME box and
    // model, which it stores itself — so it is meaningful for any checkpoint
    // and constrains none.
    intended_for: None,
    ctor: || Box::new(TtftGate::new(Mode::Warm)),
};

pub const COLD_DESCRIPTOR: BenchmarkDescriptor = BenchmarkDescriptor {
    id: "ttft-cold-gate",
    name: "Cold TTFT Regression Gate",
    summary: COLD_SUMMARY,
    detail: "Measures time-to-first-token with the prefix cache guaranteed to MISS: every sample \
             carries a unique prefix_tag, so each request pays a full prefill. This is the prefill path \
             on its own, with the cache's contribution removed — the warm gate cannot see a \
             prefill regression that caching is hiding.",
    duration_hint: "~3–6 min",
    updated: "2026-07-31",
    needs_confirmation: false,
    // A TTFT gate compares against a baseline recorded on the SAME box and
    // model, which it stores itself — so it is meaningful for any checkpoint
    // and constrains none.
    intended_for: None,
    ctor: || Box::new(TtftGate::new(Mode::Cold)),
};
