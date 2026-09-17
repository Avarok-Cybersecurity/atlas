// SPDX-License-Identifier: AGPL-3.0-only

//! The parameter schema, split out so `mod.rs` stays under the 500-LoC cap.
//!
//! Defaults live HERE and nowhere else — `ParamValues::defaults` derives the
//! starting values from these specs, so the pane and the benchmark cannot
//! disagree about what a run was configured with.

use crate::params::{ParamKind, ParamSpec, ParamValue};

/// Every editable parameter, in the order the form renders them.
pub fn specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new(
            "include",
            "Model filter",
            "Case-insensitive substring of the HF id. `all` runs every checkpoint the box can serve.",
            ParamKind::Text,
            ParamValue::Text("all".into()),
        ),
        ParamSpec::new(
            "max_seq_len",
            "Max sequence length",
            "Context each round is served with. Must fit the long-context probe plus its output.",
            ParamKind::Int {
                min: 2048,
                max: 262_144,
            },
            ParamValue::Int(32_768),
        ),
        ParamSpec::new(
            "long_ctx_tokens",
            "Long-context probe",
            "Prompt size for the needle-in-a-haystack recall probe. 0 turns it off.",
            ParamKind::Int {
                min: 0,
                max: 131_072,
            },
            ParamValue::Int(16_384),
        ),
        ParamSpec::new(
            "tps_tokens",
            "Throughput budget",
            "Output tokens the throughput probe asks for. Too few and the reply arrives in one SSE delta, leaving decode unmeasurable.",
            ParamKind::Int { min: 16, max: 4096 },
            ParamValue::Int(256),
        ),
        ParamSpec::new(
            "probe_budget",
            "Probe output tokens",
            "Output budget for the codegen and tool-call probes.",
            ParamKind::Int { min: 32, max: 4096 },
            ParamValue::Int(512),
        ),
        ParamSpec::new(
            "speculative",
            "Speculative decoding",
            "Serve each round with MTP on. Off by default: a checkpoint with no MTP head falls back to single-token decode and reports the baseline's numbers under a +MTP label.",
            ParamKind::Bool,
            ParamValue::Bool(false),
        ),
        ParamSpec::new(
            "request_timeout_s",
            "Request timeout",
            "Seconds before a single probe request is abandoned.",
            ParamKind::Int { min: 10, max: 3600 },
            ParamValue::Int(300),
        ),
        ParamSpec::new(
            "update_baselines",
            "Update baselines",
            "Record this run's tok/s as the new bar instead of gating against it. Deliberate refresh only — review the numbers first.",
            ParamKind::Bool,
            ParamValue::Bool(false),
        ),
    ]
}
