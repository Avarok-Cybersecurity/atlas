// SPDX-License-Identifier: AGPL-3.0-only

//! The closed value sets of the enumerated `spark serve` string flags.
//!
//! Two consumers read these: `validate::check_enum`, which REFUSES a value
//! outside the set, and the dashboard's option picker, which OFFERS the set.
//! They used to be able to disagree — the validator's lists lived inline in
//! `validate.rs` and anything else wanting them had to copy them, and a copy
//! drifts into the worst failure mode a picker has: offering a value the
//! server refuses, or hiding one it accepts. One module, one list per flag.
//!
//! `--kv-cache-dtype` is deliberately NOT a const here: its authority is
//! `spark_runtime::kv_cache::KvCacheDtype`, whose `catalog` module derives
//! the list from the enum under a wildcard-free match — adding a variant
//! fails the build there rather than silently missing from this picker.
//!
//! Deliberately NOT wired into clap as `PossibleValuesParser`s. That would
//! move the typo diagnosis from `validate_serve_args` — which reports every
//! problem at once, with what/why/fix — into clap, which exits on the first.
//! It would also be a second enforcement point over the same data, and two
//! enforcement points is how the two disagree. clap stays the authority on
//! the flag SURFACE (names, help, defaults, arity); this module is the
//! authority on the enumerated VALUES; `validate.rs` is the one place they
//! are enforced.

pub(crate) const LM_HEAD_DTYPES: &[&str] = &["default", "bf16", "nvfp4", "fp8"];
pub(crate) const MTP_QUANTS: &[&str] = &["bf16", "fp8", "nvfp4"];
pub(crate) const SCHEDULING_POLICIES: &[&str] = &["fifo", "slai"];
pub(crate) const SSM_H_DTYPES: &[&str] = &["f32", "f16", "f16-pool"];
pub(crate) const MTP_GATES: &[&str] = &["auto", "force"];
pub(crate) const TOOL_CALL_PARSERS: &[&str] = &[
    "hermes",
    "qwen3_coder",
    "qwen3_xml",
    "gemma4",
    "mistral",
    "minimax_xml",
    "bare_json",
    "poolside_v1",
];

/// The closed value set for a `spark serve` flag, by its long name, or `None`
/// for a free-form flag. `--kv-cache-dtype` also accepts short aliases
/// (`fp8k2v` for `fp8k_turbo2v`) that are deliberately not offered: a picker
/// lists one spelling per choice, and the canonical one is the one every
/// recipe and document uses.
pub(crate) fn options_for_flag(flag: &str) -> Option<Vec<String>> {
    let owned = |list: &[&str]| list.iter().map(|s| s.to_string()).collect();
    match flag {
        "lm-head-dtype" => Some(owned(LM_HEAD_DTYPES)),
        "mtp-quantization" => Some(owned(MTP_QUANTS)),
        "scheduling-policy" => Some(owned(SCHEDULING_POLICIES)),
        "ssm-h-dtype" => Some(owned(SSM_H_DTYPES)),
        "mtp-gate" => Some(owned(MTP_GATES)),
        "tool-call-parser" => Some(owned(TOOL_CALL_PARSERS)),
        "kv-cache-dtype" => Some(
            spark_runtime::kv_cache::KvCacheDtype::ALL
                .iter()
                .map(|d| d.name().to_string())
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
#[path = "flag_values_tests.rs"]
mod tests;
