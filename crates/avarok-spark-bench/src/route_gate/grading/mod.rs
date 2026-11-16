// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Answer extraction and comparison rules are the common conventions for these public datasets
// (final-line letter, "####" suffix, last \boxed{}), written from scratch for this lane; no upstream code copied.
//! Responsibility: pass/fail grading of the 4B's replies for the routing gate (B0).
//!
//! A label is `Some(true)` (the 4B passed), `Some(false)` (it failed) or `None` (not gradable here, for example code
//! execution is off). A reply that gives no parseable answer is a FAILURE (a user would not get an answer either),
//! recorded with the reason `no_answer` so the study can drop those rows with `--drop-no-answer` if the budget was
//! too small.

mod boxed;
mod code;
mod mc;
mod numeric;

pub use boxed::{grade_boxed, last_boxed, math_equal, normalize_math};
pub use code::{
    extract_code, grade_code, humaneval_program, mbpp_program, resolve_sandbox, run_python,
};
pub use mc::{extract_mc_letter, grade_mc};
pub use numeric::{extract_gsm8k, grade_numeric, gsm8k_gold, last_number, parse_number};

use serde_json::Value;

use super::error::{GResult, value_err};

/// Option letters, in order.
pub const LETTERS: &str = "ABCDEFGHIJ";

/// One graded reply.
#[derive(Clone, Debug, PartialEq)]
pub struct Grade {
    pub passed: Option<bool>,
    pub extracted: Option<String>,
    pub reason: String,
}

impl Grade {
    pub(crate) fn new(passed: bool, extracted: Option<String>, reason: &str) -> Self {
        Self {
            passed: Some(passed),
            extracted,
            reason: reason.to_string(),
        }
    }

    pub(crate) fn no_answer() -> Self {
        Self::new(false, None, "no_answer")
    }
}

/// Python's `str(float)` for the values the graders produce (`18.0`, not `18`).
pub(crate) fn py_float_str(v: f64) -> String {
    if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e16 {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

/// Grades one reply against its prompt record (see `prep` for the record schema).
pub fn grade(item: &Value, reply: &str, code_exec: &str, timeout: f64) -> GResult<Grade> {
    let kind = item["kind"].as_str().unwrap_or_default();
    match kind {
        "mc" => {
            let n = item["meta"]["choices"].as_array().map_or(10, Vec::len);
            Ok(grade_mc(
                reply,
                item["gold"].as_str().unwrap_or_default(),
                n,
            ))
        }
        "numeric" => {
            let gold = match &item["gold"] {
                Value::Number(n) => n.as_f64().unwrap_or(f64::NAN),
                Value::String(s) => parse_number(s).unwrap_or(f64::NAN),
                _ => f64::NAN,
            };
            Ok(grade_numeric(reply, gold))
        }
        "boxed" => Ok(grade_boxed(
            reply,
            item["gold"].as_str().unwrap_or_default(),
        )),
        "code" => grade_code(item, reply, code_exec, timeout),
        other => value_err(format!("unknown kind {other:?}")),
    }
}
