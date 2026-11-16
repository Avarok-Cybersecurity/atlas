// SPDX-License-Identifier: AGPL-3.0-only
// T-AT: Atlas-native. Number extraction for GSM8K style answers, written from scratch for this lane.
//! Responsibility: parse numbers (plain, comma-grouped, fractions, `\frac`), pull the answer out of a GSM8K reply
//! (`#### n`, else the last `\boxed{}` number, else the last number) and compare it with the gold.

use std::sync::LazyLock;

use regex::Regex;

use super::boxed::last_boxed;
use super::{Grade, py_float_str};
use crate::route_gate::error::{GResult, value_err};

static NUM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-?\$?\s*\d[\d,]*(?:\.\d+)?|-?\.\d+").expect("static regex"));
static SLASH_FRACTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\A(-?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\z").expect("static regex"));
static LATEX_FRACTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\A(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(\d+(?:\.\d+)?)\}\z").expect("static regex")
});
static HASHES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"####\s*(.+)").expect("static regex"));
static HASHES_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)####\s*(.+)$").expect("static regex"));

/// A number from text like `1,234`, `$12.50`, `45%`, `3/4` or `\frac{3}{4}`; `None` when it is not one.
pub fn parse_number(s: &str) -> Option<f64> {
    let cleaned = s.trim().replace([',', '$', '%', ' '], "");
    let s = cleaned.trim_end_matches('.');
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    if let Some(m) = SLASH_FRACTION.captures(s) {
        let den: f64 = m[2].parse().ok()?;
        if den != 0.0 {
            return Some(m[1].parse::<f64>().ok()? / den);
        }
    }
    if let Some(m) = LATEX_FRACTION.captures(s) {
        let den: f64 = m[3].parse().ok()?;
        if den != 0.0 {
            let sign = if m[1].is_empty() { 1.0 } else { -1.0 };
            return Some(sign * m[2].parse::<f64>().ok()? / den);
        }
    }
    None
}

/// The last parseable number in `text`.
pub fn last_number(text: &str) -> Option<f64> {
    NUM.find_iter(text)
        .collect::<Vec<_>>()
        .iter()
        .rev()
        .find_map(|m| parse_number(m.as_str()))
}

/// GSM8K stores the worked solution; the gold number follows `####`.
pub fn gsm8k_gold(answer_field: &str) -> GResult<f64> {
    let Some(m) = HASHES_LINE.captures(answer_field.trim()) else {
        return value_err("GSM8K answer without '####'");
    };
    parse_number(&m[1]).map_or_else(
        || value_err(format!("GSM8K gold {:?} is not a number", &m[1])),
        Ok,
    )
}

/// `#### n` (the format the prompt asks for), else the last `\boxed{...}` number, else the last number in the reply.
pub fn extract_gsm8k(text: &str) -> Option<f64> {
    if let Some(last) = HASHES.captures_iter(text).last()
        && let Some(v) = last_number(&last[1])
    {
        return Some(v);
    }
    if let Some(boxed) = last_boxed(text) {
        // Python: `parse_number(b) or last_number(b)`; a parsed 0.0 is falsy there, so it falls through too.
        let v = match parse_number(&boxed) {
            Some(v) if v != 0.0 => Some(v),
            _ => last_number(&boxed),
        };
        if v.is_some() {
            return v;
        }
    }
    last_number(text)
}

/// Grades a numeric reply: within a relative `1e-6` of the gold.
pub fn grade_numeric(text: &str, gold: f64) -> Grade {
    match extract_gsm8k(text) {
        None => Grade::no_answer(),
        Some(got) => {
            let ok = (got - gold).abs() <= 1e-6 * gold.abs().max(1.0);
            Grade::new(ok, Some(py_float_str(got)), "ok")
        }
    }
}
