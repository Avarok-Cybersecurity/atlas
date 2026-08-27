// SPDX-License-Identifier: AGPL-3.0-only

//! Shared measurement helpers: percentiles and prompt synthesis.
//!
//! Both are ports of `bench/bench_concurrency.py` and are kept bit-compatible
//! with it on purpose — the recorded sweeps we compare against were produced by
//! that script, and a different percentile rule or filler corpus would quietly
//! shift every number.

use std::fmt::Write as _;

/// Varied filler. Uniform repetition ("hello hello …") collapses attention on
/// pure-attention models and makes them emit EOS immediately, which turns an
/// input-length sweep into a measurement of degenerate decode.
const FILLER: &str = concat!(
    "The quick brown fox jumped over the lazy dog near a river bank. ",
    "Mountains rise above the clouds while birds sing their morning songs. ",
    "Science explores the universe through careful observation and experiment. ",
    "Ancient civilizations built remarkable structures that still stand today. ",
    "Music fills the air with rhythm and harmony across every culture. ",
    "Technology advances rapidly changing how people communicate and work. ",
    "Forests provide shelter for countless species of plants and animals. ",
    "Ocean waves crash upon the shore under the light of the moon. ",
);

/// Should the prompt push the model to fill the output budget?
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PromptMode {
    /// Let the model stop naturally. Measures a realistic reply length.
    Natural,
    /// Append a counting instruction so the run actually reaches `osl` tokens.
    /// The default: a sweep whose requests hit EOS after five tokens measures
    /// scheduling overhead, not decode.
    #[default]
    Count,
}

impl PromptMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "natural" | "hello" => Some(PromptMode::Natural),
            "count" => Some(PromptMode::Count),
            _ => None,
        }
    }
}

/// Build a prompt of roughly `isl_tokens` tokens.
///
/// `prefix_tag` is prefixed so callers can force a prefix-cache MISS (cold TTFT) or,
/// with a constant prefix_tag, guarantee a bit-identical prompt across runs so the
/// cache HITS (warm TTFT). It is the whole cold/warm mechanism.
pub fn make_prompt(isl_tokens: usize, mode: PromptMode, prefix_tag: &str) -> String {
    // The chat template contributes ~12 tokens of its own.
    let needed = isl_tokens.saturating_sub(12).max(1);
    let words: Vec<&str> = FILLER.split_whitespace().collect();
    let mut out = String::with_capacity(needed * 6 + prefix_tag.len() + 80);
    if !prefix_tag.is_empty() {
        let _ = write!(out, "[{prefix_tag}] ");
    }
    for i in 0..needed {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(words[i % words.len()]);
    }
    if mode == PromptMode::Count {
        out.push_str(" Count from 1 upward, one number per line, until told to stop.");
    }
    out
}

/// `p`-th percentile (0–100) of `values`, using the same nearest-rank rule as
/// the Python harness: `idx = min(int(n*p/100 + 0.5), n-1)` over sorted values.
pub fn percentile(values: &[f64], p: u32) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite"));
    let n = sorted.len();
    let idx = ((n as f64 * p as f64 / 100.0) + 0.5) as usize;
    Some(sorted[idx.min(n - 1)])
}

/// True median: middle element for odd `n`, mean of the two middle elements
/// for even `n`.
///
/// ★ NOT `percentile(values, 50)`. The nearest-rank rule above computes
/// `idx = int(n*0.5 + 0.5)`, which for n=3 is index 2 — the MAXIMUM of three
/// samples, not the middle one. A "median of 3 runs" metric built on it would
/// quietly report the best run. Kept separate rather than fixing `percentile`
/// because the percentile rule is pinned bit-compatible with the Python
/// harness the recorded sweeps came from.
pub fn median(values: &[f64]) -> Option<f64> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("filtered to finite"));
    let n = sorted.len();
    Some(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        sorted[n / 2 - 1].midpoint(sorted[n / 2])
    })
}

/// p50 / p90 / p99 in one pass over the same sorted view.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Percentiles {
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

impl Percentiles {
    pub fn of(values: &[f64]) -> Self {
        Self {
            p50: percentile(values, 50),
            p90: percentile(values, 90),
            p99: percentile(values, 99),
        }
    }
}

/// Format an optional millisecond value for a table cell.
pub fn fmt_ms(v: Option<f64>) -> String {
    match v {
        Some(ms) if !ms.is_finite() || ms < 0.0 => "—".into(),
        Some(ms) if ms >= 10_000.0 => format!("{:.1}s", ms / 1000.0),
        Some(ms) => format!("{ms:.1}"),
        None => "—".into(),
    }
}

/// Relative change `new` vs `base`, in percent. `None` when `base` is unusable.
pub fn pct_delta(new: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (new, base) {
        (Some(n), Some(b)) if n.is_finite() && b.is_finite() && b > f64::EPSILON => {
            let delta = (n - b) / b * 100.0;
            delta.is_finite().then_some(delta)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256(text: &str) -> String {
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }

    #[test]
    fn percentile_matches_the_python_nearest_rank_rule() {
        let v: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        // int(10*50/100 + 0.5) = 5 -> sorted[5] = 6
        assert_eq!(percentile(&v, 50), Some(6.0));
        // int(10*90/100 + 0.5) = 9 -> sorted[9] = 10
        assert_eq!(percentile(&v, 90), Some(10.0));
        // index clamps to the last element
        assert_eq!(percentile(&v, 100), Some(10.0));
        assert_eq!(percentile(&[], 50), None);
    }

    #[test]
    fn percentiles_ignore_non_finite_samples() {
        let v = vec![1.0, f64::NAN, 3.0, f64::INFINITY];
        assert_eq!(
            Percentiles::of(&v),
            Percentiles {
                p50: Some(3.0),
                p90: Some(3.0),
                p99: Some(3.0),
            }
        );
        assert_eq!(percentile(&[f64::NAN], 50), None);
    }

    #[test]
    fn median_handles_empty_odd_even_and_large_finite_samples() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[f64::NAN, f64::INFINITY]), None);
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
        assert_eq!(median(&[f64::MAX, f64::MAX]), Some(f64::MAX));
    }

    #[test]
    fn prompt_length_tracks_the_request_and_the_tag_changes_the_text() {
        let a = make_prompt(256, PromptMode::Natural, "");
        let b = make_prompt(1024, PromptMode::Natural, "");
        assert!(b.len() > a.len());
        assert_eq!(a.split_whitespace().count(), 256 - 12);
        assert!(
            make_prompt(256, PromptMode::Natural, "s1").starts_with("[s1] The quick brown fox")
        );
        // Same prefix_tag -> identical prompt (warm/prefix-cache hit).
        assert_eq!(
            make_prompt(256, PromptMode::Natural, "s1"),
            make_prompt(256, PromptMode::Natural, "s1")
        );
        // Different prefix_tag -> different prompt (cold/prefix-cache miss).
        assert_ne!(
            make_prompt(256, PromptMode::Natural, "s1"),
            make_prompt(256, PromptMode::Natural, "s2")
        );
    }

    #[test]
    fn prompt_modes_pin_the_complete_natural_and_forced_bytes() {
        assert_eq!(
            sha256(&make_prompt(64, PromptMode::Natural, "fixture")),
            "1d34b0e6f6c8b13f614be586f7f29ab9fcf918ed11416a3c62702d4a05bf7a54"
        );
        assert_eq!(
            sha256(&make_prompt(64, PromptMode::Count, "fixture")),
            "619330d2a0c46380c7c3e815226610d5cdea78b760736f496f64a6ec8a6efe96"
        );
    }

    #[test]
    fn millisecond_formatting_rejects_invalid_durations() {
        assert_eq!(fmt_ms(None), "—");
        assert_eq!(fmt_ms(Some(12.25)), "12.2");
        assert_eq!(fmt_ms(Some(10_000.0)), "10.0s");
        for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(fmt_ms(Some(invalid)), "—", "value={invalid:?}");
        }
    }

    #[test]
    fn pct_delta_is_none_when_there_is_no_usable_baseline() {
        assert_eq!(pct_delta(Some(110.0), Some(100.0)), Some(10.0));
        assert_eq!(pct_delta(Some(110.0), None), None);
        assert_eq!(pct_delta(Some(110.0), Some(0.0)), None);
        assert_eq!(pct_delta(Some(110.0), Some(-100.0)), None);
        assert_eq!(pct_delta(Some(110.0), Some(f64::INFINITY)), None);
        assert_eq!(pct_delta(Some(f64::INFINITY), Some(100.0)), None);
        assert_eq!(pct_delta(Some(f64::NAN), Some(100.0)), None);
        assert_eq!(pct_delta(Some(f64::MAX), Some(1.0)), None);
    }
}
