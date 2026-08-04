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
        Some(ms) if ms >= 10_000.0 => format!("{:.1}s", ms / 1000.0),
        Some(ms) => format!("{ms:.1}"),
        None => "—".into(),
    }
}

/// Relative change `new` vs `base`, in percent. `None` when `base` is unusable.
pub fn pct_delta(new: Option<f64>, base: Option<f64>) -> Option<f64> {
    match (new, base) {
        (Some(n), Some(b)) if b.abs() > f64::EPSILON => Some((n - b) / b * 100.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(Percentiles::of(&v).p50, Some(3.0));
        assert_eq!(percentile(&[f64::NAN], 50), None);
    }

    #[test]
    fn prompt_length_tracks_the_request_and_the_tag_changes_the_text() {
        let a = make_prompt(256, PromptMode::Natural, "");
        let b = make_prompt(1024, PromptMode::Natural, "");
        assert!(b.len() > a.len());
        assert_eq!(a.split_whitespace().count(), 256 - 12);
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
    fn count_mode_appends_the_forcing_instruction() {
        assert!(make_prompt(64, PromptMode::Count, "").ends_with("until told to stop."));
        assert!(!make_prompt(64, PromptMode::Natural, "").ends_with("stop."));
    }

    #[test]
    fn pct_delta_is_none_when_there_is_no_usable_baseline() {
        assert_eq!(pct_delta(Some(110.0), Some(100.0)), Some(10.0));
        assert_eq!(pct_delta(Some(110.0), None), None);
        assert_eq!(pct_delta(Some(110.0), Some(0.0)), None);
    }
}
