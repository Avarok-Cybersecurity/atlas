// SPDX-License-Identifier: AGPL-3.0-only

//! Cells and the verdict.
//!
//! Same three-state shape as the image benchmark, and for the same reason:
//! "describe this video" has a confident answer available from language priors
//! alone. A server that stopped splicing vision embeddings entirely would
//! still produce fluent video descriptions — that is exactly what the
//! 2026-08-14 splice defect did — so a run whose control also "passes" is
//! VACUOUS, not green.

use std::fmt;

/// How one ordered-color reading came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderCell {
    /// The reply named the clip's colors in the right order.
    Match {
        clip: &'static str,
        seen: String,
    },
    /// The reply named colors, but not in this clip's order. The strongest
    /// signal available: the pixels arrived, the sequence did not.
    WrongOrder {
        clip: &'static str,
        want: String,
        got: String,
    },
    /// The reply did not name the colors at all.
    NotSeen {
        clip: &'static str,
        reply: String,
    },
    /// Skipped — the clip needs a decoder the server does not have.
    Skipped {
        clip: &'static str,
        why: String,
    },
    Error {
        clip: &'static str,
        msg: String,
    },
}

/// How one numeric leg came out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountCell {
    Match { id: &'static str, detail: String },
    Mismatch { id: &'static str, detail: String },
    Skipped { id: &'static str, why: String },
    Error { id: &'static str, msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Pass,
    Fail,
    /// The control described a video it was never sent, so nothing the other
    /// legs found is evidence.
    Vacuous,
    /// Nothing could be asserted — every leg was skipped. Distinct from Pass
    /// on purpose: a run that measured nothing must not read as green.
    Inconclusive,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Verdict::Pass => "PASS",
            Verdict::Fail => "FAIL",
            Verdict::Vacuous => "VACUOUS",
            Verdict::Inconclusive => "INCONCLUSIVE",
        })
    }
}

/// Color names found in `reply`, in the order they first appear.
///
/// First-appearance order, and each color counted once: a model that says
/// "red, then green, then blue, and finally yellow — the red one was first"
/// must not be scored as red,green,blue,yellow,red. The question asked for a
/// sequence, and the sequence is what the first mentions describe.
pub fn colors_in_order(reply: &str, palette: &[&str]) -> Vec<String> {
    let lower = reply.to_lowercase();
    let mut hits: Vec<(usize, String)> = Vec::new();
    for c in palette {
        if let Some(at) = crate::benchmarks::first_standalone_term(&lower, c) {
            hits.push((at, (*c).to_string()));
        }
    }
    hits.sort_by_key(|(at, _)| *at);
    hits.into_iter().map(|(_, c)| c).collect()
}

/// Did the reply name exactly this clip's colors, in order?
pub fn order_matches(reply: &str, want: &[&str], palette: &[&str]) -> bool {
    let got = colors_in_order(reply, palette);
    got.len() == want.len() && got.iter().zip(want).all(|(g, w)| g == w)
}

/// Legs that were attempted and produced a pass/fail classification. A
/// deployment skip is not asserted; a request or decode error is a failed
/// assertion and must not masquerade as an all-skipped run.
pub fn asserted(order: &[OrderCell], counts: &[CountCell]) -> usize {
    order
        .iter()
        .filter(|c| {
            matches!(
                c,
                OrderCell::Match { .. }
                    | OrderCell::WrongOrder { .. }
                    | OrderCell::NotSeen { .. }
                    | OrderCell::Error { .. }
            )
        })
        .count()
        + counts
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    CountCell::Match { .. } | CountCell::Mismatch { .. } | CountCell::Error { .. }
                )
            })
            .count()
}

pub fn passed(order: &[OrderCell], counts: &[CountCell]) -> usize {
    order
        .iter()
        .filter(|c| matches!(c, OrderCell::Match { .. }))
        .count()
        + counts
            .iter()
            .filter(|c| matches!(c, CountCell::Match { .. }))
            .count()
}

/// `control_held` is true when the no-video control did NOT describe a clip.
pub fn verdict(order: &[OrderCell], counts: &[CountCell], control_held: bool) -> Verdict {
    let asserted_n = asserted(order, counts);
    if asserted_n == 0 {
        return Verdict::Inconclusive;
    }
    if !control_held {
        return Verdict::Vacuous;
    }
    if passed(order, counts) == asserted_n {
        Verdict::Pass
    } else {
        Verdict::Fail
    }
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
