// SPDX-License-Identifier: AGPL-3.0-only

//! The scoring core. **Pure** — no handle, no HTTP, no clock — so every case
//! below is table-testable without a GPU.
//!
//! # Why four legs and not two
//!
//! The obvious shape is "reference at C=1, then diff each rung against it".
//! That is right and incomplete, for three reasons this design answers:
//!
//! 1. **A single reference is unattributable.** Issue #435 establishes that
//!    spec-ON output is not stable even ALONE at temperature 0. Diffing a rung
//!    against one reference would blame concurrency for a defect that needs no
//!    concurrency. So the reference is run TWICE; a prompt whose two solo runs
//!    disagree is `AloneUnstable` and is excluded from contamination scoring —
//!    it still fails the verdict, under its own name.
//! 2. **Contamination can outlive the batch.** #429 reports *context*
//!    corruption, so a poisoned prefix-cache or SSM snapshot can survive into
//!    later solo work. A post-check leg at C=1 after the rungs catches that as
//!    `Persistent` — the worst class, and cheap.
//! 3. **Cache state must be equal by construction**, not by luck. The caller
//!    primes every prompt once before any measured leg, so all legs run
//!    cache-warm. `cached_prompt_tokens` rides along as evidence rather than as
//!    part of equality.

use std::collections::BTreeMap;

use super::transcript::{RequestOutcome, Transcript};

/// What happened to one prompt at one rung.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Class {
    Identical,
    /// Streams differ. `at` is the longest common prefix in characters.
    Diverged {
        at: usize,
        detail: String,
    },
    /// Another request's canary appeared in this reply — leakage on its own
    /// evidence, no reference required.
    Contaminated {
        foreign: String,
    },
    /// Diverged in the POST-CHECK leg: state survived the concurrent episode
    /// into solo execution.
    Persistent {
        at: usize,
    },
    /// The prompt's two solo runs already disagreed (#435 class). Contamination
    /// cannot be attributed for it.
    AloneUnstable,
    /// Errored, timed out, or produced too few tokens to witness anything.
    Unmeasured {
        why: String,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Score {
    pub prompts: usize,
    pub rungs: usize,
    pub compared: usize,
    pub identical: usize,
    pub diverged: usize,
    pub contaminated: usize,
    pub persistent: usize,
    pub alone_unstable: usize,
    pub unmeasured: usize,
    pub foreign_canaries: usize,
    pub tokens_compared: usize,
    pub earliest_divergence: Option<usize>,
    /// `(prompt_idx, leg_label) -> Class`, for the report table.
    pub cells: BTreeMap<(usize, String), Class>,
}

/// Longest common prefix, in characters.
fn lcp(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Classify one measured outcome against its own solo reference.
fn classify(
    reference: &Transcript,
    got: &RequestOutcome,
    own_canary: &str,
    all_canaries: &[&str],
    min_completion_tokens: usize,
    persistent_leg: bool,
) -> Class {
    let t = match got {
        RequestOutcome::Error(e) => {
            return Class::Unmeasured {
                why: format!("request failed: {e}"),
            };
        }
        RequestOutcome::Ok(t) => t,
    };
    // Liveness BEFORE equality: two replies that both stopped after three
    // tokens are equal and prove nothing.
    if t.completion_tokens < min_completion_tokens {
        return Class::Unmeasured {
            why: format!(
                "only {} completion tokens (floor {min_completion_tokens})",
                t.completion_tokens
            ),
        };
    }
    // The absolute detector runs FIRST: leakage is a stronger statement than
    // "differs from its reference", and the operator's first question on a red
    // run is which of the two this is.
    if let Some(foreign) = t.carries_foreign_canary(own_canary, all_canaries) {
        return Class::Contaminated {
            foreign: foreign.to_string(),
        };
    }
    let (a, b) = (reference.canonical(), t.canonical());
    if a != b {
        let at = lcp(&a, &b);
        return if persistent_leg {
            Class::Persistent { at }
        } else {
            Class::Diverged {
                at,
                detail: "stream differs from its solo reference".into(),
            }
        };
    }
    // Equal streams with a different server-side count is still a divergence:
    // it means the server accounted for the same text differently.
    if reference.completion_tokens != t.completion_tokens {
        return Class::Diverged {
            at: a.chars().count(),
            detail: format!(
                "identical stream but completion_tokens {} vs {}",
                reference.completion_tokens, t.completion_tokens
            ),
        };
    }
    Class::Identical
}

/// Inputs for one scoring pass. Every field is recorded data; nothing here
/// talks to a server.
pub struct Legs<'a> {
    /// Solo reference, run twice. Index = prompt.
    pub ref_a: &'a [RequestOutcome],
    pub ref_b: &'a [RequestOutcome],
    /// `(label, outcomes)` per concurrency rung.
    pub rungs: &'a [(String, Vec<RequestOutcome>)],
    /// Solo again, after the rungs.
    pub post: &'a [RequestOutcome],
    pub canaries: &'a [String],
    pub min_completion_tokens: usize,
}

pub fn score(legs: &Legs) -> Score {
    let all: Vec<&str> = legs.canaries.iter().map(String::as_str).collect();
    let mut s = Score {
        prompts: legs.ref_a.len(),
        rungs: legs.rungs.len(),
        ..Default::default()
    };

    for i in 0..legs.ref_a.len() {
        let own = legs.canaries.get(i).map(String::as_str).unwrap_or("");
        // The reference itself must be measurable and reproducible, or this
        // prompt cannot speak to contamination at all.
        let (a, b) = (
            legs.ref_a[i].transcript(),
            legs.ref_b.get(i).and_then(RequestOutcome::transcript),
        );
        let reference = match (a, b) {
            (Some(a), Some(b))
                if a.completion_tokens < legs.min_completion_tokens
                    || b.completion_tokens < legs.min_completion_tokens =>
            {
                s.unmeasured += 1;
                s.cells.insert(
                    (i, "ref".into()),
                    Class::Unmeasured {
                        why: format!(
                            "a solo reference was below the {} completion-token floor",
                            legs.min_completion_tokens
                        ),
                    },
                );
                continue;
            }
            (Some(a), Some(b))
                if a.canonical() == b.canonical() && a.completion_tokens == b.completion_tokens =>
            {
                a
            }
            (Some(_), Some(_)) => {
                s.alone_unstable += 1;
                s.cells.insert((i, "ref".into()), Class::AloneUnstable);
                continue;
            }
            _ => {
                s.unmeasured += 1;
                s.cells.insert(
                    (i, "ref".into()),
                    Class::Unmeasured {
                        why: "a solo reference leg failed".into(),
                    },
                );
                continue;
            }
        };

        let mut legs_for_prompt: Vec<(String, &RequestOutcome, bool)> = legs
            .rungs
            .iter()
            .filter_map(|(label, outs)| outs.get(i).map(|o| (label.clone(), o, false)))
            .collect();
        if let Some(p) = legs.post.get(i) {
            legs_for_prompt.push(("post".into(), p, true));
        }

        for (label, got, persistent) in legs_for_prompt {
            let c = classify(
                reference,
                got,
                own,
                &all,
                legs.min_completion_tokens,
                persistent,
            );
            s.compared += 1;
            match &c {
                Class::Identical => {
                    s.identical += 1;
                    s.tokens_compared += reference.completion_tokens;
                }
                Class::Diverged { at, .. } => {
                    s.diverged += 1;
                    s.earliest_divergence =
                        Some(s.earliest_divergence.map_or(*at, |e: usize| e.min(*at)));
                }
                Class::Persistent { at } => {
                    s.persistent += 1;
                    s.earliest_divergence =
                        Some(s.earliest_divergence.map_or(*at, |e: usize| e.min(*at)));
                }
                Class::Contaminated { .. } => {
                    s.contaminated += 1;
                    s.foreign_canaries += 1;
                }
                Class::Unmeasured { .. } => s.unmeasured += 1,
                Class::AloneUnstable => s.alone_unstable += 1,
            }
            s.cells.insert((i, label), c);
        }
    }
    s
}

/// ★ ZERO TOLERANCE, and the argument for it.
///
/// This compares TOKEN IDENTITY, not a timing statistic — there is no clock
/// being read, so there is no noise term to allow for. The one honest
/// counter-argument is batch-width numerics: at C=8 the matmuls take different
/// kernel rungs than at C=1, so an argmax near a tie can flip with no
/// cross-request leak at all. The answer is to CLASSIFY (Diverged vs
/// Contaminated) but FAIL on both — a tolerance would need a principled bound,
/// and "how many flipped tokens is a leak" has no defensible answer.
pub fn verdict(s: &Score) -> crate::result::Verdict {
    use crate::result::Verdict;
    if s.compared == 0 {
        return Verdict::fail("nothing measured: 0 comparisons (every leg failed?)");
    }
    let mut bad = Vec::new();
    if s.unmeasured > 0 {
        bad.push(format!("{} unmeasured", s.unmeasured));
    }
    if s.alone_unstable > 0 {
        bad.push(format!(
            "{} not reproducible ALONE at temp 0 (#435 class — contamination \
             unattributable for them)",
            s.alone_unstable
        ));
    }
    if s.contaminated > 0 {
        bad.push(format!(
            "{} CONTAMINATED ({} foreign canaries)",
            s.contaminated, s.foreign_canaries
        ));
    }
    if s.persistent > 0 {
        bad.push(format!("{} PERSISTENT (survived into solo)", s.persistent));
    }
    if s.diverged > 0 {
        bad.push(format!("{} diverged", s.diverged));
    }
    if bad.is_empty() {
        return Verdict::pass(format!(
            "{} prompts x {} rungs + post-check: all {} streams identical to \
             their solo reference ({} tokens compared)",
            s.prompts, s.rungs, s.compared, s.tokens_compared
        ));
    }
    let where_ = s
        .earliest_divergence
        .map(|c| format!("; earliest divergence at char {c}"))
        .unwrap_or_default();
    Verdict::fail(format!("{}{where_}", bad.join(" · ")))
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod score_tests;
