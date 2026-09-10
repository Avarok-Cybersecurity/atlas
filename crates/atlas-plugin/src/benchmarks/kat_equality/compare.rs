// SPDX-License-Identifier: AGPL-3.0-only

//! The pure core: given what each ORDER produced, decide whether the server
//! is a function of its input.
//!
//! No I/O here — the driver does the requests, this decides what they mean.
//! Every rule below is a total function of `Vec<OrderRun>`, so the whole
//! verdict is testable without a GPU, and the tests can construct the exact
//! failure the gate exists to catch.

use crate::benchmarks::transcript::RequestOutcome;

/// The order a sample set is issued in.
///
/// Deterministic and RNG-free, for the same reason the BFCL draw is: a gate
/// that shuffles randomly cannot be re-run to reproduce a failure, and the
/// first thing anyone does with a failing KAT is re-run it.
///
/// Order 0 is the canonical order — the one every recorded BFCL score was
/// produced with, so a divergence is always reported against the order the
/// baseline used. Order 1 is the REVERSE, which is the strongest single probe
/// available: it maximally changes what preceded each sample, and every
/// channel this gate hunts is a function of what preceded. Further orders are
/// rotations, which vary the neighbourhood without repeating order 1.
pub fn permutation(index: usize, len: usize) -> Vec<usize> {
    match index {
        0 => (0..len).collect(),
        1 => (0..len).rev().collect(),
        k if len == 0 => {
            let _ = k;
            Vec::new()
        }
        k => {
            // Rotate by a stride that is coprime-ish with `len` in practice
            // and never 0 mod len, so no rotation degenerates back to order 0.
            let shift = (k * len / (k + 1)).max(1) % len.max(1);
            let shift = if shift == 0 { 1 } else { shift };
            (0..len).map(|i| (i + shift) % len).collect()
        }
    }
}

/// What one sample produced under one order.
#[derive(Debug, Clone)]
pub struct Observation {
    pub sample_id: String,
    pub outcome: RequestOutcome,
}

/// One full pass over the sample set, in one order.
#[derive(Debug, Clone)]
pub struct OrderRun {
    /// How this order was produced, for the failure message.
    pub label: String,
    /// In ISSUE order, which is what makes the run reproducible; lookups go
    /// through `sample_id`, so the storage order is documentation, not index.
    pub observations: Vec<Observation>,
}

impl OrderRun {
    fn find(&self, sample_id: &str) -> Option<&Observation> {
        self.observations.iter().find(|o| o.sample_id == sample_id)
    }
}

/// What happened to one `sample_id` across every order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleVerdict {
    /// Byte-identical everywhere. The only outcome a KAT may have.
    Equal,
    /// The thing this gate exists to find.
    Diverged {
        other_order: String,
        /// Bytes of `canonical()` that matched before the first difference —
        /// it localises the divergence for a human without dumping both
        /// replies into a verdict string.
        common_prefix: usize,
    },
    /// A request that failed is not evidence of equality OR of inequality.
    /// It gets its own outcome so it can never be counted as agreement.
    Unmeasured(String),
}

/// The reduction the verdict and the report both read.
#[derive(Debug, Clone, Default)]
pub struct Score {
    pub orders: usize,
    pub samples: usize,
    pub equal: usize,
    pub diverged: usize,
    pub unmeasured: usize,
    /// Capped: a verdict string naming 400 samples helps nobody, and the full
    /// list is in the per-sample table.
    pub diverged_ids: Vec<String>,
    /// Replies whose `canonical()` carried no content at all.
    ///
    /// THE VACUITY GUARD. A server that answers every request with an empty
    /// string is perfectly "equal" across orders, and would certify. This
    /// counts those so the verdict can refuse them.
    pub empty_replies: usize,
}

/// How many diverged ids a verdict names before it stops.
const NAMED_DIVERGENCES: usize = 8;

/// Compare every sample across every order against order 0.
///
/// Order 0 is the reference because it is the order the recorded baseline was
/// measured in; "the sharded run disagrees with the whole run" is the finding,
/// not "two arbitrary orders disagree with each other".
pub fn score(runs: &[OrderRun]) -> Score {
    let mut s = Score {
        orders: runs.len(),
        ..Default::default()
    };
    let Some(reference) = runs.first() else {
        return s;
    };
    s.samples = reference.observations.len();
    for obs in &reference.observations {
        match verdict_for(&obs.sample_id, reference, &runs[1..]) {
            SampleVerdict::Equal => s.equal += 1,
            SampleVerdict::Diverged { .. } => {
                s.diverged += 1;
                if s.diverged_ids.len() < NAMED_DIVERGENCES {
                    s.diverged_ids.push(obs.sample_id.clone());
                }
            }
            SampleVerdict::Unmeasured(_) => s.unmeasured += 1,
        }
        if let RequestOutcome::Ok(t) = &obs.outcome
            && t.canonical().chars().all(|c| (c as u32) < 0x20)
        {
            // Only the \u{1}..\u{4} separators survived — nothing was said.
            s.empty_replies += 1;
        }
    }
    s
}

/// One sample's verdict across the non-reference orders.
pub fn verdict_for(sample_id: &str, reference: &OrderRun, others: &[OrderRun]) -> SampleVerdict {
    let Some(base) = reference.find(sample_id) else {
        return SampleVerdict::Unmeasured(format!("{sample_id} missing from the reference order"));
    };
    let RequestOutcome::Ok(base_t) = &base.outcome else {
        let RequestOutcome::Error(e) = &base.outcome else {
            unreachable!("RequestOutcome has two variants")
        };
        return SampleVerdict::Unmeasured(format!("reference order: {e}"));
    };
    let base_c = base_t.canonical();
    for other in others {
        let Some(obs) = other.find(sample_id) else {
            return SampleVerdict::Unmeasured(format!("{sample_id} missing from {}", other.label));
        };
        let RequestOutcome::Ok(t) = &obs.outcome else {
            let RequestOutcome::Error(e) = &obs.outcome else {
                unreachable!("RequestOutcome has two variants")
            };
            return SampleVerdict::Unmeasured(format!("{}: {e}", other.label));
        };
        let c = t.canonical();
        // `completion_tokens` is compared too: a reply can be byte-identical
        // as text while the server disagrees about how many tokens it emitted,
        // and that is a real difference in what ran.
        if c != base_c || t.completion_tokens != base_t.completion_tokens {
            return SampleVerdict::Diverged {
                other_order: other.label.clone(),
                common_prefix: common_prefix_len(&base_c, &c),
            };
        }
    }
    SampleVerdict::Equal
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// The gate's rule, in order. Anything that is not proven equal fails.
pub fn verdict(s: &Score) -> crate::result::Verdict {
    use crate::result::Verdict;
    if s.orders < 2 {
        return Verdict::fail(format!(
            "EQUALITY UNPROVEN: {} order(s) ran. Equality is a claim about TWO orders; \
             one order compares with nothing.",
            s.orders
        ));
    }
    if s.samples == 0 {
        return Verdict::fail(
            "EQUALITY UNPROVEN: the reference order issued no samples, so nothing was compared."
                .to_string(),
        );
    }
    // Before believing any equality result, refuse a run that could not have
    // seen a difference. A server answering every request with nothing is
    // byte-identical across every order and would otherwise certify.
    if s.empty_replies == s.samples {
        return Verdict::fail(format!(
            "EQUALITY VACUOUS: all {} replies were empty, so every order agrees trivially. \
             This proves the harness ran, not that the server is order-independent.",
            s.samples
        ));
    }
    if s.unmeasured > 0 {
        return Verdict::fail(format!(
            "EQUALITY UNPROVEN: {} of {} samples were unmeasured (a failed request is not \
             evidence of agreement).",
            s.unmeasured, s.samples
        ));
    }
    if s.diverged > 0 {
        let named = s.diverged_ids.join(", ");
        let more = s.diverged.saturating_sub(s.diverged_ids.len());
        let tail = if more > 0 {
            format!(" (+{more} more)")
        } else {
            String::new()
        };
        return Verdict::fail(format!(
            "ORDER-DEPENDENT: {} of {} samples changed when the request ORDER changed — \
             {named}{tail}. At temperature 0 a sample's reply must be a function of that \
             sample. A benchmark built on this cannot be sharded, and a score taken under \
             one order does not describe another.",
            s.diverged, s.samples
        ));
    }
    crate::result::Verdict::pass(format!(
        "{} samples byte-identical across {} request orders",
        s.samples, s.orders
    ))
}
