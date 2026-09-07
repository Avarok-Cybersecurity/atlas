// SPDX-License-Identifier: AGPL-3.0-only

//! How far a restricted serve's output distribution has moved from the full
//! model's, measured on identical positions.
//!
//! ## Why not compare generated text
//!
//! Greedy decoding is a step function. One flipped token diverges everything
//! after it, so string comparison reports a large difference for a small
//! change and cannot distinguish a corrupted answer from an equally good
//! paraphrase. That was measured, not assumed: in the coverage sweep the
//! restricted model answered a dict-comprehension prompt with different
//! example words — `"dict", "comprehension"` for `"python", "code"` — which
//! byte-identity scores as a failure and a reader scores as correct.
//!
//! ## What is measured instead
//!
//! Teacher forcing. Both models are shown the SAME token sequence — a prompt
//! plus the full model's own greedy continuation — and asked what
//! probability they assign at each position. Neither model gets to choose
//! the text, so they cannot drift apart, and every position is comparable.
//!
//! Two numbers come out of that:
//!
//!  * **ΔCE**, the mean increase in negative log-likelihood the restricted
//!    model assigns to the full model's own continuation, in nats per token.
//!    0 means the restricted model finds that continuation exactly as
//!    natural as the full model does. It is continuous, so it ranks
//!    configurations that byte-identity would tie.
//!  * **Top-1 agreement**, the fraction of positions where both models would
//!    have picked the same next token. This is what actually decides whether
//!    generation diverges, and it is the number to quote when someone asks
//!    "will it produce the same output".
//!
//! They answer different questions and both are reported: a configuration
//! can hold top-1 agreement while losing confidence everywhere (ΔCE rises,
//! agreement flat), which predicts fragility under sampling even though
//! greedy output is unchanged.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// One prompt's teacher-forced scores, as captured from the full model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reference {
    pub id: String,
    pub category: String,
    /// The prompt as sent.
    pub prompt: String,
    /// The full model's greedy continuation.
    pub continuation: String,
    /// Per-position log-probability the FULL model assigned, over the
    /// continuation positions only.
    pub logprobs: Vec<f32>,
    /// The full model's argmax token at each of those positions, as the
    /// token STRING the endpoint reported. Compared as strings because the
    /// completions surface reports tokens, not ids, and both sides go
    /// through the same tokenizer on the same text.
    pub argmax: Vec<String>,
}

/// One prompt's measured divergence from the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub id: String,
    pub category: String,
    /// Mean nats/token of extra surprise vs the reference. Positive means
    /// the restricted model finds the full model's continuation less likely.
    pub delta_ce: f64,
    /// Fraction of positions where the restricted model's argmax matches the
    /// full model's.
    pub top1_agreement: f64,
    pub positions: usize,
}

/// Aggregate over a run.
#[derive(Debug, Clone, PartialEq)]
pub struct Fidelity {
    pub delta_ce: f64,
    pub top1_agreement: f64,
    pub positions: usize,
    pub prompts: usize,
    /// `category -> (ΔCE, agreement, prompts)`, so a category-restricted
    /// serve can be judged on the category it was built for AND on the
    /// traffic it was not.
    pub per_category: BTreeMap<String, (f64, f64, usize)>,
    /// The worst prompts by ΔCE — where a mean hides the damage.
    pub worst: Vec<Scored>,
}

/// Extract the continuation-position logprobs and argmax tokens from a
/// `/v1/completions` echo response.
///
/// `prompt_chars` is the byte length of the prompt text; `text_offset` is
/// what separates the echoed prompt from the continuation, which is exact
/// even when the tokenizer merges across the seam.
pub fn extract(
    v: &serde_json::Value,
    prompt_chars: usize,
) -> anyhow::Result<(Vec<f32>, Vec<String>)> {
    let lp = v
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("logprobs"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "no choices[0].logprobs in the response — the endpoint must be called with \
                 echo=true and logprobs=N on /v1/completions"
            )
        })?;
    let offsets: Vec<usize> = lp
        .get("text_offset")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("logprobs has no text_offset"))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let token_lp = lp
        .get("token_logprobs")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow::anyhow!("logprobs has no token_logprobs"))?;
    let tops = lp.get("top_logprobs").and_then(|v| v.as_array());

    let mut out_lp = Vec::new();
    let mut out_argmax = Vec::new();
    for (i, off) in offsets.iter().enumerate() {
        // Positions inside the echoed prompt are context, not the thing being
        // scored; only the continuation is comparable across configurations.
        if *off < prompt_chars {
            continue;
        }
        let Some(l) = token_lp.get(i).and_then(|v| v.as_f64()) else {
            // The first position has no predecessor and reports null.
            continue;
        };
        out_lp.push(l as f32);
        let am = tops
            .and_then(|t| t.get(i))
            .and_then(|m| m.as_object())
            .and_then(|m| {
                m.iter()
                    .max_by(|a, b| {
                        a.1.as_f64()
                            .unwrap_or(f64::MIN)
                            .partial_cmp(&b.1.as_f64().unwrap_or(f64::MIN))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(k, _)| k.clone())
            })
            .unwrap_or_default();
        out_argmax.push(am);
    }
    Ok((out_lp, out_argmax))
}

/// Score one prompt against its reference.
///
/// Returns `None` when the two runs disagree about how many positions the
/// continuation has — that means they tokenized the same text differently,
/// and averaging over mismatched positions would produce a number that looks
/// fine and means nothing.
pub fn score_one(reference: &Reference, lp: &[f32], argmax: &[String]) -> Option<Scored> {
    if lp.len() != reference.logprobs.len() || lp.is_empty() {
        return None;
    }
    let n = lp.len();
    let delta: f64 = reference
        .logprobs
        .iter()
        .zip(lp.iter())
        .map(|(full, restricted)| (*full - *restricted) as f64)
        .sum::<f64>()
        / n as f64;
    let agree = if argmax.len() == reference.argmax.len() && !argmax.is_empty() {
        reference
            .argmax
            .iter()
            .zip(argmax.iter())
            .filter(|(a, b)| a == b)
            .count() as f64
            / argmax.len() as f64
    } else {
        f64::NAN
    };
    Some(Scored {
        id: reference.id.clone(),
        category: reference.category.clone(),
        delta_ce: delta,
        top1_agreement: agree,
        positions: n,
    })
}

/// Aggregate per-prompt scores.
///
/// Position-weighted, not prompt-weighted: a 60-token continuation carries
/// more evidence than a 6-token one, and averaging the per-prompt means would
/// let the short ones dominate.
pub fn aggregate(scored: &[Scored]) -> Option<Fidelity> {
    if scored.is_empty() {
        return None;
    }
    let total: usize = scored.iter().map(|s| s.positions).sum();
    if total == 0 {
        return None;
    }
    let w = |f: &dyn Fn(&Scored) -> f64| -> f64 {
        scored
            .iter()
            .filter(|s| !f(s).is_nan())
            .map(|s| f(s) * s.positions as f64)
            .sum::<f64>()
            / scored
                .iter()
                .filter(|s| !f(s).is_nan())
                .map(|s| s.positions as f64)
                .sum::<f64>()
                .max(1.0)
    };
    let mut per_category: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    let cats: std::collections::BTreeSet<&str> =
        scored.iter().map(|s| s.category.as_str()).collect();
    for c in cats {
        let rows: Vec<Scored> = scored.iter().filter(|s| s.category == c).cloned().collect();
        let n: usize = rows.iter().map(|s| s.positions).sum();
        let ce = rows
            .iter()
            .map(|s| s.delta_ce * s.positions as f64)
            .sum::<f64>()
            / n.max(1) as f64;
        let ag = rows
            .iter()
            .filter(|s| !s.top1_agreement.is_nan())
            .map(|s| s.top1_agreement * s.positions as f64)
            .sum::<f64>()
            / n.max(1) as f64;
        per_category.insert(c.to_string(), (ce, ag, rows.len()));
    }
    let mut worst = scored.to_vec();
    worst.sort_by(|a, b| {
        b.delta_ce
            .partial_cmp(&a.delta_ce)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    worst.truncate(5);

    Some(Fidelity {
        delta_ce: w(&|s: &Scored| s.delta_ce),
        top1_agreement: w(&|s: &Scored| s.top1_agreement),
        positions: total,
        prompts: scored.len(),
        per_category,
        worst,
    })
}

#[cfg(test)]
#[path = "score_tests.rs"]
mod tests;
