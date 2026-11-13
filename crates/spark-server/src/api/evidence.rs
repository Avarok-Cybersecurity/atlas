// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: request-level evidence on the chat serve path. Lets a
//! client prove "drafted equals serial", "concurrent equals solo" and "resumed
//! equals fresh" without reading logs, by returning fingerprints and the
//! confidence summary in `usage`.
//!
//! Request fields (OpenAI chat surface; all default off, so a response only
//! changes shape for a client that asks):
//!
//! * `atlas_evidence: true` adds `usage.atlas_token_sha` and
//!   `usage.atlas_output_sha256`.
//! * `atlas_confidence: true` adds `usage.atlas_confidence`
//!   (`{mean_top1, min_margin, low_frac, tokens}`, see
//!   `scheduler::conf_summary`).
//! * `atlas_speculative: false` keeps the sequence out of every speculative
//!   step (it rides the existing per-sequence `disable_mtp` switch, which the
//!   MTP gate already honors, and `seq.port.no_drafts`).
//!
//! # Fingerprints
//!
//! * `atlas_token_sha` is the SHA-256 of the VISIBLE generated token ids as
//!   little-endian `u32`, in order (reasoning and content together; the
//!   terminal stop token is excluded in both the blocking and the streaming
//!   path). Lowercase hex. With `n > 1` choices it is the SHA-256 of the
//!   per-choice digests concatenated in index order; with one choice it is
//!   that choice's digest.
//! * `atlas_output_sha256` is the SHA-256 of the SHA-256 of the content bytes
//!   followed by the SHA-256 of the reasoning bytes (UTF-8, raw 32-byte
//!   digests, content first). It does not depend on how the text was cut into
//!   chunks. Tool calls do not contribute. Multi-choice combination as above.
//!
//! The definition is the one in TensorFold `docs/cache-benchmarks.md`
//! (`tools/bench_evidence.py`, Apache-2.0, read for the method only; no code
//! is copied). Compare like with like: a blocking and a streaming request do
//! not see the same stop-token handling, so the identity checks (speculative
//! versus plain, concurrent versus solo) pair requests of the same mode.
//!
//! `prompt_tokens_details.cached_tokens` is already reported on every chat,
//! completions and responses usage block, so nothing is added there.

use sha2::{Digest as _, Sha256};

use crate::ir;
use crate::scheduler::conf_summary::{ConfAcc, ConfPlan, ConfSummary};

pub(crate) mod stream;

#[cfg(test)]
mod tests;

/// A SHA-256 digest.
pub type Digest = [u8; 32];

fn finish(h: Sha256) -> Digest {
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

/// Incremental token-id fingerprint (little-endian `u32`, in order).
#[derive(Clone, Default)]
pub struct TokenHasher(Sha256);

impl TokenHasher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, ids: &[u32]) {
        for id in ids {
            self.0.update(id.to_le_bytes());
        }
    }

    pub fn finish(self) -> Digest {
        finish(self.0)
    }
}

/// Incremental output fingerprint: content and reasoning hashed apart, so the
/// result is independent of chunk boundaries.
#[derive(Clone, Default)]
pub struct OutputHasher {
    content: Sha256,
    reasoning: Sha256,
}

impl OutputHasher {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_content(&mut self, text: &str) {
        self.content.update(text.as_bytes());
    }

    pub fn add_reasoning(&mut self, text: &str) {
        self.reasoning.update(text.as_bytes());
    }

    pub fn finish(self) -> Digest {
        let content = self.content.finalize();
        let reasoning = self.reasoning.finalize();
        let mut h = Sha256::new();
        h.update(content.as_slice());
        h.update(reasoning.as_slice());
        finish(h)
    }
}

/// Fingerprint of a token id list.
pub fn token_sha(ids: &[u32]) -> Digest {
    let mut t = TokenHasher::new();
    t.update(ids);
    t.finish()
}

/// Fingerprint of one choice's text.
pub fn output_sha(content: &str, reasoning: &str) -> Digest {
    let mut o = OutputHasher::new();
    o.add_content(content);
    o.add_reasoning(reasoning);
    o.finish()
}

/// Combine per-choice digests: one choice is its own digest, several are the
/// SHA-256 of the digests concatenated in index order.
pub fn combine(per_choice: &[Digest]) -> Digest {
    match per_choice {
        [one] => *one,
        many => {
            let mut h = Sha256::new();
            for d in many {
                h.update(d);
            }
            finish(h)
        }
    }
}

/// Lowercase hex of a digest.
pub fn to_hex(d: &Digest) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in d {
        s.push(char::from(HEX[usize::from(b >> 4)]));
        s.push(char::from(HEX[usize::from(b & 0x0f)]));
    }
    s
}

/// The Atlas additions to a response's usage block, in IR form (`Copy`, so
/// `ir::Usage` stays `Copy`). Empty unless the request asked for something.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UsageExtras {
    pub token_sha: Option<Digest>,
    pub output_sha: Option<Digest>,
    pub confidence: Option<ConfSummary>,
}

impl UsageExtras {
    /// True when nothing is attached (the wire block then adds no key).
    pub fn is_empty(&self) -> bool {
        self.token_sha.is_none() && self.output_sha.is_none() && self.confidence.is_none()
    }
}

/// What the streaming path needs from one request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct StreamPortOpts {
    /// Keep the sequence out of every speculative step.
    pub no_drafts: bool,
    /// Accumulate the confidence summary (row on, or the request asked).
    pub conf_collect: bool,
    /// Put the summary in the usage block (the request asked).
    pub conf_report: bool,
}

impl StreamPortOpts {
    pub(crate) fn for_request(req: &ir::ChatRequest, plan: &ConfPlan) -> Self {
        Self {
            no_drafts: req.no_drafts,
            conf_collect: plan.collect,
            conf_report: plan.report,
        }
    }
}

/// The usage extras of a finished blocking response. `token_digests` is one
/// digest per choice (only filled when `want_evidence`), `choices` the final
/// choices as the client sees them.
pub(crate) fn blocking_extras(
    want_evidence: bool,
    token_digests: &[Digest],
    choices: &[ir::Choice],
    confidence: Option<ConfSummary>,
) -> UsageExtras {
    let mut x = UsageExtras {
        confidence,
        ..UsageExtras::default()
    };
    if want_evidence {
        x.token_sha = Some(combine(token_digests));
        let outputs: Vec<Digest> = choices
            .iter()
            .map(|c| {
                output_sha(
                    c.content.as_deref().unwrap_or(""),
                    c.reasoning.as_deref().unwrap_or(""),
                )
            })
            .collect();
        x.output_sha = Some(combine(&outputs));
    }
    x
}

/// Trim a blocking choice's logprobs back to what the client asked for after
/// the scheduler extracted more for the confidence summary: nothing asked,
/// nothing returned; `k` asked, at most `k` alternatives per token.
pub(crate) fn trim_choice_logprobs(choice: &mut ir::Choice, client_top_logprobs: Option<u8>) {
    match client_top_logprobs {
        None => choice.logprobs = None,
        Some(k) => {
            if let Some(lp) = choice.logprobs.as_mut() {
                for t in &mut lp.content {
                    t.top.truncate(usize::from(k));
                }
            }
        }
    }
}

/// Close the streaming confidence summary at the terminal delta: record it
/// into the `atlas_conf_*` series and, when the request asked, attach it to
/// the `Finish` usage block.
pub(crate) fn finish_conf(deltas: &mut [ir::StreamDelta], acc: Option<ConfAcc>, report: bool) {
    let Some(summary) = acc.and_then(|a| a.summary()) else {
        return;
    };
    crate::scheduler::conf_summary::record(&summary);
    if report {
        for d in deltas.iter_mut() {
            if let ir::StreamDelta::Finish { usage, .. } = d {
                usage.atlas.confidence = Some(summary);
            }
        }
    }
}
