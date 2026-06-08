// SPDX-License-Identifier: AGPL-3.0-only

//! AdaDec Phase 1 — diagnostic-only Shannon entropy logger.
//!
//! Drops in between [`super::grammar_bitmask::GrammarBitmaskApply`] (which
//! sets non-legal tokens to `-inf` so the entropy reflects what the sampler
//! actually sees) and the downstream argmax / sampler. Gated on the
//! `ATLAS_ADADEC_DIAGNOSTIC` env var:
//!
//!   * `unset` or empty → no-op (zero overhead, no allocations)
//!   * non-empty path   → appends one JSONL record per decode step to
//!                         `<path>/adadec_entropy.jsonl`
//!
//! Record schema (per token):
//! ```json
//! {
//!   "t": 12345,                   // sequence offset (post-prefill)
//!   "h": 0.823,                   // Shannon entropy (nats) over masked dist
//!   "topk_ids":   [27, 9, 4321],  // top-3 grammar-legal token ids
//!   "topk_logits":[ 8.1, 7.6, 7.2],
//!   "thk": false,                 // inside_thinking
//!   "pb":  true,                  // inside_parameter_body
//!   "pbc": 17                     // param_body_chars_emitted
//! }
//! ```
//!
//! Phase 2 (lookahead + rerank) will read these records offline to learn the
//! per-model entropy threshold τ_LM (arXiv:2506.08980 §3.2) and verify the
//! paper's claim that drift positions are high-entropy on our specific model
//! + workload before any sampler-behaviour change ships.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use std::io::Write;
use std::sync::Mutex;

/// Pipeline stage that records Shannon entropy of the masked logit
/// distribution to a JSONL file. Diagnostic-only — never mutates logits.
pub struct AdaDecDiagnostic;

/// Mutex-guarded file appender. `OnceLock` so the file is opened lazily on
/// the first record write rather than at startup (zero cost when the env
/// var is unset).
static APPENDER: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();

fn appender() -> Option<&'static Mutex<std::fs::File>> {
    APPENDER
        .get_or_init(|| {
            let dir = std::env::var("ATLAS_ADADEC_DIAGNOSTIC").ok()?;
            if dir.is_empty() {
                return None;
            }
            std::fs::create_dir_all(&dir).ok()?;
            let path = std::path::Path::new(&dir).join("adadec_entropy.jsonl");
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .ok()?;
            tracing::info!(target: "atlas::adadec", "Appending entropy diagnostics to {}", path.display());
            Some(Mutex::new(f))
        })
        .as_ref()
}

/// Shannon entropy of the softmax distribution over `logits`, computed
/// in numerically-stable log-sum-exp form. Tokens at `-inf` (masked by
/// the grammar bitmask) are skipped — they contribute 0 to entropy and
/// would otherwise produce NaN under the exp.
fn masked_entropy(logits: &[f32]) -> f32 {
    // Pass 1: find max for log-sum-exp stability (skip -inf).
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l.is_finite() && l > max {
            max = l;
        }
    }
    if !max.is_finite() {
        // All masked, or NaN-only — degenerate; report 0 entropy.
        return 0.0;
    }
    // Pass 2: sum of exp(l - max) and sum of l*exp(l - max).
    // Entropy H = log(Z) - (Σ l_i * exp(l_i - max)) / Z   [in nats]
    let mut z: f64 = 0.0;
    let mut weighted_logit_sum: f64 = 0.0;
    for &l in logits {
        if !l.is_finite() {
            continue;
        }
        let e = ((l - max) as f64).exp();
        z += e;
        weighted_logit_sum += (l as f64) * e;
    }
    if z <= 0.0 {
        return 0.0;
    }
    let log_z_plus_max = z.ln() + (max as f64);
    let expectation = weighted_logit_sum / z;
    (log_z_plus_max - expectation) as f32
}

/// Top-K grammar-legal tokens by logit value. Tiny K (≤ 8) so a linear
/// scan is faster than a heap.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() {
            continue;
        }
        if out.len() < k {
            out.push((i as u32, l));
            // Keep sorted descending by logit.
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else if l > out[k - 1].1 {
            out[k - 1] = (i as u32, l);
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    out
}

/// Public entry point usable from the main decode path
/// (`decode_logits_seq::process_seq_logits`) which applies its stages
/// INLINE rather than via [`run_pipeline`]. No-op when the env var is
/// unset; otherwise writes one JSONL record per call.
pub fn log_step(logits: &[f32], seq: &ActiveSeq, path: &'static str) {
    let Some(mtx) = appender() else {
        return;
    };

    let h = masked_entropy(logits);
    let tk = top_k(logits, 3);

    let topk_ids: Vec<u32> = tk.iter().map(|(i, _)| *i).collect();
    let topk_logits: Vec<f32> = tk.iter().map(|(_, l)| *l).collect();

    let record = serde_json::json!({
        "t":    seq.output_tokens.len(),
        "h":    h,
        "topk_ids":    topk_ids,
        "topk_logits": topk_logits,
        "thk":  seq.inside_thinking,
        "pb":   seq.inside_parameter_body,
        "pbc":  seq.param_body_chars_emitted,
        "p":    path,
    });

    if let Ok(mut f) = mtx.lock() {
        let mut line = serde_json::to_string(&record).unwrap_or_default();
        line.push('\n');
        let _ = f.write_all(line.as_bytes());
    }
}

impl LogitsProcessor for AdaDecDiagnostic {
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        _ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        log_step(logits, seq, "verify");
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "adadec_diag"
    }

    fn is_argmax_invariant(&self) -> bool {
        // No logit mutation — argmax is preserved.
        true
    }
}
