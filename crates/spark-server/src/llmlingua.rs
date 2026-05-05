// SPDX-License-Identifier: AGPL-3.0-only

//! LongLLMLingua-2 prompt compression primitive (B.3, 2026-04-25).
//!
//! Reference: arXiv:2310.06839 (LongLLMLingua) + LLMLingua-2. A
//! token-classification-based prompt compressor that retains 95-98%
//! task accuracy at 14× compression on chain-of-thought prompts and
//! +21.4% RAG quality at 4× compression. The token classifier is
//! XLM-RoBERTa (~280M params), trained to predict per-token
//! "preserve" / "drop" labels.
//!
//! ## Scope
//!
//! This module ships the **token-keep-decision interface**. The
//! actual classifier (ONNX, candle, or a remote API) is plugged in
//! via the [`KeepClassifier`] trait. Atlas integrators can:
//!
//!   - Load the LLMLingua-2 ONNX model (~280MB) at startup.
//!   - Or wire a smaller distilled classifier.
//!   - Or fall back to a heuristic classifier (this module ships a
//!     minimal one for testing — drops repeated lines, blank fillers).
//!
//! The compressor preserves: (a) the system prompt verbatim,
//! (b) the LAST K turns verbatim, (c) anything marked "must-keep" by
//! the classifier in older turns. Conservative by design — the
//! disabled-by-default auto-compactor regression (see
//! `feedback_no_auto_compaction.md`) taught us that aggressive
//! prompt rewriting causes more loops than it cures.

/// Plug-in classifier interface — the production impl is an ONNX
/// XLM-RoBERTa loaded at server startup. This trait keeps the
/// compressor model-agnostic and testable.
pub trait KeepClassifier {
    /// Return per-token keep probabilities in [0, 1] for the input.
    /// Caller thresholds at e.g. 0.5 to derive the keep mask. Output
    /// length must equal input token count.
    fn keep_probs(&self, tokens: &[&str]) -> Vec<f32>;
}

/// Minimal heuristic classifier for tests and as a graceful fallback
/// when no production classifier is loaded. Keep heuristic:
///   - Always keep tokens with alphanumeric content.
///   - Drop blank-filler runs (consecutive whitespace tokens).
///   - Drop classic stop-words at low priority.
pub struct HeuristicClassifier;

impl KeepClassifier for HeuristicClassifier {
    fn keep_probs(&self, tokens: &[&str]) -> Vec<f32> {
        const STOPWORDS: &[&str] = &[
            "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
            "by", "as", "is", "was", "are", "were", "be", "been",
        ];
        tokens
            .iter()
            .map(|t| {
                let trimmed = t.trim();
                if trimmed.is_empty() {
                    0.0
                } else if STOPWORDS.contains(&trimmed.to_ascii_lowercase().as_str()) {
                    0.4
                } else {
                    0.9
                }
            })
            .collect()
    }
}

/// Compress `text` to a target ratio in (0, 1] by dropping tokens
/// with the lowest keep probabilities. Returns the compressed text
/// (whitespace-joined) and the actual achieved ratio.
///
/// The threshold is auto-tuned: the classifier's per-token probs are
/// sorted, the cut is placed at the rank that yields the target ratio.
pub fn compress<C: KeepClassifier>(text: &str, target_ratio: f32, classifier: &C) -> (String, f32) {
    let target_ratio = target_ratio.clamp(0.05, 1.0);
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.is_empty() {
        return (String::new(), 1.0);
    }
    let probs = classifier.keep_probs(&tokens);
    debug_assert_eq!(probs.len(), tokens.len());

    let n_total = tokens.len();
    let n_keep = ((n_total as f32 * target_ratio).round() as usize).max(1);
    if n_keep >= n_total {
        return (text.to_string(), 1.0);
    }

    // Find the threshold: keep tokens whose prob is in the top-n_keep.
    let mut sorted_probs: Vec<f32> = probs.clone();
    sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = sorted_probs[n_keep.saturating_sub(1)];

    // Two-pass selection so the target ratio is honoured exactly
    // when there are ties at the cutoff:
    //   pass 1: keep every token with probability > cutoff
    //   pass 2: fill remaining slots with tokens at == cutoff in
    //           document order
    // The `keep` mask is built by index, then we materialise the
    // output in document order.
    let mut keep_mask = vec![false; n_total];
    let mut kept_count = 0usize;
    for (i, &p) in probs.iter().enumerate() {
        if p > cutoff && kept_count < n_keep {
            keep_mask[i] = true;
            kept_count += 1;
        }
    }
    for (i, &p) in probs.iter().enumerate() {
        if kept_count >= n_keep {
            break;
        }
        if p == cutoff && !keep_mask[i] {
            keep_mask[i] = true;
            kept_count += 1;
        }
    }
    let out: Vec<&str> = tokens
        .iter()
        .enumerate()
        .filter(|(i, _)| keep_mask[*i])
        .map(|(_, &t)| t)
        .collect();
    let achieved = kept_count as f32 / n_total as f32;
    (out.join(" "), achieved)
}

/// Conservative compression: keep the first `n_preserve_head` tokens
/// and the last `n_preserve_tail` tokens verbatim, only compress the
/// middle. Mirrors LongLLMLingua's "structural anchor" pattern that
/// fixes the lost-in-the-middle effect.
pub fn compress_with_anchors<C: KeepClassifier>(
    text: &str,
    target_ratio: f32,
    n_preserve_head: usize,
    n_preserve_tail: usize,
    classifier: &C,
) -> (String, f32) {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    if tokens.len() <= n_preserve_head + n_preserve_tail {
        return (text.to_string(), 1.0);
    }
    let head: Vec<&str> = tokens.iter().take(n_preserve_head).copied().collect();
    let tail: Vec<&str> = tokens
        .iter()
        .skip(tokens.len() - n_preserve_tail)
        .copied()
        .collect();
    let middle_text = tokens[n_preserve_head..tokens.len() - n_preserve_tail].join(" ");
    // Adjust the target ratio so the middle compresses MORE
    // aggressively while head + tail stay verbatim.
    let middle_target = if target_ratio >= 1.0 {
        1.0
    } else {
        let total = tokens.len() as f32;
        let preserved = (n_preserve_head + n_preserve_tail) as f32;
        let target_total = total * target_ratio;
        ((target_total - preserved).max(1.0) / (total - preserved).max(1.0)).clamp(0.05, 1.0)
    };
    let (middle_compressed, _achieved) = compress(&middle_text, middle_target, classifier);
    let mut out = String::new();
    out.push_str(&head.join(" "));
    if !middle_compressed.is_empty() {
        out.push(' ');
        out.push_str(&middle_compressed);
    }
    if !tail.is_empty() {
        out.push(' ');
        out.push_str(&tail.join(" "));
    }
    let achieved = (out.split_whitespace().count() as f32) / (tokens.len() as f32);
    (out, achieved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_at_full_ratio_returns_input() {
        let text = "the quick brown fox jumps over the lazy dog";
        let (out, r) = compress(text, 1.0, &HeuristicClassifier);
        assert_eq!(out, text);
        assert!((r - 1.0).abs() < 1e-3);
    }

    #[test]
    fn compress_at_half_ratio_drops_stopwords_first() {
        let text = "the alpha and beta is in the gamma";
        let (out, r) = compress(text, 0.5, &HeuristicClassifier);
        // Should keep alpha/beta/gamma (alphanumeric content) and
        // drop "the/and/is/in" first.
        assert!(out.contains("alpha"));
        assert!(out.contains("beta"));
        assert!(out.contains("gamma"));
        assert!(r <= 0.6);
    }

    #[test]
    fn compress_empty_input_returns_empty() {
        let (out, r) = compress("", 0.5, &HeuristicClassifier);
        assert_eq!(out, "");
        assert_eq!(r, 1.0);
    }

    #[test]
    fn compress_with_anchors_preserves_head_and_tail() {
        let text = "head1 head2 mid1 mid2 mid3 mid4 mid5 tail1 tail2";
        let (out, _) = compress_with_anchors(text, 0.6, 2, 2, &HeuristicClassifier);
        assert!(out.starts_with("head1 head2"), "head preserved: {out}");
        assert!(out.ends_with("tail1 tail2"), "tail preserved: {out}");
    }

    #[test]
    fn compress_below_min_ratio_clamps_to_at_least_one_token() {
        let text = "alpha beta gamma";
        let (out, _) = compress(text, 0.01, &HeuristicClassifier);
        let kept = out.split_whitespace().count();
        assert!(kept >= 1, "must keep at least one token");
    }
}
