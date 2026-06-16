// SPDX-License-Identifier: AGPL-3.0-only

//! Logit penalty passes: repetition / presence / frequency / LZ / DRY and
//! per-token logit bias. All applied IN PLACE before the softmax/filter stage.

use super::SamplingParams;

/// LZ penalty: penalize tokens that would extend repeated n-gram patterns
/// in the recent token history. Based on arXiv:2504.20131.
///
/// For each candidate token that appears in the history, check if appending it
/// creates a repeated 3/4/5-gram. Penalize proportional to n-gram length and
/// frequency: `logit -= penalty * (ngram_len - 2) * count`.
pub fn apply_lz_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    use std::collections::HashSet;
    // Window the history to last 256 tokens to avoid penalizing
    // cross-turn structural repetition (e.g., JSON keys in tool calls).
    const LZ_WINDOW: usize = 256;
    let history = if history.len() > LZ_WINDOW {
        &history[history.len() - LZ_WINDOW..]
    } else {
        history
    };
    let n = logits.len();
    // Only check tokens that appear in history (others can't form repeats)
    let token_set: HashSet<u32> = history.iter().copied().collect();
    for &candidate in &token_set {
        if (candidate as usize) >= n {
            continue;
        }
        for ngram_len in 3..=5usize {
            if history.len() < ngram_len {
                continue;
            }
            // The n-gram that would form: history[-(ngram_len-1)..] ++ [candidate]
            let suffix = &history[history.len() - (ngram_len - 1)..];
            let count = history
                .windows(ngram_len)
                .filter(|w| w[..ngram_len - 1] == *suffix && w[ngram_len - 1] == candidate)
                .count();
            if count > 0 {
                logits[candidate as usize] -= penalty * (ngram_len as f32 - 2.0) * count as f32;
            }
        }
    }
}

/// DRY (Don't Repeat Yourself) penalty. Ported from llama.cpp PR #9702.
///
/// Uses suffix matching to find the longest repeated sequence ending at the current
/// position in the token history. For each candidate token, checks if appending it
/// would extend a previously-seen sequence. Applies exponential penalty:
///   `penalty = multiplier * base^(match_length - allowed_length)`
///
/// Sequence breakers (e.g., newlines, quotes, braces) reset tracking, preventing
/// false positives in structured output like JSON tool calls.
pub fn apply_dry_penalty(
    logits: &mut [f32],
    history: &[u32],
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    breakers: &[u32],
) {
    if history.is_empty() || multiplier == 0.0 {
        return;
    }
    let n = logits.len();
    let hist_len = history.len();
    let allowed = allowed_length as usize;

    // Build suffix match table: for each position i in history, find the length
    // of the longest suffix of history[..hist_len] that matches starting at i.
    // This is a simplified Z-function approach.
    let mut match_lengths = vec![0usize; hist_len];
    for i in (0..hist_len.saturating_sub(1)).rev() {
        // Check if history[i] is a sequence breaker — reset match length
        if breakers.contains(&history[i]) {
            match_lengths[i] = 0;
            continue;
        }
        // Match history[i..] against history[hist_len - 1 - k..] for increasing k
        let mut len = 0;
        let mut j = i;
        let mut k = hist_len - 1;
        while j < k && history[j] == history[k] {
            len += 1;
            if breakers.contains(&history[j]) {
                break;
            }
            if j == 0 {
                break;
            }
            j -= 1;
            k -= 1;
        }
        // Correction: we want the match starting at position (i) comparing with the suffix
        // This gives us: if we see history[i..i+len] == history[hist_len-len..hist_len],
        // then the token at history[i+len] (if it existed) would extend the repeat.
        match_lengths[i] = len;
    }

    // For each position where a match of length > allowed was found, the token
    // that FOLLOWS the match in history (history[i - 1] looking backward from the match start)
    // would extend a repeat if generated next. Penalize it.
    #[allow(clippy::needless_range_loop)]
    for i in 0..hist_len.saturating_sub(1) {
        let len = match_lengths[i];
        if len > allowed {
            // The token at history[i + len] (one past the match) would extend the repeat
            let extend_pos = i + len;
            if extend_pos < hist_len {
                let token = history[extend_pos] as usize;
                if token < n {
                    let penalty = multiplier * base.powi((len - allowed) as i32);
                    logits[token] -= penalty;
                }
            }
        }
    }
}

/// Apply repetition / presence / frequency / LZ / DRY penalties and
/// per-token logit bias to `logits` IN PLACE, using `token_history`.
///
/// SSOT for the pre-filter logit-modification block. Extracted verbatim
/// from `sample_with_params_seeded` (the non-MTP sampling path) so the
/// MTP verify path (`verify_pick_with_pipeline`) and bootstrap path
/// (`sample_token_with_grammar`) apply the *same* penalties+bias the
/// non-MTP path does — previously those two paths emitted tokens with no
/// penalties (hardcoded `repetition_penalty=1.0`, empty history), so the
/// configured `repetition_penalty`/`dry_multiplier` from MODEL.toml never
/// reached MTP-emitted tokens and the model degenerated into repeated
/// tool-call argument junk.
///
/// BACKWARD-COMPATIBLE / ADDITIVE: a mathematical no-op when
/// `repetition_penalty == 1.0`, `presence_penalty == 0.0`,
/// `frequency_penalty == 0.0`, `lz_penalty <= 0.0`, `dry_multiplier <= 0.0`
/// and `logit_bias` is empty — every branch below is individually gated on
/// its parameter being non-neutral, so the NVFP4 / Gemma / Mistral presets
/// (which use those neutral values) are byte-for-byte unchanged.
pub fn apply_penalties_and_bias(
    logits: &mut [f32],
    params: &SamplingParams,
    token_history: &[u32],
) {
    let n = logits.len();

    // ── 0. Windowed repetition penalty: penalize recently seen tokens ──
    // Window=0 uses full history; window>0 uses only the last N tokens.
    // Skip when rep_penalty <= 0.0 — the divide at the next branch would
    // produce inf for positive logits and 0 for negative, poisoning the
    // distribution. (Caller intent for 0.0 is unclear; treat as no-op.)
    let rep_penalty = params.repetition_penalty;
    if rep_penalty != 1.0 && rep_penalty > 0.0 && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        for &tid in effective {
            if (tid as usize) < n {
                let logit = &mut logits[tid as usize];
                if *logit > 0.0 {
                    *logit /= rep_penalty;
                } else {
                    *logit *= rep_penalty;
                }
            }
        }
    }

    // ── 0b. OpenAI-style additive penalties (presence + frequency) ──
    // Presence: z'ⱼ = zⱼ − β (flat, if token appeared at all)
    // Frequency: z'ⱼ = zⱼ − α · cⱼ (proportional to occurrence count)
    let freq_pen = params.frequency_penalty;
    let pres_pen = params.presence_penalty;
    if (freq_pen != 0.0 || pres_pen != 0.0) && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        // Count occurrences per token
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &tid in effective {
            *counts.entry(tid).or_insert(0) += 1;
        }
        for (&tid, &count) in &counts {
            if (tid as usize) < n {
                logits[tid as usize] -= freq_pen * count as f32 + pres_pen;
            }
        }
    }

    // ── 0c. LZ penalty: penalize tokens that extend repeated n-gram patterns ──
    if params.lz_penalty > 0.0 && token_history.len() >= 4 {
        apply_lz_penalty(logits, token_history, params.lz_penalty);
    }

    // ── 0d. DRY penalty: exponential penalty for extending repeated sequences ──
    if params.dry_multiplier > 0.0 && token_history.len() >= 3 {
        apply_dry_penalty(
            logits,
            token_history,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            &params.dry_sequence_breakers,
        );
    }

    // ── 0e. Logit bias: additive per-token bias ──
    for &(tid, bias) in &params.logit_bias {
        if (tid as usize) < n {
            logits[tid as usize] += bias;
        }
    }
}
