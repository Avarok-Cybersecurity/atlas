// SPDX-License-Identifier: AGPL-3.0-only

//! Logprobs extraction helpers.

use super::*;

/// Extract top-K logprobs from an FP32 logits slice for one token position.
///
/// Computes log-softmax over the logits, extracts the logprob of the sampled
/// token, and returns the top-K alternatives sorted descending by logprob.
pub fn extract_logprobs_from_f32(
    f32_logits: &[f32],
    sampled_token: u32,
    k: usize,
) -> crate::api::TokenLogprobs {
    // Log-softmax: logprob = logit - log(sum(exp(logits)))
    let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = max_logit
        + f32_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln();
    let sampled_logprob = if (sampled_token as usize) < f32_logits.len() {
        f32_logits[sampled_token as usize] - log_sum_exp
    } else {
        f32::NEG_INFINITY
    };
    // Find top-K by partial sort.
    let mut indexed: Vec<(u32, f32)> = f32_logits
        .iter()
        .enumerate()
        .map(|(j, &l)| (j as u32, l - log_sum_exp))
        .collect();
    let nth = k.min(indexed.len().saturating_sub(1));
    indexed.select_nth_unstable_by(nth, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> = indexed[..k.min(indexed.len())].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    crate::api::TokenLogprobs {
        token_id: sampled_token,
        logprob: sampled_logprob,
        top,
    }
}

/// Extract logprobs for K positions from the BF16 logits buffer on GPU.
///
/// Copies `[K, vocab_size]` BF16 logits D2H, converts to FP32, and extracts
/// top-K logprobs per position. Returns empty Vec on copy failure.
pub fn extract_verify_logprobs(
    model: &dyn Model,
    tokens: &[u32],
    k_logprobs: u8,
) -> Vec<crate::api::TokenLogprobs> {
    let k = tokens.len();
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; k * vocab * 2]; // BF16
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        return Vec::new();
    }
    tokens
        .iter()
        .enumerate()
        .map(|(i, &tok)| {
            let slice = &buf[i * vocab * 2..(i + 1) * vocab * 2];
            // BF16 → FP32 expansion
            let f32_logits: Vec<f32> = (0..vocab)
                .map(|j| {
                    let lo = slice[j * 2];
                    let hi = slice[j * 2 + 1];
                    bf16_to_f32(lo, hi)
                })
                .collect();
            extract_logprobs_from_f32(&f32_logits, tok, k_logprobs as usize)
        })
        .collect()
}

/// Extract logprobs for a single token from the BF16 logits buffer on GPU.
///
/// Copies `[1, vocab_size]` BF16 logits D2H, converts to FP32, and extracts
/// top-K logprobs. Returns None on copy failure.
pub fn extract_single_logprobs(
    model: &dyn Model,
    logits: DevicePtr,
    sampled_token: u32,
    k_logprobs: u8,
) -> Option<crate::api::TokenLogprobs> {
    let vocab = model.vocab_size();
    let mut buf = vec![0u8; vocab * 2]; // BF16
    if model.copy_logits_to_host(logits, &mut buf).is_err() {
        return None;
    }
    let f32_logits: Vec<f32> = (0..vocab)
        .map(|j| {
            let lo = buf[j * 2];
            let hi = buf[j * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    Some(extract_logprobs_from_f32(
        &f32_logits,
        sampled_token,
        k_logprobs as usize,
    ))
}
