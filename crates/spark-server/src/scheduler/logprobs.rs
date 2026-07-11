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
    // SSOT: the log-softmax + top-k math lives in spark-model
    // (`traits::logprobs`), shared with prompt-logprob collection.
    let (logprob, top) = spark_model::traits::logprob_of(f32_logits, sampled_token, k);
    crate::api::TokenLogprobs {
        token_id: sampled_token,
        logprob,
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
