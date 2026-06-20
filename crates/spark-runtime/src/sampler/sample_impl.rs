// SPDX-License-Identifier: AGPL-3.0-only
//
// Core sampling pipeline (`sample_with_params_history` + seeded
// implementation). Split out of `sampler.rs` to keep the parent file
// under the 500-line cap. The parent re-exports these via `pub use`.

use super::{SamplingParams, apply_penalties_and_bias, read_f32, record_entropy};

pub fn sample_with_params_history(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
) -> u32 {
    sample_with_params_seeded(data, params, token_history, params.seed)
}

/// Core sampling pipeline with explicit seed control.
/// `seed` overrides the RNG for deterministic sampling. None = thread_rng.
pub fn sample_with_params_seeded(
    data: &[u8],
    params: &SamplingParams,
    token_history: &[u32],
    seed: Option<u64>,
) -> u32 {
    let n = data.len() / 4;
    let top_k = params.top_k as usize;
    let top_p = params.top_p;
    let top_n_sigma = params.top_n_sigma;
    let min_p = params.min_p;

    // Read raw logits into a mutable vec for in-place modifications.
    // Penalties (repetition / presence / frequency / LZ / DRY) and
    // logit_bias are applied to `raw_logits` BEFORE the greedy bypass
    // below, so they take effect even at `temperature == 0.0`. Atlas
    // previously short-circuited to `argmax(raw_logits)` for greedy,
    // silently dropping caller-configured penalties — the 2026-05-01
    // sweep showed this caused Gemma-4-31B's haiku to enter a
    // repetition loop ("la... la... laaaL!") even with the model's
    // configured `repetition_penalty=1.1` because the harness uses
    // `temperature=0`. HF Transformers, vLLM, and llama.cpp all run
    // LogitsProcessor (penalties + bias) before greedy argmax — Atlas
    // is the outlier here.
    let mut raw_logits: Vec<f32> = (0..n).map(|i| read_f32(data, i)).collect();

    // ── 0. Penalties (repetition / presence / frequency / LZ / DRY) +
    //       logit bias, applied in place via the shared SSOT helper.
    // Identical behavior to the previous inline block; the same helper is
    // now also invoked on the MTP verify + bootstrap paths so all three
    // emit/verify sites apply the same penalties+bias+history.
    apply_penalties_and_bias(&mut raw_logits, params, token_history);

    // ── Greedy bypass (post-penalty argmax) ──
    // At `temperature == 0.0` we return argmax of the penalty/bias-modified
    // logits. We bypass top_n_sigma, temperature scaling, top_k, top_p,
    // min_p — all of those either filter (set values to -inf) or apply
    // monotonic transforms, neither of which can re-order the maximum. Only
    // penalties + logit_bias actually re-order logits, so as long as those
    // ran first, this argmax is correct AND respects caller config.
    if params.temperature <= 0.0 {
        return raw_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let temperature = params.temperature;

    // ── 1. Top-n-sigma: filter noise in logit space (temperature-invariant) ──
    // Keep tokens with logit >= mean - n*sigma. Filters NVFP4 quantization noise.
    if top_n_sigma > 0.0 {
        let sum: f32 = raw_logits.iter().sum();
        let mean = sum / n as f32;
        let var: f32 = raw_logits.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let sigma = var.sqrt();
        if sigma > 0.0 {
            let threshold = mean - top_n_sigma * sigma;
            for logit in raw_logits.iter_mut() {
                if *logit < threshold {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }
    }

    // ── 2. Temperature scaling ──
    let mut logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite()) // Skip -inf tokens from top-n-sigma
        .map(|(i, v)| (i as u32, v / temperature))
        .collect();

    if logits.is_empty() {
        // Fallback: if top-n-sigma filtered everything, use argmax of original
        return raw_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    // ── 3. Sort descending + top-k ──
    logits.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if top_k > 0 && top_k < logits.len() {
        logits.truncate(top_k);
    }

    // ── 4. Softmax ──
    let max_val = logits[0].1;
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .map(|&(idx, logit)| (idx, (logit - max_val).exp()))
        .collect();

    // ── 4b. Entropy: H = -Σ p·ln(p) over the post-softmax distribution ──
    {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            let h: f32 = probs
                .iter()
                .map(|&(_, w)| {
                    let p = w * inv;
                    if p > 1e-10 { -p * p.ln() } else { 0.0 }
                })
                .sum();
            record_entropy(h);
        }
    }

    // ── 5. Min-p: keep tokens with prob >= min_p * max_prob ──
    if min_p > 0.0 {
        let max_prob = probs[0].1; // Already sorted descending
        let threshold = min_p * max_prob;
        probs.retain(|p| p.1 >= threshold);
    }

    // ── 6. Top-p (nucleus) ──
    if top_p < 1.0 {
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        let mut cumsum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &(_, prob)) in probs.iter().enumerate() {
            cumsum += prob / sum;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    // Multinomial sample from the filtered distribution.
    let sum: f32 = probs.iter().map(|p| p.1).sum();
    let random_val: f32 = if let Some(s) = seed {
        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(s);
        rng.r#gen::<f32>()
    } else {
        rand::random::<f32>()
    };
    let threshold = random_val * sum;
    let mut cumsum = 0.0f32;
    for &(idx, prob) in &probs {
        cumsum += prob;
        if cumsum >= threshold {
            return idx;
        }
    }
    probs.last().map_or(0, |p| p.0)
}
