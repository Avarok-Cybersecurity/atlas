// SPDX-License-Identifier: AGPL-3.0-only
//
// Core sampling pipeline (`sample_with_params_history` + seeded
// implementation). Split out of `sampler.rs` to keep the parent file
// under the 500-line cap. The parent re-exports these via `pub use`.

use super::{SamplingParams, apply_dry_penalty, apply_lz_penalty, read_f32, record_entropy};

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
                let logit = &mut raw_logits[tid as usize];
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
                raw_logits[tid as usize] -= freq_pen * count as f32 + pres_pen;
            }
        }
    }

    // ── 0c. LZ penalty: penalize tokens that extend repeated n-gram patterns ──
    if params.lz_penalty > 0.0 && token_history.len() >= 4 {
        apply_lz_penalty(&mut raw_logits, token_history, params.lz_penalty);
    }

    // ── 0d. DRY penalty: exponential penalty for extending repeated sequences ──
    if params.dry_multiplier > 0.0 && token_history.len() >= 3 {
        apply_dry_penalty(
            &mut raw_logits,
            token_history,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            &params.dry_sequence_breakers,
        );
    }

    // ── 0b. Logit bias: additive per-token bias ──
    for &(tid, bias) in &params.logit_bias {
        if (tid as usize) < n {
            raw_logits[tid as usize] += bias;
        }
    }

    // ── Greedy bypass (post-penalty argmax) ──
    // At `temperature == 0.0` we return argmax of the penalty/bias-modified
    // logits. We bypass top_n_sigma, EDT, temperature scaling, top_k, top_p,
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

    // ── 1b. EDT — entropy-conditioned dynamic temperature ──
    // arXiv:2403.14541 family. Scale the effective temperature by
    // the normalised entropy of the (post-mask, pre-temperature)
    // distribution: high entropy (uncertain) → close to base T,
    // low entropy (confident) → reduced T toward `edt_floor`. Helps
    // reasoning tasks where the model "knows" tokens (low H) but
    // benefits from sampling diversity at branch points (high H).
    let effective_temperature = if params.edt_strength > 0.0 {
        let mut max_l = f32::NEG_INFINITY;
        let mut count = 0usize;
        for &v in raw_logits.iter() {
            if v.is_finite() {
                count += 1;
                if v > max_l {
                    max_l = v;
                }
            }
        }
        if count >= 2 && max_l.is_finite() {
            // Compute softmax-norm entropy with numerical stability.
            let mut sum_exp = 0.0f32;
            for &v in raw_logits.iter() {
                if v.is_finite() {
                    sum_exp += (v - max_l).exp();
                }
            }
            if sum_exp > 0.0 {
                let mut h = 0.0f32;
                for &v in raw_logits.iter() {
                    if v.is_finite() {
                        let p = (v - max_l).exp() / sum_exp;
                        if p > 1e-10 {
                            h -= p * p.ln();
                        }
                    }
                }
                let h_max = (count as f32).ln().max(1e-6);
                let h_norm = (h / h_max).clamp(0.0, 1.0);
                let scale = h_norm.powf(params.edt_strength);
                (temperature * scale).max(params.edt_floor)
            } else {
                temperature
            }
        } else {
            temperature
        }
    } else {
        temperature
    };

    // ── 2. Temperature scaling ──
    let mut logits: Vec<(u32, f32)> = raw_logits
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite()) // Skip -inf tokens from top-n-sigma
        .map(|(i, v)| (i as u32, v / effective_temperature))
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
