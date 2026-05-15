// SPDX-License-Identifier: AGPL-3.0-only

// Module is gated by parent's `#[cfg(test)] mod tests;` declaration —
// no inner `#![cfg(test)]` needed (and nesting them is a duplicated
// attribute under recent rustc).

use super::*;

#[test]
fn test_bf16_to_f32() {
    // BF16 for 1.0: 0x3F80 → f32 bits 0x3F800000 = 1.0
    assert_eq!(bf16_to_f32(0x80, 0x3F), 1.0);
    // BF16 for -1.0: 0xBF80 → f32 bits 0xBF800000 = -1.0
    assert_eq!(bf16_to_f32(0x80, 0xBF), -1.0);
    // BF16 for 0.0: 0x0000
    assert_eq!(bf16_to_f32(0x00, 0x00), 0.0);
}

#[test]
fn test_argmax_bf16() {
    // 3 values: 1.0, 2.0, 0.5
    // BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000, BF16(0.5) = 0x3F00
    let data: Vec<u8> = vec![
        0x80, 0x3F, // 1.0
        0x00, 0x40, // 2.0
        0x00, 0x3F, // 0.5
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_argmax_negative() {
    // Values: -1.0, -0.5, -2.0 → argmax should be index 1 (-0.5)
    let data: Vec<u8> = vec![
        0x80, 0xBF, // -1.0
        0x00, 0xBF, // -0.5
        0x00, 0xC0, // -2.0
    ];
    assert_eq!(argmax_bf16(&data), 1);
}

#[test]
fn test_greedy_params() {
    let params = SamplingParams::greedy(100);
    assert!(params.is_greedy());
    assert_eq!(params.max_tokens, 100);
    assert!(params.stop_token_ids.is_empty());
}

#[test]
fn test_argmax_f32() {
    // 3 values: 1.0, 2.0, 0.5 as FP32 little-endian
    let data: Vec<u8> = [1.0f32, 2.0f32, 0.5f32]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    assert_eq!(argmax_f32(&data), 1);
}

#[test]
fn test_sampler_with_mock() {
    use crate::gpu::mock::MockGpuBackend;

    let gpu = MockGpuBackend::new();
    let vocab_size = 4;
    let mut sampler = Sampler::new(vocab_size);

    // Sampler reads BF16 from device (2 bytes/element), not FP32.
    // BF16 encoding: upper 2 bytes of the IEEE 754 FP32 representation.
    // BF16(0.5) = 0x3F00, BF16(3.0) = 0x4040, BF16(1.0) = 0x3F80, BF16(2.0) = 0x4000
    let ptr = gpu.alloc(vocab_size * 2).unwrap();
    let logits: Vec<u8> = vec![
        0x00, 0x3F, // 0.5
        0x40, 0x40, // 3.0
        0x80, 0x3F, // 1.0
        0x00, 0x40, // 2.0
    ];
    gpu.copy_h2d(&logits, ptr).unwrap();

    let params = SamplingParams::greedy(10);
    let token = sampler.sample(ptr, &params, &gpu).unwrap();
    assert_eq!(token, 1); // index 1 = BF16(3.0) = max
}

#[test]
fn test_top_n_sigma_keeps_high_logits() {
    // 5 tokens with moderate spread: [2.0, 1.0, 1.0, 1.0, 1.0]
    // mean = 1.2, sigma ≈ 0.4
    // Correct threshold (mean - 1*sigma) = 0.8 → keeps ALL tokens (all >= 0.8)
    // Bug threshold (mean + 1*sigma) = 1.6 → kills tokens 1-4 (1.0 < 1.6)
    let logits_f32 = [2.0f32, 1.0, 1.0, 1.0, 1.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 1.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        edt_strength: 0.0,
        edt_floor: 0.1,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // With correct threshold (mean - sigma = 0.8), all 5 tokens survive.
    // After softmax at temp=1: P(0)=exp(2)/Z≈0.42, P(1-4)=exp(1)/Z≈0.145 each.
    // With 500 samples, P(never see non-zero) ≈ 0.42^500 ≈ 0. Very reliable.
    let mut saw_non_zero = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token != 0 {
            saw_non_zero = true;
            break;
        }
    }
    assert!(
        saw_non_zero,
        "top_n_sigma=1.0 should not filter tokens above mean-sigma"
    );
}

#[test]
fn test_top_n_sigma_disabled_at_zero() {
    // With top_n_sigma=0.0, no filtering should occur.
    // Use moderate logits so softmax gives reasonable probabilities.
    let logits_f32 = [1.0f32, 1.0, 1.0, 1.0, 1.5];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0, // disabled
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        edt_strength: 0.0,
        edt_floor: 0.1,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // P(token 0-3) = exp(1.0)/Z ≈ 0.19 each, P(token 4) = exp(1.5)/Z ≈ 0.24
    // With 500 samples, P(never see token < 4) ≈ 0.24^500 ≈ 0.
    let mut saw_low = false;
    for _ in 0..500 {
        let token = sample_with_params(&logits, &params);
        if token < 4 {
            saw_low = true;
            break;
        }
    }
    assert!(saw_low, "top_n_sigma=0.0 should not filter any tokens");
}

#[test]
fn test_sample_with_params_seeded_temperature_zero_returns_argmax() {
    // Direct call with temperature=0.0 must NOT divide-by-zero.
    // Should return the argmax of raw logits.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0; // explicit
    for _ in 0..10 {
        assert_eq!(sample_with_params_seeded(&logits, &params, &[], None), 1);
    }
}

#[test]
fn test_greedy_applies_repetition_penalty_before_argmax() {
    // Regression for 2026-05-01 Gemma-4-31B greedy creative collapse.
    // At temperature=0, repetition_penalty MUST shift argmax when the
    // previous-argmax token is in history. Before the fix, greedy
    // bypassed all penalty processing → infinite repetition loops on
    // models with `repetition_penalty=1.1` configured in MODEL.toml.
    //
    // Setup: token 1 has highest raw logit (1.7). With rep_penalty=1.5
    // applied to history [1, 1] (token 1 twice), its logit becomes
    // 1.7 / 1.5 = 1.133, which drops below token 3's 1.2. Argmax
    // should flip from 1 → 3.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    params.repetition_penalty = 1.5;
    let history = vec![1u32, 1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token, 3,
        "rep_penalty must shift greedy argmax away from history-repeated token"
    );

    // Sanity: same prompt without history → original argmax (token 1).
    let token_no_hist = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token_no_hist, 1, "no history → no penalty → raw argmax");

    // Sanity: rep_penalty=1.0 (default) → original argmax even with history.
    params.repetition_penalty = 1.0;
    let token_no_pen = sample_with_params_seeded(&logits, &params, &history, None);
    assert_eq!(
        token_no_pen, 1,
        "rep_penalty=1.0 is no-op even with history"
    );
}

#[test]
fn test_greedy_applies_logit_bias_before_argmax() {
    // logit_bias must shift greedy argmax (e.g. for tool-call grammar).
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 0.0;
    // Bias token 0 by +5.0 → it should become argmax (0.5+5.0 = 5.5 > 1.7)
    params.logit_bias = vec![(0, 5.0)];
    let token = sample_with_params_seeded(&logits, &params, &[], None);
    assert_eq!(token, 0, "logit_bias must shift greedy argmax");
}

#[test]
fn test_sample_with_params_seeded_repetition_penalty_zero_doesnt_div_by_zero() {
    // repetition_penalty=0.0 used to produce inf/0 logits. Now skipped.
    let logits_f32 = [0.5f32, 1.7, 0.3, 1.2];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut params = SamplingParams::greedy(10);
    params.temperature = 1.0;
    params.repetition_penalty = 0.0; // pathological
    // Token 1 was "seen" — under broken code its logit would have been
    // divided by 0, producing inf. With the guard, no penalty applies.
    let history = vec![1u32];
    let token = sample_with_params_seeded(&logits, &params, &history, Some(42));
    // Just assert we got a valid token (no panic, no infinite loop).
    assert!(token < 4);
}

#[test]
fn test_top_n_sigma_filters_extreme_outliers() {
    // Logits: [100.0, -100.0, -100.0, -100.0, -100.0]
    // mean = -60.0, sigma ≈ 80.0
    // threshold at n=1: mean - sigma = -140 → keeps everything
    // threshold at n=0.5: mean - 0.5*sigma = -100 → keeps token 0 only
    // With very tight sigma (n=0.1): mean - 0.1*sigma = -68 → kills tokens 1-4
    let logits_f32 = [100.0f32, -100.0, -100.0, -100.0, -100.0];
    let logits: Vec<u8> = logits_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    let params = SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.1, // tight filter
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        edt_strength: 0.0,
        edt_floor: 0.1,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        max_tokens: 10,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    // Token 0 (100.0) should always be selected since others are far below threshold
    for _ in 0..50 {
        let token = sample_with_params(&logits, &params);
        assert_eq!(
            token, 0,
            "extreme low-logit tokens should be filtered at tight sigma"
        );
    }
}

// ── Leviathan rejection-sampling primitives ────────────────────────

#[test]
fn softmax_token_prob_one_hot() {
    // Very-peaked logits → softmax should be ≈1.0 at the peak.
    let logits = [-1000.0_f32, 0.0, -1000.0, -1000.0];
    let p = super::softmax_token_prob(&logits, 1);
    assert!((p - 1.0).abs() < 1e-6, "one-hot p={p}");
    let p0 = super::softmax_token_prob(&logits, 0);
    // Non-peak logit at -1000 vs peak at 0: exp(-1000) underflows in
    // f32 to exactly 0.0; the assertion is "near zero" not "≤ 0".
    assert!(p0 < 1e-30, "non-peak p0={p0}");
}

#[test]
fn softmax_token_prob_uniform() {
    let logits = [1.5_f32; 16];
    let p = super::softmax_token_prob(&logits, 7);
    // Uniform over 16 → 1/16.
    assert!((p - 1.0 / 16.0).abs() < 1e-6, "uniform p={p}");
}

#[test]
fn softmax_token_prob_sums_to_one() {
    let logits = [0.1_f32, 0.5, -0.3, 2.0, 1.0, -1.5, 0.0, 0.8];
    let sum: f32 = (0..logits.len() as u32)
        .map(|t| super::softmax_token_prob(&logits, t))
        .sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax should sum to 1.0, got {sum}"
    );
}

#[test]
fn softmax_token_prob_out_of_bounds() {
    let logits = [0.0_f32, 1.0, 2.0];
    assert_eq!(super::softmax_token_prob(&logits, 99), 0.0);
}

#[test]
fn sample_excluding_drops_draft_token() {
    // Sharply peaked at token 5: greedy would return 5. With temp=0
    // (post-penalty argmax bypass), excluding 5 must return the
    // second-best (token 2).
    let mut logits = vec![-100.0_f32; 32];
    logits[5] = 10.0;
    logits[2] = 5.0;
    let params = SamplingParams {
        temperature: 0.0, // hits the post-penalty argmax branch
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        repetition_penalty_window: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        lz_penalty: 0.0,
        edt_strength: 0.0,
        edt_floor: 0.1,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_sequence_breakers: Vec::new(),
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: Some(42),
    };
    let token = super::sample_excluding(&logits, &params, &[], 5);
    assert_eq!(token, 2, "expected second-best after excluding peak");
}

#[test]
fn test_apply_dry_penalty_penalises_continuation_not_window_end() {
    // History contains a 3-period repeat: [10, 20, 30, 10, 20, 30].
    // The model is about to emit the 7th token. If it emits 10, the
    // pattern continues into a 4th [10, 20, 30] iteration. DRY must
    // penalise token 10 (the continuation), NOT token 30 (the last
    // token already in the window).
    //
    // Regression test for the original `extend_pos = i + len` bug
    // (penalised history[i + len] = 30) which made DRY silently
    // ineffective for backward-matched repeats — the same bug that
    // let dense Qwen 3.6 27B FP8 loop on CSS-rule output for
    // thousands of tokens despite dry_multiplier=0.8.
    let history: Vec<u32> = vec![10, 20, 30, 10, 20, 30];
    let mut logits = vec![0.0f32; 64];
    super::apply_dry_penalty(&mut logits, &history, 1.0, 2.0, 2, &[]);
    // Token 10 should be heavily penalised (would extend the repeat).
    assert!(
        logits[10] < -1.0,
        "DRY must penalise the continuation token 10, got logits[10]={}",
        logits[10]
    );
    // Token 30 is the LAST token already in history — emitting 30
    // would NOT extend the [10,20,30] pattern (it'd break it), so
    // DRY must not penalise it.
    assert!(
        logits[30] >= 0.0,
        "DRY must not penalise token 30 (already trailing in history); got logits[30]={}",
        logits[30]
    );
    // Token 50 is uninvolved; left alone.
    assert_eq!(logits[50], 0.0);
}

#[test]
fn test_apply_dry_penalty_uses_max_not_cumulative() {
    // History: 10 repetitions of [1, 2, 3]. Token 1 is the continuation token targeted
    // by multiple match positions (one per prior triple ending in '3'). The implementation
    // must apply only the MAX penalty (from the longest match), not the cumulative sum.
    // Regression: additive stacking would overflow f32 for large repeat counts and produce
    // NaN, corrupting the entire logit distribution.
    let history: Vec<u32> = std::iter::repeat([1u32, 2, 3]).take(10).flatten().collect();
    let mut logits = vec![0.0f32; 64];
    super::apply_dry_penalty(&mut logits, &history, 1.0, 2.0, 2, &[]);
    assert!(
        !logits[1].is_nan(),
        "DRY must not produce NaN (overflow from additive stacking); got logits[1]={}",
        logits[1],
    );
    assert!(
        logits[1] < -1.0,
        "Token 1 (continuation of repeat) must be heavily penalised; got {}",
        logits[1],
    );
    // Token 1's penalty must dominate over token 2's (shorter suffix match from same positions).
    assert!(
        logits[1] <= logits[2],
        "Token 1 penalty must be ≥ token 2 penalty; got logits[1]={} logits[2]={}",
        logits[1],
        logits[2],
    );
}

#[test]
fn test_apply_xtc_drops_modal_peak_when_triggered() {
    // probs sorted descending: token 0 has 0.55, token 1 has 0.25,
    // token 2 has 0.15, token 3 has 0.05.
    // With xtc_threshold=0.1, the eligible set is {0, 1, 2}.
    // XTC keeps the LAST of that set (token 2 at 0.15) and drops
    // tokens 0 and 1 (the higher-prob candidates).
    let mut probs = vec![(0u32, 0.55f32), (1, 0.25), (2, 0.15), (3, 0.05)];
    // random < probability → trigger.
    super::apply_xtc(&mut probs, 1.0, 0.1, 0.0);
    // Tokens 0 and 1 must be gone, token 2 (lowest-prob eligible) kept,
    // token 3 (below threshold) kept.
    assert!(probs.iter().all(|p| p.0 != 0), "token 0 should be dropped");
    assert!(probs.iter().all(|p| p.0 != 1), "token 1 should be dropped");
    assert!(probs.iter().any(|p| p.0 == 2), "token 2 (lowest above-thresh) kept");
    assert!(probs.iter().any(|p| p.0 == 3), "token 3 (below-thresh) kept");
}

#[test]
fn test_apply_xtc_noop_when_not_triggered() {
    let original = vec![(0u32, 0.55f32), (1, 0.25), (2, 0.15), (3, 0.05)];
    let mut probs = original.clone();
    // random >= probability → no-op.
    super::apply_xtc(&mut probs, 0.3, 0.1, 0.9);
    assert_eq!(probs, original);
}

#[test]
fn test_apply_xtc_noop_when_only_one_above_threshold() {
    // Only token 0 is above threshold; XTC needs ≥2 to do anything.
    let original = vec![(0u32, 0.95f32), (1, 0.03), (2, 0.02)];
    let mut probs = original.clone();
    super::apply_xtc(&mut probs, 1.0, 0.1, 0.0);
    assert_eq!(probs, original);
}
