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
