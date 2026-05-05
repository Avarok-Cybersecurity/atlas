// SPDX-License-Identifier: AGPL-3.0-only

//! Token sampling helpers (resample + sample + grammar-constrained sample).

use super::*;

/// Re-sample verify tokens from the logits buffer when temperature > 0.
///
/// After `decode_verify_graphed`, the logits buffer still contains valid
/// BF16 logits for each verified position (`[k, vocab_size]`). The CUDA
/// graph bakes in argmax, but when the request has temperature > 0 we need
/// stochastic sampling. This copies the logits to host and samples per
/// position, returning the temperature-sampled tokens.
///
/// Falls back to `argmax_tokens` if the D2H copy fails.
#[allow(dead_code)]
pub fn verify_resample(model: &dyn Model, argmax_tokens: &[u32], temperature: f32) -> Vec<u32> {
    if temperature == 0.0 {
        return argmax_tokens.to_vec();
    }
    let k = argmax_tokens.len();
    let vocab = model.vocab_size();
    let total_bytes = k * vocab * 2;
    let mut buf = vec![0u8; total_bytes];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut buf)
        .is_err()
    {
        return argmax_tokens.to_vec();
    }
    let params = SamplingParams {
        temperature,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        logit_bias: Vec::new(),
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty_window: 0,
        lz_penalty: DEFAULT_LZ_PENALTY,
        edt_strength: 0.0,
        edt_floor: 0.1,
        dry_multiplier: DEFAULT_DRY_MULTIPLIER,
        dry_base: DEFAULT_DRY_BASE,
        dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
        dry_sequence_breakers: Vec::new(),
        max_tokens: 0,
        stop_token_ids: Vec::new(),
        seed: None,
    };
    (0..k)
        .map(|i| {
            let slice = &buf[i * vocab * 2..(i + 1) * vocab * 2];
            sample_with_params(slice, &params)
        })
        .collect()
}

/// Sample one token from device logits, applying temperature/top-k/top-p if non-greedy.
///
/// `suppress_ids`: token IDs to mask to -inf before sampling (e.g. EOS on first token).
pub fn sample_token(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
) -> Result<u32> {
    if temperature == 0.0 && suppress_ids.is_empty() {
        return model.argmax_on_device(logits, 0);
    }
    let vocab_size = model.vocab_size();
    // Read logits from device. Gemma-4 dense single-token decode produces FP32
    // logits via the FP32 lm_head + softcap path (margin between top-1 and
    // top-2 sits on a BF16 representable boundary at value 16-32, so storing
    // BF16 there flips the greedy argmax). Other paths still produce BF16
    // and need expansion. Dispatch by `logits_ptr_is_fp32`.
    let mut f32_logits: Vec<f32> = if model.logits_ptr_is_fp32(logits) {
        let mut buf = vec![0u8; vocab_size * 4];
        model.copy_logits_to_host(logits, &mut buf)?;
        // SAFETY: buf has length vocab_size * 4 and the device kernel wrote
        // little-endian f32 values; reinterpret is byte-equivalent on x86/arm.
        let f32_slice: &[f32] =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, vocab_size) };
        f32_slice.to_vec()
    } else {
        let mut bf16_buf = vec![0u8; vocab_size * 2];
        model.copy_logits_to_host(logits, &mut bf16_buf)?;
        (0..vocab_size)
            .map(|i| {
                let lo = bf16_buf[i * 2];
                let hi = bf16_buf[i * 2 + 1];
                bf16_to_f32(lo, hi)
            })
            .collect()
    };
    // Suppress EOS tokens on first token by setting to -inf.
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    if temperature == 0.0 {
        // Greedy argmax over FP32
        let best = f32_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            edt_strength: 0.0,
            edt_floor: 0.1,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}

/// Sample one token from device logits with optional grammar constraint.
///
/// Like `sample_token` but also applies grammar bitmask when `grammar_state`
/// is provided. Always uses host-side sampling when grammar is active (can't
/// use GPU argmax since grammar bitmask is CPU-side).
pub fn sample_token_with_grammar(
    model: &dyn Model,
    logits: DevicePtr,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    suppress_ids: &[u32],
    grammar_state: Option<&mut GrammarState>,
) -> Result<u32> {
    let Some(gs) = grammar_state else {
        return sample_token(model, logits, temperature, top_k, top_p, suppress_ids);
    };
    let vocab_size = model.vocab_size();
    let mut bf16_buf = vec![0u8; vocab_size * 2];
    model.copy_logits_to_host(logits, &mut bf16_buf)?;
    let mut f32_logits: Vec<f32> = (0..vocab_size)
        .map(|i| {
            let lo = bf16_buf[i * 2];
            let hi = bf16_buf[i * 2 + 1];
            bf16_to_f32(lo, hi)
        })
        .collect();
    for &id in suppress_ids {
        if (id as usize) < vocab_size {
            f32_logits[id as usize] = f32::NEG_INFINITY;
        }
    }
    // Apply grammar bitmask.
    gs.fill_bitmask();
    gs.apply_bitmask_to_logits(&mut f32_logits);
    if temperature == 0.0 {
        let best = f32_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return Ok(best);
    }
    let f32_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, vocab_size * 4) };
    Ok(sample_with_params(
        f32_bytes,
        &SamplingParams {
            temperature,
            top_k,
            top_p,
            top_n_sigma: 0.0,
            min_p: 0.0,
            logit_bias: Vec::new(),
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repetition_penalty_window: 0,
            lz_penalty: DEFAULT_LZ_PENALTY,
            edt_strength: 0.0,
            edt_floor: 0.1,
            dry_multiplier: DEFAULT_DRY_MULTIPLIER,
            dry_base: DEFAULT_DRY_BASE,
            dry_allowed_length: DEFAULT_DRY_ALLOWED_LENGTH,
            dry_sequence_breakers: Vec::new(),
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            seed: None,
        },
    ))
}
