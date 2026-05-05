// SPDX-License-Identifier: AGPL-3.0-only

//! Inference engine — generate loop for a single request.
//!
//! Orchestrates prefill → decode → sample loop using the [`Model`] trait.
//! The engine is stateless — each call to [`generate`] creates a fresh
//! sequence, runs inference, and returns output tokens.

use anyhow::Result;
use spark_runtime::sampler::SamplingParams;

use crate::traits::Model;

/// Result of a generate call.
pub struct GenerateResult {
    /// Output tokens (does not include prompt tokens).
    pub output_tokens: Vec<u32>,
    /// Why generation stopped: "stop" (EOS/stop token) or "length" (max_tokens).
    pub finish_reason: String,
}

/// Generate response tokens from a prompt.
///
/// Runs prefill on the prompt tokens, then iteratively decodes up to
/// `params.max_tokens` output tokens, stopping early on EOS or stop tokens.
pub fn generate(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
) -> Result<GenerateResult> {
    let mut seq = model.alloc_sequence()?;
    let stream = 0u64; // Default CUDA stream.

    let result = generate_inner(model, prompt_tokens, params, &mut seq, stream);

    // Always free GPU resources, even on error
    model.free_sequence(&mut seq)?;

    result
}

fn generate_inner(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    seq: &mut crate::traits::SequenceState,
    stream: u64,
) -> Result<GenerateResult> {
    // ── Prefill ──
    let logits_ptr = model.prefill(prompt_tokens, seq, stream)?;
    let first_token = model.argmax_on_device(logits_ptr, stream)?;

    let mut output_tokens = Vec::with_capacity(params.max_tokens);
    output_tokens.push(first_token);

    if params.stop_token_ids.contains(&first_token) {
        return Ok(GenerateResult {
            output_tokens,
            finish_reason: "stop".to_string(),
        });
    }

    // ── Decode loop ──
    for _step in 1..params.max_tokens {
        let last_token = *output_tokens.last().unwrap();
        let logits_ptr = model.decode(last_token, seq, stream)?;
        let token = model.argmax_on_device(logits_ptr, stream)?;

        output_tokens.push(token);

        if params.stop_token_ids.contains(&token) {
            return Ok(GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }
    }

    Ok(GenerateResult {
        output_tokens,
        finish_reason: "length".to_string(),
    })
}

/// Generate response tokens with per-token callback.
///
/// Same as [`generate`] but calls `on_token(token_id)` after each token
/// is produced (including the first token from prefill). The callback
/// is synchronous — designed for the caller to send tokens through a
/// channel without pulling in an async runtime dependency.
pub fn generate_streaming<F>(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    mut on_token: F,
) -> Result<GenerateResult>
where
    F: FnMut(u32),
{
    let mut seq = model.alloc_sequence()?;
    let stream = 0u64;

    let result = generate_streaming_inner(
        model,
        prompt_tokens,
        params,
        &mut on_token,
        &mut seq,
        stream,
    );

    model.free_sequence(&mut seq)?;

    result
}

fn generate_streaming_inner<F>(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    on_token: &mut F,
    seq: &mut crate::traits::SequenceState,
    stream: u64,
) -> Result<GenerateResult>
where
    F: FnMut(u32),
{
    let logits_ptr = model.prefill(prompt_tokens, seq, stream)?;
    let first_token = model.argmax_on_device(logits_ptr, stream)?;

    let mut output_tokens = Vec::with_capacity(params.max_tokens);
    output_tokens.push(first_token);
    on_token(first_token);

    if params.stop_token_ids.contains(&first_token) {
        return Ok(GenerateResult {
            output_tokens,
            finish_reason: "stop".to_string(),
        });
    }

    for _step in 1..params.max_tokens {
        let last_token = *output_tokens.last().unwrap();
        let logits_ptr = model.decode(last_token, seq, stream)?;
        let token = model.argmax_on_device(logits_ptr, stream)?;

        output_tokens.push(token);
        on_token(token);

        if params.stop_token_ids.contains(&token) {
            return Ok(GenerateResult {
                output_tokens,
                finish_reason: "stop".to_string(),
            });
        }
    }

    Ok(GenerateResult {
        output_tokens,
        finish_reason: "length".to_string(),
    })
}

/// Generate with speculative decoding (MTP).
///
/// Delegates the speculative decode loop to `model.generate_speculative()`,
/// which has access to GPU/buffers needed for the MTP proposer.
pub fn generate_speculative(
    model: &dyn Model,
    prompt_tokens: &[u32],
    params: &SamplingParams,
    num_drafts: usize,
) -> Result<GenerateResult> {
    model.generate_speculative(prompt_tokens, params, num_drafts)
}

#[cfg(test)]
mod tests;
