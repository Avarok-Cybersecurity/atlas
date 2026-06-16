// SPDX-License-Identifier: AGPL-3.0-only

//! Token sampling strategies.
//!
//! Phase 1: Greedy argmax (CPU-side D2H + argmax).
//! Future: temperature, top-k, top-p, min-p, repetition penalty.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::gpu::{DevicePtr, GpuBackend};
use anyhow::Result;

// ── Global entropy tracking ──
// Stores the latest per-token entropy and counts of low-entropy streaks.
// AtomicU32 stores f32 bits for lock-free reads.

static LAST_ENTROPY: AtomicU32 = AtomicU32::new(0);
static LOW_ENTROPY_TOKENS: AtomicU64 = AtomicU64::new(0);
static TOTAL_SAMPLED_TOKENS: AtomicU64 = AtomicU64::new(0);

/// Read the most recent per-token entropy (nats).
pub fn last_entropy() -> f32 {
    f32::from_bits(LAST_ENTROPY.load(Ordering::Relaxed))
}

/// Total tokens with entropy < 0.3 (potential degeneration).
pub fn low_entropy_token_count() -> u64 {
    LOW_ENTROPY_TOKENS.load(Ordering::Relaxed)
}

/// Total tokens sampled (for computing low-entropy ratio).
pub fn total_sampled_token_count() -> u64 {
    TOTAL_SAMPLED_TOKENS.load(Ordering::Relaxed)
}

pub(super) fn record_entropy(entropy: f32) {
    LAST_ENTROPY.store(entropy.to_bits(), Ordering::Relaxed);
    TOTAL_SAMPLED_TOKENS.fetch_add(1, Ordering::Relaxed);
    if entropy < 0.3 {
        LOW_ENTROPY_TOKENS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Sampling parameters for a request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature (0.0 = greedy).
    pub temperature: f32,
    /// Top-k: keep only the k highest-probability tokens before sampling.
    /// 0 = disabled (use all tokens).
    pub top_k: u32,
    /// Top-p (nucleus): keep smallest set of tokens whose cumulative probability >= p.
    /// 1.0 = disabled.
    pub top_p: f32,
    /// Top-n-sigma: filter tokens in logit space before temperature scaling.
    /// Keep only tokens with logit >= mean - n*sigma. Temperature-invariant.
    /// 0.0 = disabled. Recommended: 1.0 for NVFP4 models.
    pub top_n_sigma: f32,
    /// Min-p: keep tokens with prob >= min_p * max_prob (post-softmax).
    /// 0.0 = disabled. Recommended: 0.05-0.1.
    pub min_p: f32,
    /// Per-token logit bias: (token_id, bias_value) pairs.
    /// Applied additively to raw logits before any filtering.
    pub logit_bias: Vec<(u32, f32)>,
    /// Repetition penalty: multiply logits of previously-seen tokens.
    /// 1.0 = disabled. Recommended: 1.05-1.1.
    pub repetition_penalty: f32,
    /// Repetition penalty window: only consider the last N tokens.
    /// 0 = full history (default). Recommended: 64 for long-form generation.
    pub repetition_penalty_window: u32,
    /// Presence penalty (OpenAI-style): flat additive penalty for each token that
    /// appeared at least once. Range [-2.0, 2.0], 0.0 = disabled.
    pub presence_penalty: f32,
    /// Frequency penalty (OpenAI-style): additive penalty proportional to occurrence
    /// count. Range [-2.0, 2.0], 0.0 = disabled.
    pub frequency_penalty: f32,
    /// LZ penalty: penalize tokens that extend repeated n-gram patterns.
    /// 0.0 = disabled. 1.0 = moderate (default). Based on arXiv:2504.20131.
    pub lz_penalty: f32,
    /// DRY (Don't Repeat Yourself) penalty multiplier. From llama.cpp.
    /// Uses Z-algorithm O(n) sequence matching with exponential penalty.
    /// 0.0 = disabled. Recommended: 0.8.
    pub dry_multiplier: f32,
    /// DRY penalty base for exponential scaling. penalty = multiplier * base^(match_len - allowed_len).
    /// Recommended: 1.75.
    pub dry_base: f32,
    /// DRY minimum match length before penalty applies. Sequences shorter than this are ignored.
    /// Recommended: 2.
    pub dry_allowed_length: u32,
    /// DRY sequence breaker token IDs. Delimiters (newlines, colons, quotes, braces) that
    /// reset sequence tracking. Critical for JSON/tool call output where structural tokens repeat.
    pub dry_sequence_breakers: Vec<u32>,
    /// Maximum tokens to generate.
    pub max_tokens: usize,
    /// Stop token IDs.
    pub stop_token_ids: Vec<u32>,
    /// Seed for deterministic sampling. When Some, the RNG is seeded with this
    /// value for reproducible output. None = non-deterministic (thread_rng).
    pub seed: Option<u64>,
}

impl SamplingParams {
    /// Greedy sampling with a max token limit.
    pub fn greedy(max_tokens: usize) -> Self {
        Self {
            temperature: 0.0,
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
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_sequence_breakers: Vec::new(),
            max_tokens,
            stop_token_ids: Vec::new(),
            seed: None,
        }
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }
}

/// Sampler that picks tokens from logits.
pub struct Sampler {
    /// Reusable host buffer for BF16 logits D2H copy.
    logits_host: Vec<u8>,
    /// FP32 expanded logits for accurate sampling.
    logits_f32: Vec<f32>,
    /// Vocab size.
    vocab_size: usize,
}

impl Sampler {
    pub fn new(vocab_size: usize) -> Self {
        let logits_host = vec![0u8; vocab_size * 2]; // BF16 from GPU
        let logits_f32 = vec![0.0f32; vocab_size]; // FP32 for sampling
        Self {
            logits_host,
            logits_f32,
            vocab_size,
        }
    }

    /// Copy BF16 logits from GPU, expand to FP32, return FP32 slice.
    fn fetch_logits_f32(&mut self, logits_ptr: DevicePtr, gpu: &dyn GpuBackend) -> Result<&[f32]> {
        let byte_len = self.vocab_size * 2;
        gpu.copy_d2h(logits_ptr, &mut self.logits_host[..byte_len])?;
        // BF16 → FP32 expansion: full precision for sampling
        for i in 0..self.vocab_size {
            self.logits_f32[i] = bf16_to_f32(self.logits_host[i * 2], self.logits_host[i * 2 + 1]);
        }
        Ok(&self.logits_f32[..self.vocab_size])
    }

    /// Sample a token from logits on the GPU.
    ///
    /// `logits_ptr` points to `[vocab_size]` BF16 values on device.
    /// Reads BF16, expands to FP32, then samples with full precision.
    pub fn sample(
        &mut self,
        logits_ptr: DevicePtr,
        params: &SamplingParams,
        gpu: &dyn GpuBackend,
    ) -> Result<u32> {
        if params.is_greedy() {
            // Greedy: BF16 argmax is fine (argmax is robust to BF16 quantization)
            let byte_len = self.vocab_size * 2;
            gpu.copy_d2h(logits_ptr, &mut self.logits_host[..byte_len])?;
            return Ok(argmax_bf16(&self.logits_host[..byte_len]));
        }
        // Stochastic: expand to FP32 for accurate sampling
        let f32_logits = self.fetch_logits_f32(logits_ptr, gpu)?;
        let f32_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(f32_logits.as_ptr() as *const u8, f32_logits.len() * 4)
        };
        Ok(sample_with_params(f32_bytes, params))
    }

    /// Sample a batch of tokens (one per sequence in the batch).
    ///
    /// `logits_ptr` points to [batch_size, vocab_size] BF16 values.
    pub fn sample_batch(
        &mut self,
        logits_ptr: DevicePtr,
        batch_size: usize,
        params: &[&SamplingParams],
        gpu: &dyn GpuBackend,
    ) -> Result<Vec<u32>> {
        let total_bytes = batch_size * self.vocab_size * 2; // BF16
        if self.logits_host.len() < total_bytes {
            self.logits_host.resize(total_bytes, 0);
        }
        gpu.copy_d2h(logits_ptr, &mut self.logits_host[..total_bytes])?;

        let stride_bf16 = self.vocab_size * 2;
        let mut tokens = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let start = i * stride_bf16;
            let end = start + stride_bf16;
            let p = params.get(i).copied().unwrap_or(params[0]);
            tokens.push(if p.is_greedy() {
                argmax_bf16(&self.logits_host[start..end])
            } else {
                // Expand BF16 → FP32 for accurate stochastic sampling
                if self.logits_f32.len() < self.vocab_size {
                    self.logits_f32.resize(self.vocab_size, 0.0);
                }
                for j in 0..self.vocab_size {
                    self.logits_f32[j] = bf16_to_f32(
                        self.logits_host[start + j * 2],
                        self.logits_host[start + j * 2 + 1],
                    );
                }
                let f32_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        self.logits_f32.as_ptr() as *const u8,
                        self.vocab_size * 4,
                    )
                };
                sample_with_params(f32_bytes, p)
            });
        }
        Ok(tokens)
    }
}

mod penalties;
pub use penalties::{apply_dry_penalty, apply_lz_penalty, apply_penalties_and_bias};

mod sample_impl;
pub use sample_impl::{sample_with_params_history, sample_with_params_seeded};

/// Convenience wrapper: sample without token history (no repetition penalty).
pub fn sample_with_params(data: &[u8], params: &SamplingParams) -> u32 {
    sample_with_params_history(data, params, &[])
}

/// Argmax over FP32 values stored as raw bytes (4 bytes per element, little-endian).
pub fn argmax_f32(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(4));
    let n = data.len() / 4;
    if n == 0 {
        return 0;
    }

    let mut best_idx: u32 = 0;
    let mut best_val = read_f32(data, 0);

    for i in 1..n {
        let val = read_f32(data, i);
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Read an f32 value from a byte slice at element index `i`.
#[inline]
pub(super) fn read_f32(data: &[u8], i: usize) -> f32 {
    let off = i * 4;
    let bytes = [data[off], data[off + 1], data[off + 2], data[off + 3]];
    f32::from_le_bytes(bytes)
}

/// Legacy: argmax over BF16 values (still used by argmax_on_device fallback).
pub fn argmax_bf16(data: &[u8]) -> u32 {
    debug_assert!(data.len().is_multiple_of(2));
    let n = data.len() / 2;
    if n == 0 {
        return 0;
    }
    let mut best_idx: u32 = 0;
    let mut best_val = bf16_to_f32(data[0], data[1]);
    for i in 1..n {
        let val = bf16_to_f32(data[i * 2], data[i * 2 + 1]);
        if val > best_val {
            best_val = val;
            best_idx = i as u32;
        }
    }
    best_idx
}

/// Convert BF16 (2 bytes, little-endian) to f32.
#[inline]
fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    let bits = (lo as u32) | ((hi as u32) << 8);
    f32::from_bits(bits << 16)
}

#[cfg(test)]
mod tests;
