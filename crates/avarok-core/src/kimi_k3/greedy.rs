// SPDX-License-Identifier: AGPL-3.0-only

//! Token-greedy helper for C1: prefill the prompt, then decode one token at a time.

use super::cache::HybridCache;
use super::cpu_forward::{forward_token, logits};
use super::cpu_weights::{Ablation, K3CpuModel};
use super::ops::argmax;

/// Greedy decode. Returns `prompt || new_tokens` (length `prompt.len() + max_new`).
pub fn greedy_decode(
    model: &K3CpuModel,
    prompt: &[u32],
    max_new: usize,
    ablation: Ablation,
) -> Vec<u32> {
    assert!(!prompt.is_empty(), "C1 greedy needs a non-empty prompt");
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let mut tokens = prompt.to_vec();
    let mut h = Vec::new();
    for (pos, &tok) in prompt.iter().enumerate() {
        h = forward_token(model, tok, pos, &mut cache, ablation);
    }
    for _ in 0..max_new {
        let next = argmax(&logits(model, &h));
        tokens.push(next);
        let pos = tokens.len() - 1;
        h = forward_token(model, next, pos, &mut cache, ablation);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_emits_requested_new_tokens() {
        let model = K3CpuModel::synthetic_tiny();
        let out = greedy_decode(&model, &[1, 2], 4, Ablation::default());
        assert_eq!(out.len(), 6);
        assert_eq!(&out[..2], &[1, 2]);
        assert!(out[2..].iter().all(|&t| (t as usize) < model.vocab));
    }

    #[test]
    fn greedy_is_deterministic() {
        let model = K3CpuModel::synthetic_tiny();
        let a = greedy_decode(&model, &[3], 8, Ablation::default());
        let b = greedy_decode(&model, &[3], 8, Ablation::default());
        assert_eq!(a, b);
    }
}
