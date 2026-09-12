// SPDX-License-Identifier: AGPL-3.0-only

//! C2: prefill-then-decode logits vs full-prefill at the last position.

use super::cache::HybridCache;
use super::cpu_forward::{forward_token, logits};
use super::cpu_weights::{Ablation, K3CpuModel};
use super::ops::argmax;

/// Written in the test file (PRD C2). f32 CPU refs should be tighter; this
/// is the bound the instrument actually checks.
const ATOL: f32 = 1e-5;
const RTOL: f32 = 1e-5;

fn close(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= atol + rtol * y.abs())
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn prefill(
    model: &K3CpuModel,
    tokens: &[u32],
    cache: &mut HybridCache,
    ablation: Ablation,
) -> Vec<f32> {
    let mut h = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        h = forward_token(model, tok, pos, cache, ablation);
    }
    h
}

fn last_logits(model: &K3CpuModel, tokens: &[u32], ablation: Ablation) -> Vec<f32> {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    logits(model, &prefill(model, tokens, &mut cache, ablation))
}

/// Prefill `prompt`, then one decode step for `next`. Logits after that
/// step vs a fresh full prefill of `prompt || next`.
fn decode_step_logits(
    model: &K3CpuModel,
    prompt: &[u32],
    next: u32,
    ablation: Ablation,
) -> Vec<f32> {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let _ = prefill(model, prompt, &mut cache, Ablation::default());
    let h = forward_token(model, next, prompt.len(), &mut cache, ablation);
    logits(model, &h)
}

fn c2_pair(model: &K3CpuModel, prompt: &[u32]) -> (Vec<f32>, Vec<f32>, u32) {
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let h = prefill(model, prompt, &mut cache, Ablation::default());
    let next = argmax(&logits(model, &h));
    let decode = decode_step_logits(model, prompt, next, Ablation::default());
    let mut full_tokens = prompt.to_vec();
    full_tokens.push(next);
    let full = last_logits(model, &full_tokens, Ablation::default());
    (decode, full, next)
}

fn assert_c2(model: &K3CpuModel, prompt: &[u32], label: &str) {
    let (decode, full, next) = c2_pair(model, prompt);
    assert_eq!(decode.len(), full.len(), "C2 {label} vocab");
    assert!(
        close(&decode, &full, ATOL, RTOL),
        "C2 {label} prefill-decode vs full-prefill logits (next={next}) max_abs={} atol={ATOL} rtol={RTOL}",
        max_abs(&decode, &full)
    );
}

#[test]
fn c2_prefill_decode_logits_match_full_prefill_tiny() {
    assert_c2(&K3CpuModel::synthetic_tiny(), &[1, 2, 3, 4], "tiny");
}

#[test]
fn c2_prefill_decode_logits_match_full_prefill_small() {
    assert_c2(&K3CpuModel::synthetic_small(), &[1, 2, 3, 4, 5], "small");
}

#[test]
fn c2_skip_layer_on_decode_path_diverges() {
    let model = K3CpuModel::synthetic_tiny();
    let prompt = [1u32, 2, 3, 4];
    let mut cache = HybridCache::from_graph(&model.graph, &model.kda);
    let h = prefill(&model, &prompt, &mut cache, Ablation::default());
    let next = argmax(&logits(&model, &h));

    let skip = Ablation {
        skip_layer: Some(0),
        ..Ablation::default()
    };
    let decode_skip = decode_step_logits(&model, &prompt, next, skip);
    let mut full_tokens = prompt.to_vec();
    full_tokens.push(next);
    let full = last_logits(&model, &full_tokens, Ablation::default());

    assert!(
        !close(&decode_skip, &full, ATOL, RTOL),
        "RST known-bad: skip layer 0 on decode only must diverge from full prefill (max_abs={})",
        max_abs(&decode_skip, &full)
    );
}

// TODO: GPU C2 — same last-position check on KDA/MLA kernels + paged hybrid cache.
