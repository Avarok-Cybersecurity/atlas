// SPDX-License-Identifier: AGPL-3.0-only

//! C6: LatentMoE top-k + mix vs frozen gate scores.
//!
//! Self-consistency / fixture gate (not HF token-exact). Uses the tiny
//! 0.40B-pattern graph (layer 1 LatentMoE: 2 routed experts, top-k=1).

use super::cpu_weights::{K3CpuModel, MlpW};
use super::latent_moe::{latent_moe_forward, sigmoid_topk};

/// Written in the test file (PRD C6).
const ATOL: f32 = 1e-5;
const EPS: f32 = 1e-5;

/// Frozen hidden + gate logits. Expert 1 is the true top-1.
const H: [f32; 4] = [1.0, 0.0, 0.0, 0.0];
const GATES: [f32; 2] = [0.0, 4.0];
const BIAS: [f32; 2] = [0.0, 0.0];

/// Recorded routed mix after RMSNorm (python3 closed form, f32).
/// Expert 1 scales w3[0]=2; down/up identity; `latent_moe_use_norm`.
const RECORDED_MIX: [f32; 4] = [1.999_980_4, 0.0, 0.0, 0.0];
const RECORDED_IDS: [usize; 1] = [1];

fn close(a: &[f32], b: &[f32], atol: f32) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= atol)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn first_moe(model: &K3CpuModel) -> &super::cpu_weights::MoeWeights {
    for layer in &model.layers {
        if let MlpW::Moe(w) = &layer.mlp {
            return w;
        }
    }
    panic!("tiny graph has no LatentMoE layer");
}

fn routed(model: &K3CpuModel, logits: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let w = first_moe(model);
    latent_moe_forward(
        &H, &w.down, &w.up, &w.norm, logits, &w.bias, &w.experts, None, &model.moe, EPS,
    )
}

#[test]
fn c6_frozen_gates_topk_and_mix_match_recorded() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.moe.n_routed, 2);
    assert_eq!(model.moe.top_k, 1);
    assert!(model.moe.use_norm);
    assert_eq!(model.graph.layers[1].mlp, super::layer::MlpKind::LatentMoe);

    let w = first_moe(&model);
    let (ids, weights) = sigmoid_topk(&GATES, &BIAS, model.moe.top_k);
    assert_eq!(ids, RECORDED_IDS, "frozen-gate top-k expert ids");
    assert_eq!(weights.len(), 1);
    assert!((weights[0] - 1.0).abs() <= ATOL);

    let (mix, mix_ids) = routed(&model, &GATES);
    assert_eq!(mix_ids, RECORDED_IDS);
    assert_eq!(w.bias, BIAS);
    assert!(
        close(&mix, &RECORDED_MIX, ATOL),
        "C6 mix vs recorded max_abs={} atol={ATOL} got={mix:?} want={RECORDED_MIX:?}",
        max_abs(&mix, &RECORDED_MIX)
    );
}

#[test]
fn c6_force_expert_zero_diverges() {
    let model = K3CpuModel::synthetic_tiny();
    let (clean, ids) = routed(&model, &GATES);
    assert_eq!(ids, RECORDED_IDS);

    // Force expert 0: saturate its gate, zero the rest (C1 Ablation::force_expert).
    let forced_logits = [8.0f32, 0.0];
    let (forced, forced_ids) = routed(&model, &forced_logits);
    assert_eq!(forced_ids, vec![0], "force expert 0 must select expert 0");
    assert_ne!(forced_ids, ids);
    assert!(
        !close(&forced, &clean, ATOL),
        "RST known-bad: force expert 0 must change the mix vs frozen top-k (max_abs={})",
        max_abs(&forced, &clean)
    );
    assert!(
        !close(&forced, &RECORDED_MIX, ATOL),
        "RST known-bad: force expert 0 must miss the recorded mix (max_abs={})",
        max_abs(&forced, &RECORDED_MIX)
    );
}

// TODO: GPU C6 — fused LatentMoE vs this frozen-gate fixture (same ids + atol).
