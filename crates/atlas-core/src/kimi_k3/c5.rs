// SPDX-License-Identifier: AGPL-3.0-only

//! C5: AttnRes mix vs a frozen residual fixture.
//!
//! Self-consistency / fixture gate (not HF token-exact). Frozen skip +
//! completed-block sources and query are mixed with the graph's AttnRes
//! (`attnres_mix` on the 0.40B-pattern tiny graph, hidden=4).

use super::attnres::attnres_mix;
use super::cpu_weights::K3CpuModel;

/// Written in the test file (PRD C5). f32 CPU refs should be tighter.
const ATOL: f32 = 1e-5;
const EPS: f32 = 1e-5;

/// Frozen residual stack (skip = sources[0]) and query. Independent of HF.
const SKIP: [f32; 4] = [1.0, 0.0, -0.5, 2.0];
const BLOCK: [f32; 4] = [0.0, 1.0, 4.0, -3.0];
const QUERY: [f32; 4] = [0.2, -0.1, 0.4, 0.3];
const NORM_W: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// Recorded mix=1 softmax mixture (python3 closed form, f32).
const RECORDED_MIX1: [f32; 4] = [0.571_599_9, 0.428_400_1, 1.427_800_4, -0.142_000_4];

fn sources() -> [Vec<f32>; 2] {
    [SKIP.to_vec(), BLOCK.to_vec()]
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn close(a: &[f32], b: &[f32], atol: f32) -> bool {
    a.len() == b.len() && max_abs(a, b) <= atol
}

fn mix(mix_w: f32) -> Vec<f32> {
    attnres_mix(&sources(), &QUERY, &NORM_W, EPS, mix_w)
}

#[test]
fn c5_attnres_mix_matches_recorded_fixture() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(
        model.graph.hidden,
        SKIP.len(),
        "fixture is tiny-graph hidden"
    );
    assert_eq!(model.graph.attn_res_block_size, 4);
    let got = mix(1.0);
    assert_eq!(got.len(), RECORDED_MIX1.len());
    assert!(
        close(&got, &RECORDED_MIX1, ATOL),
        "C5 mix=1 vs recorded max_abs={} atol={ATOL} got={got:?} want={RECORDED_MIX1:?}",
        max_abs(&got, &RECORDED_MIX1)
    );
}

#[test]
fn c5_zero_mix_weights_diverges_from_recorded() {
    // mix=0 is identity skip; zeroed softmax weights yield the zero vector.
    let recorded = RECORDED_MIX1;
    let mix0 = mix(0.0);
    let zeroed = vec![0.0f32; SKIP.len()];
    assert_eq!(mix0, SKIP, "mix=0 must return the skip source");
    assert!(
        !close(&mix0, &recorded, ATOL),
        "RST known-bad: mix=0 must diverge from recorded mix=1 (max_abs={})",
        max_abs(&mix0, &recorded)
    );
    assert!(
        !close(&zeroed, &recorded, ATOL),
        "RST known-bad: zero mix weights must diverge from recorded mix=1 (max_abs={})",
        max_abs(&zeroed, &recorded)
    );
    let mix1 = mix(1.0);
    assert!(
        !close(&mix0, &mix1, ATOL),
        "RST known-bad: mix=0 vs mix=1 must diverge (max_abs={})",
        max_abs(&mix0, &mix1)
    );
}

// TODO: GPU C5 — AttnRes kernel vs this frozen residual fixture (same atol).
