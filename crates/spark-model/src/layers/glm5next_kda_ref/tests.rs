// SPDX-License-Identifier: AGPL-3.0-only

//! Slice 2B/2C numeric gate: every KDA sub-op checked independently against HuggingFace
//! `transformers` 5.16.1 goldens.
//!
//! Goldens: `kda_golden.json`, produced by `gen_kda_golden.py` running the real HF module inside
//! `glm53:sm121-v8-fp8only` (torch 2.13.0). Regenerate with that script; the fixture is RNG-free
//! and every input tensor is stored alongside the outputs, so these tests never re-derive an input.

use super::*;
use serde_json::Value;

const GOLDEN: &str = include_str!("kda_golden.json");

/// FP32 elementwise tolerance. Ops that are a single pass over the data hold far tighter than
/// this; the chunked prefill path accumulates over a different order than the recurrent one, so it
/// gets its own looser bound at the call site.
const TOL: f32 = 1e-6;

struct Golden(Value);

impl Golden {
    fn load() -> Self {
        Golden(serde_json::from_str(GOLDEN).expect("kda_golden.json parses"))
    }

    fn get(&self, section: &str, name: &str) -> Vec<f32> {
        self.0[section][name]["data"]
            .as_array()
            .unwrap_or_else(|| panic!("missing {section}.{name}"))
            .iter()
            .map(|v| v.as_f64().expect("numeric") as f32)
            .collect()
    }

    fn dims(&self) -> KdaDims {
        let f = &self.0["fixture"];
        KdaDims {
            hidden: f["hidden"].as_u64().unwrap() as usize,
            heads: f["heads"].as_u64().unwrap() as usize,
            head_dim: f["head_dim"].as_u64().unwrap() as usize,
            tokens: f["tokens"].as_u64().unwrap() as usize,
        }
    }

    fn scalar(&self, name: &str) -> f32 {
        self.0["fixture"][name].as_f64().unwrap() as f32
    }
}

#[track_caller]
fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f32;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max abs diff {worst:e} > {tol:e} at index {at} (got {}, want {})",
        got[at],
        want[at]
    );
    eprintln!("{what}: max abs diff {worst:e} (tol {tol:e})");
}

// ---------------------------------------------------------------- 1. q/k L2 normalisation

#[test]
fn sub_op_1_qk_l2norm() {
    let g = Golden::load();
    let d = g.dims().head_dim;
    assert_close(
        "q_l2",
        &l2norm_rows(&g.get("inputs", "q_in"), d, 1e-6),
        &g.get("outputs", "q_l2"),
        TOL,
    );
    assert_close(
        "k_l2",
        &l2norm_rows(&g.get("inputs", "k_in"), d, 1e-6),
        &g.get("outputs", "k_l2"),
        TOL,
    );
}

/// The eps goes *inside* the sqrt. Guard against a future "fix" to `x / max(norm, eps)`, which
/// agrees on well-conditioned rows and silently diverges on small-norm ones.
#[test]
fn sub_op_1_l2norm_eps_is_inside_the_sqrt() {
    let x = [1e-4f32, 0.0, 0.0, 0.0];
    let got = l2norm_rows(&x, 4, 1e-6)[0];
    let inside = 1e-4 / (1e-8f32 + 1e-6).sqrt();
    let clamped = 1e-4 / 1e-4f32.max(1e-6);
    assert!(
        (got - inside).abs() < 1e-6,
        "expected inside-sqrt form, got {got}"
    );
    assert!(
        (got - clamped).abs() > 0.5,
        "the two forms must be distinguishable here"
    );
}

// ---------------------------------------------------------------- 2. q scale

#[test]
fn sub_op_2_q_scale() {
    let g = Golden::load();
    let d = g.dims().head_dim;
    let scale = 1.0 / (d as f32).sqrt();
    let scaled: Vec<f32> = g.get("outputs", "q_l2").iter().map(|v| v * scale).collect();
    assert_close("q_scaled", &scaled, &g.get("outputs", "q_scaled"), TOL);
}

// ---------------------------------------------------------------- 3. f_a/f_b low-rank decay

#[test]
fn sub_op_3_lowrank_decay_projection() {
    let g = Golden::load();
    let d = g.dims();
    let f_a = linear(
        &g.get("inputs", "hidden_states"),
        d.tokens,
        d.hidden,
        &g.get("inputs", "W_f_a"),
        d.head_dim,
    );
    let g_lr = linear(
        &f_a,
        d.tokens,
        d.head_dim,
        &g.get("inputs", "W_f_b"),
        d.heads * d.head_dim,
    );
    assert_close("g_lowrank", &g_lr, &g.get("outputs", "g_lowrank"), TOL);
}

// ---------------------------------------------------------------- 4. per-channel dt_bias

/// `dt_bias` has length `H * head_dim`, not `H`. This is the load-bearing shape difference against
/// Atlas's Qwen GDN, where `dt_bias` is one scalar per head.
#[test]
fn sub_op_4_dt_bias_is_per_channel() {
    let g = Golden::load();
    let d = g.dims();
    let dt = g.get("inputs", "dt_bias");
    assert_eq!(
        dt.len(),
        d.heads * d.head_dim,
        "dt_bias must be per-channel"
    );
    assert_eq!(
        g.get("inputs", "A_log").len(),
        d.heads,
        "A_log must be per-head"
    );

    let biased: Vec<f32> = g
        .get("outputs", "g_lowrank")
        .iter()
        .enumerate()
        .map(|(i, v)| v + dt[i % (d.heads * d.head_dim)])
        .collect();
    assert_close("g_biased", &biased, &g.get("outputs", "g_biased"), TOL);
}

// ---------------------------------------------------------------- 5. exp(A_log)

#[test]
fn sub_op_5_exp_a_log() {
    let g = Golden::load();
    let decay: Vec<f32> = g.get("inputs", "A_log").iter().map(|v| v.exp()).collect();
    assert_close("decay", &decay, &g.get("outputs", "decay"), TOL);
}

// ---------------------------------------------------------------- 6. bounded gate

#[test]
fn sub_op_6_bounded_gate() {
    let g = Golden::load();
    let d = g.dims();
    let gate = bounded_gate(
        &g.get("outputs", "g_lowrank"),
        &g.get("inputs", "dt_bias"),
        &g.get("inputs", "A_log"),
        d,
        g.scalar("lower_bound"),
    );
    assert_close("gate", &gate, &g.get("outputs", "gate"), TOL);
}

/// The bounded gate saturates at `lower_bound`; the Qwen GDN law does not. Substituting one for the
/// other would not crash — it would silently change decay dynamics. This pins the divergence.
#[test]
fn sub_op_6_bounded_and_unbounded_laws_differ() {
    let (a_log, dt) = (0.3f32, 0.1f32);
    for &raw in &[-8.0f32, -1.0, 0.0, 1.0, 8.0] {
        let bounded = -5.0 * sigmoid(a_log.exp() * (raw + dt));
        let unbounded = unbounded_gdn_gate(raw, dt, a_log);
        assert!(
            bounded >= -5.0,
            "bounded gate must not exceed lower_bound, got {bounded}"
        );
        if raw >= 1.0 {
            assert!(
                (bounded - unbounded).abs() > 0.5,
                "laws must diverge at raw={raw}: bounded {bounded}, unbounded {unbounded}"
            );
        }
    }
    // Saturation: the unbounded law grows without limit, the bounded one cannot.
    assert!(unbounded_gdn_gate(50.0, 0.0, 0.3) < -50.0);
    assert!(-5.0 * sigmoid(0.3f32.exp() * 50.0) > -5.001);
}

// ---------------------------------------------------------------- 7. beta

#[test]
fn sub_op_7_beta_sigmoid() {
    let g = Golden::load();
    let d = g.dims();
    let beta: Vec<f32> = linear(
        &g.get("inputs", "hidden_states"),
        d.tokens,
        d.hidden,
        &g.get("inputs", "W_b"),
        d.heads,
    )
    .iter()
    .map(|x| sigmoid(*x))
    .collect();
    assert_eq!(
        beta.len(),
        d.tokens * d.heads,
        "beta is per-head, not per-channel"
    );
    assert_close("beta", &beta, &g.get("outputs", "beta"), TOL);
}

#[path = "tests_recurrence.rs"]
mod tests_recurrence;
