// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! Layer 0's MoE against the golden: routing weights and indices exactly (the gate is f32 in
//! the reference), the block's FFN output within bf16. Input is the `L0.ffn_in` full capture.

use super::super::testutil::*;
use super::super::{Golden, regen_param};
use super::*;

struct Fx {
    g: Golden,
    regimes: Vec<String>,
    cfg: MoeCfg,
}

fn fx() -> Fx {
    let g = Golden::load();
    let cfg = MoeCfg {
        dim: g.fixture_u64("dim") as usize,
        inter: g.fixture_u64("moe_inter_dim") as usize,
        n_routed: g.fixture_u64("n_routed_experts") as usize,
        topk: g.fixture_u64("n_activated_experts") as usize,
        gate_temp: g.fixture_f64("gate_temp") as f32,
        norm_topk_prob: true,
        route_scale: g.fixture_f64("route_scale") as f32,
        swiglu_limit: g.fixture_f64("swiglu_limit") as f32,
    };
    Fx {
        regimes: g.regimes(),
        cfg,
        g,
    }
}

struct Params {
    gate_w: Vec<f32>,
    gate_bias: Vec<f32>,
    experts: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    shared: (Vec<f32>, Vec<f32>, Vec<f32>),
}

fn params(f: &Fx) -> Params {
    let p = |n: &str| regen_param(&f.g, &format!("layers.0.ffn.{n}"));
    Params {
        gate_w: p("gate.weight"),
        gate_bias: p("gate.bias"),
        experts: (0..f.cfg.n_routed)
            .map(|e| {
                (
                    p(&format!("experts.{e}.w1.weight")),
                    p(&format!("experts.{e}.w2.weight")),
                    p(&format!("experts.{e}.w3.weight")),
                )
            })
            .collect(),
        shared: (
            p("shared_experts.w1.weight"),
            p("shared_experts.w2.weight"),
            p("shared_experts.w3.weight"),
        ),
    }
}

fn weights(p: &Params) -> MoeWeights<'_> {
    MoeWeights {
        gate_w: &p.gate_w,
        gate_bias: &p.gate_bias,
        experts: p
            .experts
            .iter()
            .map(|(a, b, c)| (a.as_slice(), b.as_slice(), c.as_slice()))
            .collect(),
        shared: (&p.shared.0, &p.shared.1, &p.shared.2),
    }
}

#[test]
fn params_have_the_expected_geometry() {
    let f = fx();
    let p = params(&f);
    let (dim, inter, n) = (f.cfg.dim, f.cfg.inter, f.cfg.n_routed);
    assert_eq!((dim, inter, n, f.cfg.topk), (64, 24, 4, 2));
    assert_eq!(p.gate_w.len(), n * dim);
    assert_eq!(p.gate_bias.len(), n);
    for (w1, w2, w3) in &p.experts {
        assert_eq!(
            (w1.len(), w2.len(), w3.len()),
            (inter * dim, dim * inter, inter * dim)
        );
    }
    assert_eq!(p.shared.0.len(), inter * dim);
    assert!((f.cfg.route_scale - 1.5).abs() < 1e-6 && (f.cfg.swiglu_limit - 10.0).abs() < 1e-6);
}

#[test]
fn routing_matches_exactly() {
    let f = fx();
    let p = params(&f);
    let w = weights(&p);
    for r in &f.regimes {
        let x = full_f32(&f.g.tensor(r, "L0.ffn_in"), "L0.ffn_in");
        let tokens = x.len() / f.cfg.dim;
        let (_, wt, idx) = moe(&x, tokens, &w, &f.cfg);
        let gi = f.g.tensor(r, "L0.moe_indices");
        let got_idx: Vec<f64> = idx.iter().map(|&i| i as f64).collect();
        check_capture(&format!("{r}.L0.moe_indices"), &got_idx, &gi, EXACT, EXACT);
        let gw = f.g.tensor(r, "L0.moe_weights");
        check_capture(
            &format!("{r}.L0.moe_weights"),
            &as_f64(&wt),
            &gw,
            F32_TOL,
            F32_TOL,
        );
        // renormalised top-k times route_scale sums to route_scale per token
        for t in 0..tokens {
            let s: f32 = wt[t * f.cfg.topk..(t + 1) * f.cfg.topk].iter().sum();
            assert!(
                (s - f.cfg.route_scale).abs() < 1e-5,
                "{r} token {t}: weights sum {s}"
            );
        }
    }
}

#[test]
fn ffn_output_matches_within_bf16() {
    let f = fx();
    let p = params(&f);
    let w = weights(&p);
    for r in &f.regimes {
        let x = full_f32(&f.g.tensor(r, "L0.ffn_in"), "L0.ffn_in");
        let tokens = x.len() / f.cfg.dim;
        let (y, _, _) = moe(&x, tokens, &w, &f.cfg);
        let gt = f.g.tensor(r, "L0.ffn_out");
        check_capture(
            &format!("{r}.L0.ffn_out"),
            &as_f64(&y),
            &gt,
            bf16_tol(&gt),
            BF16_CK_REL,
        );
    }
}

#[test]
fn shared_expert_alone_is_not_the_answer() {
    // guards against a routed path that silently contributes nothing
    let f = fx();
    let p = params(&f);
    let r = &f.regimes[0];
    let x = full_f32(&f.g.tensor(r, "L0.ffn_in"), "L0.ffn_in");
    let tokens = x.len() / f.cfg.dim;
    let shared = expert(
        &x,
        &p.shared.0,
        &p.shared.1,
        &p.shared.2,
        tokens,
        f.cfg.dim,
        f.cfg.inter,
        f.cfg.swiglu_limit,
        None,
    );
    let gt = f.g.tensor(r, "L0.ffn_out");
    let sample: Vec<f64> = shared
        .iter()
        .step_by(gt.stride)
        .map(|&v| v as f64)
        .collect();
    let diff = sample
        .iter()
        .zip(&gt.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0f64, f64::max);
    assert!(
        diff > bf16_tol(&gt),
        "shared expert alone already matches ffn_out; the routed path is untested"
    );
}
