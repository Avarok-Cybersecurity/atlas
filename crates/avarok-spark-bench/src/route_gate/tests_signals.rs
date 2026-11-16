// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of the float64 signal heads (port of the Python `SignalTests`: probabilities against a
//! hand-written scalar loop, the activation grid, `present` and `expected_probs`, plus the erf series).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde_json::json;

use super::error::GateError;
use super::mat::{Mat, dot};
use super::mt::NumpyRandom;
use super::signals::{
    Activation, PlattOrder, ScaleMode, SignalHead, Thresholds, Variant, erf, expected_probs, grid,
};
use super::tensorio::Tensor;

fn rn(rng: &mut NumpyRandom, n: usize, s: f64) -> Vec<f64> {
    rng.randn_vec(n).into_iter().map(|v| v * s).collect()
}

fn classes() -> Vec<String> {
    vec!["c0".to_string(), "c1".to_string()]
}

/// The toy head: 4 inputs, 3 hidden units, 2 classes.
fn toy_head() -> (BTreeMap<String, Tensor>, SignalHead) {
    let mut rng = NumpyRandom::new(3);
    let mut t: BTreeMap<String, Tensor> = BTreeMap::new();
    let mut put = |name: &str, shape: &[usize], data: Vec<f64>| {
        t.insert(name.to_string(), Tensor::new(shape.to_vec(), data));
    };
    put("mean", &[4], rn(&mut rng, 4, 0.1));
    let scale: Vec<f64> = rng
        .randn_vec(4)
        .into_iter()
        .map(|v| v.abs() + 0.5)
        .collect();
    put("scale", &[4], scale);
    put("net.0.weight", &[3, 4], rn(&mut rng, 12, 1.0));
    put("net.0.bias", &[3], rn(&mut rng, 3, 0.1));
    put("net.2.weight", &[2, 3], rn(&mut rng, 6, 1.0));
    put("net.2.bias", &[2], rn(&mut rng, 2, 0.1));
    put("platt", &[2, 2], vec![1.5, 0.2, 0.8, -0.3]);
    let head = SignalHead::new("toy", &t, classes(), None).unwrap();
    (t, head)
}

fn randn_mat(seed: u64, rows: usize) -> Mat {
    Mat::new(rows, 4, NumpyRandom::new(seed).randn_vec(rows * 4)).unwrap()
}

fn max_abs_diff(a: &Mat, b: &Mat) -> f64 {
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn probs_match_manual() {
    let (t, head) = toy_head();
    let x = randn_mat(4, 3);
    let v = Variant {
        activation: Activation::Relu,
        scale_mode: ScaleMode::Mul,
        platt_order: PlattOrder::Ab,
    };
    let got = head.probs(&x, v).unwrap();
    let (mean, scale) = (&t["mean"].data, &t["scale"].data);
    let (w0, b0, w2, b2, pl) = (
        &t["net.0.weight"].data,
        &t["net.0.bias"].data,
        &t["net.2.weight"].data,
        &t["net.2.bias"].data,
        &t["platt"].data,
    );
    for n in 0..3 {
        let z: Vec<f64> = (0..4).map(|k| (x.at(n, k) - mean[k]) * scale[k]).collect();
        let h: Vec<f64> = (0..3)
            .map(|j| (dot(&z, &w0[j * 4..(j + 1) * 4]) + b0[j]).max(0.0))
            .collect();
        for c in 0..2 {
            let p = 1.0 / (1.0 + (-(dot(&h, &w2[c * 3..(c + 1) * 3]) + b2[c])).exp());
            let (a, b) = (pl[c * 2], pl[c * 2 + 1]);
            let q = p.clamp(1e-6, 1.0 - 1e-6);
            let want = 1.0 / (1.0 + (-(a * (q / (1.0 - q)).ln() + b)).exp());
            assert!(
                (got.at(n, c) - want).abs() < 5e-13,
                "row {n} class {c}: {} vs {want}",
                got.at(n, c)
            );
        }
    }
    let div = head
        .probs(
            &x,
            Variant {
                scale_mode: ScaleMode::Div,
                ..v
            },
        )
        .unwrap();
    assert!(max_abs_diff(&div, &got) > 0.0);
    let swapped = head
        .probs(
            &x,
            Variant {
                platt_order: PlattOrder::Ba,
                ..v
            },
        )
        .unwrap();
    assert!(max_abs_diff(&swapped, &got) > 0.0);
    // the default variant is the first guess of the search: relu, mul, ab
    assert_eq!(Variant::default(), v);
}

#[test]
fn activations_and_grid() {
    let (_, head) = toy_head();
    let x = randn_mat(5, 2);
    let mut seen = BTreeSet::new();
    let variants = grid();
    for v in &variants {
        let pr = head.probs(&x, *v).unwrap();
        assert!(pr.all_finite());
        seen.insert((
            v.activation.name(),
            v.scale_mode.name(),
            v.platt_order.name(),
        ));
    }
    assert_eq!(seen.len(), Activation::ALL.len() * 2 * 2);
    assert_eq!(variants.len(), Activation::ALL.len() * 2 * 2);
    assert!((Activation::Gelu.apply(1.0) - 0.8413447460685429).abs() < 5e-13);
    assert!(Activation::Silu.apply(0.0).abs() < 5e-16);
}

#[test]
fn present_and_expected() {
    let (_, head) = toy_head();
    let mut per_class = HashMap::new();
    per_class.insert("c0".to_string(), 0.5);
    per_class.insert("c1".to_string(), 0.2);
    assert_eq!(
        head.present(&[0.6, 0.1], &Thresholds::PerClass(&per_class)),
        ["c0"]
    );
    assert_eq!(
        head.present(&[0.6, 0.3], &Thresholds::Scalar(0.25)),
        ["c0", "c1"]
    );
    let scope = json!({"scopes": {"conversation": {"probabilities": {"c1": 0.25, "c0": 0.75}}}});
    assert_eq!(expected_probs(&scope, &classes()), Some(vec![0.75, 0.25]));
    assert_eq!(
        expected_probs(&json!({"scopes": {}}), &["c0".to_string()]),
        None
    );
    let bad = SignalHead::new("bad", &BTreeMap::new(), vec!["c0".to_string()], None);
    assert!(matches!(bad, Err(GateError::Value(_))));
}

#[test]
fn erf_series_known_values() {
    assert_eq!(erf(0.0), 0.0);
    assert!((erf(1.0) - 0.8427007929497149).abs() < 5e-15);
    assert!((erf(-0.5) + 0.5204998778130465).abs() < 5e-15);
    assert_eq!(erf(6.0), 1.0);
    assert_eq!(erf(-7.5), -1.0);
}
