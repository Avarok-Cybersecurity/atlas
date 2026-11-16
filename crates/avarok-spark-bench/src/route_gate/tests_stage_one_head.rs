// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of the Stage-1 head as a whole (port of the Python `StageOneTests` forward-shape,
//! usage-clamp, refusal and release-loader cases), on the toy release fixture of `tests_stage_one`.

use std::collections::BTreeMap;

use serde_json::json;

use super::error::GateError;
use super::mat::Mat;
use super::stage_one::{Candidate, StageOne, release_sha256};
use super::tensorio::{OutTensor, Tensor, write_safetensors};
use super::tests_stage_one::{
    C, D, H, L, M, REGIMES, assert_close, head, randn_mat, regime_names, tiny_stage_one,
};

#[test]
fn forward_shapes_batch_invariance_and_usage_bounds() {
    let t = tiny_stage_one(9, true);
    let h = head(&t);
    let n = 5;
    let x = randn_mat(10, n);
    let out = h.forward(&x, &["r0"; 5]).unwrap();
    assert_eq!((out.rows, out.n, out.c, out.l), (1, n, C, L));
    assert_eq!(out.p.len(), n * C);
    assert_eq!(out.usage.len(), n * C * L);
    assert!(out.p.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert!(out.usage.iter().all(|v| v.is_finite()));
    assert!(out.usage.iter().all(|&v| v >= 0.0));

    // batch invariance: row 2 alone gives the same answer as row 2 of the batch
    let one = Mat::new(1, D, x.row(2).to_vec()).unwrap();
    let o2 = h.forward(&one, &["r0"]).unwrap();
    for ci in 0..C {
        assert_close(o2.p[ci], out.p[2 * C + ci], 1e-9, 1e-12);
        for k in 0..L {
            assert_close(
                o2.usage[ci * L + k],
                out.usage[(2 * C + ci) * L + k],
                1e-9,
                1e-12,
            );
        }
    }

    // a regime per row: row 0 uses r0 in both calls, and a different regime changes the answer
    let pm = h.forward(&x, &["r0", "r1", "r0", "r1", "r0"]).unwrap();
    for ci in 0..C {
        assert_close(pm.p[ci], out.p[ci], 1e-12, 0.0);
    }
    let max_diff = (0..C)
        .map(|ci| (pm.p[C + ci] - out.p[C + ci]).abs())
        .fold(0.0, f64::max);
    assert!(max_diff > 0.0);
}

#[test]
fn usage_clamp_and_not_ok() {
    let mut t = tiny_stage_one(11, true);
    t.insert(
        "w_usage".to_string(),
        Tensor::new(vec![M, C, L, D], vec![0.0; M * C * L * D]),
    );
    t.insert(
        "b_usage".to_string(),
        Tensor::new(vec![M, C, L], vec![50.0; M * C * L]),
    ); // exp(50) is enormous
    t.insert(
        "high".to_string(),
        Tensor::new(vec![M, C, L], vec![123.0; M * C * L]),
    );
    let mut ok = vec![1.0; M * C * L];
    for m in 0..M {
        for k in 0..L {
            ok[(m * C + 1) * L + k] = 0.0;
        }
    }
    t.insert("usage_ok".to_string(), Tensor::new(vec![M, C, L], ok));
    // the orderer has no net: ratio is 1, recalibration only (and it is switched off here)
    t.insert(
        "ord_seen".to_string(),
        Tensor::new(vec![L, C], vec![0.0; L * C]),
    );
    t.insert(
        "ord_recal_n".to_string(),
        Tensor::new(vec![C, L], vec![0.0; C * L]),
    );
    let h = head(&t);
    let out = h.forward(&Mat::zeros(1, D), &["r0"]).unwrap();
    for k in 0..L {
        assert!(
            out.usage[L + k].is_nan(),
            "candidate 1 leg {k} should be NaN"
        );
        assert_close(out.usage[k], 123.0, 1e-7, 0.0);
    }
}

fn unsupported<T>(r: Result<T, GateError>) -> bool {
    matches!(r, Err(GateError::Stage1Unsupported(_)))
}

fn build(t: BTreeMap<String, Tensor>) -> Result<StageOne, GateError> {
    StageOne::new(t, None, Some(regime_names()))
}

#[test]
fn refuses_unported_unknown_missing_and_wrong_shapes() {
    let t = tiny_stage_one(12, true);

    let mut with = t.clone();
    with.insert(
        "components".to_string(),
        Tensor::new(vec![2, D], vec![0.0; 2 * D]),
    );
    assert!(
        unsupported(build(with)),
        "an unported branch must be refused"
    );

    let mut with = t.clone();
    with.insert("mystery".to_string(), Tensor::new(vec![2], vec![0.0; 2]));
    assert!(
        unsupported(build(with)),
        "an unknown tensor must be refused"
    );

    let mut miss = t.clone();
    miss.remove("w1");
    assert!(unsupported(build(miss)), "a missing tensor must be refused");

    let mut bad = t.clone();
    bad.insert(
        "c1".to_string(),
        Tensor::new(vec![M, H + 1], vec![0.0; M * (H + 1)]),
    );
    assert!(unsupported(build(bad)), "a wrong shape must be refused");

    assert!(build(t.clone()).is_ok(), "`auxiliary` alone is fine");

    let r = StageOne::new(t, Some(vec![Candidate::named("a")]), Some(regime_names()));
    assert!(
        unsupported(r),
        "one candidate name for four candidates must be refused"
    );
}

/// Writes every tensor in the dtype the shipped file uses (flags as u8, indexes as i64, the rest as f64).
fn out_tensor(name: &str, t: &Tensor) -> OutTensor {
    match name {
        "usage_ok" | "p_ok" | "ord_seen" | "cal_platt" | "auxiliary" => {
            OutTensor::u8(&t.shape, t.data.iter().map(|&v| v as u8).collect())
        }
        "ens_point" | "ens_draws" | "ens_owner" | "ord_pool_n" | "ord_recal_n" => {
            OutTensor::i64(&t.shape, t.data.iter().map(|&v| v as i64).collect())
        }
        _ => OutTensor::f64(&t.shape, t.data.clone()),
    }
}

#[test]
fn release_loader_pins_and_candidates() {
    let t = tiny_stage_one(13, true);
    let d = tempfile::tempdir().unwrap();
    let dir = d.path();
    let tensors: Vec<(String, OutTensor)> = t
        .iter()
        .map(|(k, v)| (format!("stage_one.{k}"), out_tensor(k, v)))
        .collect();
    let mut meta = BTreeMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    write_safetensors(dir.join("model.safetensors"), &tensors, &meta).unwrap();

    let cands = json!([
        {"model": "alpha", "effort": null},
        {"model": "beta"},
        {"model": "withheld-model-00"},
        {"model": "gamma", "effort": "high"}
    ]);
    let cfg = json!({"arc": {"release": "arc-test", "package_sha256": "abc",
        "stage_one": {"candidates": cands, "regimes": REGIMES}}});
    std::fs::write(dir.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let manifest = |config_hash: &str, weights_hash: &str| {
        json!({"name": "arc-test", "package_sha256": "abc",
               "files": {"config.json": config_hash, "model.safetensors": weights_hash}})
    };
    let config_hash = release_sha256(dir.join("config.json")).unwrap();
    let weights_hash = release_sha256(dir.join("model.safetensors")).unwrap();
    let write_manifest = |m: serde_json::Value| {
        std::fs::write(
            dir.join("arc_release.json"),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
    };
    write_manifest(manifest(&config_hash, &weights_hash));

    let h = StageOne::from_release(dir, true).unwrap();
    assert_eq!(
        h.candidate_names(),
        ["alpha", "beta", "withheld-model-00", "gamma@high"]
    );
    assert_eq!(h.candidate_index("beta").unwrap(), 1);
    assert_eq!(h.candidate_index("withheld").unwrap(), 2);
    assert!(matches!(h.candidate_index("a"), Err(GateError::Key(_))));

    write_manifest(manifest(&"0".repeat(64), &weights_hash));
    assert!(matches!(
        StageOne::from_release(dir, true),
        Err(GateError::ReleaseRefused(_))
    ));
    assert_eq!(StageOne::from_release(dir, false).unwrap().dims.m, M);
}
