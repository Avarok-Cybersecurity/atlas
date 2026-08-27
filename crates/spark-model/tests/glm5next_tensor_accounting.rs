// SPDX-License-Identifier: AGPL-3.0-only

//! Slice-1 acceptance: every tensor in the reference GLM-5.3 checkpoint is
//! classified deliberately, with zero silent skips.
//!
//! Fixture `fixtures/glm53-nvfp4-9e0d74e3-patterns.tsv` was produced by reading
//! the safetensors header of all **120 shards** of
//! `LibertAIDAI/GLM-5.3-Flash-NVFP4` snapshot `9e0d74e3…` and canonicalising
//! layer/expert indices. It is `count<TAB>pattern`, 407 rows, and the counts sum
//! to the checkpoint's **113,074** tensors. Claims are scoped to that checkpoint.

use spark_model::weight_loader::glm5_next::{TensorRole, account, classify};

const FIXTURE: &str = include_str!("fixtures/glm53-nvfp4-9e0d74e3-patterns.tsv");
const EXPECTED_TENSORS: usize = 113_074;
const EXPECTED_PATTERNS: usize = 407;

fn rows() -> Vec<(&'static str, usize)> {
    FIXTURE
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let (c, name) = l.split_once('\t').expect("fixture row is count<TAB>name");
            (name, c.parse::<usize>().expect("count"))
        })
        .collect()
}

#[test]
fn fixture_is_the_reference_checkpoint() {
    let r = rows();
    assert_eq!(r.len(), EXPECTED_PATTERNS, "pattern count drifted");
    let total: usize = r.iter().map(|(_, c)| c).sum();
    assert_eq!(total, EXPECTED_TENSORS, "tensor count drifted");
}

/// THE acceptance criterion: 113,074 / 113,074 classified, zero unknown.
#[test]
fn every_tensor_is_intentionally_classified() {
    let acc = account(rows());
    assert!(
        acc.unknown.is_empty(),
        "{} unclassified pattern(s), first 10: {:?}",
        acc.unknown.len(),
        &acc.unknown[..acc.unknown.len().min(10)]
    );
    assert_eq!(acc.total, EXPECTED_TENSORS);
    let sum: usize = acc.by_role.values().sum();
    assert_eq!(
        sum, EXPECTED_TENSORS,
        "role totals must reconcile to the checkpoint"
    );
}

/// Reconciled census — see
/// `.planning/ATLAS-GLM5NEXT-SKILL-RECONCILIATION-20260826.md`.
/// The per-layer mixer tensors are the discriminator: KDA layers carry `A_log`,
/// sparse-MLA layers carry `kv_b_proj`.
#[test]
fn layer_census_matches_reconciled_counts() {
    let r = rows();
    let count_of = |pat: &str| -> usize {
        r.iter()
            .find(|(n, _)| *n == pat)
            .map(|(_, c)| *c)
            .unwrap_or_else(|| panic!("pattern absent from fixture: {pat}"))
    };

    // 34 KDA layers — one A_log each, text layers only.
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.A_log"),
        34,
        "KDA layer count"
    );
    // 12 sparse-MLA layers over the WHOLE checkpoint = 11 text + layer 45 (MTP).
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.kv_b_proj.weight"),
        12,
        "DSA layers incl. the MTP layer"
    );
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.indexer.wk.weight"),
        12,
        "indexer instances incl. the MTP layer"
    );
    // o_proj is on every mixer: 34 + 12 = 46.
    assert_eq!(
        count_of("model.language_model.layers.N.self_attn.o_proj.weight"),
        46
    );
    // 3 dense FFN layers (first_k_dense_replace = 3).
    assert_eq!(
        count_of("model.language_model.layers.N.mlp.gate_proj.weight"),
        3,
        "dense FFN layers"
    );
    // 43 MoE layers over the whole checkpoint = 42 text + layer 45.
    assert_eq!(
        count_of("model.language_model.layers.N.mlp.gate.weight"),
        43,
        "MoE router instances incl. the MTP layer"
    );
}

/// MTP naming proven from the checkpoint, not assumed.
#[test]
fn mtp_lives_at_layer_45_and_not_under_mtp_prefix() {
    let r = rows();
    assert!(
        r.iter().any(|(n, _)| n.contains("eh_proj")),
        "no eh_proj in fixture"
    );
    assert!(
        !r.iter().any(|(n, _)| n.contains("mtp.0.")),
        "checkpoint must contain zero mtp.0.* tensors"
    );
    // eh_proj / enorm / hnorm occur exactly once each => exactly one MTP layer.
    for p in ["eh_proj.weight", "enorm.weight", "hnorm.weight"] {
        let c: usize = r
            .iter()
            .filter(|(n, _)| n.ends_with(p))
            .map(|(_, c)| *c)
            .sum();
        assert_eq!(c, 1, "{p} should occur on exactly one layer");
    }
}

/// RMSNorm hazard guard: no norm tensor may hide inside a non-norm role.
#[test]
fn norm_sanity_pass() {
    let mut checked = 0usize;
    for (name, _) in rows() {
        let looks_like_norm = name.contains("norm") || name.ends_with(".norm.weight");
        if !looks_like_norm {
            continue;
        }
        let role = classify(name).unwrap_or_else(|| panic!("unclassified norm: {name}"));
        // The vision tower is bucketed wholesale and is out of scope for the
        // text port; its norms are classified, just not individually typed.
        if role == TensorRole::Vision {
            continue;
        }
        checked += 1;
        assert!(
            role.is_norm(),
            "text-model norm tensor {name} classified as non-norm {role:?}"
        );
    }
    assert!(checked >= 8, "expected several norm patterns, saw {checked}");
}

/// The vision tower is present and must be classified, not silently ignored —
/// but it is explicitly out of scope for the text port.
#[test]
fn vision_tower_is_classified_and_separable() {
    let acc = account(rows());
    let vision = acc.by_role.get("Vision").copied().unwrap_or(0);
    assert!(vision > 0, "vision tensors should be present and classified");
    let text: usize = acc
        .by_role
        .iter()
        .filter(|(k, _)| k.as_str() != "Vision")
        .map(|(_, v)| *v)
        .sum();
    assert_eq!(text + vision, EXPECTED_TENSORS);
    assert!(
        text > vision,
        "text model should dominate: text={text} vision={vision}"
    );
}

/// Emit the accounting table for the record (visible with `--nocapture`).
#[test]
fn print_accounting_table() {
    let acc = account(rows());
    println!("\nGLM-5.3-Flash-NVFP4 @ 9e0d74e3 — tensor accounting");
    println!("  shards 120 · patterns {EXPECTED_PATTERNS} · tensors {}", acc.total);
    for (role, n) in &acc.by_role {
        println!("  {role:>18} : {n:>7}");
    }
    println!("  {:>18} : {:>7}", "UNKNOWN", acc.unknown.len());
    println!("  MTP layer indices: {:?}", acc.mtp_layers);
    assert!(matches!(classify("lm_head.weight"), Some(TensorRole::LmHead)));
}
