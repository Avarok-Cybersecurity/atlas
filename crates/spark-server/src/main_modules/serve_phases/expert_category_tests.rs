// SPDX-License-Identifier: AGPL-3.0-only

//! Resolving `--expert-category` at boot.
//!
//! Every rejection here is one that, if it instead resolved to *something*,
//! would put a serve into production loading the wrong experts. Nothing
//! downstream can detect that: the model still answers, just worse.

use super::*;
use atlas_kernels::ExpertCategory;

const PY_LAYERS: &[(usize, &[u16])] = &[(0, &[1, 2, 3, 4]), (2, &[0, 5, 6, 7])];

fn ptx_set(categories: &'static [ExpertCategory]) -> atlas_kernels::TargetPtxSet {
    atlas_kernels::TargetPtxSet {
        target: atlas_core::target::KernelTarget {
            arch: "sm_121",
            model: "test",
            quant: "nvfp4",
        },
        modules: Vec::new(),
        sampling: atlas_kernels::SamplingPresets::default(),
        behavior: atlas_kernels::ModelBehavior::default(),
        model_type_matches: Vec::new(),
        match_names: &[],
        dflash: None,
        expert_categories: categories,
        shadowed_dropped: &[],
        expected_absent: &[],
    }
}

fn moe_config() -> atlas_core::config::ModelConfig {
    // The 80B preset is the repo's ready-made MoE config; only the three
    // fields this resolver reads are overridden, so a change to any other
    // field cannot silently alter what these tests exercise.
    let mut c = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    c.num_experts = 8;
    c.num_experts_per_tok = 2;
    c.num_hidden_layers = 4;
    c
}

static CATS: &[ExpertCategory] = &[ExpertCategory {
    name: "code-python",
    coverage: 0.9,
    layers: PY_LAYERS,
}];

// ---------------------------------------------------------------- Path A

#[test]
fn a_known_category_installs_a_plan() {
    let mut c = moe_config();
    resolve_expert_category(Some("code-python"), &ptx_set(CATS), &mut c).expect("resolves");
    let plan = c.bel.as_ref().expect("plan installed");
    assert_eq!(plan.category, "code-python");
    assert!(plan.is_loaded(0, 1));
    assert!(!plan.is_loaded(0, 7));
    // Layer 1 is not in the table — a dense layer, unrestricted.
    assert!(!plan.restricts_layer(1));
    assert_eq!(plan.totals(), (8, 16));
}

#[test]
fn no_flag_installs_no_plan() {
    let mut c = moe_config();
    resolve_expert_category(None, &ptx_set(CATS), &mut c).expect("resolves");
    assert!(
        c.bel.is_none(),
        "every expert loads when the flag is absent"
    );
}

// ---------------------------------------------------------------- Path B

#[test]
fn a_layer_keeping_fewer_experts_than_top_k_is_refused() {
    // top-k would have to name a masked expert to fill its slots. The
    // resulting serve routes into weights that were never loaded.
    static THIN: &[ExpertCategory] = &[ExpertCategory {
        name: "thin",
        coverage: 0.5,
        layers: &[(0, &[3])],
    }];
    let mut c = moe_config(); // num_experts_per_tok = 2
    let err = resolve_expert_category(Some("thin"), &ptx_set(THIN), &mut c)
        .unwrap_err()
        .to_string();
    assert!(err.contains("keeps only 1 experts"), "got: {err}");
    assert!(
        err.contains("higher coverage"),
        "the fix must be named: {err}"
    );
    assert!(c.bel.is_none(), "a refused plan must not be installed");
}

#[test]
fn a_table_measured_on_a_different_checkpoint_is_refused() {
    // The ids index an expert space this model does not have.
    static WIDE: &[ExpertCategory] = &[ExpertCategory {
        name: "wide",
        coverage: 0.9,
        layers: &[(0, &[1, 2, 300])],
    }];
    let mut c = moe_config();
    let err = resolve_expert_category(Some("wide"), &ptx_set(WIDE), &mut c)
        .unwrap_err()
        .to_string();
    assert!(err.contains("different checkpoint"), "got: {err}");
}

// ---------------------------------------------------------------- Path C

#[test]
fn an_unknown_category_lists_what_the_model_declares() {
    let mut c = moe_config();
    let err = resolve_expert_category(Some("klingon"), &ptx_set(CATS), &mut c)
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown category"), "got: {err}");
    assert!(
        err.contains("code-python"),
        "must list what IS there: {err}"
    );
}

#[test]
fn a_model_with_no_table_says_how_to_produce_one() {
    // The fix is a whole workflow — measure, paste, rebuild — so the error
    // has to carry it or the flag looks broken rather than unprepared.
    let mut c = moe_config();
    let err = resolve_expert_category(Some("code-python"), &ptx_set(&[]), &mut c)
        .unwrap_err()
        .to_string();
    assert!(err.contains("expert-categories` benchmark"), "got: {err}");
    assert!(err.contains("--expert-telemetry"), "got: {err}");
    assert!(err.contains("rebuild"), "got: {err}");
}

#[test]
fn a_dense_model_is_refused_before_anything_else() {
    let mut c = moe_config();
    c.num_experts = 0;
    let err = resolve_expert_category(Some("code-python"), &ptx_set(CATS), &mut c)
        .unwrap_err()
        .to_string();
    assert!(err.contains("dense checkpoint"), "got: {err}");
}
