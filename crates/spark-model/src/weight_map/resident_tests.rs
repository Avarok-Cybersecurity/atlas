// SPDX-License-Identifier: AGPL-3.0-only

//! Residency must mean the same thing to the loader and to the load loop.
//!
//! The loader decides per tensor name; the loops decide per `(prefix,
//! expert)`. A disagreement is not a wrong answer — it is either a failed
//! load ("tensor not found") or a silently nulled expert, depending on the
//! direction. These tests pin that the two derivations coincide.

use super::expert_resident;
use atlas_core::config::ModelConfig;
use atlas_core::config::bel::BelPlan;

fn config_with_plan() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.num_experts = 8;
    c.num_hidden_layers = 4;
    c.ep_world_size = 1;
    c.ep_rank = 0;
    // Layer 3 keeps 1 and 5; layer 2 keeps 0. Layer 1 unmentioned.
    c.bel = Some(std::sync::Arc::new(
        BelPlan::new(
            "code-python",
            0.9,
            4,
            8,
            vec![(3usize, vec![1u16, 5]), (2usize, vec![0u16])],
        )
        .unwrap(),
    ));
    c
}

// ---------------------------------------------------------------- Path A

#[test]
fn a_listed_expert_is_resident() {
    let c = config_with_plan();
    assert!(expert_resident(&c, "model.layers.3.mlp", 1));
    assert!(expert_resident(&c, "model.layers.3.mlp", 5));
    assert!(expert_resident(&c, "model.layers.2.mlp", 0));
}

#[test]
fn an_unlisted_expert_is_not_resident() {
    let c = config_with_plan();
    assert!(!expert_resident(&c, "model.layers.3.mlp", 2));
    assert!(!expert_resident(&c, "model.layers.2.mlp", 7));
}

#[test]
fn residency_matches_the_loader_decision_tensor_for_tensor() {
    // The agreement that matters. Same plan, same names: what the loader
    // skips must be exactly what the loop nulls.
    let c = config_with_plan();
    let loader = spark_runtime::fast_weights::FastSafetensorsLoader {
        num_experts: 8,
        bel: c.bel.clone(),
        ..Default::default()
    };

    for layer in 0..4 {
        let prefix = format!("model.layers.{layer}.mlp");
        for e in 0..8 {
            let name = format!("{prefix}.experts.{e}.gate_proj.weight");
            assert_eq!(
                expert_resident(&c, &prefix, e),
                !loader.should_skip_tensor(&name),
                "layer {layer} expert {e}: loop and loader disagree"
            );
        }
    }
}

// ---------------------------------------------------------------- Path B

#[test]
fn an_unmentioned_layer_keeps_all_its_experts() {
    let c = config_with_plan();
    for e in 0..8 {
        assert!(expert_resident(&c, "model.layers.1.mlp", e));
    }
}

#[test]
fn an_unparseable_prefix_reads_as_resident() {
    // The loader could not parse it either, so it skipped nothing. Claiming
    // "not resident" here would null an expert whose weights ARE loaded —
    // silent, and wrong in the dangerous direction.
    let c = config_with_plan();
    assert!(expert_resident(&c, "block_sparse_moe", 3));
}

#[test]
fn ep_and_bel_compose() {
    // EP owns half the experts; the category restricts within them. An
    // expert must clear BOTH to be resident.
    let mut c = config_with_plan();
    c.ep_world_size = 2;
    c.ep_rank = 0; // owns experts 0..4
    assert!(
        expert_resident(&c, "model.layers.3.mlp", 1),
        "local and listed"
    );
    assert!(
        !expert_resident(&c, "model.layers.3.mlp", 5),
        "listed but owned by rank 1"
    );
    assert!(
        !expert_resident(&c, "model.layers.3.mlp", 2),
        "local but not listed"
    );
}

// ---------------------------------------------------------------- Path C

#[test]
fn without_a_plan_only_ep_decides() {
    let mut c = config_with_plan();
    c.bel = None;
    for e in 0..8 {
        assert!(
            expert_resident(&c, "model.layers.3.mlp", e),
            "no plan means every local expert is resident"
        );
    }
}
