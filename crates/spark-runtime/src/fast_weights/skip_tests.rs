// SPDX-License-Identifier: AGPL-3.0-only

//! Which tensors boot-time expert loading withholds.
//!
//! The rule is paired with a router mask built from the same plan. If they
//! disagreed in the direction of "skipped but selectable", a request routing
//! to that expert dereferences a null pointer in a kernel; in the direction
//! of "loaded but masked", the serve silently wastes the memory the feature
//! exists to save. So these tests are about the skip decision meaning
//! exactly what the mask means.

use super::FastSafetensorsLoader;
use atlas_core::config::bel::BelPlan;
use std::sync::Arc;

/// Layer 3 keeps experts 1 and 5 of 8; layer 4 keeps expert 0. Layer 9 is
/// unmentioned — a dense layer of a hybrid model.
fn loader_with_plan() -> FastSafetensorsLoader {
    let plan = BelPlan::new(
        "code-python",
        0.9,
        16,
        8,
        vec![(3usize, vec![1u16, 5]), (4usize, vec![0u16])],
    )
    .expect("valid plan");
    FastSafetensorsLoader {
        num_experts: 8,
        bel: Some(Arc::new(plan)),
        ..Default::default()
    }
}

// ---------------------------------------------------------------- Path A

#[test]
fn a_listed_expert_is_kept() {
    let l = loader_with_plan();
    assert!(!l.should_skip_tensor("model.layers.3.mlp.experts.1.gate_proj.weight"));
    assert!(!l.should_skip_tensor("model.layers.3.mlp.experts.5.down_proj.weight"));
    assert!(!l.should_skip_tensor("model.layers.4.mlp.experts.0.up_proj.weight"));
}

#[test]
fn an_unlisted_expert_is_skipped() {
    let l = loader_with_plan();
    assert!(l.should_skip_tensor("model.layers.3.mlp.experts.2.gate_proj.weight"));
    assert!(l.should_skip_tensor("model.layers.3.mlp.experts.7.down_proj.weight"));
}

// ---------------------------------------------------------------- Path B

#[test]
fn the_decision_is_per_layer_not_per_expert() {
    // The whole reason the plan is keyed by (layer, expert): expert 1 is
    // resident in layer 3 and absent from layer 4. A rule that looked only
    // at the expert index would keep or drop it in both.
    let l = loader_with_plan();
    assert!(!l.should_skip_tensor("model.layers.3.mlp.experts.1.gate_proj.weight"));
    assert!(l.should_skip_tensor("model.layers.4.mlp.experts.1.gate_proj.weight"));
}

#[test]
fn a_layer_the_plan_does_not_mention_keeps_everything() {
    // A dense layer, or one the category table has nothing to say about.
    // Skipping it would strand weights the loader goes on to read.
    let l = loader_with_plan();
    for e in 0..8 {
        assert!(
            !l.should_skip_tensor(&format!("model.layers.9.mlp.experts.{e}.gate_proj.weight")),
            "expert {e} of an unlisted layer must be kept"
        );
    }
}

#[test]
fn non_expert_tensors_are_untouched() {
    // Attention, norms and the router gate itself are not per-expert; a rule
    // that caught them would remove weights every layer needs.
    let l = loader_with_plan();
    for name in [
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.input_layernorm.weight",
        "model.layers.3.mlp.gate.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ] {
        assert!(!l.should_skip_tensor(name), "{name} must be kept");
    }
}

#[test]
fn mtp_experts_are_always_kept() {
    // The drafter's experts are its own; a category table describes the
    // TARGET model's routing and says nothing about them.
    let l = loader_with_plan();
    assert!(!l.should_skip_tensor("mtp.layers.3.mlp.experts.7.gate_proj.weight"));
}

// ---------------------------------------------------------------- Path C

#[test]
fn without_a_plan_nothing_is_skipped() {
    // The negative control: no --expert-category means byte-identical
    // loading to before this feature existed.
    let l = FastSafetensorsLoader {
        num_experts: 8,
        ..Default::default()
    };
    for e in 0..8 {
        assert!(!l.should_skip_tensor(&format!("model.layers.3.mlp.experts.{e}.gate_proj.weight")));
    }
}

#[test]
fn a_plan_listing_every_expert_skips_nothing() {
    // The other negative control, and the one a BEL run is judged against:
    // a category covering the whole expert set must load exactly what a
    // no-flag serve loads.
    let plan = BelPlan::new("all", 1.0, 8, 4, vec![(0usize, vec![0u16, 1, 2, 3])]).unwrap();
    let l = FastSafetensorsLoader {
        num_experts: 4,
        bel: Some(Arc::new(plan)),
        ..Default::default()
    };
    for e in 0..4 {
        assert!(!l.should_skip_tensor(&format!("model.layers.0.mlp.experts.{e}.gate_proj.weight")));
    }
}
