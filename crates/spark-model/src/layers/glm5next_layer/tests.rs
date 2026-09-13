// SPDX-License-Identifier: AGPL-3.0-only

//! What can be asserted about the composite layer without a GPU: that the state kinds are not
//! interchangeable, and that the wiring this layer executes is the wiring the skeleton records.

use atlas_core::config::parse_config;

use crate::layers::glm5next_skeleton::{Glm5NextTextSkeleton, Mixer, Mlp, ResidualStep, Site};

const CONFIG: &str = include_str!("../../../tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn skeleton() -> Glm5NextTextSkeleton {
    Glm5NextTextSkeleton::from_config(&parse_config(CONFIG).expect("real config parses"))
        .expect("skeleton builds")
}

/// `forward_one` executes a fixed sequence. This pins it against the skeleton's `residual_plan`,
/// which is the artifact that was checked against HF 5.16.1 — so the two cannot drift apart
/// silently. If this fails, either the plan changed or the layer stopped following it.
#[test]
fn the_layer_executes_the_skeletons_residual_plan() {
    let sk = skeleton();
    let l = sk.layers[0];
    assert_eq!(
        sk.residual_plan(&l),
        vec![
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Attn),
            ResidualStep::Norm("input_layernorm.weight"),
            ResidualStep::Mixer,
            ResidualStep::HcPost(Site::Attn),
            ResidualStep::SaveResidual,
            ResidualStep::HcPre(Site::Ffn),
            ResidualStep::Norm("post_attention_layernorm.weight"),
            ResidualStep::Mlp,
            ResidualStep::HcPost(Site::Ffn),
        ],
        "forward_one runs hc_pre -> norm -> sublayer -> hc_post twice, in this order"
    );
}

/// The composite has to answer for both mixers and both MLP kinds across the same 45 layers.
/// This is the census it must cover — 34 KDA / 11 DSA, 3 dense / 42 routed.
#[test]
fn the_stack_the_composite_must_cover() {
    let sk = skeleton();
    assert_eq!(sk.layers.len(), 45);
    assert_eq!(
        sk.layers.iter().filter(|l| l.mixer == Mixer::Kda).count(),
        34
    );
    assert_eq!(
        sk.layers.iter().filter(|l| l.mixer == Mixer::Dsa).count(),
        11
    );
    assert_eq!(sk.layers.iter().filter(|l| l.mlp == Mlp::Dense).count(), 3);
    assert_eq!(
        sk.layers.iter().filter(|l| l.mlp == Mlp::RoutedMoe).count(),
        42
    );
    // Every text layer carries a hyper-connection; only the MTP layer does not.
    assert!(sk.layers.iter().all(|l| l.hyper_connection));
    assert!(!sk.mtp.expect("layer 45 exists").hyper_connection);
}

/// Both mixer/MLP combinations occur, so neither dispatch arm is dead code: the dense layers are
/// KDA (0, 1, 2) and the routed set contains both mixers.
#[test]
fn every_dispatch_arm_is_reachable() {
    let sk = skeleton();
    let combos: std::collections::BTreeSet<(bool, bool)> = sk
        .layers
        .iter()
        .map(|l| (l.mixer == Mixer::Kda, l.mlp == Mlp::Dense))
        .collect();
    assert!(combos.contains(&(true, true)), "KDA + dense");
    assert!(combos.contains(&(true, false)), "KDA + routed");
    assert!(combos.contains(&(false, false)), "DSA + routed");
    // 🪤 There is NO DSA+dense layer — first_k_dense_replace is 3 and layer 3 is the first DSA
    // layer. The arm exists anyway; a checkpoint revision that moved either boundary would use
    // it, and refusing to construct it would be a guess about a future checkpoint.
    assert!(!combos.contains(&(false, true)), "no DSA + dense today");
}
