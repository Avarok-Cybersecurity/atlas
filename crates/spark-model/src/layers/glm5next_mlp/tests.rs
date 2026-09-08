// SPDX-License-Identifier: AGPL-3.0-only

//! `Glm5NextMlpConfig` derivation and refusals. No GPU.

use atlas_core::config::{Glm5NextRouterMode, ModelConfig, parse_config};

use super::*;

/// The REAL checkpoint config of `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`, parsed by Atlas's
/// own `glm5_next` parser — the same fixture the Slice-9 skeleton acceptance test uses.
///
/// A hand-built `ModelConfig` would prove the arithmetic and nothing about the checkpoint; this
/// also pins that the parser still reads `swiglu_limit` and the group-routing keys off the real
/// file rather than off a fixture written to agree with it.
const CONFIG: &str = include_str!("../../../tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn glm_config() -> ModelConfig {
    parse_config(CONFIG).expect("the real checkpoint config parses")
}

#[test]
fn reads_the_checkpoint_geometry_at_world_one() {
    let c = Glm5NextMlpConfig::from_config(&glm_config()).unwrap();
    assert_eq!(c.hidden, 4096);
    assert_eq!(c.local_dense_intermediate, 12288);
    assert_eq!(c.moe_intermediate, 2048);
    assert_eq!(c.local_shared_intermediate, 2048);
    assert_eq!(c.num_experts, 288);
    assert_eq!(c.local_experts, 288);
    assert_eq!(c.top_k, 8);
    assert_eq!(c.routed_scale, 2.5);
    assert!(c.renormalize);
    assert_eq!(c.swiglu_limit, 10.0);
    assert!(!c.router_bf16_ladder);
    assert!(!c.needs_all_reduce());
}

/// The campaign topology: world 2, TP 2, EP 2 on the same two ranks. The dense/shared widths
/// halve, the expert SET halves, and one expert is never split.
#[test]
fn tp2_ep2_halves_widths_and_the_expert_set() {
    let mut base = glm_config();
    base.tp_world_size = 2;
    base.ep_world_size = 2;

    for rank in 0..2 {
        let mut m = base.clone();
        m.tp_rank = rank;
        m.ep_rank = rank;
        let c = Glm5NextMlpConfig::from_config(&m).unwrap();
        assert_eq!(c.local_dense_intermediate, 6144);
        assert_eq!(c.local_shared_intermediate, 1024);
        // 🪤 An expert is owned WHOLE. Its width is untouched by TP.
        assert_eq!(c.moe_intermediate, 2048, "rank {rank}");
        assert_eq!(c.local_experts, 144);
        assert!(c.needs_all_reduce());
    }
}

/// The EP residency contract Slice 12 proved: ids 0..143 and 144..287, intersection empty,
/// union complete. Every id is owned by exactly one rank, and no rank can index a remote one.
#[test]
fn every_expert_id_is_owned_by_exactly_one_rank() {
    let mut base = glm_config();
    base.tp_world_size = 2;
    base.ep_world_size = 2;

    let cfgs: Vec<Glm5NextMlpConfig> = (0..2)
        .map(|r| {
            let mut m = base.clone();
            m.ep_rank = r;
            Glm5NextMlpConfig::from_config(&m).unwrap()
        })
        .collect();

    assert_eq!(cfgs[0].local_expert_range(), 0..144);
    assert_eq!(cfgs[1].local_expert_range(), 144..288);

    for id in 0..288usize {
        let owners: Vec<usize> = cfgs
            .iter()
            .enumerate()
            .filter(|(_, c)| c.local_slot(id).is_some())
            .map(|(r, _)| r)
            .collect();
        assert_eq!(owners.len(), 1, "expert {id} owned by {owners:?}");
    }
    // The local slot is the OFFSET, not the global id — indexing a 144-entry array with 200
    // would panic, and indexing it with 200 % 144 would silently run the wrong expert.
    assert_eq!(cfgs[1].local_slot(200), Some(56));
    assert_eq!(cfgs[0].local_slot(200), None);
}

/// A zero clamp is not "no clamp": `min(gate, 0)` forces the gate non-positive, so a defaulted
/// limit is a wrong answer rather than a disabled feature.
#[test]
fn a_zero_swiglu_limit_is_refused() {
    let mut c = glm_config();
    c.swiglu_limit = 0.0;
    let err = Glm5NextMlpConfig::from_config(&c).unwrap_err();
    assert!(err.to_string().contains("swiglu_limit"), "{err}");
}

/// `glm5next_router_topk` holds the selected scores in `float best_w[16]`.
#[test]
fn a_top_k_past_the_kernels_register_budget_is_refused() {
    let mut c = glm_config();
    c.num_experts_per_tok = KERNEL_MAX_TOP_K + 1;
    let err = Glm5NextMlpConfig::from_config(&c).unwrap_err();
    assert!(err.to_string().contains("16-slot"), "{err}");

    let mut ok = glm_config();
    ok.num_experts_per_tok = KERNEL_MAX_TOP_K;
    assert!(Glm5NextMlpConfig::from_config(&ok).is_ok(), "16 is legal");
}

/// A ragged expert split would leave ids owned by nobody — which under masked-local EP is a
/// silently missing contribution, not an error.
#[test]
fn an_expert_count_that_does_not_divide_over_ep_is_refused() {
    let mut c = glm_config();
    c.ep_world_size = 7;
    let err = Glm5NextMlpConfig::from_config(&c).unwrap_err();
    assert!(err.to_string().contains("owned by nobody"), "{err}");
}

/// The router mode is SEMANTIC — the two ladders select different experts, so it must come
/// from the config and never from a precision preference.
#[test]
fn the_router_ladder_comes_from_the_config() {
    let mut c = glm_config();
    c.glm5next_router_mode = Glm5NextRouterMode::VllmBf16;
    assert!(
        Glm5NextMlpConfig::from_config(&c)
            .unwrap()
            .router_bf16_ladder
    );
}
