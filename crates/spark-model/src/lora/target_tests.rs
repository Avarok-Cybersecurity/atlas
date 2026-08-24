// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 target-surface tests: expert/router dims (per-layer MoE
//! intermediate, NOT the dense intermediate), the pool-byte golden, and the
//! sparse per-expert coverage view. GPU-free.

use crate::lora::test_support::*;
use crate::lora::*;

#[test]
fn expert_and_router_dims_use_moe_intermediate() {
    // Factory cfg: hidden 2048, moe_intermediate_size 512, num_experts 512.
    let cfg = cfg();
    assert_eq!(ExpertProj::Gate.dims(&cfg, 7), (512, 2048));
    assert_eq!(ExpertProj::Up.dims(&cfg, 7), (512, 2048));
    assert_eq!(ExpertProj::Down.dims(&cfg, 7), (2048, 512));
    // Router base weight is [num_experts, hidden].
    assert_eq!(router_dims(&cfg), (512, 2048));
    // peft_name leaves.
    assert_eq!(ExpertProj::Gate.peft_name(), "gate_proj");
    assert_eq!(ExpertProj::Up.peft_name(), "up_proj");
    assert_eq!(ExpertProj::Down.peft_name(), "down_proj");
}

#[test]
fn expert_router_bytes_golden() {
    let cfg = cfg();
    // gate: (16*2048 + 512*16)*2 = 81920 ; down: (16*512 + 2048*16)*2 = 81920
    // router: (16*2048 + 512*16)*2 = 81920
    let ek = vec![
        (7usize, ExpertProj::Gate),
        (7usize, ExpertProj::Gate),
        (7usize, ExpertProj::Down),
    ];
    let rl = vec![3usize];
    assert_eq!(expert_router_bytes(&cfg, &ek, &rl, 16), 81_920 * 4);
    // A non-multiple-of-8 rank cap sizes at the uint4-PADDED stride (12 → 16),
    // matching the pack loop's derived stride byte-for-byte (SSOT).
    assert_eq!(expert_router_bytes(&cfg, &ek, &rl, 12), 81_920 * 4);
    // Empty audit → zero bytes (no expert pool allocated).
    assert_eq!(expert_router_bytes(&cfg, &[], &[], 16), 0);
}

#[test]
fn expert_layer_adapted_experts_sorted_deduped() {
    let mut el = ExpertLoraLayer::default();
    el.pairs
        .insert((5, ExpertProj::Gate), dummy_pair(1, 2048, 512));
    el.pairs
        .insert((5, ExpertProj::Down), dummy_pair(2, 512, 2048));
    el.pairs
        .insert((2, ExpertProj::Up), dummy_pair(3, 2048, 512));
    assert_eq!(el.adapted_experts(), vec![2, 5]);
    assert_eq!(el.pair(5, ExpertProj::Gate).map(|p| p.a.weight.0), Some(1));
    assert!(el.pair(5, ExpertProj::Up).is_none());
    assert!(!el.is_empty());
    assert!(ExpertLoraLayer::default().is_empty());
}
