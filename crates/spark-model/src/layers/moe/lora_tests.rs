// SPDX-License-Identifier: AGPL-3.0-only

//! Host tests for the `delta` scratch sizing (`lora_delta_cols`) — the
//! correctness-critical column count that used to over-allocate 4-8x by taking
//! `max(all n_out, all k_in)`. `delta` has exactly two consumers (router expand
//! = num_experts, decode down x-recompute = moe_inter); gate/up's `k_in=hidden`
//! must NOT size it. GPU-free (pure function).

use super::super::lora_router::router_expert_entry;
use super::lora_delta_cols;
use crate::layers::ops::lora_delta::LoraPair;
use crate::layers::ops::moe_lora_grouped::pack_expert_tables;
use crate::weight_map::DenseWeight;
use spark_runtime::gpu::DevicePtr;

// Holo-3.1-35B-A3B shapes: hidden=2048, moe_inter=512, num_experts=256.
const NUM_EXPERTS: u32 = 256;
const MOE_INTER: u32 = 512;
const HIDDEN: u32 = 2048;

/// Build a GPU-free router `LoraPair` with distinct dummy A/B device addresses
/// (`DevicePtr` is a thin `u64` newtype, so no allocation is needed to test the
/// pure entry mapping). `k_in=hidden`, `n_out=num_experts` — the router pair's
/// real dims.
fn dummy_router_pair(a_addr: u64, b_addr: u64, scale: f32, rank: u32) -> LoraPair {
    LoraPair {
        a: DenseWeight {
            weight: DevicePtr(a_addr),
        },
        b: DenseWeight {
            weight: DevicePtr(b_addr),
        },
        rank,
        k_in: HIDDEN,
        n_out: NUM_EXPERTS,
        scale,
        max_rank: rank,
    }
}

#[test]
fn router_entry_is_expert_zero_from_pair() {
    // The router pair maps to a single (expert_id=0, a, b, scale) entry — the
    // router owns the only slot of the degenerate 1-"expert" gather table.
    let rp = dummy_router_pair(0xA000, 0xB000, 0.5, 16);
    assert_eq!(
        router_expert_entry(&rp),
        (0u16, 0xA000u64, 0xB000u64, 0.5f32)
    );
}

#[test]
fn router_entry_packs_to_len_one_table() {
    // Packing that single entry yields a length-1 dense route (n_experts == 1),
    // so `moe_lora_gather_bgmv` sees exactly one "expert" and every all-zero
    // `indices` row resolves to it.
    let rp = dummy_router_pair(0xAAAA, 0xBBBB, 2.0, 8);
    let t = pack_expert_tables(&[router_expert_entry(&rp)]).unwrap();
    assert_eq!(t.n_experts, 1);
    assert_eq!(t.a, vec![0xAAAA]);
    assert_eq!(t.b, vec![0xBBBB]);
    assert_eq!(t.scale, vec![2.0]);
}

#[test]
fn full_adapter_takes_max_of_router_and_down() {
    // down+gate+up+router: router n_out=num_experts, down k_in=moe_inter.
    // max(256, 512) = 512 — NOT hidden (2048), which gate/up's k_in would give.
    assert_eq!(
        lora_delta_cols(Some(NUM_EXPERTS), Some(MOE_INTER)),
        MOE_INTER as usize
    );
}

#[test]
fn router_only_is_num_experts() {
    // No Down pair -> only the router expand consumes delta.
    assert_eq!(
        lora_delta_cols(Some(NUM_EXPERTS), None),
        NUM_EXPERTS as usize
    );
}

#[test]
fn gateup_only_no_router_is_unit() {
    // gate/up-only with no router: delta has no real consumer -> 1 (never 0).
    assert_eq!(lora_delta_cols(None, None), 1);
}

#[test]
fn down_only_is_moe_inter() {
    // down-only (no router): decode x-recompute needs moe_inter cols.
    assert_eq!(lora_delta_cols(None, Some(MOE_INTER)), MOE_INTER as usize);
}

#[test]
fn router_wins_when_experts_exceed_inter() {
    // A model with num_experts > moe_inter routes the max the other way.
    assert_eq!(lora_delta_cols(Some(1024), Some(MOE_INTER)), 1024);
}

#[test]
fn gate_up_k_in_hidden_never_inflates() {
    // Regression: the old formula was max(n_out, k_in) over ALL pairs, so
    // gate/up k_in=hidden=2048 blew delta to 2048 (4x). The pure fn only sees
    // router n_out and down k_in, so hidden can never enter.
    let sized = lora_delta_cols(Some(NUM_EXPERTS), Some(MOE_INTER));
    assert!(
        sized < HIDDEN as usize,
        "delta_cols {sized} must not be inflated to hidden {HIDDEN}"
    );
}
