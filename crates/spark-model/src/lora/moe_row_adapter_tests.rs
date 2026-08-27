// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the pure MoE-LoRA per-request routing primitives (no GPU).

use super::*;
use crate::layer::MoeLoraRoute;

#[test]
fn route_off_when_no_moe_lora() {
    // Value is inert (the fold hook no-ops on self.lora == None), but must stay
    // Fold so an off run is byte-identical.
    assert_eq!(resolve_moe_lora_route(-1, -1, false), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(3, 0, false), MoeLoraRoute::Fold);
}

#[test]
fn route_base_request_skips() {
    // adapter_slot < 0 with an adapter installed => base pays nothing.
    assert_eq!(resolve_moe_lora_route(-1, 0, true), MoeLoraRoute::Skip);
    assert_eq!(resolve_moe_lora_route(-5, 2, true), MoeLoraRoute::Skip);
}

#[test]
fn route_active_adapter_folds() {
    assert_eq!(resolve_moe_lora_route(0, 0, true), MoeLoraRoute::Fold);
    assert_eq!(resolve_moe_lora_route(2, 2, true), MoeLoraRoute::Fold);
}

#[test]
fn route_non_active_adapter_refuses() {
    // Phase-1 installs one active MoE adapter; a request for a different slot
    // cannot be served correctly => refuse loudly, never fold the wrong one.
    assert_eq!(resolve_moe_lora_route(1, 0, true), MoeLoraRoute::Refuse);
    assert_eq!(resolve_moe_lora_route(0, 3, true), MoeLoraRoute::Refuse);
}

#[test]
fn row_adapter_uniform_single_stream() {
    // One stream of 4 tokens on slot 2 -> all rows 2.
    let map = build_moe_row_adapter_host(&[0, 4], &[2]).unwrap();
    assert_eq!(map, vec![2, 2, 2, 2]);
}

#[test]
fn row_adapter_varlen_mixed_streams() {
    // Three streams of unequal length: base, adapter 1, adapter 0.
    // cu_seqlens = [0, 2, 5, 6] -> rows: [base,base, 1,1,1, 0].
    let map = build_moe_row_adapter_host(&[0, 2, 5, 6], &[-1, 1, 0]).unwrap();
    assert_eq!(map, vec![-1, -1, 1, 1, 1, 0]);
}

#[test]
fn row_adapter_empty_stream_span() {
    // A zero-length stream (partial-prefix-cache hit: all tokens cached) leaves
    // no rows for that stream; neighbors still align.
    let map = build_moe_row_adapter_host(&[0, 2, 2, 5], &[7, 9, -1]).unwrap();
    assert_eq!(map, vec![7, 7, -1, -1, -1]);
}

#[test]
fn row_adapter_rejects_malformed() {
    // Empty / single-element boundary.
    assert!(build_moe_row_adapter_host(&[], &[]).is_none());
    assert!(build_moe_row_adapter_host(&[0], &[]).is_none());
    // adapter_slots length mismatch.
    assert!(build_moe_row_adapter_host(&[0, 2, 4], &[0]).is_none());
    // Non-zero first boundary.
    assert!(build_moe_row_adapter_host(&[1, 3], &[0]).is_none());
    // Non-monotonic boundary.
    assert!(build_moe_row_adapter_host(&[0, 4, 2], &[0, 1]).is_none());
}

// ── SOLID Incr-4 batched-decode per-row map (build_moe_row_adapter_decode) ──

#[test]
fn decode_map_off_when_no_moe_lora() {
    // No adapter installed (active = -1, has = false): every row must skip so an
    // off run folds NOTHING even though resolve() returns the inert Fold.
    let map = build_moe_row_adapter_decode(&[-1, 0, 2], 4, -1, false);
    assert_eq!(map, vec![-1, -1, -1, -1]);
}

#[test]
fn decode_map_all_base_skips() {
    // Concurrent base-only batch with an adapter resident: all rows -1, pads -1.
    let map = build_moe_row_adapter_decode(&[-1, -1], 4, 0, true);
    assert_eq!(map, vec![-1, -1, -1, -1]);
}

#[test]
fn decode_map_mixed_base_and_active() {
    // active adapter = slot 3. Rows: [active, base, active] padded to 4.
    // Active rows carry `active` (>=0 => fold); base + pad carry -1 (skip).
    let map = build_moe_row_adapter_decode(&[3, -1, 3], 4, 3, true);
    assert_eq!(map, vec![3, -1, 3, -1]);
}

#[test]
fn decode_map_refuse_row_defensively_skips() {
    // A non-active adapter row (slot 1, active 0) resolves Refuse; the batch is
    // bailed host-side before upload, but if it leaked, the map skips (-1) rather
    // than folding the wrong adapter. Active-owning rows still fold.
    let map = build_moe_row_adapter_decode(&[0, 1], 2, 0, true);
    assert_eq!(map, vec![0, -1]);
}

#[test]
fn decode_map_padding_widths() {
    // The uploaded buffer is always [padded_n]; unused rows pad with -1.
    for padded_n in [2usize, 4, 8] {
        let map = build_moe_row_adapter_decode(&[0], padded_n, 0, true);
        assert_eq!(map.len(), padded_n);
        assert_eq!(map[0], 0);
        assert!(map[1..].iter().all(|&v| v == -1));
    }
}

#[test]
fn decode_map_mixed_batch_16_exact() {
    // SOLID Incr-4 cap 8 → 32: with moe_row_adapter relocated to its own buffer,
    // batch 16 folds cleanly. active adapter = slot 2. Interleave active/base
    // rows (13 real + 3 pad → padded_n 16). Active rows carry 2 (>=0 => fold);
    // base rows AND pad rows carry -1 (per-row skip stays exact — base rows in a
    // mixed batch must NOT fold the adapter).
    let slots = [2, -1, 2, 2, -1, -1, 2, -1, 2, 2, -1, 2, -1];
    let map = build_moe_row_adapter_decode(&slots, 16, 2, true);
    let expect = vec![2, -1, 2, 2, -1, -1, 2, -1, 2, 2, -1, 2, -1, -1, -1, -1];
    assert_eq!(map, expect);
    assert_eq!(map.len(), 16);
}

#[test]
fn decode_map_full_cap_32_all_active() {
    // At the new cap (padded_n = 32) every real row folds; the builder produces
    // a full-width [32] map with no collision (the +160-gap squat is gone).
    let slots = [0i32; 32];
    let map = build_moe_row_adapter_decode(&slots, 32, 0, true);
    assert_eq!(map, vec![0i32; 32]);
    assert_eq!(map.len(), 32);
}

// ── Item-1 (PR #335 gate): batched-decode pre-lookup Refuse guard ───────────

#[test]
fn decode_route_guard_refuses_only_refuse() {
    use super::ensure_decode_route_servable as guard;
    // Negative pair: Skip (pure-base batch) and Fold (active-adapter batch)
    // must pass — the fold/no-op paths are servable.
    assert!(guard(MoeLoraRoute::Skip, "decode_batch_compute_main").is_ok());
    assert!(guard(MoeLoraRoute::Fold, "decode_batch_compute_main").is_ok());
    // Positive: a Refuse batch (row routed to a NON-active adapter) must bail
    // loudly BEFORE the fold — the row map would silently serve base weights.
    let err = guard(MoeLoraRoute::Refuse, "decode_batch_compute_main")
        .expect_err("Refuse must not be servable");
    assert!(
        err.to_string().contains("non-active adapter"),
        "guard must explain the refusal: {err}"
    );
    assert!(
        err.to_string().contains("decode_batch_compute_main"),
        "guard must name the refusing path: {err}"
    );
}
