// SPDX-License-Identifier: AGPL-3.0-only

//! Fingerprint golden pins + determinism + override precedence.
//!
//! L1 pins the hash primitive to external FNV reference vectors, L2 pins two
//! full fingerprints to frozen literals, L3 proves per-field sensitivity (a
//! refactor cannot silently drop a field), L4 proves encoding injectivity.

use std::num::NonZeroU64;

use atlas_core::config::{LayerType, ModelConfig, QuantizationConfig};

use super::*;

const BLOB: usize = 4096;

fn hybrid() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

fn dense() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3".to_string();
    c.num_hidden_layers = 28;
    c.layer_types = vec![LayerType::FullAttention; 28];
    c.num_experts = 0;
    c.linear_num_key_heads = 0;
    c.linear_key_head_dim = 0;
    c.linear_num_value_heads = 0;
    c.linear_value_head_dim = 0;
    c
}

fn fp(cfg: &ModelConfig) -> u64 {
    ModelFingerprint::derive_with_id(cfg, BLOB, "")
        .unwrap()
        .get()
}

// ── L1: pin the primitive to published FNV-1a/64 reference vectors ────
// A swapped or "optimized" implementation cannot pass someone else's
// constants.
#[test]
fn fnv1a_64_matches_reference_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
}

// ── L2: golden pins ────────────────────────────────────────────────────
// DO NOT update these literals to make a test pass — changing them
// invalidates every persisted cache key on every shared paging peer. An
// encoding change is a deliberate fleet-wide cache flush and bumps
// FP_VERSION (and gets a changelog entry).
#[test]
fn golden_fingerprint_hybrid_moe_is_pinned() {
    // FP_VERSION 2 (2026-07-10): rotated by the addition of the width fields
    // (hidden_size / num_attention_heads / intermediate_size /
    // moe_intermediate_size / num_experts_per_tok). Deliberate cache flush.
    assert_eq!(fp(&hybrid()), 0x5629_922c_51a1_6a10);
}

#[test]
fn golden_fingerprint_dense_is_pinned() {
    // FP_VERSION 2 — see the hybrid pin above.
    assert_eq!(fp(&dense()), 0x971e_b3b4_bd13_22f1);
}

/// Frozen `mix64` output pins: these EXACT literals ARE the durable wire-key
/// contract. The fold turns (key, ns) into the on-peer durable key on both the
/// SSM and KV paging paths, so any consumer that mirrors this hash (the KV
/// paging namespace) must assert the same literals — a drifted constant on
/// either side rotates or collides deployed keys.
#[test]
fn mix64_frozen_literals() {
    assert_eq!(mix64(0, 0), 0x0); // splitmix64's fixed point at zero
    assert_eq!(mix64(1, 2), 0xbeeb_8da1_658e_ec67);
    assert_eq!(
        mix64(0x5EED_F00D_CAFE_D00D, 0xD3C0_DE12_A5B6_C7D8),
        0xe567_5d86_f750_1640
    );
}

/// The KV-paging fp convention: same encoding, blob_bytes = 0 — distinct
/// from every SSM instance (whose blob is real and non-zero) and frozen like
/// the other goldens.
#[test]
fn golden_kv_fingerprint_is_pinned_and_distinct() {
    let kv = ModelFingerprint::derive_with_id(&hybrid(), 0, "")
        .unwrap()
        .get();
    assert_eq!(kv, 0x8dea_f1c5_3d0f_3540);
    assert_ne!(
        kv,
        fp(&hybrid()),
        "KV instance must not equal the SSM instance"
    );
}

// ── Determinism: same config → same u64, every time ───────────────────
#[test]
fn fingerprint_is_deterministic() {
    assert_eq!(fp(&hybrid()), fp(&hybrid()));
    assert_eq!(fp(&dense()), fp(&dense()));
    assert_ne!(fp(&hybrid()), fp(&dense()));
}

// ── L3: per-field sensitivity ──────────────────────────────────────────
// Mutating any single fingerprint field must change the value — this is
// what prevents a refactor from silently DROPPING a field from the
// encoding (the golden pin alone would still pass whenever the dropped
// field is unchanged in the fixture).
#[test]
/// ⚠ This is a HAND-MAINTAINED list, so despite the name it CANNOT prove the
/// fingerprint is exhaustive — it only proves the fields listed below are
/// load-bearing. A `ModelConfig` field that is in neither `derive_with_id` nor
/// this list is invisible to it. That is not hypothetical: `hidden_size`,
/// `num_attention_heads`, `intermediate_size`, `moe_intermediate_size` and
/// `num_experts_per_tok` were absent from BOTH, so this test was green while
/// two distinct models silently shared a namespace on a paging peer.
/// When you add a field to `ModelConfig` that changes the produced bytes, add it
/// to `derive_with_id` AND here, and bump `FP_VERSION`.
fn every_fingerprint_field_is_load_bearing() {
    let base = fp(&hybrid());
    let muts: Vec<(&str, Box<dyn Fn(&mut ModelConfig)>)> = vec![
        ("model_type", Box::new(|c| c.model_type = "other".into())),
        (
            "num_hidden_layers",
            Box::new(|c| {
                c.num_hidden_layers += 1;
                // keep layer_types-driven counts moving too
                c.layer_types.push(LayerType::FullAttention);
            }),
        ),
        (
            "layer mix (ssm/attn split)",
            Box::new(|c| {
                c.layer_types[0] = LayerType::FullAttention;
            }),
        ),
        ("head_dim", Box::new(|c| c.head_dim += 1)),
        (
            "num_key_value_heads",
            Box::new(|c| c.num_key_value_heads += 1),
        ),
        ("num_experts", Box::new(|c| c.num_experts = 0)),
        // FP_VERSION 2. These five were MISSING from the fingerprint and this
        // list simultaneously — so this test passed green while two models
        // differing only in `hidden_size` / `num_attention_heads` collided on a
        // shared paging peer. Adding a field to `derive_with_id` without adding
        // it here reproduces exactly that false confidence.
        ("hidden_size", Box::new(|c| c.hidden_size += 1)),
        (
            "num_attention_heads",
            Box::new(|c| c.num_attention_heads += 1),
        ),
        ("intermediate_size", Box::new(|c| c.intermediate_size += 1)),
        (
            "moe_intermediate_size",
            Box::new(|c| c.moe_intermediate_size += 1),
        ),
        (
            "num_experts_per_tok",
            Box::new(|c| c.num_experts_per_tok += 1),
        ),
        (
            "linear_num_key_heads",
            Box::new(|c| c.linear_num_key_heads += 1),
        ),
        (
            "linear_key_head_dim",
            Box::new(|c| c.linear_key_head_dim += 1),
        ),
        (
            "linear_num_value_heads",
            Box::new(|c| c.linear_num_value_heads += 1),
        ),
        (
            "linear_value_head_dim",
            Box::new(|c| c.linear_value_head_dim += 1),
        ),
        (
            "linear_conv_kernel_dim",
            Box::new(|c| c.linear_conv_kernel_dim += 1),
        ),
        ("mamba_num_heads", Box::new(|c| c.mamba_num_heads = 8)),
        ("mamba_head_dim", Box::new(|c| c.mamba_head_dim = 64)),
        ("ssm_state_size", Box::new(|c| c.ssm_state_size = 128)),
        ("n_groups", Box::new(|c| c.n_groups = 8)),
        (
            "quantization_config",
            Box::new(|c| {
                c.quantization_config = Some(QuantizationConfig {
                    quant_method: "modelopt".into(),
                    quant_algo: "NVFP4".into(),
                    format: String::new(),
                    ignore_modules: Vec::new(),
                });
            }),
        ),
        (
            "kv_layer_dims",
            Box::new(|c| c.kv_layer_dims = vec![(2, 256), (4, 128)]),
        ),
    ];
    for (name, m) in muts {
        let mut c = hybrid();
        m(&mut c);
        assert_ne!(fp(&c), base, "field {name} dropped from the fingerprint");
    }
    // blob_bytes is a fingerprint input too (task hard requirement).
    let b2 = ModelFingerprint::derive_with_id(&hybrid(), BLOB + 1, "")
        .unwrap()
        .get();
    assert_ne!(b2, base, "blob_bytes dropped from the fingerprint");
    // Optional ATLAS_MODEL_ID salt (fine-tune with identical geometry).
    let salted = ModelFingerprint::derive_with_id(&hybrid(), BLOB, "ft-v2")
        .unwrap()
        .get();
    assert_ne!(salted, base, "model_id salt dropped from the fingerprint");
}

// kv_layer_dims order is canonical (layer order) — a reorder must rotate.
#[test]
fn kv_layer_dims_order_is_canonical() {
    let mut a = hybrid();
    a.kv_layer_dims = vec![(2, 256), (4, 128)];
    let mut b = hybrid();
    b.kv_layer_dims = vec![(4, 128), (2, 256)];
    assert_ne!(fp(&a), fp(&b));
}

// ── L4: encoding injectivity (length-prefixed strings) ─────────────────
// Under naive concatenation ("ab" + "c") == ("a" + "bc"); the tagged,
// length-prefixed encoding must keep them distinct.
#[test]
fn string_encoding_is_injective() {
    let mut a = hybrid();
    a.model_type = "ab".into();
    a.quantization_config = Some(QuantizationConfig {
        quant_method: "c".into(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    let mut b = hybrid();
    b.model_type = "a".into();
    b.quantization_config = Some(QuantizationConfig {
        quant_method: "bc".into(),
        quant_algo: String::new(),
        format: String::new(),
        ignore_modules: Vec::new(),
    });
    assert_ne!(fp(&a), fp(&b));
}

// ── Fail-fast: an underivable config is a hard error (PCND) ───────────
#[test]
fn underivable_config_fails_fast() {
    let mut c = hybrid();
    c.model_type = String::new();
    c.num_hidden_layers = 0;
    c.layer_types = Vec::new();
    let err = ModelFingerprint::derive_with_id(&c, BLOB, "").unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("ATLAS_SSM_SWAP_NS"),
        "actionable message: {msg}"
    );
}

// ── Decode namespace: domain-separated mix, never the bare constant ────
#[test]
fn decode_ns_mixes_fingerprint_and_domain() {
    let fa = ModelFingerprint::derive_with_id(&hybrid(), BLOB, "").unwrap();
    let fb = ModelFingerprint::derive_with_id(&dense(), BLOB, "").unwrap();
    let da = mix64(fa.get(), atlas_kernels::DECODE_DOMAIN);
    let db = mix64(fb.get(), atlas_kernels::DECODE_DOMAIN);
    // Separated from the same model's Marconi keys (shared peer residency)…
    assert_ne!(da, fa.get());
    // …never the bare constant (the old cross-model decode collision)…
    assert_ne!(da, atlas_kernels::DECODE_DOMAIN);
    // …and distinct across models.
    assert_ne!(da, db);
}

// ── Override precedence + strict parsing ───────────────────────────────
#[test]
fn override_precedence_and_strict_parse() {
    let derived = NonZeroU64::new(0xFEED).unwrap();
    // No override → derived fingerprint namespace.
    assert_eq!(resolve_ns_from(None, "V", derived).unwrap(), derived);
    // Explicit decimal and 0x-hex overrides win.
    assert_eq!(resolve_ns_from(Some("42"), "V", derived).unwrap().get(), 42);
    assert_eq!(
        resolve_ns_from(Some("0xD3C0"), "V", derived).unwrap().get(),
        0xD3C0
    );
    // Unparseable is a hard error, not a silent fallthrough (PCND).
    assert!(resolve_ns_from(Some("banana"), "V", derived).is_err());
    assert!(resolve_ns_from(Some("-1"), "V", derived).is_err());
    assert!(resolve_ns_from(Some("18446744073709551616"), "V", derived).is_err());
    // 0 is a hard error: the passthrough is removed.
    let err = resolve_ns_from(Some("0"), "V", derived).unwrap_err();
    assert!(format!("{err:#}").contains("passthrough"));
}

/// The decode namespace must never equal the Marconi (swap) namespace for the
/// same model: the two tiers share a peer, and an alias would let a decode
/// rollback blob be served as a Marconi snapshot. The zero-avoidance fallback in
/// `resolve_decode_ns` previously collapsed onto `fp` itself, which IS the swap
/// namespace — so the guard is on the fallback, not just the happy path.
#[test]
fn decode_ns_never_aliases_the_swap_ns() {
    for cfg in [hybrid(), dense()] {
        let f = ModelFingerprint::derive_with_id(&cfg, BLOB, "").unwrap();
        let swap = resolve_swap_ns(f).unwrap();
        let decode = resolve_decode_ns(f).unwrap();
        assert_ne!(
            swap, decode,
            "decode namespace aliased the swap namespace for {}",
            cfg.model_type
        );
        // And the swap ns is the fingerprint itself (no override set).
        assert_eq!(swap.get(), f.get());
    }
}
