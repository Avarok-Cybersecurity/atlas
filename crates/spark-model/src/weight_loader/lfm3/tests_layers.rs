// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: layer assembly and the loader's trait surface on the mock
//! backend: the attention layer is wired like the HF block (norm order,
//! sigmoid head gate, window, RoPE table per layer kind), the vendor FP8
//! checkpoint leaves the BF16 slots null, `lfm3_fp8_all` quantizes everything
//! but the gate, and the tied head is the embedding table.
//!
//! The conv layers (lane L38) are not built here: the models under test hold
//! attention layers only, so these tests do not depend on `Lfm3ConvLayer`
//! behavior (the crate must still compile against it).

use std::collections::HashMap;

use avarok_core::config::ModelConfig;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightStore, WeightTensor};

use super::fp8::{Fp8Loader, Routes};
use super::load_layers::{Levers, load_with};
use super::names::Geometry;
use super::rope_tables::RopeTables;
use super::tests::{Ckpt, QUANT, bf16, mini, ptr_of, quantize_launches, store_for};
use super::{Lfm3WeightLoader, SLIDING_ROPE_THETA, attention, ffn};
use crate::layers::Qwen3AttentionLayer;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::weight_loader::ModelWeightLoader;

fn tables(gpu: &MockGpuBackend, store: &WeightStore, c: &ModelConfig) -> RopeTables {
    let g = Geometry::of(c);
    RopeTables::upload(
        gpu,
        store,
        g.head_dim,
        c.rope_theta,
        c.partial_rotary_factor,
        SLIDING_ROPE_THETA,
    )
    .expect("rope tables")
}

fn build_attention(
    store: &WeightStore,
    gpu: &MockGpuBackend,
    c: &ModelConfig,
    fp8: &Fp8Loader<'_>,
    layer: usize,
    rope: RopeTables,
) -> Qwen3AttentionLayer {
    let g = Geometry::of(c);
    let ffn = ffn::build(fp8, gpu, &g, &c.layer_prefix(layer)).expect("ffn");
    let args = attention::AttnArgs {
        config: c,
        geometry: g,
        layer,
        attn_idx: layer,
        kind: c.layer_type(layer),
        kv_dtype: KvCacheDtype::Bf16,
        rope,
    };
    attention::build(store, gpu, fp8, &args, ffn).expect("attention layer")
}

#[test]
fn a_bf16_attention_layer_is_wired_like_the_hf_block() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let store = store_for(&gpu, &c, Ckpt::Bf16);
    let fp8 = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let rope = tables(&gpu, &store, &c);

    // Layer 0: full attention.
    let full = build_attention(&store, &gpu, &c, &fp8, 0, rope);
    assert_eq!(full.sliding_window, None);
    assert_eq!(full.rope_theta_override, Some(1_000_000.0));
    assert_eq!(
        full.rotary_dim_override,
        Some(64),
        "the whole head; the half rotation is the table's zero tail"
    );
    assert_eq!(full.yarn_inv_freq, rope.full);
    assert_eq!(full.yarn_attention_factor, 1.0);
    assert_eq!(full.num_q_heads_override, Some(4));
    assert_eq!(full.head_dim_override, Some(64));
    assert_eq!(full.num_kv_heads_override, Some(2));
    assert_eq!(full.head_gate_activation, HeadGateActivation::Sigmoid);
    assert_eq!(
        full.head_gate_weight.unwrap().weight,
        ptr_of(&store, "model.layers.0.self_attn.gate_proj.weight")
    );
    // Sandwich norms: HF post_operator_norm / post_ffn_norm are the post norms.
    assert_eq!(
        full.post_attn_out_norm.unwrap().weight,
        ptr_of(&store, "model.layers.0.post_operator_norm.weight")
    );
    assert_eq!(
        full.post_ffn_out_norm.unwrap().weight,
        ptr_of(&store, "model.layers.0.post_ffn_norm.weight")
    );
    // BF16 projections and QK norms are the checkpoint's own bytes.
    assert_eq!(
        full.attn.q_proj.weight,
        ptr_of(&store, "model.layers.0.self_attn.q_proj.weight")
    );
    assert_eq!(
        full.attn.q_norm.weight,
        ptr_of(&store, "model.layers.0.self_attn.q_norm.weight")
    );

    // Layer 1: sliding attention.
    let sliding = build_attention(&store, &gpu, &c, &fp8, 1, rope);
    assert_eq!(sliding.sliding_window, Some(32));
    assert_eq!(sliding.rope_theta_override, Some(10_000.0));
    assert_eq!(sliding.rotary_dim_override, Some(64));
    assert_eq!(sliding.yarn_inv_freq, rope.sliding);
    assert_ne!(rope.sliding, rope.full);
    assert!(quantize_launches(&gpu).is_empty());
}

#[test]
fn the_vendor_checkpoint_leaves_bf16_slots_null_and_quantizes_nothing() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let store = store_for(&gpu, &c, Ckpt::Vendor);
    let fp8 = Fp8Loader::with_routes(&store, &gpu, Routes::default(), QUANT);
    let rope = tables(&gpu, &store, &c);
    let layer = build_attention(&store, &gpu, &c, &fp8, 0, rope);
    assert!(
        layer.attn.q_proj.weight.is_null(),
        "the E4M3 overlay replaces the BF16 slot"
    );
    assert!(layer.attn.k_proj.weight.is_null());
    assert!(layer.attn.v_proj.weight.is_null());
    assert!(quantize_launches(&gpu).is_empty());
    // The head gate is never quantized: it is the checkpoint's BF16 tensor.
    assert_eq!(
        layer.head_gate_weight.unwrap().weight,
        ptr_of(&store, "model.layers.0.self_attn.gate_proj.weight")
    );
}

#[test]
fn lfm3_fp8_all_on_a_bf16_checkpoint_quantizes_ffn_and_attention_but_not_the_gate() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let store = store_for(&gpu, &c, Ckpt::Bf16);
    let routes = Routes {
        ffn: true,
        attn: true,
        conv: true,
    };
    let fp8 = Fp8Loader::with_routes(&store, &gpu, routes, QUANT);
    let rope = tables(&gpu, &store, &c);
    let gate_ptr = ptr_of(&store, "model.layers.0.self_attn.gate_proj.weight");
    let q_ptr = ptr_of(&store, "model.layers.0.self_attn.q_proj.weight");
    let layer = build_attention(&store, &gpu, &c, &fp8, 0, rope);
    // 3 FFN + 4 attention projections.
    assert_eq!(quantize_launches(&gpu).len(), 7);
    assert!(layer.attn.q_proj.weight.is_null());
    assert_eq!(layer.head_gate_weight.unwrap().weight, gate_ptr);
    assert!(
        gpu.read_alloc(gate_ptr).is_some(),
        "the gate is never freed"
    );
    assert!(
        gpu.read_alloc(q_ptr).is_none(),
        "the BF16 q_proj source is freed after its quantize"
    );
}

// ── the trait surface ──

#[test]
fn the_loader_reads_the_lfm3_names_and_ties_the_head() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let store = store_for(&gpu, &c, Ckpt::Bf16);
    let loader = Lfm3WeightLoader;
    assert!(!loader.supports_tp());
    assert!(loader.load_mtp_weights(&store, &c, &gpu).unwrap().is_none());
    let embed = ptr_of(&store, "model.embed_tokens.weight");
    assert_eq!(
        loader.load_embedding(&store, &c, &gpu).unwrap().weight,
        embed
    );
    assert_eq!(
        loader.load_final_norm(&store, &c, &gpu).unwrap().weight,
        ptr_of(&store, "model.embedding_norm.weight")
    );
    // Tied: no lm_head.weight, the head IS the embedding table.
    assert_eq!(loader.load_lm_head(&store, &c, &gpu).unwrap().weight, embed);
}

#[test]
fn a_shipped_head_wins_over_the_embedding() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let head: WeightTensor = bf16(&gpu, &[384, 256]);
    let head_ptr = head.ptr;
    let mut map = HashMap::new();
    map.insert(
        "model.embed_tokens.weight".to_string(),
        bf16(&gpu, &[384, 256]),
    );
    map.insert("lm_head.weight".to_string(), head);
    let store = WeightStore::from_map(map);
    let got = Lfm3WeightLoader.load_lm_head(&store, &c, &gpu).unwrap();
    assert_eq!(got.weight, head_ptr);
}

fn off() -> Levers {
    Levers {
        all: false,
        ffn: false,
        attn: false,
    }
}

#[test]
fn load_layers_checks_the_kv_dtype_count_before_touching_weights() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let store = store_for(&gpu, &c, Ckpt::Bf16);
    let e = load_with(&store, &c, &gpu, &[KvCacheDtype::Bf16], off())
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("one KV dtype per attention layer"), "{e}");
    assert!(quantize_launches(&gpu).is_empty());
}

#[test]
fn load_layers_builds_every_attention_layer_of_a_conv_free_model() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let store = store_for(&gpu, &c, Ckpt::Bf16);
    let dtypes = [KvCacheDtype::Bf16, KvCacheDtype::Bf16];
    let layers = load_with(&store, &c, &gpu, &dtypes, off()).expect("load");
    assert_eq!(layers.len(), 2);
    // The two RoPE tables are the only derived allocations on a BF16 load.
    let owned: usize = store.derived().by_label().iter().map(|(_, b, _)| b).sum();
    assert_eq!(owned, 2 * (64 / 2) * 4);
}

#[test]
fn a_checkpoint_that_does_not_match_its_config_fails_before_any_layer_is_built() {
    let gpu = MockGpuBackend::new();
    let c = mini();
    let mut map = HashMap::new();
    map.insert(
        "model.embed_tokens.weight".to_string(),
        bf16(&gpu, &[384, 256]),
    );
    let store = WeightStore::from_map(map);
    let dtypes = [KvCacheDtype::Bf16, KvCacheDtype::Bf16];
    let e = load_with(&store, &c, &gpu, &dtypes, off())
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("does not match its config"), "{e}");
}
