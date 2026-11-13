// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: CPU tests of the LFM3 loader's pure parts (the RoPE
//! tables, the tensor map against the real header's 286 tensors, the FP8 route
//! table) and the mock-backend fixtures `tests_fp8.rs` and `tests_layers.rs`
//! share. No GPU: `MockGpuBackend`.

use std::collections::HashMap;

use avarok_core::config::{LayerType, ModelConfig, parse_config};
use spark_runtime::gpu::mock::{MockGpuBackend, MockLaunch};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::SLIDING_ROPE_THETA;
use super::fp8::{Inputs, Routes, fp8_weight_bytes, qualifies, routes};
use super::names::{self, Geometry, Role, global_expectations, layer_expectations};
use super::rope_tables::{RopeTables, full_table, rotated_pairs, sliding_table};

/// `LiquidAI/LFM3-1.7B` `config.json`.
const REAL_CONFIG: &str = r#"{
  "bos_token_id": 124894, "conv_L_cache": 7, "conv_bias": false, "eos_token_id": 124900,
  "head_dim": 128, "hidden_size": 2048, "intermediate_size": 7680,
  "layer_types": ["conv","conv","full_attention","conv","sliding_attention","conv",
    "full_attention","conv","sliding_attention","conv","full_attention","conv",
    "sliding_attention","conv","full_attention","conv","sliding_attention","conv",
    "full_attention","conv","sliding_attention","conv","full_attention","conv"],
  "max_position_embeddings": 262144, "model_type": "lfm3", "norm_eps": 1e-05,
  "num_attention_heads": 16, "num_hidden_layers": 24, "num_key_value_heads": 4,
  "rope_parameters": {
    "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}},
  "sliding_window": 256, "tie_word_embeddings": true, "use_sandwich_norm": true,
  "vocab_size": 128000
}"#;

/// A two-layer (full, sliding) model with 128-multiple extents, small enough
/// to allocate on the mock: hidden 256, 4 query heads, 2 KV heads, head 64.
const MINI_CONFIG: &str = r#"{
  "conv_L_cache": 7, "eos_token_id": 3, "head_dim": 64, "hidden_size": 256,
  "intermediate_size": 512, "layer_types": ["full_attention", "sliding_attention"],
  "max_position_embeddings": 4096, "model_type": "lfm3", "norm_eps": 1e-05,
  "num_attention_heads": 4, "num_hidden_layers": 2, "num_key_value_heads": 2,
  "sliding_window": 32, "tie_word_embeddings": true, "vocab_size": 384
}"#;

pub(super) fn real() -> ModelConfig {
    parse_config(REAL_CONFIG).expect("real LFM3 config")
}

pub(super) fn mini() -> ModelConfig {
    parse_config(MINI_CONFIG).expect("mini LFM3 config")
}

// ── mock-backend fixtures (shared) ──

/// The quantizer handle the FP8 tests install (distinct from the mock's
/// default 0xDEAD, so its launches can be told apart from the widen kernel's).
pub(super) const QUANT: KernelHandle = KernelHandle(0xF8);

pub(super) fn bf16(gpu: &MockGpuBackend, shape: &[usize]) -> WeightTensor {
    let bytes: usize = shape.iter().product::<usize>() * 2;
    WeightTensor {
        ptr: gpu.alloc(bytes).unwrap(),
        shape: shape.to_vec(),
        dtype: WeightDtype::BF16,
    }
}

pub(super) fn fp8_tensor(gpu: &MockGpuBackend, n: usize, k: usize) -> WeightTensor {
    WeightTensor {
        ptr: gpu.alloc(n * k).unwrap(),
        shape: vec![n, k],
        dtype: WeightDtype::FP8E4M3,
    }
}

pub(super) fn store_of(entries: Vec<(&str, WeightTensor)>) -> WeightStore {
    WeightStore::from_map(
        entries
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
    )
}

pub(super) fn quantize_launches(gpu: &MockGpuBackend) -> Vec<MockLaunch> {
    gpu.launches_snapshot()
        .into_iter()
        .filter(|l| l.func == QUANT.0)
        .collect()
}

pub(super) fn ptr_of(store: &WeightStore, name: &str) -> DevicePtr {
    store.get(name).unwrap().ptr
}

/// Which flavor of checkpoint [`store_for`] builds.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Ckpt {
    /// Everything BF16 (`LiquidAI/LFM3-1.7B`).
    Bf16,
    /// E4M3 FFN and attention projections with their block-scale grids, the
    /// conv projections, gate, norms and embedding BF16 (`-FP8`).
    Vendor,
}

/// A store holding every tensor `config` expects.
pub(super) fn store_for(gpu: &MockGpuBackend, config: &ModelConfig, ckpt: Ckpt) -> WeightStore {
    let g = Geometry::of(config);
    let mut expects = global_expectations(&g);
    for i in 0..config.num_hidden_layers {
        expects.extend(layer_expectations(
            &g,
            &config.layer_prefix(i),
            config.layer_type(i),
        ));
    }
    let mut map = HashMap::new();
    for e in expects {
        let fp8 = ckpt == Ckpt::Vendor
            && e.role == Role::Proj
            && (e.name.contains("feed_forward") || e.name.contains("self_attn"));
        let dtype = if fp8 {
            WeightDtype::FP8E4M3
        } else {
            WeightDtype::BF16
        };
        let numel: usize = e.shape.iter().product();
        if fp8 {
            let prefix = e.name.strip_suffix(".weight").unwrap().to_string();
            let grid = [e.shape[0].div_ceil(128), e.shape[1].div_ceil(128)];
            map.insert(format!("{prefix}.weight_scale"), bf16(gpu, &grid));
        }
        let ptr = gpu.alloc(numel * dtype.byte_size()).unwrap();
        map.insert(
            e.name,
            WeightTensor {
                ptr,
                shape: e.shape,
                dtype,
            },
        );
    }
    WeightStore::from_map(map)
}

// ── RoPE tables ──

#[test]
fn rotated_pairs_follow_hf_int_division() {
    assert_eq!(rotated_pairs(128, 0.5), 32);
    assert_eq!(rotated_pairs(128, 1.0), 64);
    assert_eq!(rotated_pairs(128, 0.25), 16);
    assert_eq!(rotated_pairs(64, 0.5), 16);
}

#[test]
fn sliding_table_is_the_full_head_progression() {
    let t = sliding_table(10_000.0, 128);
    assert_eq!(t.len(), 64);
    assert_eq!(t[0], 1.0);
    for (j, v) in t.iter().enumerate() {
        let want = (1.0f64 / 10_000f64.powf((2 * j) as f64 / 128.0)) as f32;
        assert_eq!(*v, want, "pair {j}");
        assert!(*v > 0.0);
    }
    assert!(t.windows(2).all(|w| w[1] < w[0]), "strictly decreasing");
}

#[test]
fn full_table_keeps_the_whole_head_progression_then_exact_zeros() {
    let t = full_table(1_000_000.0, 128, 0.5);
    assert_eq!(t.len(), 64);
    for (j, v) in t.iter().enumerate().take(32) {
        let want = (1.0f64 / 1_000_000f64.powf((2 * j) as f64 / 128.0)) as f32;
        assert_eq!(*v, want, "pair {j}");
    }
    for (j, v) in t.iter().enumerate().skip(32) {
        assert_eq!(
            v.to_bits(),
            0.0f32.to_bits(),
            "pair {j} must be exactly +0.0"
        );
    }
    // Not the usual "rotary_dim = 64" progression (which divides by 64).
    let shrunk = (1.0f64 / 1_000_000f64.powf(2.0 / 64.0)) as f32;
    assert_ne!(t[1], shrunk);
}

#[test]
fn a_zero_frequency_pair_is_an_exact_identity_rotation() {
    for pos in [0u32, 1, 255, 4096, 262_143] {
        let angle = pos as f32 * 0.0f32;
        assert_eq!(angle.cos(), 1.0);
        assert_eq!(angle.sin(), 0.0);
    }
}

#[test]
fn the_tables_upload_once_and_the_store_owns_them() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::empty();
    let r = RopeTables::upload(&gpu, &store, 128, 1_000_000.0, 0.5, SLIDING_ROPE_THETA)
        .expect("upload");
    assert_eq!(r.rotary_dim, 128);
    let read = |p: DevicePtr| -> Vec<f32> {
        gpu.read_alloc(p)
            .expect("live table")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    assert_eq!(read(r.sliding), sliding_table(SLIDING_ROPE_THETA, 128));
    assert_eq!(read(r.full), full_table(1_000_000.0, 128, 0.5));
    let owned: usize = store.derived().by_label().iter().map(|(_, b, _)| b).sum();
    assert_eq!(owned, 2 * 64 * 4);
    assert!(RopeTables::upload(&gpu, &store, 127, 1e6, 0.5, 1e4).is_err());
}

// ── tensor map ──

/// Run `names::problems` over a lookup that serves every expected tensor as
/// BF16 with its expected shape, after `mutate` has had a go at each answer.
fn problems_with(config: &ModelConfig, mutate: impl Fn(&str, &mut names::Found)) -> Vec<String> {
    let g = Geometry::of(config);
    let mut expects = global_expectations(&g);
    for i in 0..config.num_hidden_layers {
        expects.extend(layer_expectations(
            &g,
            &config.layer_prefix(i),
            config.layer_type(i),
        ));
    }
    let table: HashMap<String, (Vec<usize>, WeightDtype)> = expects
        .iter()
        .map(|e| (e.name.clone(), (e.shape.clone(), WeightDtype::BF16)))
        .collect();
    names::problems(config, &|name: &str| {
        let mut found = table.get(name).cloned();
        mutate(name, &mut found);
        found
    })
}

#[test]
fn the_real_config_expects_exactly_the_286_tensors_of_the_header() {
    let c = real();
    let g = Geometry::of(&c);
    let mut n = global_expectations(&g).len();
    for i in 0..c.num_hidden_layers {
        n += layer_expectations(&g, &c.layer_prefix(i), c.layer_type(i)).len();
    }
    // 2 global + 13 conv x 10 + 11 attention x 14 (the header counts 286).
    assert_eq!(n, 286);
    assert!(problems_with(&c, |_, _| {}).is_empty());
}

#[test]
fn conv_taps_are_accepted_flat_or_with_the_unit_dim() {
    let c = real();
    let flat = problems_with(&c, |n, f| {
        if n.ends_with("conv.conv.weight") {
            *f = Some((vec![2048, 7], WeightDtype::BF16));
        }
    });
    assert!(flat.is_empty(), "{flat:?}");
    let wrong = problems_with(&c, |n, f| {
        if n.ends_with("layers.0.conv.conv.weight") {
            *f = Some((vec![2048, 1, 5], WeightDtype::BF16));
        }
    });
    assert_eq!(wrong.len(), 1);
    assert!(wrong[0].contains("layers.0.conv.conv.weight"), "{wrong:?}");
}

#[test]
fn missing_wrong_shape_and_wrong_dtype_are_all_reported() {
    let c = real();
    let out = problems_with(&c, |n, f| {
        if n == "model.layers.2.self_attn.gate_proj.weight" {
            *f = None;
        }
        if n == "model.layers.2.self_attn.k_proj.weight" {
            *f = Some((vec![2048, 2048], WeightDtype::BF16));
        }
        if n == "model.layers.3.ffn_norm.weight" {
            *f = Some((vec![2048], WeightDtype::FP8E4M3));
        }
    });
    assert_eq!(out.len(), 3, "{out:?}");
    assert!(
        out.iter()
            .any(|s| s.contains("gate_proj") && s.contains("missing"))
    );
    assert!(
        out.iter()
            .any(|s| s.contains("k_proj") && s.contains("[2048, 2048]"))
    );
    assert!(
        out.iter()
            .any(|s| s.contains("ffn_norm") && s.contains("dtype"))
    );
}

#[test]
fn fp8_projections_are_fine_but_fp8_embeddings_are_not() {
    let c = real();
    let ok = problems_with(&c, |n, f| {
        if (n.contains("feed_forward") || n.contains("self_attn.q_proj"))
            && let Some((shape, _)) = f.clone()
        {
            *f = Some((shape, WeightDtype::FP8E4M3));
        }
    });
    assert!(ok.is_empty(), "{ok:?}");
    let bad = problems_with(&c, |n, f| {
        if n == "model.embed_tokens.weight" {
            *f = Some((vec![128_000, 2048], WeightDtype::FP8E4M3));
        }
    });
    assert_eq!(bad.len(), 1, "{bad:?}");
}

#[test]
fn an_untied_config_needs_its_head_and_a_tied_one_may_ship_it() {
    let mut c = real();
    c.tie_word_embeddings = false;
    let missing = problems_with(&c, |_, _| {});
    assert_eq!(missing.len(), 1);
    assert!(missing[0].contains("lm_head.weight"), "{missing:?}");
    let with_head = problems_with(&c, |n, f| {
        if n == names::LM_HEAD {
            *f = Some((vec![128_000, 2048], WeightDtype::BF16));
        }
    });
    assert!(with_head.is_empty(), "{with_head:?}");
    let bad_head = problems_with(&real(), |n, f| {
        if n == names::LM_HEAD {
            *f = Some((vec![128_000, 1024], WeightDtype::BF16));
        }
    });
    assert_eq!(bad_head.len(), 1, "{bad_head:?}");
}

#[test]
fn validate_lists_at_most_twelve_problems_and_counts_the_rest() {
    let c = real();
    let store = WeightStore::from_map(HashMap::new());
    let err = names::validate(&store, &c)
        .expect_err("empty store")
        .to_string();
    assert!(err.contains("286 problem(s)"), "{err}");
    assert!(err.contains("and 274 more"), "{err}");
}

#[test]
fn the_layer_schedule_maps_to_the_existing_layer_types() {
    let c = real();
    assert_eq!(c.layer_type(0), LayerType::LinearAttention);
    assert_eq!(c.layer_type(2), LayerType::FullAttention);
    assert_eq!(c.layer_type(4), LayerType::SlidingAttention);
    assert_eq!(c.num_attention_layers(), 11);
}

// ── FP8 route table ──

fn inputs() -> Inputs {
    Inputs {
        all: false,
        ffn_lever: false,
        attn_lever: false,
        single_gpu: true,
        lora: false,
        kernels: true,
    }
}

#[test]
fn the_route_is_off_unless_a_lever_asks() {
    assert_eq!(routes(inputs()), Routes::default());
}

#[test]
fn lfm3_fp8_all_turns_every_class_on_and_the_old_levers_their_own() {
    let all = routes(Inputs {
        all: true,
        ..inputs()
    });
    assert_eq!(
        all,
        Routes {
            ffn: true,
            attn: true,
            conv: true
        }
    );
    let ffn = routes(Inputs {
        ffn_lever: true,
        ..inputs()
    });
    assert_eq!(
        ffn,
        Routes {
            ffn: true,
            attn: false,
            conv: false
        }
    );
    let attn = routes(Inputs {
        attn_lever: true,
        ..inputs()
    });
    assert_eq!(
        attn,
        Routes {
            ffn: false,
            attn: true,
            conv: false
        }
    );
}

#[test]
fn the_route_declines_whole_for_tp_lora_or_missing_kernels() {
    for declined in [
        Inputs {
            all: true,
            single_gpu: false,
            ..inputs()
        },
        Inputs {
            all: true,
            lora: true,
            ..inputs()
        },
        Inputs {
            all: true,
            kernels: false,
            ..inputs()
        },
    ] {
        assert_eq!(routes(declined), Routes::default(), "{declined:?}");
    }
}

#[test]
fn every_lfm3_projection_is_a_whole_128_block() {
    for (n, k) in [
        (2048, 2048),
        (512, 2048),
        (7680, 2048),
        (2048, 7680),
        (6144, 2048),
    ] {
        assert!(qualifies(n, k), "[{n}, {k}]");
    }
    assert!(!qualifies(16, 2048), "the head gate is never quantized");
    assert!(!qualifies(2048, 100));
    assert_eq!(fp8_weight_bytes(7680, 2048), (7680 * 2048, 60 * 16 * 4));
}
