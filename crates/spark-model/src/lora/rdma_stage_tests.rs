// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for `rdma_stage`: manifest → pool-slot landing targets and the
//! post-reload per-layer pair rebuild.

use super::*;
use spark_storage::weight_peer::{WeightManifest, WeightTensorRecord};

// Real factory config (layer 3,7,… are FullAttention). Offset math only
// needs layer_type + projection dims, so the family (MoE here) is irrelevant.
fn cfg() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

fn rec(name: &str, shape: Vec<u64>) -> WeightTensorRecord {
    WeightTensorRecord {
        name: name.into(),
        dtype: "F32".into(),
        shape,
        offset_in_shard: 0,
        len: 0,
        shard_index: 0,
        extra: false,
    }
}

#[test]
fn land_targets_map_to_slot_subregions() {
    let cfg = cfg();
    // Layer 3 is FullAttention in the factory config.
    let layer = 3usize;
    assert_eq!(
        cfg.layer_type(layer),
        atlas_core::config::LayerType::FullAttention
    );
    let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
    let max_rank = 8;
    let r = 4u64;
    let pool = DevicePtr(0x1_0000);
    let manifest = WeightManifest {
        version: WeightManifest::VERSION,
        model_id: "adp".into(),
        shard_files: vec!["adapter_model.safetensors".into()],
        shard_lens: vec![0],
        tensors: vec![
            rec(
                &format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_A.weight"),
                vec![r, in_dim as u64],
            ),
            rec(
                &format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_B.weight"),
                vec![out_dim as u64, r],
            ),
        ],
    };
    let targets = build_land_targets(&manifest, &cfg, pool, 1, max_rank).unwrap();
    assert_eq!(targets.len(), 2);
    // Slot 1 base = pool + 1*slot_bytes.
    let base = pool.0 + slot_base_offset(1, &cfg, max_rank) as u64;
    let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
    let a = targets.iter().find(|t| t.kind == LoraAbKind::A).unwrap();
    let b = targets.iter().find(|t| t.kind == LoraAbKind::B).unwrap();
    assert_eq!(a.dst, base + a_off as u64);
    assert_eq!(b.dst, base + b_off as u64);
    assert_eq!(a.rank, r as usize);
    assert_eq!(b.rank, r as usize);
    assert_eq!(a.max_rank, max_rank);
}

#[test]
fn rebuild_slot_layers_sets_rank_and_pointers() {
    let cfg = cfg();
    let layer = 3usize;
    let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
    let max_rank = 8;
    let pool = DevicePtr(0x2_0000);
    let base = pool.0 + slot_base_offset(2, &cfg, max_rank) as u64;
    let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
    let targets = vec![
        LoraLandTarget {
            tensor_name: "a".into(),
            kind: LoraAbKind::A,
            dst: base + a_off as u64,
            out_dim,
            in_dim,
            rank: 4,
            max_rank,
        },
        LoraLandTarget {
            tensor_name: "b".into(),
            kind: LoraAbKind::B,
            dst: base + b_off as u64,
            out_dim,
            in_dim,
            rank: 4,
            max_rank,
        },
    ];
    let peft = PeftAdapterConfig {
        r: 4,
        lora_alpha: 8.0,
        target_modules: vec!["k_proj".into()],
        use_rslora: false,
        layers_to_transform: None,
    };
    let layers = rebuild_slot_layers(&targets, &cfg, &peft, pool, 2, max_rank).unwrap();
    let lw = layers[layer].as_ref().expect("layer 3 rebuilt");
    let pair = lw.k_proj.expect("k_proj pair");
    assert_eq!(pair.rank, 4);
    assert_eq!(pair.max_rank, max_rank as u32);
    assert_eq!(pair.a.weight.0, base + a_off as u64);
    assert_eq!(pair.b.weight.0, base + b_off as u64);
}
