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

fn manifest(tensors: Vec<WeightTensorRecord>) -> WeightManifest {
    WeightManifest {
        version: WeightManifest::VERSION,
        model_id: "adp".into(),
        shard_files: vec!["adapter_model.safetensors".into()],
        shard_lens: vec![0],
        tensors,
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
    let manifest = manifest(vec![
        rec(
            &format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_A.weight"),
            vec![r, in_dim as u64],
        ),
        rec(
            &format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_B.weight"),
            vec![out_dim as u64, r],
        ),
    ]);
    let targets = build_land_targets(&manifest, &cfg, pool, 1, max_rank).unwrap();
    assert_eq!(targets.len(), 2);
    // Slot 1 base = pool + 1*slot_bytes.
    let base = pool.0 + slot_base_offset(1, &cfg, max_rank) as u64;
    let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
    let a = targets.iter().find(|t| t.kind == LoraAbKind::A).unwrap();
    let b = targets.iter().find(|t| t.kind == LoraAbKind::B).unwrap();
    assert_eq!(
        a.tensor_name,
        format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_A.weight")
    );
    assert_eq!(
        b.tensor_name,
        format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_B.weight")
    );
    assert_eq!(a.dst, base + a_off as u64);
    assert_eq!(b.dst, base + b_off as u64);
    for target in [a, b] {
        assert_eq!(target.out_dim, out_dim);
        assert_eq!(target.in_dim, in_dim);
        assert_eq!(target.rank, r as usize);
        assert_eq!(target.max_rank, max_rank);
    }
}

#[test]
fn land_targets_reject_malformed_or_incomplete_pairs() {
    let cfg = cfg();
    let layer = 3usize;
    let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
    let a_name = format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_A.weight");
    let b_name = format!("base_model.model.model.layers.{layer}.self_attn.k_proj.lora_B.weight");
    let valid_a = || rec(&a_name, vec![4, in_dim as u64]);
    let valid_b = || rec(&b_name, vec![out_dim as u64, 4]);
    let cases = [
        (
            "wrong A input width",
            vec![rec(&a_name, vec![4, in_dim as u64 + 1]), valid_b()],
            "REJECT[shape-mismatch]",
        ),
        (
            "wrong B output width",
            vec![valid_a(), rec(&b_name, vec![out_dim as u64 + 1, 4])],
            "REJECT[shape-mismatch]",
        ),
        (
            "extra A dimension",
            vec![rec(&a_name, vec![4, in_dim as u64, 1]), valid_b()],
            "REJECT[shape-mismatch]",
        ),
        (
            "zero rank",
            vec![
                rec(&a_name, vec![0, in_dim as u64]),
                rec(&b_name, vec![out_dim as u64, 0]),
            ],
            "REJECT[shape-mismatch]",
        ),
        (
            "mismatched A/B ranks",
            vec![valid_a(), rec(&b_name, vec![out_dim as u64, 3])],
            "REJECT[rank-mismatch]",
        ),
        ("missing B half", vec![valid_a()], "REJECT[unpaired-tensor]"),
        (
            "duplicate A half",
            vec![valid_a(), valid_a(), valid_b()],
            "REJECT[duplicate-tensor]",
        ),
    ];

    for (case, tensors, expected) in cases {
        let err = build_land_targets(&manifest(tensors), &cfg, DevicePtr(0x1_0000), 1, 8)
            .expect_err(case);
        assert!(
            err.to_string().contains(expected),
            "{case}: expected {expected}, got {err}"
        );
    }
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
        target_modules_pattern: None,
        use_rslora: false,
        layers_to_transform: None,
        trainable_token_indices: Vec::new(),
        modules_to_save: Vec::new(),
        lora_embedding: false,
    };
    let layers = rebuild_slot_layers(&targets, &cfg, &peft, pool, 2, max_rank).unwrap();
    assert_eq!(layers.len(), cfg.num_hidden_layers);
    assert_eq!(
        layers.iter().filter(|layer| layer.is_some()).count(),
        1,
        "only the target's layer should be rebuilt"
    );
    let lw = layers[layer].as_ref().expect("layer 3 rebuilt");
    assert_eq!(lw.layer_idx, layer);
    assert!(lw.q_proj.is_none());
    assert!(lw.v_proj.is_none());
    assert!(lw.o_proj.is_none());
    assert!(lw.gate_proj.is_none());
    assert!(lw.up_proj.is_none());
    assert!(lw.down_proj.is_none());
    let pair = lw.k_proj.expect("k_proj pair");
    assert_eq!(pair.rank, 4);
    assert_eq!(pair.k_in, in_dim as u32);
    assert_eq!(pair.n_out, out_dim as u32);
    assert_eq!(pair.scale, peft.scaling());
    assert_eq!(pair.max_rank, max_rank as u32);
    assert_eq!(pair.a.weight.0, base + a_off as u64);
    assert_eq!(pair.b.weight.0, base + b_off as u64);
}

#[test]
fn rebuild_slot_layers_rejects_config_rank_mismatch() {
    let cfg = cfg();
    let layer = 3usize;
    let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
    let max_rank = 8;
    let pool = DevicePtr(0x2_0000);
    let base = pool.0 + slot_base_offset(2, &cfg, max_rank) as u64;
    let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
    let target = |kind, dst| LoraLandTarget {
        tensor_name: "tensor".into(),
        kind,
        dst,
        out_dim,
        in_dim,
        rank: 4,
        max_rank,
    };
    let targets = vec![
        target(LoraAbKind::A, base + a_off as u64),
        target(LoraAbKind::B, base + b_off as u64),
    ];
    let peft = PeftAdapterConfig {
        r: 3,
        lora_alpha: 6.0,
        target_modules: vec!["k_proj".into()],
        // An explicit list, not a pattern: these fixtures exercise the
        // name-level path, which is the one the rank check runs under.
        target_modules_pattern: None,
        use_rslora: false,
        layers_to_transform: None,
        trainable_token_indices: Vec::new(),
        modules_to_save: Vec::new(),
        lora_embedding: false,
    };

    let err = match rebuild_slot_layers(&targets, &cfg, &peft, pool, 2, max_rank) {
        Ok(_) => panic!("config rank mismatch accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("REJECT[rank-mismatch]"));
}

#[test]
fn rebuild_slot_layers_rejects_inconsistent_target_geometry() {
    let cfg = cfg();
    let layer = 3usize;
    let (out_dim, in_dim) = LoraModule::KProj.dims(&cfg);
    let max_rank = 8;
    let pool = DevicePtr(0x2_0000);
    let base = pool.0 + slot_base_offset(2, &cfg, max_rank) as u64;
    let (a_off, b_off) = module_slot_offsets(&cfg, max_rank, layer, LoraModule::KProj).unwrap();
    let mut targets = vec![
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
    targets[1].max_rank -= 1;
    let peft = PeftAdapterConfig {
        r: 4,
        lora_alpha: 8.0,
        target_modules: vec!["k_proj".into()],
        // An explicit list, not a pattern: these fixtures exercise the
        // name-level path, which is the one the rank check runs under.
        target_modules_pattern: None,
        use_rslora: false,
        layers_to_transform: None,
        trainable_token_indices: Vec::new(),
        modules_to_save: Vec::new(),
        lora_embedding: false,
    };

    let err = match rebuild_slot_layers(&targets, &cfg, &peft, pool, 2, max_rank) {
        Ok(_) => panic!("inconsistent target geometry accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("REJECT[landing-geometry]"));
}
