// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the out-of-index sidecar registration (split from
//! `exl3_sidecar_shards.rs` for the 500-LoC cap). The discovery rule is
//! tested purely; the registration end to end against real safetensors
//! bytes in a temp dir + the mock backend, through `materialize_exl3`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;

fn set(names: &[&str]) -> HashSet<String> {
    names.iter().map(|s| s.to_string()).collect()
}

#[test]
fn discovery_rule_selects_unindexed_sidecars_only() {
    // The 4.05bpw_h6_ng6 tree: 9 index-listed shards + ngram + vision_k6.
    let listing = [
        "config.json",
        "model-00001-of-00009.safetensors",
        "model-00009-of-00009.safetensors",
        "model.safetensors.index.json",
        "ngram_embedding.safetensors",
        "ngram_embedding.safetensors.tank-orig",
        "preprocessor_config.json",
        "vision_k6.safetensors",
    ];
    let index = set(&[
        "model-00001-of-00009.safetensors",
        "model-00009-of-00009.safetensors",
    ]);
    assert_eq!(
        select_exl3_sidecar_shards(listing, &index),
        vec!["vision_k6.safetensors".to_string()]
    );

    // The 3.05/2.05 trees: the mixer patch is the one un-indexed file.
    let listing = [
        "model-00007-of-00007.safetensors",
        "mtp_hyper_connection_mixer_patch.safetensors",
        "ngram_embedding.safetensors",
        "extra_weights.safetensors",
    ];
    let index = set(&["model-00007-of-00007.safetensors"]);
    assert_eq!(
        select_exl3_sidecar_shards(listing, &index),
        vec!["mtp_hyper_connection_mixer_patch.safetensors".to_string()]
    );
}

#[test]
fn discovery_rule_never_double_reads_the_main_loaders_files() {
    // Un-indexed checkpoints: the main loader reads model.safetensors or the
    // bare model.safetensors-*/consolidated-* shards itself.
    let listing = [
        "model.safetensors",
        "model.safetensors-00001-of-00002",
        "consolidated-00001-of-00002.safetensors",
        "consolidated.safetensors",
        "extra_weights.safetensors",
        "ngram_embedding.safetensors",
        "vision.safetensors",
        "vision_k6.safetensors",
        "vision_k6.safetensors", // duplicate listing entries collapse
    ];
    assert_eq!(
        select_exl3_sidecar_shards(listing, &HashSet::new()),
        vec![
            "vision.safetensors".to_string(),
            "vision_k6.safetensors".to_string()
        ]
    );
    // An index-listed shard is never a sidecar even under an odd name.
    let index = set(&["vision_k6.safetensors"]);
    assert!(select_exl3_sidecar_shards(["vision_k6.safetensors"], &index).is_empty());
}

#[test]
fn vision_sidecar_name_shape() {
    assert!(is_exl3_vision_sidecar_name("vision_k6.safetensors"));
    assert!(is_exl3_vision_sidecar_name("vision.safetensors"));
    assert!(!is_exl3_vision_sidecar_name("vision_k6.safetensors.bak"));
    assert!(!is_exl3_vision_sidecar_name(
        "mtp_hyper_connection_mixer_patch.safetensors"
    ));
    assert!(!is_exl3_vision_sidecar_name(
        "model-00001-of-00009.safetensors"
    ));
}

// ── End-to-end against real safetensors bytes ──

/// Minimal safetensors writer: `(name, dtype, shape, bytes)` in order,
/// contiguous offsets, header padded to 8 bytes (what the crate's
/// `deserialize` validates).
fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !hdr.len().is_multiple_of(8) {
        hdr.push(b' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(&hdr);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn temp_model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atlas-exl3-sidecar-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A 4.05-shaped directory: an index naming one (absent) shard, a bogus
/// `ngram_embedding.safetensors` that must never be opened, the vision
/// sidecar and the MTP mixer patch.
fn write_fixture_dir(tag: &str) -> PathBuf {
    let dir = temp_model_dir(tag);
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"metadata":{},"weight_map":{"model.embed_tokens.weight":"model-00001-of-00001.safetensors"}}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("ngram_embedding.safetensors"),
        b"not a safetensors file",
    )
    .unwrap();
    std::fs::write(dir.join("config.json"), b"{}").unwrap();

    // f16 1.0 / -2.0 / 0.5 / 0.0 — exactly representable in bf16.
    let qkv: Vec<u16> = vec![
        0x3C00, 0xC000, 0x3800, 0x0000, 0x3C00, 0xC000, 0x3800, 0x0000,
    ];
    // One K=6 trellis linear at [in=128 -> out=256]: trellis I16
    // [in/16, out/16, 16*K], suh f16 [in], svh f16 [out], mul1 I32 scalar.
    let fc1 = "model.visual.blocks.0.mlp.linear_fc1";
    let (trellis, suh, svh, mul1) = (
        format!("{fc1}.trellis"),
        format!("{fc1}.suh"),
        format!("{fc1}.svh"),
        format!("{fc1}.mul1"),
    );
    write_safetensors(
        &dir.join("vision_k6.safetensors"),
        &[
            (
                "model.visual.blocks.0.attn.qkv.weight",
                "F16",
                vec![2, 4],
                le16(&qkv),
            ),
            (
                "model.visual.blocks.0.attn.qkv.bias",
                "F16",
                vec![2],
                le16(&[0x3C00, 0x3C00]),
            ),
            (
                "model.visual.pos_embed.weight",
                "F16",
                vec![2, 4],
                le16(&qkv),
            ),
            (
                trellis.as_str(),
                "I16",
                vec![8, 16, 96],
                vec![0u8; 8 * 16 * 96 * 2],
            ),
            (suh.as_str(), "F16", vec![128], le16(&[0x3C00; 128])),
            (svh.as_str(), "F16", vec![256], le16(&[0x3C00; 256])),
            (
                mul1.as_str(),
                "I32",
                vec![],
                0x83DC_D12Du32.to_le_bytes().to_vec(),
            ),
        ],
    );
    write_safetensors(
        &dir.join("mtp_hyper_connection_mixer_patch.safetensors"),
        &[(
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "F16",
            vec![4],
            le16(&[0x3C00; 4]),
        )],
    );
    dir
}

fn read_u16s(gpu: &MockGpuBackend, t: &WeightTensor) -> Vec<u16> {
    let mut buf = vec![0u8; t.num_elements() * 2];
    gpu.copy_d2h(t.ptr, &mut buf).unwrap();
    buf.chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[test]
fn registers_vision_sidecar_then_materialize_sees_its_trellis() {
    let dir = write_fixture_dir("e2e");
    let gpu = MockGpuBackend::new();
    // The store as the main loader left it: the index-listed embed plus a
    // pos_embed that ALSO appears in the sidecar (index wins).
    let mut m = HashMap::new();
    let pre = WeightTensor {
        ptr: gpu.alloc(16).unwrap(),
        shape: vec![2, 4],
        dtype: WeightDtype::BF16,
    };
    let pre_ptr = pre.ptr;
    m.insert("model.visual.pos_embed.weight".to_string(), pre);
    m.insert(
        "model.embed_tokens.weight".to_string(),
        WeightTensor {
            ptr: gpu.alloc(16).unwrap(),
            shape: vec![2, 4],
            dtype: WeightDtype::BF16,
        },
    );
    let mut store = WeightStore::from_map(m);

    // qwen4_exp's loader policy: no `mtp.*`.
    let skip_mtp = |n: &str| n.starts_with("mtp.");
    let stats = register_exl3_sidecar_shards(&gpu, &mut store, &dir, 0, &skip_mtp).unwrap();
    assert_eq!(
        stats,
        Exl3SidecarStats {
            files: 2,
            tensors: 6,
            vision_tensors: 6,
            already_present: 1,
            skipped_by_policy: 1,
        }
    );

    // Idempotent: a second registration finds every name present and
    // uploads nothing (the mixer patch is still refused by policy).
    let again = register_exl3_sidecar_shards(&gpu, &mut store, &dir, 0, &skip_mtp).unwrap();
    assert_eq!(
        again,
        Exl3SidecarStats {
            files: 2,
            tensors: 0,
            vision_tensors: 0,
            already_present: 7,
            skipped_by_policy: 1,
        }
    );

    // (3) The namespace probe's exact key is now in the store; the mixer
    // patch tensor is not (policy); the pre-existing pos_embed survived.
    assert!(store.contains("model.visual.blocks.0.attn.qkv.weight"));
    assert!(!store.contains("mtp.hyper_connection_mixer.hc_norm.weight"));
    assert_eq!(
        store.get("model.visual.pos_embed.weight").unwrap().ptr,
        pre_ptr
    );
    let report = crate::weight_loader::qwen4_exp::audit_namespace(
        &store,
        &atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4(),
    );
    assert_eq!(report.vision_tensors, 1);

    // Dtypes: F16 fused qkv -> BF16 with converted bits; `.suh` stays F16;
    // trellis I16 -> UInt16; mul1 -> Int32.
    let qkv = store.get("model.visual.blocks.0.attn.qkv.weight").unwrap();
    assert_eq!(qkv.dtype, WeightDtype::BF16);
    assert_eq!(qkv.shape, vec![2, 4]);
    assert_eq!(
        read_u16s(&gpu, qkv),
        vec![
            0x3F80, 0xC000, 0x3F00, 0x0000, 0x3F80, 0xC000, 0x3F00, 0x0000
        ]
    );
    let fc1 = "model.visual.blocks.0.mlp.linear_fc1";
    assert_eq!(
        store.get(&format!("{fc1}.suh")).unwrap().dtype,
        WeightDtype::F16
    );
    assert_eq!(
        store.get(&format!("{fc1}.trellis")).unwrap().dtype,
        WeightDtype::UInt16
    );
    assert_eq!(
        store.get(&format!("{fc1}.mul1")).unwrap().dtype,
        WeightDtype::Int32
    );

    // (1)+(2): materialize now walks the sidecar's trellis like an
    // index-listed one — the ViT fc1 lands as BF16 `[out, in]` with the
    // TRELLIS width (what `load_vision_encoder` sizes the MLP from), and the
    // packed sources are gone.
    let mstats = super::super::materialize_exl3_impl(
        &gpu,
        &mut store,
        false,
        false,
        crate::weight_map::Exl3DenseFamilies::OFF,
    )
    .unwrap();
    assert_eq!(mstats.bf16, 1);
    let w = store.get(&format!("{fc1}.weight")).unwrap();
    assert_eq!(w.dtype, WeightDtype::BF16);
    assert_eq!(w.shape, vec![256, 128]);
    for sfx in ["trellis", "suh", "svh", "mul1"] {
        assert!(
            !store.contains(&format!("{fc1}.{sfx}")),
            "{fc1}.{sfx} remains"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn no_policy_registers_the_mixer_patch_too() {
    let dir = write_fixture_dir("nopolicy");
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::from_map(HashMap::new());
    let stats = register_exl3_sidecar_shards(&gpu, &mut store, &dir, 0, &|_| false).unwrap();
    assert_eq!(stats.files, 2);
    assert_eq!(stats.tensors, 8);
    assert_eq!(stats.vision_tensors, 7);
    assert_eq!(stats.skipped_by_policy, 0);
    let mixer = store
        .get("mtp.hyper_connection_mixer.hc_norm.weight")
        .unwrap();
    assert_eq!(mixer.dtype, WeightDtype::BF16); // F16 patch -> BF16 ingest
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn directory_without_sidecars_is_a_noop() {
    let dir = temp_model_dir("none");
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("ngram_embedding.safetensors"), b"never opened").unwrap();
    std::fs::write(dir.join("extra_weights.safetensors"), b"never opened here").unwrap();
    let gpu = MockGpuBackend::new();
    let mut store = WeightStore::from_map(HashMap::new());
    let stats = register_exl3_sidecar_shards(&gpu, &mut store, &dir, 0, &|_| false).unwrap();
    assert_eq!(stats, Exl3SidecarStats::default());
    assert!(store.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
