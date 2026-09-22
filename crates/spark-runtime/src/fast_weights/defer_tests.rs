// SPDX-License-Identifier: AGPL-3.0-only

//! The model-supplied defer hook, end to end through a real shard.
//!
//! 🔴 Two claims, and the second is the one that cannot be checked by reading
//! the predicate: a deferred tensor is NOT on the device, and it IS in the
//! store's `deferred` map with an offset that reads back the right bytes. A
//! predicate that matched and then dropped the locator would look identical
//! from the allocation ledger and fail at bind time with a missing-name error.

use std::sync::Arc;

use super::FastSafetensorsLoader;
use crate::gpu::mock::MockGpuBackend;
use crate::weights::{WeightDtype, WeightLoader};

/// Write a one-shard safetensors model dir. Each entry is
/// `(name, dtype string, shape, bytes)`, laid out in the order given.
fn write_shard(tag: &str, entries: &[(&str, &str, Vec<usize>, Vec<u8>)]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("avarok-defer-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut blob = (header.len() as u64).to_le_bytes().to_vec();
    blob.extend_from_slice(&header);
    blob.extend_from_slice(&data);
    std::fs::write(dir.join("model.safetensors"), blob).unwrap();
    dir
}

fn bf16(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect()
}

#[test]
fn no_hook_means_nothing_is_deferred() {
    let l = FastSafetensorsLoader::new();
    assert!(!l.is_deferred("anything.weight", WeightDtype::BF16));
    assert!(FastSafetensorsLoader::new().defer.is_none());
}

/// The claimed tensor stays on disk with a working locator; its neighbour is
/// uploaded exactly as before.
#[test]
fn a_claimed_tensor_is_recorded_and_not_uploaded() {
    let claimed = bf16(&[1.0, -2.0, 0.5, 0.25]);
    let kept = bf16(&[3.0, 4.0]);
    let dir = write_shard(
        "claimed",
        &[
            ("big.weight", "BF16", vec![2, 2], claimed.clone()),
            ("small.weight", "BF16", vec![2], kept.clone()),
        ],
    );

    let mut loader = FastSafetensorsLoader::new();
    loader.defer = Some(Arc::new(|name: &str, dtype: WeightDtype| {
        name == "big.weight" && dtype == WeightDtype::BF16
    }));
    let gpu = MockGpuBackend::new();
    let store = loader.load(&dir, &gpu, 0).unwrap();

    assert!(
        !store.contains("big.weight"),
        "a deferred tensor must not be uploaded"
    );
    assert!(store.contains("small.weight"));
    let d = store.deferred("big.weight").expect("locator recorded");
    assert_eq!(d.shape, vec![2, 2]);
    assert_eq!(d.dtype, WeightDtype::BF16);
    assert_eq!(d.read_host_bytes().unwrap(), claimed);

    let _ = std::fs::remove_dir_all(&dir);
}

/// 🪤 The hook is asked only about tensors this rank KEEPS. Asked first, it
/// would record a remote expert's on-disk location and invite a binder to read
/// weights EP gave to another rank.
#[test]
fn a_tensor_this_rank_skips_is_never_deferred() {
    let w = bf16(&[1.0, 2.0]);
    let dir = write_shard(
        "ep",
        &[
            (
                "m.layers.0.mlp.experts.0.gate_proj.weight",
                "BF16",
                vec![2],
                w.clone(),
            ),
            (
                "m.layers.0.mlp.experts.3.gate_proj.weight",
                "BF16",
                vec![2],
                w.clone(),
            ),
        ],
    );

    // Rank 0 of 2 over 4 experts owns 0..2, so expert 3 is skipped outright.
    let mut loader = FastSafetensorsLoader::with_ep(0, 2, 4);
    loader.defer = Some(Arc::new(|_: &str, _: WeightDtype| true));
    let gpu = MockGpuBackend::new();
    let store = loader.load(&dir, &gpu, 0).unwrap();

    assert!(
        store
            .deferred("m.layers.0.mlp.experts.0.gate_proj.weight")
            .is_some(),
        "a local expert the hook claims IS deferred"
    );
    assert!(
        store
            .deferred("m.layers.0.mlp.experts.3.gate_proj.weight")
            .is_none(),
        "a remote expert belongs to the other rank and must not be recorded"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// 🪤 F16 is the one width the loaders REWRITE on the way to the store, so a
/// (path, offset) locator would hand its reader F16 where the engine expects
/// BF16. It is never deferred — and the pre-flight counts it for the same
/// reason, which is what keeps the two halves the same rule.
#[test]
fn an_f16_tensor_is_uploaded_rather_than_deferred() {
    let raw: Vec<u8> = [1.0f32, 2.0]
        .iter()
        .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
        .collect();
    let dir = write_shard("f16", &[("h.weight", "F16", vec![2], raw)]);

    let mut loader = FastSafetensorsLoader::new();
    loader.defer = Some(Arc::new(|_: &str, _: WeightDtype| true));
    let gpu = MockGpuBackend::new();
    let store = loader.load(&dir, &gpu, 0).unwrap();

    assert!(store.deferred("h.weight").is_none());
    assert_eq!(store.get("h.weight").unwrap().dtype, WeightDtype::BF16);

    let _ = std::fs::remove_dir_all(&dir);
}
