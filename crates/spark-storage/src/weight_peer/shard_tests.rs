// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use std::io::Write;

/// Write a minimal safetensors file: `[u64 LE header_len][header][data]`.
fn write_st(path: &Path, header: &str, data: &[u8]) {
    let hb = header.as_bytes();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    f.write_all(hb).unwrap();
    f.write_all(data).unwrap();
}

/// A shard header may list tensors the index `weight_map` does not route
/// (orphan/tied/aux). The disk loaders iterate weight_map keys and never
/// load those; `build_manifest` must filter identically so the RDMA store
/// is byte-identical (same key set, not a superset).
#[test]
fn build_manifest_filters_orphan_tensors() {
    let dir = std::env::temp_dir().join(format!("wpeer-orphan-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let shard = "model-00001.safetensors";
    // Two f32[2] tensors in the header; only `a.weight` is in the index.
    let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"orphan.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    write_st(&dir.join(shard), header, &[0u8; 16]);
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        format!(r#"{{"weight_map":{{"a.weight":"{shard}"}}}}"#),
    )
    .unwrap();

    let (_paths, manifest) = build_manifest(&dir, "test").unwrap();
    let names: Vec<&str> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["a.weight"],
        "orphan tensor (not in weight_map) must not be published"
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// With no index (single-file checkpoint) every header tensor is kept —
/// there is no weight_map to filter against.
#[test]
fn build_manifest_keeps_all_when_no_index() {
    let dir = std::env::temp_dir().join(format!("wpeer-single-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let header = r#"{"a.weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b.weight":{"dtype":"F32","shape":[2],"data_offsets":[8,16]}}"#;
    write_st(&dir.join("model.safetensors"), header, &[0u8; 16]);

    let (_paths, manifest) = build_manifest(&dir, "test").unwrap();
    let mut names: Vec<&str> = manifest.tensors.iter().map(|t| t.name.as_str()).collect();
    names.sort();
    assert_eq!(names, vec!["a.weight", "b.weight"]);

    std::fs::remove_dir_all(&dir).ok();
}
