// SPDX-License-Identifier: AGPL-3.0-only

use super::{
    MODEL_REQUEST_MAX, read_model_request, read_weight_manifest, write_model_request,
    write_weight_manifest,
};
use crate::weight_peer::{WeightManifest, WeightTensorRecord};

fn sample_manifest() -> WeightManifest {
    WeightManifest {
        version: WeightManifest::VERSION,
        model_id: "qwen3.6-35b-a3b".to_string(),
        shard_files: vec![
            "model.safetensors-00001-of-00002.safetensors".to_string(),
            "model.safetensors-00002-of-00002.safetensors".to_string(),
        ],
        shard_lens: vec![1_000_000, 2_000_000],
        tensors: vec![
            WeightTensorRecord {
                name: "model.embed_tokens.weight".to_string(),
                dtype: "BF16".to_string(),
                shape: vec![152064, 4096],
                offset_in_shard: 4096,
                len: 152064 * 4096 * 2,
                shard_index: 0,
                extra: false,
            },
            WeightTensorRecord {
                name: "mtp.experts.5.gate_proj.weight_packed".to_string(),
                dtype: "I8".to_string(),
                shape: vec![2048, 1024],
                offset_in_shard: 8192,
                len: 2048 * 1024,
                shard_index: 1,
                extra: true,
            },
        ],
    }
}

#[test]
fn manifest_round_trips() {
    let m = sample_manifest();
    let mut buf = Vec::new();
    write_weight_manifest(&mut buf, &m).unwrap();
    let back = read_weight_manifest(&mut &buf[..]).unwrap();
    assert_eq!(m, back);
}

#[test]
fn manifest_rejects_bad_version() {
    let mut m = sample_manifest();
    m.version = 999;
    let mut buf = Vec::new();
    write_weight_manifest(&mut buf, &m).unwrap();
    assert!(read_weight_manifest(&mut &buf[..]).is_err());
}

#[test]
fn manifest_rejects_shard_len_mismatch() {
    let mut m = sample_manifest();
    m.shard_lens.pop(); // now 2 files, 1 len
    let mut buf = Vec::new();
    write_weight_manifest(&mut buf, &m).unwrap();
    assert!(read_weight_manifest(&mut &buf[..]).is_err());
}

#[test]
fn model_request_round_trips() {
    let mut buf = Vec::new();
    write_model_request(&mut buf, "/tank/models/qwen3.6-35b-a3b").unwrap();
    let back = read_model_request(&mut &buf[..]).unwrap();
    assert_eq!(back, "/tank/models/qwen3.6-35b-a3b");
}

#[test]
fn model_request_rejects_empty() {
    let mut buf = Vec::new();
    assert!(write_model_request(&mut buf, "").is_err());
}

#[test]
fn model_request_rejects_oversize_read() {
    // A hostile length prefix must not attempt a giant allocation.
    let mut buf = Vec::new();
    buf.extend_from_slice(&(MODEL_REQUEST_MAX as u32 + 1).to_le_bytes());
    assert!(read_model_request(&mut &buf[..]).is_err());
}
