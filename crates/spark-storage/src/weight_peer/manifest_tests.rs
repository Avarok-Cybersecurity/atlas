// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

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
fn total_shard_bytes_sums_lens() {
    let m = sample_manifest();
    assert_eq!(m.total_shard_bytes(), 3_000_000);
    assert_eq!(m.num_shards(), 2);
}

#[test]
fn tensor_remote_addr_adds_absolute_offset() {
    // remote_addr = shard MR base + the tensor's ABSOLUTE in-shard offset.
    // The offset already includes the 8-byte size prefix + header, so no
    // extra rebasing — a bug here reads header bytes off the shard front.
    assert_eq!(tensor_remote_addr(0x1_0000_0000, 4096), 0x1_0000_1000);
    assert_eq!(tensor_remote_addr(0, 0), 0);
    // Matches the manifest records verbatim (offset_in_shard is absolute).
    let m = sample_manifest();
    let base = 0xdead_0000u64;
    let t = &m.tensors[0];
    assert_eq!(tensor_remote_addr(base, t.offset_in_shard), base + 4096);
}

#[test]
fn rail_for_tensor_stripes_round_robin() {
    // Single rail: everything on rail 0.
    for i in 0..5 {
        assert_eq!(rail_for_tensor(i, 1), 0);
    }
    // Dual rail: even → 0, odd → 1.
    assert_eq!(rail_for_tensor(0, 2), 0);
    assert_eq!(rail_for_tensor(1, 2), 1);
    assert_eq!(rail_for_tensor(2, 2), 0);
    assert_eq!(rail_for_tensor(3, 2), 1);
    // n_rails == 0 must not divide-by-zero (clamped to 1).
    assert_eq!(rail_for_tensor(7, 0), 0);
}
