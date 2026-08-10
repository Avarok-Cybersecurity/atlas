// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the standalone GGUF reader, driven by a synthetic in-memory
//! container (no on-disk model required).

use super::read_gguf_f32;
use half::f16;

/// Build a minimal GGUF v3 buffer with one F32 tensor and one F16 tensor,
/// plus a single `general.alignment=32` metadata pair.
fn synth_gguf() -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic "GGUF"
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&2u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&1u64.to_le_bytes()); // kv_count

    // metadata: general.alignment (u32=4) = 32
    let key = b"general.alignment";
    b.extend_from_slice(&(key.len() as u64).to_le_bytes());
    b.extend_from_slice(key);
    b.extend_from_slice(&4u32.to_le_bytes()); // value type u32
    b.extend_from_slice(&32u32.to_le_bytes());

    // tensor dir: two entries, offsets are relative to the data section.
    let push_tensor = |b: &mut Vec<u8>, name: &[u8], dims: &[u64], type_id: u32, off: u64| {
        b.extend_from_slice(&(name.len() as u64).to_le_bytes());
        b.extend_from_slice(name);
        b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend_from_slice(&d.to_le_bytes());
        }
        b.extend_from_slice(&type_id.to_le_bytes());
        b.extend_from_slice(&off.to_le_bytes());
    };
    // F32 tensor "a": dims [3], values 1,2,3 at offset 0
    push_tensor(&mut b, b"a", &[3], 0, 0);
    // F16 tensor "b": dims [2], values 4,5 at offset 16 (32-aligned after 3xf32)
    push_tensor(&mut b, b"b", &[2], 1, 16);

    // pad to 32-byte alignment for the data section
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    let data_start = b.len();
    // tensor "a" data (3 x f32) at rel offset 0
    for v in [1.0f32, 2.0, 3.0] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    // pad to rel offset 16 (where tensor "b" begins)
    while b.len() - data_start < 16 {
        b.push(0);
    }
    // tensor "b" data (2 x f16) at rel offset 16
    for v in [4.0f32, 5.0] {
        b.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    b
}

#[test]
fn reads_f32_and_f16_tensors() {
    let bytes = synth_gguf();
    let dir = std::env::temp_dir().join(format!("nllb_gguf_test_{}.gguf", std::process::id()));
    std::fs::write(&dir, &bytes).unwrap();
    let tensors = read_gguf_f32(&dir).unwrap();
    std::fs::remove_file(&dir).ok();

    assert_eq!(tensors.len(), 2);
    let a = tensors.iter().find(|t| t.name == "a").unwrap();
    assert_eq!(a.dims, vec![3]);
    assert_eq!(a.data, vec![1.0, 2.0, 3.0]);
    let bt = tensors.iter().find(|t| t.name == "b").unwrap();
    assert_eq!(bt.dims, vec![2]);
    assert_eq!(bt.data, vec![4.0, 5.0]);
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = synth_gguf();
    bytes[0] = 0;
    let p = std::env::temp_dir().join(format!("nllb_bad_{}.gguf", std::process::id()));
    std::fs::write(&p, &bytes).unwrap();
    let err = read_gguf_f32(&p).unwrap_err().to_string();
    std::fs::remove_file(&p).ok();
    assert!(err.contains("not a GGUF file"), "got: {err}");
}
