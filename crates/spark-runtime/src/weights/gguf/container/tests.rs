// SPDX-License-Identifier: AGPL-3.0-only
//! Synthetic-container parse tests for the GGUF v3 parser.

use super::*;

fn push_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Build a minimal synthetic GGUF (header + 4 KVs + 1 tensor, no data).
/// Returns `(bytes, header_end)` where `header_end` is the position just
/// after the tensor directory, before alignment padding.
fn synthetic() -> (Vec<u8>, usize) {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC.to_le_bytes()); // magic
    b.extend_from_slice(&3u32.to_le_bytes()); // version
    b.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    b.extend_from_slice(&4u64.to_le_bytes()); // kv_count

    // 1) general.architecture : string
    push_str(&mut b, "general.architecture");
    b.extend_from_slice(&8u32.to_le_bytes());
    push_str(&mut b, "llama");

    // 2) llama.block_count : u32
    push_str(&mut b, "llama.block_count");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&32u32.to_le_bytes());

    // 3) test.answer : array<u32> [1,2,3]
    push_str(&mut b, "test.answer");
    b.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
    b.extend_from_slice(&4u32.to_le_bytes()); // elem type = u32
    b.extend_from_slice(&3u64.to_le_bytes()); // len
    for v in [1u32, 2, 3] {
        b.extend_from_slice(&v.to_le_bytes());
    }

    // 4) test.flag : bool true
    push_str(&mut b, "test.flag");
    b.extend_from_slice(&7u32.to_le_bytes());
    b.push(1);

    // tensor: token_embd.weight, ndim=2, dims=[8,4], type F32(0), offset 0
    push_str(&mut b, "token_embd.weight");
    b.extend_from_slice(&2u32.to_le_bytes());
    b.extend_from_slice(&8u64.to_le_bytes());
    b.extend_from_slice(&4u64.to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());

    let header_end = b.len();
    // Pad to alignment 32 and append a byte of "data" so data_offset is in-bounds.
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }
    b.push(0);
    (b, header_end)
}

#[test]
fn parses_metadata_and_tensor_dir() {
    let (bytes, header_end) = synthetic();
    let f = GgufFile::parse(&bytes).expect("parse ok");

    assert_eq!(f.version, 3);
    assert_eq!(f.get_str("general.architecture"), Some("llama"));
    assert_eq!(f.get_u32("llama.block_count"), Some(32));
    assert_eq!(f.get_u32_array("test.answer"), Some(vec![1, 2, 3]));
    assert_eq!(f.get_bool("test.flag"), Some(true));
    assert_eq!(f.get_str("missing"), None);

    assert_eq!(f.tensors.len(), 1);
    let t = f.tensor("token_embd.weight").expect("tensor present");
    assert_eq!(t.dims, vec![8, 4]);
    assert_eq!(t.ggml_type, GgmlType::F32);
    assert_eq!(t.offset, 0);
    assert_eq!(t.num_elements(), 32);

    // default alignment, data section aligned right after the header.
    assert_eq!(f.alignment, 32);
    assert_eq!(f.data_offset, align_up(header_end, 32));
    assert_eq!(f.data_offset % 32, 0);
    assert_eq!(f.tensor_abs_offset(t), f.data_offset);

    // F32 byte size = 32 elems * 4 bytes.
    assert_eq!(f.tensor_byte_size(t, Q2Group::G128).unwrap(), 128);
}

#[test]
fn typed_getters_widen_integers() {
    let f = GgufFile {
        version: 3,
        metadata: vec![
            ("a".into(), MetaValue::U8(7)),
            ("b".into(), MetaValue::I32(-5)),
            ("c".into(), MetaValue::F32(1.5)),
            ("d".into(), MetaValue::U64(9)),
        ],
        tensors: vec![],
        alignment: 32,
        data_offset: 0,
    };
    assert_eq!(f.get_u32("a"), Some(7));
    assert_eq!(f.get_u32("b"), None); // negative -> not a u32
    assert_eq!(f.get_f64("c"), Some(1.5));
    assert_eq!(f.get_u64("d"), Some(9));
    assert_eq!(f.get_f64("d"), Some(9.0)); // integer widened to float
}

#[test]
fn nested_array_of_arrays() {
    let mut b = Vec::new();
    b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    b.extend_from_slice(&1u64.to_le_bytes()); // 1 kv

    push_str(&mut b, "nested");
    b.extend_from_slice(&9u32.to_le_bytes()); // outer ARRAY
    b.extend_from_slice(&9u32.to_le_bytes()); // elem type = ARRAY
    b.extend_from_slice(&2u64.to_le_bytes()); // 2 inner arrays
    for pair in [[10u32, 11], [12, 13]] {
        b.extend_from_slice(&4u32.to_le_bytes()); // inner elem = u32
        b.extend_from_slice(&2u64.to_le_bytes()); // inner len
        for v in pair {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    while !b.len().is_multiple_of(32) {
        b.push(0);
    }

    let f = GgufFile::parse(&b).expect("parse nested");
    let outer = f.get("nested").and_then(MetaValue::as_array).unwrap();
    assert_eq!(outer.len(), 2);
    let inner0 = outer[0].as_array().unwrap();
    assert_eq!(inner0[0], MetaValue::U32(10));
    assert_eq!(inner0[1], MetaValue::U32(11));
    assert_eq!(f.arr_len("nested"), Some(2));
}

#[test]
fn rejects_bad_magic() {
    let mut b = vec![0u8; 24];
    b[0] = b'X';
    assert!(GgufFile::parse(&b).is_err());
}

#[test]
fn truncated_buffer_errors_not_panics() {
    let (bytes, _) = synthetic();
    // Chop mid-metadata; must return Err, never panic.
    assert!(GgufFile::parse(&bytes[..30]).is_err());
}

#[test]
fn ggml_type_ids_roundtrip() {
    for id in [0u32, 1, 2, 3, 6, 7, 8, 9, 10, 14, 30, 34, 35, 41, 42] {
        let t = GgmlType::from_id(id).unwrap();
        assert_eq!(t.id(), id);
    }
    assert!(GgmlType::from_id(999).is_err());
}

#[test]
fn q2_0_group_selects_block_layout() {
    assert_eq!(
        GgmlType::Q2_0.block_layout(Q2Group::G128).unwrap(),
        (128, 34)
    );
    assert_eq!(GgmlType::Q2_0.block_layout(Q2Group::G64).unwrap(), (64, 18));
    assert_eq!(
        GgmlType::Q6_K.block_layout(Q2Group::G128).unwrap(),
        (256, 210)
    );
    assert_eq!(
        GgmlType::Q8_0.block_layout(Q2Group::G128).unwrap(),
        (32, 34)
    );
    // IQ family intentionally has no block layout here.
    assert!(GgmlType::Iq2Xxs.block_layout(Q2Group::G128).is_err());
}
