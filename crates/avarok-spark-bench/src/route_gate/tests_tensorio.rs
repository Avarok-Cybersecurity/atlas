// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: unit tests of the safetensors reader and writer (port of the Python `SafetensorsTests`, plus the
//! Rust-only accessors: names, has, raw, BF16 bit patterns and `need`).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::json;

use super::error::GateError;
use super::tensorio::{OutTensor, Tensor, TensorFile, need, write_safetensors};

fn meta(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn named(name: &str, t: OutTensor) -> (String, OutTensor) {
    (name.to_string(), t)
}

fn roundtrip_file(dir: &Path) -> std::path::PathBuf {
    let p = dir.join("t.safetensors");
    let a: Vec<f64> = (0..6).map(|i| f64::from(i) / 7.0).collect();
    let tensors = vec![
        named("a.f64", OutTensor::f64(&[2, 3], a)),
        named("b.f32", OutTensor::f32(&[2], vec![1.5, -2.0])),
        named("c.i64", OutTensor::i64(&[2, 2], vec![1, 2, 3, 4])),
        named("d.u8", OutTensor::u8(&[3], vec![0, 1, 255])),
        named("e.empty", OutTensor::i64(&[0], vec![])),
        named(
            "f.bf16",
            OutTensor::bf16(&[4], vec![1.0, -2.5, 0.15625, 0.0]),
        ),
    ];
    write_safetensors(&p, &tensors, &meta(&[("k", "v")])).unwrap();
    p
}

#[test]
fn roundtrip() {
    let d = tempfile::tempdir().unwrap();
    let p = roundtrip_file(d.path());
    let f = TensorFile::open(&p).unwrap();
    let mut want_meta = HashMap::new();
    want_meta.insert("k".to_string(), "v".to_string());
    assert_eq!(f.metadata(), want_meta);
    assert_eq!(f.shape("a.f64").unwrap(), vec![2, 3]);
    assert_eq!(f.dtype("f.bf16").unwrap(), "BF16");

    let want: Vec<(&str, Vec<usize>, Vec<f64>)> = vec![
        (
            "a.f64",
            vec![2, 3],
            (0..6).map(|i| f64::from(i) / 7.0).collect(),
        ),
        ("b.f32", vec![2], vec![1.5, -2.0]),
        ("c.i64", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ("d.u8", vec![3], vec![0.0, 1.0, 255.0]),
        ("e.empty", vec![0], vec![]),
        ("f.bf16", vec![4], vec![1.0, -2.5, 0.15625, 0.0]), // these values are exact in BF16
    ];
    for (name, shape, data) in want {
        let got = f.read(name).unwrap();
        assert_eq!(got.shape, shape, "{name}");
        assert_eq!(got.data, data, "{name}");
    }

    let stripped: Vec<String> = f.read_prefix("a.", true).unwrap().into_keys().collect();
    assert_eq!(stripped, ["f64"]);
    let kept: Vec<String> = f.read_prefix("a.", false).unwrap().into_keys().collect();
    assert_eq!(kept, ["a.f64"]);
    assert!(matches!(f.read("missing"), Err(GateError::Safetensors(_))));
}

#[test]
fn bf16_rounds_to_nearest_even() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("bf.safetensors");
    // between two BF16 values (spacing 2^-7): 1/4 and 3/4 of a step round down and up
    let x = vec![1.0 + 2.0f32.powi(-9), 1.0 + 3.0 * 2.0f32.powi(-9)];
    write_safetensors(
        &p,
        &[named("x", OutTensor::bf16(&[2], x))],
        &BTreeMap::new(),
    )
    .unwrap();
    let f = TensorFile::open(&p).unwrap();
    assert_eq!(f.read("x").unwrap().data, vec![1.0, 1.0 + 2.0f64.powi(-7)]);

    // exact ties go to the even mantissa: 1 + 1/2 step -> 1.0, 1 + 3/2 steps -> 1 + 2 steps
    let ties = vec![1.0 + 2.0f32.powi(-8), 1.0 + 3.0 * 2.0f32.powi(-8)];
    write_safetensors(
        &p,
        &[named("x", OutTensor::bf16(&[2], ties))],
        &BTreeMap::new(),
    )
    .unwrap();
    let f = TensorFile::open(&p).unwrap();
    assert_eq!(f.read("x").unwrap().data, vec![1.0, 1.0 + 2.0f64.powi(-6)]);
}

fn raw_file(dir: &Path, header: serde_json::Value, payload: &[u8]) -> std::path::PathBuf {
    let p = dir.join("bad.safetensors");
    let mut h = header.to_string().into_bytes();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&h);
    bytes.extend_from_slice(payload);
    std::fs::write(&p, bytes).unwrap();
    p
}

fn refused(p: &Path) -> bool {
    matches!(TensorFile::open(p), Err(GateError::Safetensors(_)))
}

#[test]
fn rejects_bad_offsets_and_overlap() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path();
    // offsets beyond the buffer
    let p = raw_file(
        dir,
        json!({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}),
        &[0u8; 4],
    );
    assert!(refused(&p), "offsets beyond the buffer");
    // size does not match shape
    let p = raw_file(
        dir,
        json!({"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}}),
        &[0u8; 8],
    );
    assert!(refused(&p), "size does not match shape");
    // overlap
    let p = raw_file(
        dir,
        json!({"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
               "y": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]}}),
        &[0u8; 12],
    );
    assert!(refused(&p), "overlap");
    // unsupported dtype
    let p = raw_file(
        dir,
        json!({"x": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}),
        &[0u8; 1],
    );
    assert!(refused(&p), "unsupported dtype");
}

#[test]
fn names_has_raw_bf16_bits_and_need() {
    let d = tempfile::tempdir().unwrap();
    let p = roundtrip_file(d.path());
    let f = TensorFile::open(&p).unwrap();
    assert_eq!(
        f.names(),
        ["a.f64", "b.f32", "c.i64", "d.u8", "e.empty", "f.bf16"]
    );
    assert!(f.has("d.u8") && !f.has("missing"));
    assert_eq!(f.raw("d.u8").unwrap(), [0u8, 1, 255]);
    assert_eq!(f.raw("b.f32").unwrap().len(), 8);
    assert!(matches!(f.shape("missing"), Err(GateError::Safetensors(_))));

    let (shape, bits) = f.read_bf16_bits("f.bf16").unwrap();
    assert_eq!(shape, vec![4]);
    assert_eq!(bits, vec![0x3F80, 0xC020, 0x3E20, 0x0000]); // 1.0, -2.5, 0.15625, 0.0
    assert!(matches!(
        f.read_bf16_bits("b.f32"),
        Err(GateError::Safetensors(_))
    ));

    let (shape32, v32) = f.read_f32("b.f32").unwrap();
    assert_eq!((shape32, v32), (vec![2], vec![1.5f32, -2.0]));

    let map = f.read_prefix("", false).unwrap();
    assert_eq!(
        need(&map, "d.u8").unwrap(),
        &Tensor::new(vec![3], vec![0.0, 1.0, 255.0])
    );
    assert!(matches!(need(&map, "nope"), Err(GateError::Key(_))));
}
