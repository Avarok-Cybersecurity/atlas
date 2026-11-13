// SPDX-License-Identifier: AGPL-3.0-only
// After TensorFold zig/src/core/exl3_load.zig:179-350 (the hand-built safetensors admission tests;
// Apache-2.0, ashhart/TensorFold, commit 3885dce). Changes: translated to Rust over synthetic shards
// built from the documented format; one test per refusal reason; shard discovery, plain tensors and
// config tests added. See EXL3-ATTRIBUTION.md.

//! Responsibility: tests of EXL3 checkpoint admission and loading on synthetic safetensors shards: what
//! is admitted, every refusal reason, packed sign scales, sharded checkpoints, plain tensors, header
//! validation, and the `config.json` block.

use std::path::Path;

use super::config::config_from_json;
use super::fixture::{
    StTensor, SynthGroup, group_tensors, synth_group, u16_bytes, write_safetensors,
};
use super::format::{Codebook, MCG_MUL, MUL1_MUL, Width};
use super::fp16;
use super::load::{Exl3Scan, Refusal};
use super::shards::ShardSet;
use super::tensor::ScaleForm;

fn w(halves: u16) -> Width {
    Width::from_halves(halves).unwrap()
}

fn scan_of(dir: &Path, tensors: &[StTensor]) -> Exl3Scan {
    write_safetensors(&dir.join("model.safetensors"), tensors);
    Exl3Scan::open(dir).expect("scan")
}

fn refusal(tensors: Vec<StTensor>) -> Option<Refusal> {
    let dir = tempfile::tempdir().unwrap();
    let scan = scan_of(dir.path(), &tensors);
    assert!(scan.groups().is_empty(), "the group must not be admitted");
    scan.refused().first().map(|(_, r)| *r)
}

fn base(prefix: &str, halves: u16, g: &SynthGroup, marker: Option<u32>) -> Vec<StTensor> {
    group_tensors(prefix, w(halves), 128, 128, g, false, false, marker)
}

#[test]
fn a_full_group_is_admitted_and_a_scaleless_one_refused() {
    let dir = tempfile::tempdir().unwrap();
    let g = synth_group(w(4), 128, 128, 1);
    let mut tensors = group_tensors("a", w(4), 128, 128, &g, false, false, Some(MUL1_MUL));
    tensors.push(StTensor::new(
        "b.trellis",
        "I16",
        &[8, 8, 32],
        vec![0; 4096],
    ));
    let scan = scan_of(dir.path(), &tensors);
    assert_eq!(scan.groups().len(), 1);
    assert_eq!(scan.refused(), &[("b".to_string(), Refusal::MissingScales)]);
    let s = scan.summary();
    assert_eq!(
        (s.groups, s.mul1, s.fp16_scales, s.by_halves[4]),
        (1, 1, 1, 1)
    );
    assert_eq!(s.trellis_bytes, 4096);
    assert_eq!(s.average_bits(), 2.0);
    let info = &scan.groups()[0];
    assert_eq!(
        (info.k, info.n, info.codebook, info.bias),
        (128, 128, Codebook::Mul1, false)
    );
    let t = scan.load_prefix("a").unwrap();
    assert_eq!(t.trellis, g.trellis);
    assert_eq!(t.suh, g.suh);
    assert_eq!(t.svh, g.svh);
    assert_eq!(t.scale_form, ScaleForm::Fp16);
    assert_eq!(t.bias, None);
    assert!(scan.load_prefix("b").is_err());
    assert!(scan.load(1).is_err());
}

#[test]
fn packed_sign_planes_load_as_plus_minus_one() {
    let dir = tempfile::tempdir().unwrap();
    let g = synth_group(w(6), 128, 256, 2);
    let tensors = group_tensors("p", w(6), 128, 256, &g, true, true, Some(MCG_MUL));
    let scan = scan_of(dir.path(), &tensors);
    assert_eq!(scan.summary().sign_scales, 1);
    assert_eq!(scan.summary().bias, 1);
    let t = scan.load(0).unwrap();
    assert_eq!(t.scale_form, ScaleForm::Signs);
    assert_eq!(t.codebook, Codebook::Mcg);
    assert_eq!(t.suh.len(), 128);
    assert_eq!(t.svh.len(), 256);
    for (i, &v) in t.suh.iter().enumerate() {
        let negative = ((g.su[i / 16] as u16) >> (i % 16)) & 1 == 1;
        assert_eq!(
            v,
            if negative { fp16::MINUS_ONE } else { fp16::ONE },
            "suh[{i}]"
        );
    }
    assert_eq!(t.bias.as_deref(), Some(&g.bias[..]));
}

#[test]
fn no_marker_is_3inst_and_a_scalar_marker_of_rank_zero_or_one_is_read() {
    let dir = tempfile::tempdir().unwrap();
    let g = synth_group(w(8), 128, 128, 3);
    let mut tensors = base("plain3", 8, &g, None);
    tensors.extend(base("m", 8, &g, Some(MCG_MUL)));
    tensors.extend(base("r1", 8, &g, None));
    tensors.push(StTensor::new(
        "r1.mul1",
        "I32",
        &[1],
        MUL1_MUL.to_le_bytes().to_vec(),
    ));
    tensors.extend(base("u", 8, &g, None));
    tensors.push(StTensor::new(
        "u.mcg",
        "U32",
        &[],
        MCG_MUL.to_le_bytes().to_vec(),
    ));
    let scan = scan_of(dir.path(), &tensors);
    let cb = |p: &str| scan.groups()[scan.find(p).unwrap()].codebook;
    assert_eq!(cb("plain3"), Codebook::Inst3);
    assert_eq!(cb("m"), Codebook::Mcg);
    assert_eq!(cb("r1"), Codebook::Mul1);
    assert_eq!(cb("u"), Codebook::Mcg);
    assert!(scan.refused().is_empty());
}

#[test]
fn every_refusal_reason() {
    let g = synth_group(w(4), 128, 128, 4);
    let g3 = synth_group(w(3), 128, 128, 4);
    // Odd halves need mul1.
    assert_eq!(
        refusal(base("x", 3, &g3, None)),
        Some(Refusal::HalfWidthNeedsMul1)
    );
    assert_eq!(
        refusal(base("x", 3, &g3, Some(MCG_MUL))),
        Some(Refusal::HalfWidthNeedsMul1)
    );
    // Both markers.
    let mut two = base("x", 4, &g, Some(MCG_MUL));
    two.push(StTensor::new(
        "x.mul1",
        "I32",
        &[],
        MUL1_MUL.to_le_bytes().to_vec(),
    ));
    assert_eq!(refusal(two), Some(Refusal::TwoMarkers));
    // A marker with the other codebook's value, or not a 4-byte scalar.
    let mut wrong = base("x", 4, &g, None);
    wrong.push(StTensor::new(
        "x.mcg",
        "I32",
        &[],
        MUL1_MUL.to_le_bytes().to_vec(),
    ));
    assert_eq!(refusal(wrong), Some(Refusal::BadMarker));
    let mut wide = base("x", 4, &g, None);
    wide.push(StTensor::new(
        "x.mul1",
        "I64",
        &[],
        u64::from(MUL1_MUL).to_le_bytes().to_vec(),
    ));
    assert_eq!(refusal(wide), Some(Refusal::BadMarker));
    // Mixed scale forms.
    let mut mixed = base("x", 4, &g, None);
    mixed.retain(|t| t.name != "x.svh");
    mixed.push(StTensor::new("x.sv", "I16", &[8], vec![0; 16]));
    assert_eq!(refusal(mixed), Some(Refusal::MixedScales));
    // Bias of the wrong dtype.
    let mut bias = base("x", 4, &g, None);
    bias.push(StTensor::new("x.bias", "BF16", &[128], vec![0; 256]));
    assert_eq!(refusal(bias), Some(Refusal::BadBias));
    // suh of the wrong length, or BF16.
    let mut short = base("x", 4, &g, None);
    short.retain(|t| t.name != "x.suh");
    short.push(StTensor::new("x.suh", "F16", &[64], vec![0; 128]));
    assert_eq!(refusal(short), Some(Refusal::BadScales));
    let mut bf = base("x", 4, &g, None);
    bf.retain(|t| t.name != "x.svh");
    bf.push(StTensor::new("x.svh", "BF16", &[128], vec![0; 256]));
    assert_eq!(refusal(bf), Some(Refusal::BadScales));
    // A packed sign plane of the wrong length.
    let mut packed = group_tensors("x", w(4), 128, 128, &g, true, false, None);
    packed.retain(|t| t.name != "x.su");
    packed.push(StTensor::new("x.su", "I16", &[4], vec![0; 8]));
    assert_eq!(refusal(packed), Some(Refusal::BadScales));
    // Trellis dtype, rank, width and alignment.
    let scales = |p: &str| {
        vec![
            StTensor::new(&format!("{p}.suh"), "F16", &[128], u16_bytes(&g.suh)),
            StTensor::new(&format!("{p}.svh"), "F16", &[128], u16_bytes(&g.svh)),
        ]
    };
    let with = |t: StTensor| {
        let mut v = vec![t];
        v.extend(scales("x"));
        v
    };
    let f16 = StTensor::new("x.trellis", "F16", &[8, 8, 32], vec![0; 4096]);
    assert_eq!(refusal(with(f16)), Some(Refusal::NotI16Rank3));
    let flat = StTensor::new("x.trellis", "I16", &[64, 32], vec![0; 4096]);
    assert_eq!(refusal(with(flat)), Some(Refusal::NotI16Rank3));
    let odd = StTensor::new("x.trellis", "I16", &[8, 8, 12], vec![0; 1536]);
    assert_eq!(refusal(with(odd)), Some(Refusal::BadWidth));
    let nine = StTensor::new("x.trellis", "I16", &[8, 8, 136], vec![0; 17408]);
    assert_eq!(refusal(with(nine)), Some(Refusal::BadWidth));
    let small = StTensor::new("x.trellis", "I16", &[4, 8, 32], vec![0; 2048]);
    assert_eq!(refusal(with(small)), Some(Refusal::NotBlockAligned));
}

/// A trellis of no bytes, `[2^60 + 8, 0, 32]`, whose `K = 16 * shape[0]` wraps to 128 unchecked: it
/// must be refused, not admitted at the wrapped size (`exl3_load.zig:325-350`).
#[cfg(target_pointer_width = "64")]
#[test]
fn a_trellis_whose_k_wraps_is_refused() {
    let wrapped = StTensor::new("x.trellis", "I16", &[(1usize << 60) + 8, 0, 32], Vec::new());
    let tensors = vec![
        wrapped,
        StTensor::new("x.suh", "F16", &[128], vec![0; 256]),
        StTensor::new("x.svh", "F16", &[0], Vec::new()),
    ];
    assert_eq!(refusal(tensors), Some(Refusal::Overflow));
}

#[test]
fn groups_sort_by_prefix_and_plain_tensors_are_listed() {
    let dir = tempfile::tempdir().unwrap();
    let g = synth_group(w(4), 128, 128, 5);
    let mut tensors = base("a-b", 4, &g, None);
    tensors.extend(base("a", 4, &g, None));
    tensors.extend(base("model.layers.0.mlp.experts.3.down_proj", 4, &g, None));
    tensors.push(StTensor::new(
        "bad.trellis",
        "I16",
        &[8, 8, 32],
        vec![0; 4096],
    ));
    tensors.push(StTensor::new("bad.suh", "F16", &[128], vec![0; 256]));
    tensors.push(StTensor::new(
        "model.embed_tokens.weight",
        "BF16",
        &[4, 128],
        vec![0; 1024],
    ));
    tensors.push(StTensor::new(
        "model.norm.weight",
        "BF16",
        &[128],
        vec![0; 256],
    ));
    let scan = scan_of(dir.path(), &tensors);
    let prefixes: Vec<&str> = scan.groups().iter().map(|g| g.prefix.as_str()).collect();
    assert_eq!(
        prefixes,
        ["a", "a-b", "model.layers.0.mlp.experts.3.down_proj"]
    );
    assert_eq!(scan.find("a-b"), Some(1));
    assert_eq!(scan.summary().experts_down[4], 1);
    assert_eq!(
        scan.plain_tensors(),
        ["model.embed_tokens.weight", "model.norm.weight"]
    );
    assert!(scan.is_group_tensor("a.trellis"));
    assert!(
        scan.is_group_tensor("bad.suh"),
        "a refused group's tensors are not plain either"
    );
    assert!(!scan.is_group_tensor("model.norm.weight"));
    assert!(!scan.is_group_tensor("a.weight"));
}

#[test]
fn a_sharded_checkpoint_reads_through_its_index() {
    let dir = tempfile::tempdir().unwrap();
    let g = synth_group(w(5), 256, 128, 6);
    let mut all = group_tensors("s", w(5), 256, 128, &g, false, true, Some(MUL1_MUL));
    let second = all.split_off(2);
    write_safetensors(&dir.path().join("model-00001-of-00002.safetensors"), &all);
    write_safetensors(
        &dir.path().join("model-00002-of-00002.safetensors"),
        &second,
    );
    let mut map = serde_json::Map::new();
    for t in &all {
        map.insert(t.name.clone(), "model-00001-of-00002.safetensors".into());
    }
    for t in &second {
        map.insert(t.name.clone(), "model-00002-of-00002.safetensors".into());
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": map });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        index.to_string(),
    )
    .unwrap();
    let scan = Exl3Scan::open(dir.path()).unwrap();
    assert_eq!(scan.shards().files().len(), 2);
    let t = scan.load_prefix("s").unwrap();
    assert_eq!(
        (t.trellis.clone(), t.bias.clone()),
        (g.trellis.clone(), Some(g.bias.clone()))
    );
    assert_eq!(t.width.halves(), 5);
    assert_eq!(t.bpw_x100(), 250);
    scan.shards().release_page_cache();

    let evil = serde_json::json!({ "weight_map": { "x": "../model-00001-of-00002.safetensors" } });
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        evil.to_string(),
    )
    .unwrap();
    assert!(
        Exl3Scan::open(dir.path()).is_err(),
        "an index naming a file outside the directory"
    );
}

#[test]
fn shard_headers_are_validated() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let a = StTensor::new("t", "F16", &[4], vec![0; 8]);
    // The same tensor in two unindexed shards.
    write_safetensors(&d.join("a.safetensors"), std::slice::from_ref(&a));
    write_safetensors(&d.join("b.safetensors"), &[a]);
    assert!(ShardSet::open(d).is_err());
    std::fs::remove_file(d.join("b.safetensors")).unwrap();
    let set = ShardSet::open(d).unwrap();
    assert_eq!(set.len(), 1);
    assert!(!set.is_empty());
    // The data is the file's last 8 bytes.
    let file_len = std::fs::metadata(d.join("a.safetensors")).unwrap().len();
    assert_eq!(
        set.get("t").map(|e| (e.offset, e.len)),
        Some((file_len - 8, 8))
    );
    assert_eq!(set.read(set.get("t").unwrap()).unwrap(), vec![0u8; 8]);
    // A size that does not match the shape, and data past the end of the file.
    let bad = d.join("bad");
    std::fs::create_dir(&bad).unwrap();
    write_safetensors(
        &bad.join("x.safetensors"),
        &[StTensor::new("t", "F16", &[5], vec![0; 8])],
    );
    assert!(ShardSet::open(&bad).is_err());
    let mut bytes =
        super::fixture::safetensors_bytes(&[StTensor::new("t", "F16", &[4], vec![0; 8])]);
    bytes.truncate(bytes.len() - 1);
    std::fs::write(bad.join("x.safetensors"), bytes).unwrap();
    assert!(ShardSet::open(&bad).is_err());
    std::fs::write(bad.join("x.safetensors"), [9u8, 0, 0, 0, 0, 0, 0, 0, b'{']).unwrap();
    assert!(ShardSet::open(&bad).is_err(), "header longer than the file");
    let empty = d.join("empty");
    std::fs::create_dir(&empty).unwrap();
    assert!(ShardSet::open(&empty).is_err(), "no shards");
}

#[test]
fn config_json_exl3_block() {
    let top = serde_json::json!({
        "model_type": "qwen3_5",
        "quantization_config": {
            "quant_method": "exl3", "version": "0.0.20", "bits": 3.5, "head_bits": 6,
            "codebook": "mcg", "out_scales": "auto"
        }
    });
    let c = config_from_json(&top).unwrap();
    assert_eq!(c.bits, Some(3.5));
    assert_eq!(c.head_bits, Some(6.0));
    assert_eq!(c.version.as_deref(), Some("0.0.20"));
    assert_eq!(c.codebook.as_deref(), Some("mcg"));
    assert_eq!(c.out_scales.as_deref(), Some("auto"));
    let nested =
        serde_json::json!({ "text_config": { "quantization_config": { "quant_method": "EXL3" } } });
    assert_eq!(config_from_json(&nested).map(|c| c.bits), Some(None));
    let other = serde_json::json!({ "quantization_config": { "quant_method": "gptq", "bits": 4 } });
    assert_eq!(config_from_json(&other), None);
    assert_eq!(
        config_from_json(&serde_json::json!({ "model_type": "x" })),
        None
    );

    let dir = tempfile::tempdir().unwrap();
    assert!(!super::is_exl3_checkpoint(dir.path()), "no config.json");
    std::fs::write(dir.path().join("config.json"), top.to_string()).unwrap();
    assert!(super::is_exl3_checkpoint(dir.path()));
    assert_eq!(super::read_config(dir.path()).unwrap(), Some(c));
}
