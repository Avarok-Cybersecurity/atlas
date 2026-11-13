// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: CPU tests of the compressed-tensors `pack-quantized` parser: a synthetic packed
//! tensor (packed here by an independent loop) dequantizes to the naive `(v) * s` product bit for bit,
//! equals the GPTQ parse of the same codes, survives the Marlin repack round trip, the `g_idx` rules,
//! and the scheme detection from sample `config.json` documents.

use serde_json::json;

use super::super::checkpoint::{self, GptqWeights, GptqZeroFormat, ScaleDtype, le_bytes};
use super::super::reference::{bf16_to_f32, f32_to_bf16};
use super::super::repack::{from_host, to_host};
use super::*;

/// A deterministic xorshift stream.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// A signed int4 value, -8..=7.
    fn int4(&mut self) -> i8 {
        (self.next() % 16) as i8 - 8
    }
    /// A BF16 scale of either sign, magnitude about 2^-10..2^-3.
    fn scale(&mut self) -> u16 {
        let exp = -((self.next() % 8) as i32) - 3;
        let frac = 1.0 + (self.next() % 128) as f32 / 128.0;
        let mag = 2f32.powi(exp) * frac;
        let v = if self.next().is_multiple_of(4) {
            -mag
        } else {
            mag
        };
        f32_to_bf16(v)
    }
}

/// A synthetic compressed-tensors linear: signed values `[n][k]`, scales `[n][groups]`.
struct Synth {
    n: usize,
    k: usize,
    group: usize,
    v: Vec<i8>,
    s: Vec<u16>,
}

impl Synth {
    fn new(n: usize, k: usize, group: usize, seed: u64) -> Self {
        let mut r = Lcg::new(seed);
        let v = (0..n * k).map(|_| r.int4()).collect();
        let s = (0..n * (k / group)).map(|_| r.scale()).collect();
        Self { n, k, group, v, s }
    }

    /// `pack_to_int32` restated independently: `+ 8`, then eight per word, element `j` at `4 j`.
    fn packed_bytes(&self) -> Vec<u8> {
        let (n, k) = (self.n, self.k);
        let mut out = Vec::with_capacity(n * k / 2);
        for nn in 0..n {
            for w in 0..k / 8 {
                let mut word = 0u32;
                for j in 0..8 {
                    let code = (i32::from(self.v[nn * k + w * 8 + j]) + 8) as u32;
                    word += code << (4 * j);
                }
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    }

    fn scale_bytes(&self) -> Vec<u8> {
        self.s.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The naive dequantization `bf16(v * s)`, `[n][k]` row-major.
    fn naive_nk(&self) -> Vec<u16> {
        let (n, k) = (self.n, self.k);
        let groups = k / self.group;
        let mut out = vec![0u16; n * k];
        for nn in 0..n {
            for kk in 0..k {
                let s = bf16_to_f32(self.s[nn * groups + kk / self.group]);
                out[nn * k + kk] = f32_to_bf16(f32::from(self.v[nn * k + kk]) * s);
            }
        }
        out
    }

    fn logical(&self, g_idx: Option<&[u8]>) -> Result<LogicalInt4> {
        let packed = self.packed_bytes();
        let scales = self.scale_bytes();
        ct_to_logical(&CtPackedWeights {
            packed: &packed,
            scales: &scales,
            scale_dtype: ScaleDtype::Bf16,
            g_idx,
            n: self.n,
            k: self.k,
            group: self.group,
        })
    }
}

#[test]
fn unpack_and_dequant_equal_the_naive_dequantization() {
    for (n, k, seed) in [(64, 256, 1), (128, 384, 2), (64, 1024, 3)] {
        let syn = Synth::new(n, k, 128, seed);
        let l = syn.logical(None).unwrap();
        assert_eq!((l.n, l.k, l.group), (n, k, 128));
        assert!(l.zeros.is_none(), "symmetric is kU4B8");
        assert_eq!(l.btype(), super::super::MarlinType::U4B8);
        assert_eq!(l.dequant_nk(), syn.naive_nk(), "[{n}x{k}] seed {seed}");
    }
}

#[test]
fn ct_pack_matches_the_independent_packer() {
    let syn = Synth::new(64, 256, 128, 7);
    let codes: Vec<u8> = syn.v.iter().map(|&v| (i32::from(v) + 8) as u8).collect();
    assert_eq!(le_bytes(&ct_pack(&codes, 64, 256)), syn.packed_bytes());
}

#[test]
fn equals_the_gptq_parse_of_the_same_codes() {
    // compressed-tensors is the GPTQ layout transposed: same codes, same scales, same LogicalInt4.
    let syn = Synth::new(64, 256, 128, 11);
    let ct = syn.logical(None).unwrap();
    let qweight = le_bytes(&checkpoint::gptq_pack(&ct.q, ct.k, ct.n));
    let scales: Vec<u8> = ct.scales.iter().flat_map(|x| x.to_le_bytes()).collect();
    let gptq = checkpoint::gptq_to_logical(&GptqWeights {
        qweight: &qweight,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        qzeros: None,
        g_idx: None,
        n: 64,
        k: 256,
        group: 128,
        zero_format: GptqZeroFormat::V1,
    })
    .unwrap();
    assert_eq!(ct, gptq);
}

#[test]
fn f16_scales_are_converted() {
    let syn = Synth::new(64, 128, 128, 5);
    let packed = syn.packed_bytes();
    let f16: Vec<u8> = syn
        .s
        .iter()
        .flat_map(|&b| half::f16::from_f32(bf16_to_f32(b)).to_bits().to_le_bytes())
        .collect();
    let l = ct_to_logical(&CtPackedWeights {
        packed: &packed,
        scales: &f16,
        scale_dtype: ScaleDtype::F16,
        g_idx: None,
        n: 64,
        k: 128,
        group: 128,
    })
    .unwrap();
    // The synthetic scales are exact in F16, so the round trip is exact.
    assert_eq!(l.dequant_nk(), syn.naive_nk());
}

#[test]
fn marlin_repack_round_trip() {
    let syn = Synth::new(128, 512, 128, 13);
    let l = syn.logical(None).unwrap();
    let h = to_host(&l).unwrap();
    assert_eq!(from_host(&h).unwrap(), l);
}

fn g_idx_bytes(g: &[u32]) -> Vec<u8> {
    g.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
fn contiguous_g_idx_is_served_and_equals_no_g_idx() {
    let syn = Synth::new(64, 256, 128, 17);
    let trivial: Vec<u32> = (0..256u32).map(|kk| kk / 128).collect();
    let with = syn.logical(Some(&g_idx_bytes(&trivial))).unwrap();
    let without = syn.logical(None).unwrap();
    assert_eq!(with, without);
    // And it survives the repack round trip like any other weight.
    assert_eq!(from_host(&to_host(&with).unwrap()).unwrap(), without);
}

#[test]
fn act_order_g_idx_is_refused_by_name() {
    let syn = Synth::new(64, 256, 128, 19);
    // A real act-order permutation: input columns interleaved between the two groups.
    let act: Vec<u32> = (0..256u32).map(|kk| kk % 2).collect();
    let err = syn.logical(Some(&g_idx_bytes(&act))).unwrap_err();
    let msg = format!("{err:#}");
    assert!(msg.contains("actorder"), "{msg}");
    assert!(msg.contains("act-order"), "{msg}");
    // A g_idx of the wrong length is refused too.
    let short: Vec<u32> = (0..128u32).map(|kk| kk / 128).collect();
    assert!(syn.logical(Some(&g_idx_bytes(&short))).is_err());
}

#[test]
fn wrong_sizes_are_refused() {
    let syn = Synth::new(64, 256, 128, 23);
    let packed = syn.packed_bytes();
    let scales = syn.scale_bytes();
    let base = CtPackedWeights {
        packed: &packed,
        scales: &scales,
        scale_dtype: ScaleDtype::Bf16,
        g_idx: None,
        n: 64,
        k: 256,
        group: 128,
    };
    assert!(ct_to_logical(&CtPackedWeights { k: 512, ..base }).is_err());
    assert!(ct_to_logical(&CtPackedWeights { n: 128, ..base }).is_err());
    assert!(ct_to_logical(&CtPackedWeights { group: 96, ..base }).is_err());
    assert!(
        ct_to_logical(&CtPackedWeights {
            scales: &scales[2..],
            ..base
        })
        .is_err()
    );
    assert!(ct_to_logical(&base).is_ok());
}

/// The RedHatAI Qwen3.5-4B `quantized.w4a16` block, trimmed to the keys that matter.
fn sample_config() -> Value {
    json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": "static",
                        "block_structure": null,
                        "dynamic": false,
                        "group_size": 128,
                        "num_bits": 4,
                        "observer": "memoryless_minmax",
                        "strategy": "group",
                        "symmetric": true,
                        "type": "int"
                    }
                }
            },
            "format": "pack-quantized",
            "ignore": [
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
                "lm_head",
                "re:^mtp.*"
            ],
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed"
        }
    })
}

fn with_weights(key: &str, v: Value) -> Value {
    let mut c = sample_config();
    c["quantization_config"]["config_groups"]["group_0"]["weights"][key] = v;
    c
}

#[test]
fn detects_the_w4a16_scheme_from_config() {
    let s = CtW4Scheme::from_config_json(&sample_config())
        .unwrap()
        .expect("pack-quantized W4A16 is detected");
    assert_eq!(s.group, 128);
    assert_eq!(s.actorder, ActOrder::Static);
    assert_eq!(s.ignore.len(), 3);
    assert!(is_int_pack_format("pack-quantized"));
    assert!(is_int_pack_format(" Pack-Quantized "));
}

#[test]
fn other_checkpoints_are_not_detected() {
    // No quantization block.
    assert!(
        CtW4Scheme::from_config_json(&json!({"model_type": "qwen3_5"}))
            .unwrap()
            .is_none()
    );
    // The FP4 packers keep their NVFP4 / MXFP4 routes.
    for fmt in [
        "nvfp4-pack-quantized",
        "mxfp4-pack-quantized",
        "float-quantized",
    ] {
        let mut c = sample_config();
        c["quantization_config"]["format"] = json!(fmt);
        assert!(CtW4Scheme::from_config_json(&c).unwrap().is_none(), "{fmt}");
        assert!(!is_int_pack_format(fmt), "{fmt}");
    }
    // Another method.
    let mut c = sample_config();
    c["quantization_config"]["quant_method"] = json!("modelopt");
    assert!(CtW4Scheme::from_config_json(&c).unwrap().is_none());
}

#[test]
fn unserved_schemes_fail_with_a_reason() {
    let cases: [(Value, &str); 6] = [
        (with_weights("symmetric", json!(false)), "asymmetric"),
        (with_weights("group_size", json!(64)), "group_size 64"),
        (with_weights("num_bits", json!(8)), "8-bit"),
        (with_weights("strategy", json!("channel")), "strategy"),
        (with_weights("dynamic", json!(true)), "dynamic"),
        (with_weights("actorder", json!("sideways")), "actorder"),
    ];
    for (cfg, want) in cases {
        let err = CtW4Scheme::from_config_json(&cfg).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains(want), "expected {want:?} in {msg}");
    }
    let mut c = sample_config();
    c["quantization_config"]["config_groups"]["group_0"]["input_activations"] =
        json!({"num_bits": 8, "type": "int"});
    let msg = format!("{:#}", CtW4Scheme::from_config_json(&c).unwrap_err());
    assert!(msg.contains("W4A16"), "{msg}");
}

#[test]
fn actorder_spellings() {
    for (v, want) in [
        (json!(null), ActOrder::None),
        (json!("weight"), ActOrder::Static),
        (json!("group"), ActOrder::Group),
        (json!("dynamic"), ActOrder::Group),
    ] {
        let s = CtW4Scheme::from_config_json(&with_weights("actorder", v.clone()))
            .unwrap()
            .unwrap();
        assert_eq!(s.actorder, want, "{v}");
    }
}

#[test]
fn scheme_tensor_names() {
    let m = "model.language_model.layers.3.self_attn.q_proj";
    for suf in [PACKED, SCALE, SHAPE, G_IDX, ZERO_POINT] {
        assert!(is_scheme_tensor(&tensor_name(m, suf)), "{suf}");
    }
    assert!(!is_scheme_tensor(&format!("{m}.weight")));
    assert!(!is_scheme_tensor(&format!("{m}.weight_scale_inv")));
    assert!(!is_scheme_tensor(&format!("{m}.weight_scale_2")));
    assert!(!is_scheme_tensor("weight_packed"));
}
