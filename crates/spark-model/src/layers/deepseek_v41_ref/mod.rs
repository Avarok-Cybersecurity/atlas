// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! DeepSeek-V4.1 Flash **tiny-graph reference harness** — stage S2 of the single-Spark Q2_K
//! bring-up (#1059, under #970).
//!
//! Design artifact, **not** a production path. Nothing here runs on a GPU and nothing here is
//! wired into a forward pass. Its job is to give the V4.1 graph a checkable truth BEFORE it is
//! written, so engram, the shared-attention runtime and the reworked indexer get built once
//! against DeepSeek's own reference instead of twice.
//!
//! # Where the truth comes from
//!
//! `ds41_tiny_golden.json` is produced by `gen_ds41_tiny_golden.py` (kept beside it), which runs
//! DeepSeek's OWN `inference/model.py` (deepseek-ai/DeepSeek-V4.1-Flash @ `dba1be0a`) on a tiny
//! synthetic model that keeps every V4.1 structural feature: hc_mult=4 hyper-connections with the
//! delayed mixes, engram on two layers with DeepSeek's hash (`engram.py`), shared compressed
//! attention with two kv/index sources and a candidate prefilter whose consumers share its ratio,
//! sqrt-softplus routing with correction bias, route_scale, swiglu clamp and a shared expert.
//! No equation is re-derived on the Python side. The five tilelang kernels the reference calls
//! unconditionally are replaced by `ds41_ref_shims.py`, pure torch written from the tilelang
//! source, line-referenced.
//!
//! # Determinism, and why only outputs are committed
//!
//! Weights and inputs are RNG-free and order-free: `fixed_value(name, i, scale, offset)` here
//! reproduces the generator bit for bit (fnv1a64 salt, splitmix64, 24-bit uniform, f64
//! arithmetic, f32 result, bf16 RNE where the parameter is stored bf16). So the golden carries
//! only OUTPUTS: each capture as `{shape, n, stride, ck, data}` where `data` is a prime-strided
//! sample and `ck` is the fp64 index-weighted checksum over the whole tensor. The tests below
//! prove the regeneration matches before any graph code exists.
//!
//! # Tier 2: the engram hash is pinned (proven 2026-09-15)
//!
//! `gen_ds41_engram_hash_check.py` recomputes, from the real V4.1 tokenizer through
//! `engram.py`, the compressed token map (129,280 -> 99,092), the 48 bucket primes, the 48 slot
//! offsets and the 8 hash multipliers, and they are IDENTICAL to the `deepseek41.engram.*`
//! metadata the Q2_K GGUF ships. The sum of each layer's primes equals its table's row count
//! exactly (384,006,168 / 384,016,682), so `engram_num_embeddings` is derivable from the file.
//! An Atlas engram can therefore be built from GGUF metadata alone, with nothing reverse-engineered.

use serde_json::Value;

/// The tracked golden. Regenerate with the script beside it; the fixture is RNG-free.
pub const GOLDEN_JSON: &str = include_str!("ds41_tiny_golden.json");

const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64 finaliser, exactly as `gen_ds41_tiny_golden.py::splitmix64`.
pub fn splitmix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(GOLD);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// FNV-1a 64 over the UTF-8 bytes of `s`; the per-tensor salt.
pub fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// Raw stream value for element `i` of the tensor named `name`.
pub fn fixed_raw(name: &str, i: u64) -> u64 {
    splitmix64(fnv1a64(name) ^ i.wrapping_mul(GOLD))
}

/// The generator's filler: `u = (z >> 40) / 2^24`, `f32((2u - 1) * scale + offset)` in f64.
pub fn fixed_value(name: &str, i: u64, scale: f64, offset: f64) -> f32 {
    let u = (fixed_raw(name, i) >> 40) as f64 / (1u64 << 24) as f64;
    ((2.0 * u - 1.0) * scale + offset) as f32
}

/// The generator's integer filler: `splitmix64(...) % modulus`.
pub fn fixed_int(name: &str, i: u64, modulus: u64) -> u64 {
    fixed_raw(name, i) % modulus
}

/// Round an f32 to bf16 precision with round-to-nearest-even, as torch's bf16 cast does.
pub fn to_bf16_rne(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    f32::from_bits(b.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000)
}

/// fp64 index-weighted checksum `sum(v[i] * (i + 1))`, the generator's `ck`. Summed
/// sequentially here; torch reduces in a different order, so compare with a relative tolerance.
pub fn checksum<I: IntoIterator<Item = f64>>(v: I) -> f64 {
    v.into_iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 + 1.0))
        .sum()
}

/// One committed capture.
pub struct GoldenTensor {
    pub shape: Vec<usize>,
    pub n: usize,
    pub stride: usize,
    pub ck: f64,
    pub data: Vec<f64>,
}

/// One parameter's regeneration rule, from `weights_meta`.
pub struct WeightMeta {
    pub name: String,
    pub n: usize,
    pub scale: f64,
    pub offset: f64,
    pub kind: String,
    pub dtype: String,
    pub ck: f64,
}

pub struct Golden(Value);

impl Golden {
    pub fn load() -> Self {
        Golden(serde_json::from_str(GOLDEN_JSON).expect("ds41_tiny_golden.json parses"))
    }

    pub fn regimes(&self) -> Vec<String> {
        self.0["fixture"]["regimes"]
            .as_array()
            .expect("fixture.regimes")
            .iter()
            .map(|v| v.as_str().expect("regime name").to_string())
            .collect()
    }

    pub fn fixture_u64(&self, key: &str) -> u64 {
        self.0["fixture"][key]
            .as_u64()
            .unwrap_or_else(|| panic!("fixture.{key} is not an integer"))
    }

    pub fn fixture_f64(&self, key: &str) -> f64 {
        self.0["fixture"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("fixture.{key} is not numeric"))
    }

    pub fn lcg_probe(&self) -> Vec<u64> {
        self.0["fixture"]["lcg_probe"]
            .as_array()
            .expect("fixture.lcg_probe")
            .iter()
            .map(|v| v.as_u64().expect("u64 probe"))
            .collect()
    }

    pub fn capture_names(&self, regime: &str) -> Vec<String> {
        self.0[regime]
            .as_object()
            .unwrap_or_else(|| panic!("regime {regime} missing"))
            .keys()
            .cloned()
            .collect()
    }

    pub fn tensor(&self, regime: &str, name: &str) -> GoldenTensor {
        let t = &self.0[regime][name];
        assert!(!t.is_null(), "missing capture {regime}.{name}");
        let as_usize = |k: &str| t[k].as_u64().unwrap_or_else(|| panic!("{regime}.{name}.{k}")) as usize;
        GoldenTensor {
            shape: t["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|v| v.as_u64().expect("dim") as usize)
                .collect(),
            n: as_usize("n"),
            stride: as_usize("stride"),
            ck: t["ck"].as_f64().expect("ck"),
            data: t["data"]
                .as_array()
                .expect("data")
                .iter()
                .map(|v| v.as_f64().expect("numeric"))
                .collect(),
        }
    }

    pub fn weights_meta(&self) -> Vec<WeightMeta> {
        self.0["weights_meta"]
            .as_object()
            .expect("weights_meta")
            .iter()
            .map(|(name, m)| WeightMeta {
                name: name.clone(),
                n: m["n"].as_u64().expect("n") as usize,
                scale: m["scale"].as_f64().expect("scale"),
                offset: m["offset"].as_f64().expect("offset"),
                kind: m["kind"].as_str().expect("kind").to_string(),
                dtype: m["dtype"].as_str().expect("dtype").to_string(),
                ck: m["ck"].as_f64().expect("ck"),
            })
            .collect()
    }
}

/// Elementwise max-abs comparison that names the worst index and both values.
#[track_caller]
pub fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length {} vs {}", got.len(), want.len());
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max abs diff {worst:e} > {tol:e} at index {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

pub mod engram;

#[cfg(test)]
mod tests;
