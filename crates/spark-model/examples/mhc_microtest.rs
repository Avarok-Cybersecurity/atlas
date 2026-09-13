// SPDX-License-Identifier: AGPL-3.0-only
//! Slice 9 Gate 1 — GLM-5.3-Flash **mHC (Manifold-Constrained Hyper-Connections)** numeric oracle
//! against HF `transformers` 5.16.1.
//!
//! Slice 2 down-graded mHC to REUSE because Atlas's `hc_mult = 4` and `hc_sinkhorn_iters = 20`
//! equal GLM's. That is a **config** match. This runs Atlas's real `hc_pre` / `hc_post` CUDA
//! kernels — written for DeepSeek-V4 — against goldens produced by GLM-5.3's own reference module
//! on real `hc_{attn,ffn}_{fn,base,scale}` weights, which is the **arithmetic** check.
//!
//! Real layer 0 / 3 / 22 / 44 weights of `LibertAIDAI/GLM-5.3-Flash-NVFP4` @ `9e0d74e3`.
//! The mHC parameter surface is BF16 (`fn`) + F32 (`base`, `scale`) with zero F8/U8/scale
//! tensors, so **floor C is N/A here too** and this golden IS the production numerics.
//!
//! 🪤 `hc_*_fn` is **BF16 on disk**, not F32. The handoff's "hc_* are F32" is true only of
//! `base`/`scale` (180 tensors); `fn` is the other 90. Atlas's kernel takes an `f32*`, so the
//! loader must upcast — exact, but it is an upcast, not a reinterpret.
//!
//! Each site is checked at both halves of the residual write:
//!   `hc_pre`  -> `post` [T,hc], `comb` [T,hc,hc], `collapsed` [T,H]
//!   `hc_post` -> `site_out` [T,hc,H]
//! and the two sites are **chained** (attn then ffn) exactly as the decoder layer chains them, so
//! a per-site pass that does not compose still fails here.
//!
//!   MHC_PACKET_DIR=/home/msi1/atlas-scratch/mhc-family \
//!   cargo run -p spark-model --release --example mhc_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "common/mhc_run.rs"]
pub(crate) mod mhc_run;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/spark-model/src/layers/glm5next_mhc_ref/mhc_golden.json",
        "gen_mhc_golden.py",
    )
});

pub(crate) const LAYERS: [usize; 4] = [0, 3, 22, 44];
pub(crate) const SITES: [&str; 2] = ["attn", "ffn"];
/// `(name, T)`. mHC is strictly per-token, so the regimes only vary the token count.
pub(crate) const REGIMES: [(&str, usize); 4] = [
    ("decode1", 1),
    ("short7", 7),
    ("medium64", 64),
    ("long2176", 2176),
];

// ───────────────────────────────────────────────────────────────────── plumbing
pub(crate) fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// The generator's LCG, bit-for-bit. Inputs are never read from the golden — reproducing them is
/// what proves the two sides are looking at the same tensor.
pub(crate) struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0) as f32
    }
    fn t(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
}

// ───────────────────────────────────────────────────── golden accessors
pub(crate) struct Golden(Value);
impl Golden {
    fn load() -> Result<Self> {
        Ok(Golden(serde_json::from_str(&GOLDEN)?))
    }
    fn fixture(&self, k: &str) -> Result<f64> {
        self.0["fixture"][k]
            .as_f64()
            .with_context(|| format!("fixture.{k}"))
    }
    /// Returns `(values, stride, n)` — the golden stores every tensor strided, so a comparison
    /// must walk the produced tensor with the same stride, and must check `n` first.
    fn get(
        &self,
        layer: usize,
        arm: &str,
        regime: &str,
        name: &str,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let sec = &self.0["by_layer"][layer.to_string()][format!("{arm}__{regime}")][name];
        if sec.is_null() {
            bail!("golden missing {layer}/{arm}__{regime}/{name}");
        }
        let n = sec["n"].as_u64().context("n")? as usize;
        let stride = sec["stride"].as_u64().context("stride")? as usize;
        let v = sec["data"]
            .as_array()
            .context("data")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        Ok((v, stride, n))
    }
}

/// Max abs difference between a produced tensor and a strided golden row.
/// Length is checked against the golden's own element count first — a silent shape drift must be
/// a failure, not a comparison over a prefix.
pub(crate) fn residual(what: &str, got: &[f32], g: &(Vec<f32>, usize, usize)) -> Result<f32> {
    let (want, stride, n) = g;
    if got.len() != *n {
        bail!(
            "{what}: golden describes {n} elements, produced {}",
            got.len()
        );
    }
    let mut worst = 0.0f32;
    for (i, w) in want.iter().enumerate() {
        let d = (got[i * stride] - w).abs();
        if d > worst {
            worst = d;
        }
    }
    Ok(worst)
}

/// Floor B for a stage: the reference's own bf16-vs-f32 activation spread. A residual is
/// unreadable without it — `E/B < 1` is normal, and B is a scale, not a bound.
pub(crate) fn floor_b(g: &Golden, layer: usize, regime: &str, name: &str) -> Result<(f32, f32)> {
    let (a, _, _) = g.get(layer, "bf16", regime, name)?;
    let (b, _, _) = g.get(layer, "f32", regime, name)?;
    let mut worst = 0.0f32;
    let mut mag = 0.0f32;
    for (x, y) in a.iter().zip(&b) {
        worst = worst.max((x - y).abs());
        mag = mag.max(y.abs());
    }
    Ok((worst, mag))
}

// ───────────────────────────────────────────────────── packet
pub(crate) struct Packet {
    pub(crate) raw: Vec<u8>,
    pub(crate) base: usize,
    pub(crate) hdr: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
}
impl Packet {
    fn open(path: &str) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().context("packet header")? {
            if k == "__metadata__" {
                continue;
            }
            let dt = m["dtype"].as_str().context("dtype")?.to_string();
            if dt != "BF16" && dt != "F32" {
                bail!("{k}: mHC params are BF16/F32 only, saw {dt}");
            }
            let shape = m["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let a = m["data_offsets"][0].as_u64().context("off0")? as usize;
            let b = m["data_offsets"][1].as_u64().context("off1")? as usize;
            hdr.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    /// Upcasts BF16 to f32 exactly. `hc_*_fn` lives here — see the module trap note.
    fn f32s(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>, String)> {
        let (dt, shape, a, b) = self
            .hdr
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        let by = &self.raw[self.base + a..self.base + b];
        let v = match dt.as_str() {
            "BF16" => by
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => by
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        };
        Ok((v, shape.clone(), dt.clone()))
    }
}

pub(crate) struct Row {
    pub(crate) arm: &'static str,
    pub(crate) layer: usize,
    pub(crate) regime: &'static str,
    pub(crate) site: &'static str,
    pub(crate) stage: &'static str,
    pub(crate) e: f32,
    pub(crate) b: f32,
    pub(crate) mag: f32,
}

fn main() -> Result<()> {
    mhc_run::run()
}
