// SPDX-License-Identifier: AGPL-3.0-only
//! Slice 10 gates 6 + 7 — a COMPLETE GLM-5.3 routed MoE layer on real checkpoint weights.
//!
//! `hidden → router logits → top-k ids/weights → selected NVFP4 experts → weighted routed
//! mixture → shared BF16 expert → final`, against HF `transformers` 5.16.1.
//!
//! **Both router dtype ladders run independently and are NEVER pooled into one tolerance.**
//! `HfFp32` is the canonical production semantics; `VllmBf16` reproduces what vLLM currently
//! does for `glm5_next_text`. A residual in one says nothing about the other.
//!
//! 🔴 `apply_routed_scale_to_output = false`: `routed_scaling_factor` rides on the top-k
//! weights and the **shared expert is not multiplied by it**. The shared-expert stage is
//! therefore expected to be BIT-IDENTICAL across the two router modes — it does not depend on
//! routing at all — and that is asserted as a built-in control.
//!
//!   MOE_PACKET_DIR=/home/msi1/atlas-scratch/moe-family \
//!   cargo run -p spark-model --release --example glm5next_moe_microtest \
//!       --features cuda,gpu-examples -- 3

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use serde_json::Value;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

#[path = "common/glm5next_moe_run.rs"]
pub(crate) mod glm5next_moe_run;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/spark-model/src/layers/glm5next_moe_ref/moe_golden.json",
        "gen_moe_golden.py",
    )
});
pub(crate) const MODES: [&str; 2] = ["hf_fp32", "vllm_bf16"];
pub(crate) const REGIMES: [&str; 4] = ["t1", "t7", "real32", "nearcut"];

pub(crate) fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
pub(crate) fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    up(g, &b)
}
pub(crate) fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    up(g, &b)
}
pub(crate) fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
pub(crate) fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn dn_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub(crate) fn bf16_ulp(x: f32) -> f32 {
    if x == 0.0 {
        return f32::MIN_POSITIVE;
    }
    (2.0f32).powi(x.abs().log2().floor() as i32 - 7)
}
pub(crate) const ULP_BUDGET: f32 = 4.0;

pub(crate) struct G(Value);
impl G {
    fn f(&self, k: &str) -> Result<f64> {
        self.0["fixture"][k].as_f64().with_context(|| k.to_string())
    }
    fn get(&self, l: usize, sec: &str, n: &str) -> Result<(Vec<f32>, usize, usize)> {
        let s = &self.0["by_layer"][l.to_string()][sec][n];
        if s.is_null() {
            bail!("golden missing {l}/{sec}/{n}");
        }
        Ok((
            s["data"]
                .as_array()
                .context("d")?
                .iter()
                .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                .collect(),
            s["stride"].as_u64().context("s")? as usize,
            s["n"].as_u64().context("n")? as usize,
        ))
    }
}
pub(crate) fn resid(w: &str, got: &[f32], g: &(Vec<f32>, usize, usize)) -> Result<f32> {
    if got.len() != g.2 {
        bail!("{w}: golden has {} elements, produced {}", g.2, got.len());
    }
    Ok(g.0
        .iter()
        .enumerate()
        .fold(0.0f32, |a, (i, x)| a.max((got[i * g.1] - x).abs())))
}
pub(crate) fn mag(g: &(Vec<f32>, usize, usize)) -> f32 {
    g.0.iter().fold(0.0f32, |a, b| a.max(b.abs()))
}

pub(crate) struct Pk {
    pub(crate) raw: Vec<u8>,
    pub(crate) base: usize,
    pub(crate) hdr: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
}
impl Pk {
    fn open(p: &str) -> Result<Self> {
        let raw = std::fs::read(p).with_context(|| p.to_string())?;
        let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().context("h")? {
            if k == "__metadata__" {
                continue;
            }
            hdr.insert(
                k.clone(),
                (
                    m["dtype"].as_str().context("dt")?.to_string(),
                    m["shape"]
                        .as_array()
                        .context("sh")?
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect(),
                    m["data_offsets"][0].as_u64().context("a")? as usize,
                    m["data_offsets"][1].as_u64().context("b")? as usize,
                ),
            );
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    fn m(&self, n: &str) -> Result<&(String, Vec<usize>, usize, usize)> {
        self.hdr.get(n).with_context(|| format!("missing {n}"))
    }
    fn b(&self, n: &str) -> Result<&[u8]> {
        let (_, _, a, b) = self.m(n)?;
        Ok(&self.raw[self.base + a..self.base + b])
    }
    fn f32b(&self, n: &str) -> Result<Vec<f32>> {
        let (dt, ..) = self.m(n)?;
        Ok(match dt.as_str() {
            "BF16" => self
                .b(n)?
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => self
                .b(n)?
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }
}

pub(crate) struct K {
    pub(crate) gemm: KernelHandle,
    pub(crate) gemm_f32: KernelHandle,
    pub(crate) w4: KernelHandle,
    pub(crate) act: KernelHandle,
    pub(crate) router: KernelHandle,
    pub(crate) combine: KernelHandle,
}

pub(crate) struct Row {
    pub(crate) layer: usize,
    pub(crate) mode: &'static str,
    pub(crate) regime: &'static str,
    pub(crate) stage: &'static str,
    pub(crate) e: f32,
    pub(crate) b: f32,
    pub(crate) mag: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w4(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    bp: DevicePtr,
    bs: DevicePtr,
    s2: f32,
    c: DevicePtr,
    m: u32,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(64), m.div_ceil(64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(bp)
        .arg_ptr(bs)
        .arg_f32(s2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}
pub(crate) fn gemm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(16), m.div_ceil(16), 1])
        .block([16, 16, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}
pub(crate) fn act(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    o: DevicePtr,
    n: u32,
    l: f32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(o)
        .arg_u32(n)
        .arg_f32(l)
        .launch(0)
}

fn main() -> Result<()> {
    glm5next_moe_run::run()
}
