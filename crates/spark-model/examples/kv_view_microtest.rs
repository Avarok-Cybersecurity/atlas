// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).
//! GPU receipt for the bounded KV view core (lane L31; the row `kv_view_selection` is lane
//! L32's and stays off): the page sketch, the query unrope and the page-softmax scorer on
//! the real driver against the f64 oracles in `spark_runtime::kv_select::oracle`.
//!
//!   AVAROK_TARGET_HW=ada cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example kv_view_microtest -- [--only sketch|score|all] [--reps 5] [--quick]
//!
//! Gates (exit code 1 on any FAIL):
//! * S1 BF16 sketch: keys written by the real write kernel (`fused_k_norm_rope_cache_write_bf16`)
//!   at positions up to 200K (rotary 64 of 256, theta 1e7), sketched by
//!   `kv_page_sketch_jobs_bf16`: relative L2 error per (page, head) <= 1e-2 against the f64
//!   mean of the normed pre-RoPE keys; the partial tail page included.
//! * S2 FP8 sketch: the same through `fused_k_norm_rope_cache_write_fp8`; the error is
//!   MEASURED and printed (gate <= 0.1, a sanity bound, not a quality claim).
//! * S3 the graph-safe tail path, token by token and with 4-row verify groups, reproduces the
//!   job path's sketch bit for bit (recompute semantics).
//! * S4 unrope: RoPE'd BF16 query rows rotated back within 1e-2 relative L2.
//! * C1 scorer vs the f64 oracle at 100, 7,813 and 40,000 view blocks, both GQA modes:
//!   L1 relative <= 1e-5 and worst per-block relative <= 1e-4.
//! * C2 two runs give equal bits; C3 a planted needle block is picked at the smallest budget.
//! * info: scorer milliseconds at 1M tokens (7,813 blocks) and 512 query rows on the 4B shape.

use anyhow::{Context, Result, bail};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "kv_view_microtest/score.rs"]
mod score;
#[path = "kv_view_microtest/sketch.rs"]
mod sketch;

pub struct Args {
    pub only: String,
    pub reps: usize,
    pub quick: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        only: "all".into(),
        reps: 5,
        quick: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--only" => a.only = val()?,
            "--reps" => a.reps = val()?.parse::<usize>()?.max(1),
            "--quick" => a.quick = true,
            other => bail!("unknown flag {other}"),
        }
    }
    if !["all", "sketch", "score"].contains(&a.only.as_str()) {
        bail!("--only {}: expected sketch, score or all", a.only);
    }
    Ok(a)
}

pub struct Gates {
    pub failed: usize,
}

impl Gates {
    pub fn check(&mut self, name: &str, ok: bool, detail: &str) {
        println!("{} {name}: {detail}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            self.failed += 1;
        }
    }
}

/// Deterministic generator (xorshift64*).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly normal, unit variance.
    pub fn normal(&mut self) -> f64 {
        ((0..4).map(|_| self.unit()).sum::<f64>() - 2.0) * 3f64.sqrt()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }
}

pub fn bf16_bits(x: f64) -> u16 {
    half::bf16::from_f64(x).to_bits()
}

pub fn bf16_f64(b: u16) -> f64 {
    half::bf16::from_bits(b).to_f64()
}

pub fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

pub fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn download(gpu: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut out = vec![0u8; bytes];
    gpu.synchronize(gpu.default_stream())?;
    gpu.copy_d2h(p, &mut out)?;
    Ok(out)
}

pub fn as_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

pub fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Relative L2 error of `got` against `want`.
pub fn rel_l2(got: &[f64], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(g, w)| (g - w).powi(2)).sum();
    let den: f64 = want.iter().map(|w| w * w).sum();
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    for m in [
        spark_runtime::kv_cache::sketch::KERNEL_MODULE,
        spark_runtime::kv_select::KERNEL_MODULE,
    ] {
        if !g.has_module(m) {
            bail!("module {m} is not in this image: build with AVAROK_TARGET_HW=ada");
        }
    }
    println!(
        "kv_view_microtest: only={} reps={} quick={}",
        a.only, a.reps, a.quick
    );
    let mut gates = Gates { failed: 0 };
    if a.only != "score" {
        sketch::run(g, &mut gates)?;
    }
    if a.only != "sketch" {
        score::run(g, &a, &mut gates)?;
    }
    if gates.failed > 0 {
        println!("kv_view_microtest: {} gate(s) FAILED", gates.failed);
        std::process::exit(1);
    }
    println!("kv_view_microtest: all gates passed");
    Ok(())
}
