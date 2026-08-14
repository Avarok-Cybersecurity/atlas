// SPDX-License-Identifier: AGPL-3.0-only
//! Cold-DRAM bandwidth oracle for `w4a16_gemv_batch2` and one optional
//! candidate kernel.
//!
//! This is the kill-gate for prefetch experiments. `#497` cp.async lost
//! here (195 vs 227 GB/s on GDN in_proj 12288×2048). A candidate that is
//! >3% slower than template batch2 on a production shape must not default-on.
//!
//! Weight copies cycle past L2 so every launch streams from DRAM the way a
//! real K=2 verify step does. GB/s is packed+scale bytes / GPU time.
//!
//! Env:
//!   ATLAS_GEMV_BATCH2_CANDIDATE  module:kernel (default none — baseline only)
//!   ATLAS_STREAM_GBPS            STREAM read ceiling (default 230)
//!   ATLAS_PEAK_GBPS              datasheet peak (default 273)
//!
//! Exit 0 if no candidate, or candidate is not >3% slower.
//! Exit 1 if candidate loses the 3% bar.
//! Exit 2 if template batch2 is absent.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-35b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w4a16_batch2_bw_oracle

use anyhow::Result;
use spark_model::layers::ops::gemv_batch2_oracle::{
    FAIL_SLOWER, OracleVerdict, PEAK_GBPS_DEFAULT, STREAM_GBPS_DEFAULT, gbps_from, parse_candidate,
    verdict,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const WARMUP: usize = 8;
const ITERS: usize = 40;
const COLD_CYCLE_BYTES: usize = 256 << 20;
const M: usize = 2;

const SHAPES: &[(&str, u32, u32)] = &[
    ("gdn in_proj N=12288 K=2048", 12288, 2048),
    ("attn Q      N=8192  K=2048", 8192, 2048),
    ("gdn out     N=2048  K=4096", 2048, 4096),
];

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit_f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 23) as f32) * 2.0 - 1.0
    }
}

fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(1.0)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

struct Timed {
    name: String,
    us: f64,
    gbps: f64,
}

fn time_kernel(
    g: &dyn GpuBackend,
    name: &str,
    kh: KernelHandle,
    a: DevicePtr,
    copies: &[(DevicePtr, DevicePtr)],
    c: DevicePtr,
    n: u32,
    k: u32,
    weight_bytes: usize,
) -> Result<Timed> {
    let go = |i: usize| {
        launch(
            g,
            kh,
            a,
            copies[i % copies.len()].0,
            copies[i % copies.len()].1,
            c,
            n,
            k,
        )
    };
    for i in 0..WARMUP {
        go(i)?;
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for i in 0..ITERS {
        go(WARMUP + i)?;
    }
    g.synchronize(0)?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let gbps = weight_bytes as f64 / (us * 1e-6) / 1e9;
    Ok(Timed {
        name: name.to_string(),
        us,
        gbps,
    })
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let peak = gbps_from(
        std::env::var("ATLAS_PEAK_GBPS").ok().as_deref(),
        PEAK_GBPS_DEFAULT,
    );
    let stream_bw = gbps_from(
        std::env::var("ATLAS_STREAM_GBPS").ok().as_deref(),
        STREAM_GBPS_DEFAULT,
    );

    let Ok(b2) = g.kernel("w4a16_gemv", "w4a16_gemv_batch2") else {
        eprintln!("w4a16_gemv_batch2 absent — SKIP");
        std::process::exit(2);
    };
    let cand_spec = std::env::var("ATLAS_GEMV_BATCH2_CANDIDATE").ok();
    let cand = cand_spec
        .as_deref()
        .and_then(parse_candidate)
        .and_then(|(m, k)| g.kernel(m, k).ok().map(|h| (format!("{m}:{k}"), h)));

    eprintln!(
        "W4A16 batch2 cold-DRAM oracle  STREAM {stream_bw:.0}  peak {peak:.0}  {ITERS} iters"
    );
    if let Some((name, _)) = &cand {
        eprintln!("candidate {name}");
    } else {
        eprintln!("no candidate (baseline only — set ATLAS_GEMV_BATCH2_CANDIDATE=module:kernel)");
    }
    eprintln!();

    let mut rng = XorShift(0x9E3779B97F4A7C15);
    let mut ok = true;
    for &(label, n, k) in SHAPES {
        let (n_us, k_us) = (n as usize, k as usize);
        let packed_bytes = n_us * k_us / 2;
        let scale_bytes = n_us * k_us / 16;
        let weight_bytes = packed_bytes + scale_bytes;
        let copies_n = (COLD_CYCLE_BYTES.div_ceil(weight_bytes)).clamp(1, 16);

        let a_host: Vec<u16> = (0..M * k_us)
            .map(|_| f32_to_bf16_bits(rng.unit_f32()))
            .collect();
        let b_host: Vec<u8> = (0..packed_bytes).map(|_| rng.byte()).collect();
        let bs_host: Vec<u8> = (0..scale_bytes)
            .map(|_| 0x18 + (rng.byte() & 0x0F))
            .collect();

        let a = g.alloc(M * k_us * 2)?;
        let c = g.alloc(M * n_us * 2)?;
        let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&a_bytes, a)?;
        let mut copies = Vec::with_capacity(copies_n);
        for _ in 0..copies_n {
            let b = g.alloc(packed_bytes)?;
            let bs = g.alloc(scale_bytes)?;
            g.copy_h2d(&b_host, b)?;
            g.copy_h2d(&bs_host, bs)?;
            copies.push((b, bs));
        }

        let floor_us = weight_bytes as f64 / (stream_bw * 1e9) * 1e6;
        eprintln!(
            "── {label}  weights {:.2} MB × {copies_n} copies, STREAM floor {floor_us:.1} us ──",
            weight_bytes as f64 / 1e6
        );

        let t_b2 = time_kernel(g, "batch2", b2, a, &copies, c, n, k, weight_bytes)?;
        eprintln!(
            "  {:<28} {:>8.2} us  {:>6.1} GB/s  {:>5.1}% STREAM  {:>5.1}% peak",
            t_b2.name,
            t_b2.us,
            t_b2.gbps,
            100.0 * t_b2.gbps / stream_bw,
            100.0 * t_b2.gbps / peak,
        );
        if let Some((name, kh)) = &cand {
            let t_c = time_kernel(g, name, *kh, a, &copies, c, n, k, weight_bytes)?;
            let vs = t_c.us / t_b2.us;
            let shape = verdict(t_c.us, t_b2.us);
            eprintln!(
                "  {:<28} {:>8.2} us  {:>6.1} GB/s  {:>5.1}% STREAM  {:>5.1}% peak",
                t_c.name,
                t_c.us,
                t_c.gbps,
                100.0 * t_c.gbps / stream_bw,
                100.0 * t_c.gbps / peak,
            );
            eprintln!("  candidate / batch2 = {vs:.3}x   {shape:?}   (fail if >{FAIL_SLOWER})");
            if shape == OracleVerdict::Fail {
                ok = false;
            }
        }
        eprintln!();

        g.free(a)?;
        g.free(c)?;
        for (b, bs) in copies {
            g.free(b)?;
            g.free(bs)?;
        }
    }

    if cand.is_none() {
        eprintln!("PASS — baseline recorded (no candidate)");
        return Ok(());
    }
    if ok {
        eprintln!("PASS — candidate is not a >3% regression vs batch2");
        Ok(())
    } else {
        eprintln!("FAIL — candidate is >3% slower than batch2 on a production shape");
        std::process::exit(1);
    }
}
