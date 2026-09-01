// SPDX-License-Identifier: AGPL-3.0-only
//! EXL3_BENCH=1 leg: launch timing at qwen4_exp shapes.
//!
//! NOT an apples-to-apples per-kernel comparison against the shipping
//! reconstruct+NVFP4 requant path (that path pays its cost at LOAD time and
//! serves NVFP4 kernels) — this times the native fused launches themselves:
//!   (i) exl3_gemv m=1 [2560 -> 10240] K=4 cb2 (decode dense proj shape),
//!  (ii) exl3_gemm m=2048 same shape (prefill),
//! plus gemm m=1 for the gemv-vs-gemm crossover question on GB10.
//! 20 warmup + 200 timed launches, wall-clock us/launch (sync at end).

use anyhow::Result;
use std::time::Instant;

use spark_model::layers::ops::{exl3_gemm, exl3_gemv};

use crate::util::{Ctx, DevWeight, Lcg, as_bytes, up};

const WARMUP: usize = 20;
const ITERS: usize = 200;

#[allow(clippy::too_many_arguments)]
fn time_arm(
    ctx: &Ctx,
    label: &str,
    m: usize,
    k: usize,
    n: usize,
    k_bits: u32,
    launch: &dyn Fn() -> Result<()>,
) -> Result<f64> {
    let g = ctx.g;
    let stream = g.default_stream();
    for _ in 0..WARMUP {
        launch()?;
    }
    g.synchronize(stream)?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        launch()?;
    }
    g.synchronize(stream)?;
    let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    let weight_bytes = (n * k * k_bits as usize) as f64 / 8.0;
    let gbs = weight_bytes / (us * 1e-6) / 1e9;
    println!("bench {label:<44} m={m:<5} {us:>9.2} us/launch  ({gbs:>6.1} GB/s quantized-weight)");
    Ok(us)
}

pub fn run(ctx: &Ctx, rng: &mut Lcg) -> Result<()> {
    let g = ctx.g;
    let (k, n) = (2560usize, 10240usize);
    let (k_bits, cb) = (4u32, 2u32);
    println!("--- bench [{k} -> {n}] K={k_bits} cb{cb} ({WARMUP} warmup + {ITERS} iters) ---");

    let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * k_bits as usize)
        .map(|_| rng.u16())
        .collect();
    let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
    let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
    let w = DevWeight::upload(g, &trellis, &suh, &svh)?;

    let m_big = 2048usize;
    let a: Vec<u16> = (0..m_big * k).map(|_| rng.act_f16()).collect();
    let a_d = up(g, &as_bytes(&a))?;
    let a_had_d = g.alloc(m_big * k * 2)?;
    let c_d = g.alloc(m_big * n * 4)?;
    let stream = g.default_stream();

    // (i) decode-shape gemv, both configs, f32 and f16 C.
    for (cfg, c_fp32) in [(0u32, true), (1, true), (1, false)] {
        let label = format!("gemv cfg{cfg} {}", if c_fp32 { "f32" } else { "f16" });
        time_arm(ctx, &label, 1, k, n, k_bits, &|| {
            let launched = exl3_gemv(
                g, a_d, w.trellis, c_d, 1, k, n, k_bits, cb, c_fp32, ctx.locks, w.suh,
                a_had_d, w.svh, Some(cfg), ctx.sms, stream,
            )?;
            anyhow::ensure!(launched, "gemv refused");
            Ok(())
        })?;
    }

    // gemv-vs-gemm crossover: gemm at m=1 (heuristic shape).
    let gemm_arm = |m: usize, label: &str| -> Result<f64> {
        time_arm(ctx, label, m, k, n, k_bits, &|| {
            exl3_gemm(
                g, a_d, w.trellis, c_d, m, k, n, k_bits, cb, true, ctx.locks, w.suh, a_had_d,
                w.svh, None, ctx.sms, stream,
            )
        })
    };
    gemm_arm(1, "gemm f32 (heuristic shape)")?;

    // (ii) prefill-shape gemm.
    gemm_arm(m_big, "gemm f32 (heuristic shape)")?;

    for p in [a_d, a_had_d, c_d] {
        g.free(p).ok();
    }
    w.free(g);

    // Memory table for this tensor.
    let exl3 = (n * k * k_bits as usize) as f64 / 8.0 + 2.0 * (n + k) as f64 + 4.0;
    let bf16 = (n * k * 2) as f64;
    let nvfp4 = (n * k) as f64 / 2.0 + (n * k / 16) as f64 + 4.0;
    println!(
        "memory [{k}x{n}]: EXL3 K={k_bits} {:.1} MB | NVFP4-materialized {:.1} MB | BF16-materialized {:.1} MB",
        exl3 / 1e6,
        nvfp4 / 1e6,
        bf16 / 1e6
    );
    Ok(())
}
