// SPDX-License-Identifier: AGPL-3.0-only

//! Atlas CUTLASS NVFP4 GEMM throughput at the Holo MoE expert shapes — the clean
//! apples-to-apples vs vLLM's MARLIN NVFP4 GEMM (rand_marlin_weight_nvfp4_like /
//! apply_fp4_marlin_linear). Timing only (values irrelevant); weight layout per
//! gemm.rs: packed [K/2,N], scales [K/16,N], global scale f32.
use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::cutlass::{bf16_gemm_act_weight_t, nvfp4_gemm_bf16_act_weight_t};
use spark_runtime::gpu::GpuBackend;

/// W16A16 baseline: pure BF16 act @ BF16 weight^T (no FP4, no act-pack). This is
/// the upper bound a W4A16 path approaches (W4A16 = this + in-kernel FP4 dequant of
/// the weight). Same MoE shapes as the FP4 bench → small-M head-to-head.
fn bench_bf16(g: &dyn GpuBackend, n: usize, k: usize, label: &str) -> Result<()> {
    let stream = 0u64;
    let w = g.alloc(n * k * 2)?; // BF16 weight [N,K]
    g.memset(w, 0x3c, n * k * 2)?;
    for &m in &[512usize, 2048, 8192, 16384] {
        let act = g.alloc(m * k * 2)?;
        g.memset(act, 0x3c, m * k * 2)?;
        let out = g.alloc(m * n * 2)?;
        g.memset(out, 0, m * n * 2)?;
        let call =
            || bf16_gemm_act_weight_t(act.0, w.0, out.0, m as u32, n as u32, k as u32, stream);
        for _ in 0..10 {
            if let Err(e) = call() {
                println!(
                    "  {label} M={m}: ERR {}",
                    format!("{e}").chars().take(120).collect::<String>()
                );
                let _ = g.free(act);
                let _ = g.free(out);
                continue;
            }
        }
        g.synchronize(stream)?;
        let it = 100u32;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            call()?;
        }
        g.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
        let tflops = 2.0 * (m * n * k) as f64 / (us * 1e-6) / 1e12;
        println!("  {label} M={m:5} N={n} K={k}: {us:8.1}us  {tflops:6.1} TFLOP/s");
        let _ = g.free(act);
        let _ = g.free(out);
    }
    let _ = g.free(w);
    Ok(())
}

fn bench(g: &dyn GpuBackend, n: usize, k: usize, label: &str) -> Result<()> {
    let stream = 0u64;
    let wpt = g.alloc((k / 2) * n)?;
    g.memset(wpt, 0x11, (k / 2) * n)?;
    let wst = g.alloc((k / 16) * n)?;
    g.memset(wst, 0x3c, (k / 16) * n)?; // arbitrary nonzero e4m3 bytes
    for &m in &[512usize, 2048, 8192, 16384] {
        let act = g.alloc(m * k * 2)?;
        g.memset(act, 0x10, m * k * 2)?;
        let out = g.alloc(m * n * 2)?;
        g.memset(out, 0, m * n * 2)?;
        let call = || {
            nvfp4_gemm_bf16_act_weight_t(
                act.0, wpt.0, wst.0, 1.0, out.0, m as u32, n as u32, k as u32, stream,
            )
        };
        // warmup
        for _ in 0..10 {
            if let Err(e) = call() {
                println!(
                    "  {label} M={m}: ERR {}",
                    format!("{e}").chars().take(120).collect::<String>()
                );
                let _ = g.free(act);
                let _ = g.free(out);
                continue;
            }
        }
        g.synchronize(stream)?;
        let it = 100u32;
        let t0 = std::time::Instant::now();
        for _ in 0..it {
            call()?;
        }
        g.synchronize(stream)?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / it as f64;
        let tflops = 2.0 * (m * n * k) as f64 / (us * 1e-6) / 1e12;
        println!("  {label} M={m:5} N={n} K={k}: {us:8.1}us  {tflops:6.1} TFLOP/s");
        let _ = g.free(act);
        let _ = g.free(out);
    }
    let _ = g.free(wpt);
    let _ = g.free(wst);
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    if let Ok(spec) = std::env::var("ATLAS_BENCH_SHAPE") {
        let v: Vec<usize> = spec.split(',').map(|x| x.parse().unwrap()).collect();
        let (n, k, m) = (v[0], v[1], v[2]);
        // single shape, few launches (for ncu): warmup 3 + 5 timed
        let wpt = g.alloc((k / 2) * n)?;
        g.memset(wpt, 0x11, (k / 2) * n)?;
        let wst = g.alloc((k / 16) * n)?;
        g.memset(wst, 0x3c, (k / 16) * n)?;
        let act = g.alloc(m * k * 2)?;
        g.memset(act, 0x10, m * k * 2)?;
        let out = g.alloc(m * n * 2)?;
        g.memset(out, 0, m * n * 2)?;
        for _ in 0..8 {
            nvfp4_gemm_bf16_act_weight_t(
                act.0, wpt.0, wst.0, 1.0, out.0, m as u32, n as u32, k as u32, 0,
            )?;
        }
        g.synchronize(0)?;
        println!("single-shape done N={n} K={k} M={m}");
        return Ok(());
    }
    println!("=== Atlas CUTLASS NVFP4 GEMM (W4A4) @ Holo expert shapes ===");
    bench(g, 512, 2048, "gate_up(N=512,K=2048)")?;
    bench(g, 2048, 512, "down   (N=2048,K=512)")?;
    println!("=== Atlas CUTLASS BF16 GEMM (W16A16 baseline) @ same shapes ===");
    bench_bf16(g, 512, 2048, "gate_up(N=512,K=2048)")?;
    bench_bf16(g, 2048, 512, "down   (N=2048,K=512)")?;
    Ok(())
}
