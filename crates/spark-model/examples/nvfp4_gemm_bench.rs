//! Atlas CUTLASS NVFP4 GEMM throughput at the Holo MoE expert shapes — the clean
//! apples-to-apples vs vLLM's MARLIN NVFP4 GEMM (rand_marlin_weight_nvfp4_like /
//! apply_fp4_marlin_linear). Timing only (values irrelevant); weight layout per
//! gemm.rs: packed [K/2,N], scales [K/16,N], global scale f32.
use anyhow::Result;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::cutlass::nvfp4_gemm_bf16_act_weight_t;
use spark_runtime::gpu::GpuBackend;

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
            nvfp4_gemm_bf16_act_weight_t(act.0, wpt.0, wst.0, 1.0, out.0, m as u32, n as u32, k as u32, stream)
        };
        // warmup
        for _ in 0..10 {
            if let Err(e) = call() {
                println!("  {label} M={m}: ERR {}", format!("{e}").chars().take(120).collect::<String>());
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
    println!("=== Atlas CUTLASS NVFP4 GEMM @ Holo expert shapes ===");
    bench(g, 512, 2048, "gate_up(N=512,K=2048)")?;
    bench(g, 2048, 512, "down   (N=2048,K=512)")?;
    Ok(())
}
