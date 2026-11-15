// SPDX-License-Identifier: AGPL-3.0-only

//! Timing-only A/B of the DECODE projection GEMVs: `w8a16_gemv` (FP8 weight,
//! 1 byte) vs `w4a16_gemv` (NVFP4 weight, 0.5 byte) at the real Holo projection
//! shapes (M=1 decode). Answers ONE question: on the current CUDA 13.2 / sm_121
//! kernels, is the FP4 decode GEMV actually faster than the FP8 LUT GEMV? The
//! code keeps decode on FP8 to avoid FP8->BF16->FP4 double-quant (a *quality*
//! argument); the *speed* claim ("FP8 LUT beats w4a16 at M=1") may predate 13.2.
//!
//! Timing-only: buffers hold synthetic but finite weights (FP8 bytes without NaN
//! codes, NVFP4 scales in the finite positive range), correctly SIZED so the
//! kernel's memory traffic is real; we never check the output (the correctness
//! gates are `w4a16_ada_microtest` and the bit-parity microtests). Both kernels
//! are launched via the same KernelLaunch path the production wrappers use
//! (identical grid/block/arg order).
//!
//! Timings: the weights ROTATE through distinct
//! copies totaling more than twice the L2 (`common/cold_l2.rs`), so every call
//! is a DRAM read, and the two arms are timed in rotated, interleaved order
//! (five rounds, 3 warm-ups, 20 calls per sample, CUDA events). The run refuses
//! to report a number whose ring fits the L2; `--hot` prints the L2-resident
//! contrast, marked as not a bandwidth number. Effective GB/s uses each
//! kernel's real weight bytes (plus its scales).
//!
//! Usage: cargo run --release -p spark-model --example gemv_fp4_vs_fp8_microtest -- [N] [K] [--hot]
//! Default N=12288 K=2048 (Holo in_proj_qkv). Other shapes: q_proj 8192/2048,
//! out_proj 2048/4096. N and K come first when given.

use anyhow::Result;
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

#[path = "common/bench_args.rs"]
mod bench_args;
#[path = "common/cold_l2.rs"]
mod cold_l2;
#[path = "common/profiler.rs"]
mod profiler;
#[path = "common/weight_fill.rs"]
mod weight_fill;

use cold_l2::{Arm, bench_arms, build_ring, gbs, plan_hot, plan_ring, temperature_label};
use weight_fill::{Fill, alloc_filled};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let positional = |i: usize, default: u32| {
        args.get(i)
            .filter(|s| !s.starts_with("--"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    let (n, k) = (positional(1, 12288), positional(2, 2048));
    let hot = bench_args::has_flag("--hot");

    let backend = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;
    let info = profiler::device_info();
    let l2 = if info.l2_bytes > 0 {
        info.l2_bytes
    } else {
        cold_l2::FALLBACK_L2_BYTES
    };
    let (nn, kk) = (n as usize, k as usize);

    // Bytes one call reads: the weight plus its scales (the activation is tiny
    // and stays cached, as in decode).
    let fp8_scale_bytes = nn.div_ceil(128) * kk.div_ceil(128) * 4;
    let fp8_bytes = nn * kk + fp8_scale_bytes;
    let fp4_scale_bytes = nn * (kk / 16);
    let fp4_bytes = nn * kk / 2 + fp4_scale_bytes;
    let (plan8, plan4) = if hot {
        (plan_hot(fp8_bytes, l2), plan_hot(fp4_bytes, l2))
    } else {
        (
            plan_ring(fp8_bytes, l2, cold_l2::DEFAULT_RING_CAP)?,
            plan_ring(fp4_bytes, l2, cold_l2::DEFAULT_RING_CAP)?,
        )
    };
    let (temp8, temp4) = (
        temperature_label(&plan8, hot)?,
        temperature_label(&plan4, hot)?,
    );
    println!(
        "=== decode GEMV A/B: N={n} K={k} (M=1)  L2 {} MiB | FP8 ring {} x {:.1} MB, FP4 ring {} x {:.1} MB ({temp8})",
        l2 >> 20,
        plan8.copies,
        fp8_bytes as f64 / 1e6,
        plan4.copies,
        fp4_bytes as f64 / 1e6
    );

    let input = alloc_filled(gpu, kk * 2, Fill::Bf16, 1)?; // [1,K] BF16
    let output = gpu.alloc(nn * 2)?; // [1,N] BF16
    let ring8 = build_ring(&plan8, |c| {
        Ok((
            alloc_filled(gpu, nn * kk, Fill::Fp8E4m3, 100 + c as u64)?,
            alloc_filled(gpu, fp8_scale_bytes, Fill::ScaleF32, 200 + c as u64)?,
        ))
    })?;
    let ring4 = build_ring(&plan4, |c| {
        Ok((
            alloc_filled(gpu, nn * kk / 2, Fill::Nvfp4Packed, 300 + c as u64)?,
            alloc_filled(gpu, fp4_scale_bytes, Fill::Nvfp4Scale, 400 + c as u64)?,
        ))
    })?;

    let k_w8 = gpu.kernel("w8a16_gemv", "w8a16_gemv")?;
    let k_w4 = gpu.kernel("w4a16_gemv", "w4a16_gemv")?;
    let mut arms = vec![
        // FP8 (w8a16_gemv): input, weight, row_scale, output, n, k
        Arm::new("fp8", || {
            let (w, s) = ring8.pick();
            KernelLaunch::new(gpu, k_w8)
                .grid([div_ceil(n, 4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(w)
                .arg_ptr(s)
                .arg_ptr(output)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        }),
        // FP4 (w4a16_gemv): input, weight, weight_scale, scale2(f32), output, n, k
        Arm::new("fp4", || {
            let (w, s) = ring4.pick();
            KernelLaunch::new(gpu, k_w4)
                .grid([div_ceil(n, 4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(w)
                .arg_ptr(s)
                .arg_f32(1.0)
                .arg_ptr(output)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        }),
    ];
    let r = bench_arms(gpu, &mut arms, &bench_args::bench_cfg(stream)?)?;
    drop(arms);
    let (fp8_us, fp4_us) = (r[0].median_us, r[1].median_us);

    println!(
        "FP8 w8a16_gemv : {:.4} ms/iter   {:.0} GB/s (weight {:.1} MB, spread {:.1}%)",
        fp8_us / 1e3,
        gbs(fp8_bytes as f64, fp8_us),
        fp8_bytes as f64 / 1e6,
        r[0].spread_pct
    );
    println!(
        "FP4 w4a16_gemv : {:.4} ms/iter   {:.0} GB/s (weight {:.1} MB, spread {:.1}%)",
        fp4_us / 1e3,
        gbs(fp4_bytes as f64, fp4_us),
        fp4_bytes as f64 / 1e6,
        r[1].spread_pct
    );
    println!(
        "SPEEDUP (FP8/FP4): {:.2}x  {}",
        fp8_us / fp4_us,
        if fp4_us < fp8_us {
            "-> FP4 decode is FASTER on this stack"
        } else {
            "-> FP8 decode still wins"
        }
    );
    if temp4 != "cold" {
        println!("NOTE: {temp4}; do not read these as bandwidth.");
    }

    for p in [input, output] {
        gpu.free(p)?;
    }
    for (w, s) in ring8.all().iter().chain(ring4.all()) {
        gpu.free(*w)?;
        gpu.free(*s)?;
    }
    Ok(())
}
