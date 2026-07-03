// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness + rough-throughput microtest for the `w8a16_gemm`
//! family of kernels (FP8 E4M3 weights × BF16 activations, 2D block scales).
//!
//! This is the grounding oracle for the Fix-A pipelined-GEMM rewrite: every
//! kernel iteration is validated here (seconds) against an independent CPU
//! reference BEFORE any full build→deploy→cosine cycle. It launches the real
//! kernel via the production `GpuBackend`/`AtlasRegistry` path (SBIO/SSOT) and
//! compares the BF16 output to a CPU recompute that mirrors the kernel's exact
//! two-level FP32 accumulation (inner over a 128-K block, then `outer +=
//! inner * block_scale`) — the accumulation order that holds the L31-39
//! precision floor and MUST be preserved by any rewrite.
//!
//! GPU vs CPU are NOT bit-identical (tensor-core MMA sums in a different order
//! than the sequential CPU loop, and both narrow to BF16 at the end), so the
//! gate is cosine similarity + relative error, not byte-equality.
//!
//! Usage:
//!   cargo run --release -p spark-model --example w8a16_microtest \
//!       -- [kernel_name] [M] [N] [K] [seed]
//! Defaults: w8a16_gemm 128 512 2048 0x51A7
//!
//! Exit code 0 = PASS (cosine >= threshold), 1 = FAIL — so it is scriptable.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

// Raw CUDA driver event API for kernel-only timing. The wall-clock `Instant`
// metric below includes per-launch host overhead (~0.3 ms floor) which swamps
// the small per-lever deltas on the compute-bound large shapes. CUDA events
// recorded on the launch stream measure GPU execution time only, so the
// optimization signal is trustworthy. Signatures mirror atlas-spark-bench's
// gpu.rs (the SSOT for these decls in the benchmark crate).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// FP8 block size along both N and K (matches `FP8_BLOCK` in w8a16_gemm.cu).
const FP8_BLOCK: usize = 128;
/// Cosine gate. A correct kernel matches the CPU reference to ~1e-5; the
/// remaining gap is a few BF16-ULP rounding flips near tie boundaries. 0.9995
/// is loose enough not to false-fail on BF16 noise, tight enough to catch a
/// real GEMM bug (a transposed index or dropped K-block collapses cosine).
const COSINE_GATE: f64 = 0.9995;

// ───────────────────────── deterministic PRNG ─────────────────────────
// splitmix64 — reproducible test inputs without a `rand` dependency (PCND:
// seed is an explicit arg, so a failure is always reproducible).
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        // 24-bit mantissa of randomness in [0, 1)
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }
}

// ───────────────────────── number-format helpers ─────────────────────────
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 → BF16 bits, round-to-nearest-even — matches CUDA `__float2bfloat16`.
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16; // NaN → quiet NaN
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// OCP E4M3 (e4m3fn) decode: 1 sign, 4 exp (bias 7), 3 mantissa; no Inf;
/// S.1111.111 is the only NaN. Independent re-derivation of the kernel's
/// `E4M3_LUT` (an oracle must not import the artifact it validates).
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        // subnormal: 2^(1-7) * (mant/8)
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        // normal: 2^(exp-7) * (1 + mant/8)
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

// ───────────────────────── upload helpers (via GpuBackend) ─────────────────
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?; // synchronous on the backend's default stream
    Ok(ptr)
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// CPU reference mirroring the kernel's two-level FP32 accumulation exactly:
/// inner accumulates one full 128-K block, then `outer += inner * scale`.
fn cpu_reference(
    a_bf16: &[u16],
    b_fp8: &[u8],
    scale: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<u16> {
    let k_blocks = k / FP8_BLOCK;
    let n_blocks = n.div_ceil(FP8_BLOCK);
    let _ = n_blocks;
    let mut out = vec![0u16; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut outer = 0.0f32;
            for kb in 0..k_blocks {
                let mut inner = 0.0f32;
                for kk in 0..FP8_BLOCK {
                    let g_k = kb * FP8_BLOCK + kk;
                    let a = bf16_bits_to_f32(a_bf16[row * k + g_k]);
                    let b = e4m3_to_f32(b_fp8[col * k + g_k]);
                    inner += a * b;
                }
                let scl = scale[(col / FP8_BLOCK) * k_blocks + kb];
                outer += inner * scl;
            }
            out[row * n + col] = f32_to_bf16_bits(outer);
        }
    }
    out
}

/// Per-kernel launch geometry. Add a new arm when a rewritten kernel lands so
/// the harness can A/B old vs new on identical inputs.
fn launch(
    gpu: &dyn GpuBackend,
    name: &str,
    ptrs: [DevicePtr; 4],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let [a, b, scale, c] = ptrs;
    let handle = gpu.kernel(name, name)?;
    let (grid, block) = match name {
        // Geometry MUST match the production launcher (ops::w8a16_gemm) and the
        // per-target .cu: native-HIP (gfx1151) is a 256×128 tile / 512-thread
        // kernel; every other target (GB10 kernels/gb10/common/w8a16_gemm.cu,
        // M_TILE=N_TILE=64, 128 threads) is 64×64 / 128. Using the HIP geometry
        // on GB10 launches a 64×64 kernel with 512 threads → smem/stride skew =
        // false validation. (Bug found 2026-07-03 while isolating the 397B-V2
        // out_proj fault — the microtest passed the kernel at out_proj dims once
        // corrected, exonerating the kernel.)
        #[cfg(atlas_hip)]
        "w8a16_gemm" => ([n.div_ceil(128), m.div_ceil(256), 1], [512u32, 1, 1]),
        #[cfg(not(atlas_hip))]
        "w8a16_gemm" => ([n.div_ceil(64), m.div_ceil(64), 1], [128u32, 1, 1]),
        // Fix-A pipelined rewrite: 128×64 tile (M×N), 256-thread block (8 warps).
        "w8a16_gemm_pipelined" => ([n.div_ceil(32), m.div_ceil(128), 1], [256u32, 1, 1]),
        other => bail!("no launch geometry registered for kernel '{other}' — add an arm"),
    };
    KernelLaunch::new(gpu, handle)
        .grid(grid)
        .block(block)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(scale)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(())
}

/// Launch WITHOUT a trailing stream synchronize — for the CUDA-event timing
/// loop, where the per-launch host sync would serialize iterations and inflate
/// the measured GPU time. The bracketing CUDA events (recorded on the same
/// stream) capture only kernel execution. Geometry mirrors `launch`.
fn launch_no_sync(
    gpu: &dyn GpuBackend,
    name: &str,
    ptrs: [DevicePtr; 4],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let [a, b, scale, c] = ptrs;
    let handle = gpu.kernel(name, name)?;
    let (grid, block) = match name {
        // Geometry MUST match the production launcher (ops::w8a16_gemm) and the
        // per-target .cu: native-HIP (gfx1151) is a 256×128 tile / 512-thread
        // kernel; every other target (GB10 kernels/gb10/common/w8a16_gemm.cu,
        // M_TILE=N_TILE=64, 128 threads) is 64×64 / 128. Using the HIP geometry
        // on GB10 launches a 64×64 kernel with 512 threads → smem/stride skew =
        // false validation. (Bug found 2026-07-03 while isolating the 397B-V2
        // out_proj fault — the microtest passed the kernel at out_proj dims once
        // corrected, exonerating the kernel.)
        #[cfg(atlas_hip)]
        "w8a16_gemm" => ([n.div_ceil(128), m.div_ceil(256), 1], [512u32, 1, 1]),
        #[cfg(not(atlas_hip))]
        "w8a16_gemm" => ([n.div_ceil(64), m.div_ceil(64), 1], [128u32, 1, 1]),
        "w8a16_gemm_pipelined" => ([n.div_ceil(32), m.div_ceil(128), 1], [256u32, 1, 1]),
        other => bail!("no launch geometry registered for kernel '{other}' — add an arm"),
    };
    KernelLaunch::new(gpu, handle)
        .grid(grid)
        .block(block)
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(scale)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kernel = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "w8a16_gemm".to_string());
    let m: usize = args.get(2).map_or(128, |s| s.parse().unwrap());
    let n: usize = args.get(3).map_or(512, |s| s.parse().unwrap());
    let k: usize = args.get(4).map_or(2048, |s| s.parse().unwrap());
    let seed: u64 = args.get(5).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    if k % FP8_BLOCK != 0 {
        bail!("K ({k}) must be a multiple of FP8_BLOCK ({FP8_BLOCK}) for the clean-block path");
    }
    println!("=== w8a16 microtest: kernel='{kernel}' M={m} N={n} K={k} seed=0x{seed:X} ===");

    // ── generate inputs ──
    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // FP8 weights restricted to exp<=7 (magnitude <= ~1.875) — realistic small
    // post-scale weights, and impossible to encode a NaN (exp never 15).
    let b_fp8: Vec<u8> = (0..n * k)
        .map(|_| {
            let sign = (rng.next_u64() & 1) as u8;
            let exp = (rng.next_u64() % 8) as u8;
            let mant = (rng.next_u64() % 8) as u8;
            (sign << 7) | (exp << 3) | mant
        })
        .collect();
    let k_blocks = k / FP8_BLOCK;
    let n_blocks = n.div_ceil(FP8_BLOCK);
    let scale: Vec<f32> = (0..n_blocks * k_blocks)
        .map(|_| rng.uniform(0.5, 1.5))
        .collect();

    // ── GPU ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &b_fp8)?;
    let s_ptr = upload_bytes(gpu, &f32s_to_le(&scale))?;
    let c_ptr = gpu.alloc(m * n * 2)?;
    let ptrs = [a_ptr, b_ptr, s_ptr, c_ptr];

    launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    let mut c_raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_ptr, &mut c_raw)?;
    let c_gpu: Vec<u16> = c_raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    // ── CPU reference ──
    let c_cpu = cpu_reference(&a_bf16, &b_fp8, &scale, m, n, k);

    // ── compare (cosine + relative error in f32 space) ──
    let (mut dot, mut ng, mut nc, mut max_rel, mut sum_rel) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for i in 0..m * n {
        let g = bf16_bits_to_f32(c_gpu[i]) as f64;
        let c = bf16_bits_to_f32(c_cpu[i]) as f64;
        dot += g * c;
        ng += g * g;
        nc += c * c;
        let rel = (g - c).abs() / c.abs().max(1e-3);
        max_rel = max_rel.max(rel);
        sum_rel += rel;
    }
    let cosine = dot / (ng.sqrt() * nc.sqrt());
    let mean_rel = sum_rel / (m * n) as f64;

    // ── rough throughput (wall-clock, includes launch overhead; relative A/B) ──
    let iters = 50;
    for _ in 0..5 {
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    }
    let per_iter_s = t0.elapsed().as_secs_f64() / iters as f64;
    let tflops = (2.0 * m as f64 * n as f64 * k as f64) / per_iter_s / 1e12;

    // ── kernel-only throughput (CUDA events on the launch stream) ──
    // Brackets the 50 back-to-back launches (no intervening host sync) with
    // two events; cuEventElapsedTime gives total GPU time for the batch. This
    // excludes per-launch host overhead, so it is the trustworthy signal for
    // the small per-lever deltas on the compute-bound large shapes.
    let (mut ev_start, mut ev_end): (u64, u64) = (0, 0);
    let rc = unsafe { cuEventCreate(&mut ev_start, 0) };
    if rc != 0 {
        bail!("cuEventCreate(start) failed: status {rc}");
    }
    let rc = unsafe { cuEventCreate(&mut ev_end, 0) };
    if rc != 0 {
        bail!("cuEventCreate(end) failed: status {rc}");
    }
    // Warm-up already done above. Record start, fire all iters, record end.
    let rc = unsafe { cuEventRecord(ev_start, stream) };
    if rc != 0 {
        bail!("cuEventRecord(start) failed: status {rc}");
    }
    for _ in 0..iters {
        launch_no_sync(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream)?;
    }
    let rc = unsafe { cuEventRecord(ev_end, stream) };
    if rc != 0 {
        bail!("cuEventRecord(end) failed: status {rc}");
    }
    let rc = unsafe { cuEventSynchronize(ev_end) };
    if rc != 0 {
        bail!("cuEventSynchronize(end) failed: status {rc}");
    }
    let mut elapsed_ms: f32 = 0.0;
    let rc = unsafe { cuEventElapsedTime(&mut elapsed_ms, ev_start, ev_end) };
    if rc != 0 {
        bail!("cuEventElapsedTime failed: status {rc}");
    }
    unsafe {
        cuEventDestroy_v2(ev_start);
        cuEventDestroy_v2(ev_end);
    }
    let kernel_s = (elapsed_ms as f64 / 1e3) / iters as f64;
    let kernel_tflops = (2.0 * m as f64 * n as f64 * k as f64) / kernel_s / 1e12;

    for p in ptrs {
        gpu.free(p).ok();
    }

    println!("cosine={cosine:.6}  mean_rel={mean_rel:.2e}  max_rel={max_rel:.2e}");
    println!(
        "perf: {:.3} ms/iter  ~{tflops:.2} TFLOP/s (wall-clock incl. launch)",
        per_iter_s * 1e3
    );
    println!(
        "kernel-only: {:.4} ms/iter  ~{kernel_tflops:.2} TFLOP/s (CUDA events)",
        kernel_s * 1e3
    );

    if cosine >= COSINE_GATE && cosine.is_finite() {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        eprintln!(
            "RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE}) — layout/dequant/accumulation mismatch"
        );
        std::process::exit(1);
    }
}
