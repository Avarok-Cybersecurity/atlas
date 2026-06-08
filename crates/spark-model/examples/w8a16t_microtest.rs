// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness + kernel-only-throughput microtest for the
//! TRANSPOSED `w8a16_gemm_t` family (FP8 E4M3 weights × BF16 activations, 2D
//! transposed block scales).
//!
//! This is the grounding oracle for the `w8a16_gemm_t_pipelined` rewrite:
//! every kernel iteration is validated here (seconds) against an independent
//! CPU reference BEFORE any full build→deploy→cosine cycle. It launches the
//! real kernel via the production `GpuBackend`/`AtlasRegistry` path (SBIO/SSOT)
//! and compares the BF16 output to a CPU recompute that mirrors the kernel's
//! exact two-level FP32 accumulation (inner over a 128-K block, then `outer +=
//! inner * block_scale`) — the accumulation order that holds the L31-39
//! precision floor and MUST be preserved by any rewrite.
//!
//! TRANSPOSED CONTRACT (differs from `w8a16_microtest`):
//!   C[M,N] = A[M,K] (BF16) * dequant(B_t[K,N] FP8 E4M3, N-contiguous) with
//!   block_scale_t[K/128, N/128] FP32. We generate the weights in the natural
//!   `B[N,K]` + `scale[N/128,K/128]` layout (so the CPU reference is shared
//!   with the non-transposed oracle), then host-transpose to `B_t[K,N]` +
//!   `scale_t[K/128,N/128]` for the GPU upload — exactly what the production
//!   `transpose_fp8` / `transpose_block_scale` kernels produce at load time.
//!
//! GPU vs CPU are NOT bit-identical (tensor-core MMA sums in a different order
//! than the sequential CPU loop, and both narrow to BF16 at the end), so the
//! gate is cosine similarity + relative error, not byte-equality.
//!
//! Usage:
//!   cargo run --release -p spark-model --example w8a16t_microtest \
//!       -- [kernel_name] [M] [N] [K] [seed]
//! Defaults: w8a16_gemm_t 512 2048 4096 0x51A7
//!
//! Exit code 0 = PASS (cosine >= threshold), 1 = FAIL — so it is scriptable.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

// Raw CUDA driver event API for kernel-only timing (see w8a16_microtest.rs).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

/// FP8 block size along both N and K (matches `FP8_BLOCK` in w8a16_gemm_t.cu).
const FP8_BLOCK: usize = 128;
/// Cosine gate — a correct kernel matches the CPU reference to ~1e-5.
const COSINE_GATE: f64 = 0.9995;

// ───────────────────────── deterministic PRNG ─────────────────────────
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

/// OCP E4M3 (e4m3fn) decode — independent re-derivation of the kernel's LUT.
fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as i32;
    if exp == 0 {
        sign * (mant as f32 / 8.0) * 2f32.powi(-6)
    } else if exp == 0x0F && mant == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mant as f32 / 8.0) * 2f32.powi(exp - 7)
    }
}

// ───────────────────────── upload helpers (via GpuBackend) ─────────────────
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(bytes, ptr)?;
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
/// Inputs are in the NATURAL (non-transposed) layout: `b_fp8` is `[N,K]`
/// (K-contiguous, `b_fp8[col*k + g_k]`) and `scale` is `[N/128, K/128]`.
/// This matches the transposed kernel's math exactly — transposing B and the
/// scales is a pure index relabel, not a numerical change.
fn cpu_reference(a_bf16: &[u16], b_fp8: &[u8], scale: &[f32], m: usize, n: usize, k: usize) -> Vec<u16> {
    let k_blocks = k / FP8_BLOCK;
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

/// Host transpose B[N,K] → B_t[K,N] (mirrors `transpose_fp8` kernel).
fn transpose_b(b: &[u8], n: usize, k: usize) -> Vec<u8> {
    let mut bt = vec![0u8; k * n];
    for nn in 0..n {
        for kk in 0..k {
            bt[kk * n + nn] = b[nn * k + kk];
        }
    }
    bt
}

/// Host transpose scale[N/128,K/128] → scale_t[K/128,N/128]
/// (mirrors `transpose_block_scale` kernel).
fn transpose_scale(scale: &[f32], n_blocks: usize, k_blocks: usize) -> Vec<f32> {
    let mut st = vec![0f32; k_blocks * n_blocks];
    for nb in 0..n_blocks {
        for kb in 0..k_blocks {
            st[kb * n_blocks + nb] = scale[nb * k_blocks + kb];
        }
    }
    st
}

/// Per-kernel launch geometry. Both transposed kernels take the SAME args
/// (A, B_t, scale_t, C, M, N, K); only the grid/block tiling differs.
fn launch_geometry(name: &str, m: u32, n: u32, k: u32) -> Result<([u32; 3], [u32; 3])> {
    let _ = k;
    Ok(match name {
        // Current production transposed kernel: 64×64 tile, 128-thread block.
        "w8a16_gemm_t" => ([n.div_ceil(64), m.div_ceil(64), 1], [128u32, 1, 1]),
        // Pipelined rewrite: 128×32 tile (M×N), 256-thread block (8 warps).
        "w8a16_gemm_t_pipelined" => ([n.div_ceil(32), m.div_ceil(128), 1], [256u32, 1, 1]),
        other => bail!("no launch geometry registered for kernel '{other}' — add an arm"),
    })
}

#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &dyn GpuBackend,
    name: &str,
    ptrs: [DevicePtr; 4],
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    sync: bool,
) -> Result<()> {
    let [a, b, scale, c] = ptrs;
    let handle = gpu.kernel("w8a16_gemm_t", name)?;
    let (grid, block) = launch_geometry(name, m, n, k)?;
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
    if sync {
        gpu.synchronize(stream)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kernel = args.get(1).cloned().unwrap_or_else(|| "w8a16_gemm_t".to_string());
    let m: usize = args.get(2).map_or(512, |s| s.parse().unwrap());
    let n: usize = args.get(3).map_or(2048, |s| s.parse().unwrap());
    let k: usize = args.get(4).map_or(4096, |s| s.parse().unwrap());
    let seed: u64 = args
        .get(5)
        .map_or(0x51A7, |s| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7));

    if k % FP8_BLOCK != 0 {
        bail!("K ({k}) must be a multiple of FP8_BLOCK ({FP8_BLOCK}) for the clean-block path");
    }
    println!("=== w8a16t microtest: kernel='{kernel}' M={m} N={n} K={k} seed=0x{seed:X} ===");

    // ── generate inputs (natural B[N,K] / scale[N/128,K/128] layout) ──
    let mut rng = Rng(seed);
    let a_bf16: Vec<u16> = (0..m * k).map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0))).collect();
    // FP8 weights restricted to exp<=7 (magnitude <= ~1.875), no NaN.
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
    let scale: Vec<f32> = (0..n_blocks * k_blocks).map(|_| rng.uniform(0.5, 1.5)).collect();

    // ── transpose for the TRANSPOSED kernel contract (host-side, matches the
    //    production transpose_fp8 / transpose_block_scale kernels) ──
    let b_t = transpose_b(&b_fp8, n, k); // [K, N]
    let scale_t = transpose_scale(&scale, n_blocks, k_blocks); // [K/128, N/128]

    // ── GPU ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_to_le(&a_bf16))?;
    let b_ptr = upload_bytes(gpu, &b_t)?;
    let s_ptr = upload_bytes(gpu, &f32s_to_le(&scale_t))?;
    let c_ptr = gpu.alloc(m * n * 2)?;
    let ptrs = [a_ptr, b_ptr, s_ptr, c_ptr];

    launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, true)?;
    let mut c_raw = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_ptr, &mut c_raw)?;
    let c_gpu: Vec<u16> = c_raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();

    // ── CPU reference (natural layout — math is layout-invariant) ──
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
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, true)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, true)?;
    }
    let per_iter_s = t0.elapsed().as_secs_f64() / iters as f64;
    let tflops = (2.0 * m as f64 * n as f64 * k as f64) / per_iter_s / 1e12;

    // ── kernel-only throughput (CUDA events on the launch stream) ──
    let (mut ev_start, mut ev_end): (u64, u64) = (0, 0);
    if unsafe { cuEventCreate(&mut ev_start, 0) } != 0 {
        bail!("cuEventCreate(start) failed");
    }
    if unsafe { cuEventCreate(&mut ev_end, 0) } != 0 {
        bail!("cuEventCreate(end) failed");
    }
    if unsafe { cuEventRecord(ev_start, stream) } != 0 {
        bail!("cuEventRecord(start) failed");
    }
    for _ in 0..iters {
        launch(gpu, &kernel, ptrs, m as u32, n as u32, k as u32, stream, false)?;
    }
    if unsafe { cuEventRecord(ev_end, stream) } != 0 {
        bail!("cuEventRecord(end) failed");
    }
    if unsafe { cuEventSynchronize(ev_end) } != 0 {
        bail!("cuEventSynchronize(end) failed");
    }
    let mut elapsed_ms: f32 = 0.0;
    if unsafe { cuEventElapsedTime(&mut elapsed_ms, ev_start, ev_end) } != 0 {
        bail!("cuEventElapsedTime failed");
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
    println!("perf: {:.3} ms/iter  ~{tflops:.2} TFLOP/s (wall-clock incl. launch)", per_iter_s * 1e3);
    println!("kernel-only: {:.4} ms/iter  ~{kernel_tflops:.2} TFLOP/s (CUDA events)", kernel_s * 1e3);

    if cosine >= COSINE_GATE && cosine.is_finite() {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        eprintln!("RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE}) — layout/dequant/accumulation mismatch");
        std::process::exit(1);
    }
}
