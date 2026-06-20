// SPDX-License-Identifier: AGPL-3.0-only

//! Standalone correctness + rough-throughput oracle for the MoE FP8 grouped
//! GEMM (`moe_fp8_grouped_gemm`, grid-compaction) — analogous to the w8a16
//! projection-GEMM oracle. It launches the real grouped-GEMM via the production
//! GpuBackend (per-expert FP8 weight + 128x128 block-scale pointer tables,
//! expert_offsets, sorted_token_ids) and compares the BF16 output to a CPU
//! reference that mirrors the kernel's exact two-level FP32 accumulation (inner
//! over a 128-K block, then `outer += inner * block_scale`) and OCP e4m3fn
//! decode — the accumulation order that holds the deep-layer FP8 floor.
//!
//! GPU vs CPU are NOT bit-identical (MMA sums in a different order + a final
//! BF16 narrowing), so the gate is cosine similarity, not byte-equality.
//!
//! Usage:
//!   cargo run --release -p spark-model --example moe_microtest \
//!       -- [kernel] [num_experts] [tokens_per_expert] [N] [K] [seed] [mode]
//! Defaults: moe_fp8_grouped_gemm 4 20 256 256 0x9E3
//!
//! [kernel] is `moe_fp8_grouped_gemm` (the only routed-expert FP8 prefill
//! kernel). [mode] (arg 7):
//!   perf      — skip the CPU reference (representative-size perf sweeps).
//!   skew      — non-uniform expert load (one expert 7× the average), to
//!               exercise the work-list under static-tail imbalance + the
//!               multi-m-tile-per-expert path.
//! The kernel builds the (expert,m_tile,n_tile) work-list on the HOST (mirroring
//! the device `moe_build_tile_worklist`) and launches the persistent 96-CTA
//! grid.
//!
//! NOTE (SSOT): the number-format + RNG + upload helpers are duplicated from
//! w8a16_microtest.rs ONLY because a shared example-module would require
//! editing that file while the round-2 Fix-A kernel agent owns it. Unify into
//! a shared `examples/` module once no agent holds w8a16_microtest.rs.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

// Raw CUDA driver event API for kernel-only timing. The wall-clock `Instant`
// metric below includes per-launch host overhead which swamps small per-lever
// deltas at the representative compute-bound size. CUDA events recorded on the
// launch stream measure GPU execution time only, so the optimization signal is
// trustworthy. Signatures mirror atlas-spark-bench's gpu.rs (the SSOT).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

const FP8_BLOCK: usize = 128;
const COSINE_GATE: f64 = 0.9995;

// ───────────────────────── deterministic PRNG (splitmix64) ─────────────────
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
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
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

// ───────────────────────── upload helpers ─────────────────────────
fn upload_bytes(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}
fn u16s_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32s_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn i32s_le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64s_le(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let kernel = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "moe_fp8_grouped_gemm".to_string());
    let num_experts: usize = args.get(2).map_or(4, |s| s.parse().unwrap());
    let tpe: usize = args.get(3).map_or(20, |s| s.parse().unwrap()); // tokens per expert
    let n: usize = args.get(4).map_or(256, |s| s.parse().unwrap());
    let k: usize = args.get(5).map_or(256, |s| s.parse().unwrap());
    let seed: u64 = args.get(6).map_or(0x9E3, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x9E3)
    });
    // arg 7 modes (mutually exclusive flags, any subset of the recognized words):
    //   "perf"/"--perf"   : skip the O(experts*tok*N*K) CPU reference so a
    //                       representative-size perf sweep runs in milliseconds.
    //   "skew"/"--skew"   : non-uniform expert load — one expert gets 7× the
    //                       average token count, the rest share the remainder.
    //                       Exercises the work-list under static-tail
    //                       imbalance (R4) + the multi-m-tile path per expert.
    let mode = args.get(7).map(|s| s.as_str()).unwrap_or("");
    let perf_only = mode == "perf" || mode == "--perf";
    let skew = mode == "skew" || mode == "--skew";

    if k % FP8_BLOCK != 0 {
        bail!("K ({k}) must be a multiple of {FP8_BLOCK}");
    }
    let total = num_experts * tpe; // total_expanded; tokens are 1:1 (identity sort)
    let k_blocks = k / FP8_BLOCK;
    let n_blocks = n.div_ceil(FP8_BLOCK);
    println!(
        "=== moe microtest: kernel='{kernel}' experts={num_experts} tok/expert={tpe} N={n} K={k} \
         seed=0x{seed:X} mode='{mode}' ==="
    );

    // ── inputs ──
    let mut rng = Rng(seed);
    // A[total, K] BF16 (each sorted row is its own token; identity sorted_token_ids)
    let a_bf16: Vec<u16> = (0..total * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    // Per-expert token counts → expert_offsets prefix sum. Uniform by default;
    // `skew` mode loads expert 0 with 7× the average (clamped to `total`) and
    // spreads the remainder across the rest. `total` is unchanged so the A
    // buffer + identity sorted_token_ids stay valid.
    let counts: Vec<usize> = if skew {
        let mut c = vec![0usize; num_experts];
        let heavy = (7 * tpe).min(total);
        c[0] = heavy;
        let mut rem = total - heavy;
        let mut e = 1usize;
        while rem > 0 && num_experts > 1 {
            c[e] += 1;
            rem -= 1;
            e += 1;
            if e >= num_experts {
                e = 1;
            }
        }
        c
    } else {
        vec![tpe; num_experts]
    };
    let expert_offsets: Vec<i32> = {
        let mut o = Vec::with_capacity(num_experts + 1);
        let mut acc = 0i32;
        o.push(0);
        for &c in &counts {
            acc += c as i32;
            o.push(acc);
        }
        o
    };
    let sorted_token_ids: Vec<i32> = (0..total as i32).collect();
    // per-expert weights [N,K] FP8 (exp<=7) + scales [ceil(N/128), K/128] FP32
    let mut weights: Vec<Vec<u8>> = Vec::with_capacity(num_experts);
    let mut scales: Vec<Vec<f32>> = Vec::with_capacity(num_experts);
    for _ in 0..num_experts {
        let w: Vec<u8> = (0..n * k)
            .map(|_| {
                let s = (rng.next_u64() & 1) as u8;
                let e = (rng.next_u64() % 8) as u8;
                let m = (rng.next_u64() % 8) as u8;
                (s << 7) | (e << 3) | m
            })
            .collect();
        let sc: Vec<f32> = (0..n_blocks * k_blocks)
            .map(|_| rng.uniform(0.5, 1.5))
            .collect();
        weights.push(w);
        scales.push(sc);
    }

    // ── GPU setup ──
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let a_ptr = upload_bytes(gpu, &u16s_le(&a_bf16))?;
    // per-expert device buffers + pointer tables
    let mut w_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    let mut s_ptrs: Vec<u64> = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        w_ptrs.push(upload_bytes(gpu, &weights[e])?.0);
        s_ptrs.push(upload_bytes(gpu, &f32s_le(&scales[e]))?.0);
    }
    let w_tbl = upload_bytes(gpu, &u64s_le(&w_ptrs))?;
    let s_tbl = upload_bytes(gpu, &u64s_le(&s_ptrs))?;
    let off_ptr = upload_bytes(gpu, &i32s_le(&expert_offsets))?;
    let sid_ptr = upload_bytes(gpu, &i32s_le(&sorted_token_ids))?;
    // C[total, N] BF16, zero-initialized (matches production memset)
    let c_ptr = upload_bytes(gpu, &vec![0u8; total * n * 2])?;

    // ── v5 work-list (built on the HOST, mirroring moe_build_tile_worklist) ──
    // PM4 geometry SSOT mirror: M_TILE=128, N_TILE=64. The device builder packs
    // (m_tile<<6)|n_tile and skips experts with M_e<=0 (weights are never NULL
    // here). The grid is sized to the work-list tile count (mirrors the
    // production launcher); PM5_PERSIST_CTAS is only a floor for tiny lists.
    // SSOT mirror of kernels/strix-hip/common/moe_fp8_grouped_gemm.cu geometry.
    const PM4_M_TILE: usize = 128;
    const PM4_N_TILE: usize = 64;
    const PM5_PERSIST_CTAS: u32 = 96;
    let n_tiles = n.div_ceil(PM4_N_TILE);
    let mut worklist: Vec<u32> = Vec::new();
    for (e, &cnt) in counts.iter().enumerate() {
        if cnt == 0 {
            continue; // mirror device M_e<=0 skip
        }
        let mt_e = cnt.div_ceil(PM4_M_TILE);
        for mt in 0..mt_e {
            for nt in 0..n_tiles {
                assert!(mt < (1 << 26) && nt < 64, "work-list packing overflow");
                worklist.push(e as u32);
                worklist.push(((mt as u32) << 6) | nt as u32);
            }
        }
    }
    let host_total_tiles = (worklist.len() / 2) as i32;
    let wl_ptr = upload_bytes(
        gpu,
        &worklist
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let tt_ptr = upload_bytes(gpu, &i32s_le(&[host_total_tiles]))?;

    // Launch geometry: a persistent 1D grid of PM5_PERSIST_CTAS CTAs (256
    // threads) striding over the compacted work-list.
    // Size the grid to the work-list tile count (mirrors the production
    // launcher, which sizes from wl_cap_items). The kernel grid-strides by
    // gridDim.x; PM5_PERSIST_CTAS is only a floor for tiny work-lists.
    let grid_ctas = (host_total_tiles as u32).max(PM5_PERSIST_CTAS).min(16384);
    let pm4_threads: u32 = 512;
    let grid_block = || -> ([u32; 3], [u32; 3]) { ([grid_ctas, 1, 1], [pm4_threads, 1, 1]) };
    // Launch the canonical grouped-GEMM into `out_ptr`. The work-list + total
    // pointers are appended after the base arg list.
    let launch_named = |name: &str, out_ptr: DevicePtr, stream: u64, sync: bool| -> Result<()> {
        let handle = gpu.kernel("moe_fp8_grouped_gemm", name)?;
        let (grid, block) = grid_block();
        KernelLaunch::new(gpu, handle)
            .grid(grid)
            .block(block)
            .arg_ptr(a_ptr)
            .arg_ptr(w_tbl)
            .arg_ptr(s_tbl)
            .arg_ptr(out_ptr)
            .arg_ptr(off_ptr)
            .arg_ptr(sid_ptr)
            .arg_u32(num_experts as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .arg_ptr(wl_ptr)
            .arg_ptr(tt_ptr)
            .launch(stream)?;
        if sync {
            gpu.synchronize(stream)
        } else {
            Ok(())
        }
    };
    let do_launch =
        |stream: u64, sync: bool| -> Result<()> { launch_named(&kernel, c_ptr, stream, sync) };
    let launch = |stream: u64| -> Result<()> { do_launch(stream, true) };
    launch(stream)?;

    // ── correctness vs CPU reference (skipped in PERF-ONLY mode) ──
    let cosine = if perf_only {
        f64::NAN
    } else {
        let mut c_raw = vec![0u8; total * n * 2];
        gpu.copy_d2h(c_ptr, &mut c_raw)?;
        let c_gpu: Vec<u16> = c_raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();

        // CPU reference (per expert; two-level FP32 block-scale accumulation).
        let mut c_cpu = vec![0u16; total * n];
        for e in 0..num_experts {
            let m_start = expert_offsets[e] as usize;
            let m_end = expert_offsets[e + 1] as usize;
            for m in m_start..m_end {
                let token = sorted_token_ids[m] as usize;
                for col in 0..n {
                    let mut outer = 0.0f32;
                    for kb in 0..k_blocks {
                        let mut inner = 0.0f32;
                        for kk in 0..FP8_BLOCK {
                            let gk = kb * FP8_BLOCK + kk;
                            let a = bf16_bits_to_f32(a_bf16[token * k + gk]);
                            let b = e4m3_to_f32(weights[e][col * k + gk]);
                            inner += a * b;
                        }
                        outer += inner * scales[e][(col / FP8_BLOCK) * k_blocks + kb];
                    }
                    c_cpu[m * n + col] = f32_to_bf16_bits(outer);
                }
            }
        }

        // compare
        let (mut dot, mut ng, mut nc, mut max_rel, mut sum_rel) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for i in 0..total * n {
            let g = bf16_bits_to_f32(c_gpu[i]) as f64;
            let c = bf16_bits_to_f32(c_cpu[i]) as f64;
            dot += g * c;
            ng += g * g;
            nc += c * c;
            let rel = (g - c).abs() / c.abs().max(1e-3);
            max_rel = max_rel.max(rel);
            sum_rel += rel;
        }
        let cos = dot / (ng.sqrt() * nc.sqrt());
        let mean_rel = sum_rel / (total * n) as f64;
        println!("cosine={cos:.6}  mean_rel={mean_rel:.2e}  max_rel={max_rel:.2e}");
        cos
    };

    // ── rough throughput (wall-clock; includes launch overhead, relative A/B) ──
    let iters = 50;
    for _ in 0..5 {
        launch(stream)?;
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        launch(stream)?;
    }
    let per_iter = t0.elapsed().as_secs_f64() / iters as f64;
    let tflops = (2.0 * total as f64 * n as f64 * k as f64) / per_iter / 1e12;
    println!(
        "perf: {:.3} ms/iter  ~{tflops:.2} TFLOP/s (wall-clock incl. launch)",
        per_iter * 1e3
    );

    // ── kernel-only throughput (CUDA events on the launch stream) ──
    // Brackets `iters` back-to-back launches (no intervening host sync) with two
    // events; cuEventElapsedTime gives total GPU time for the batch, excluding
    // per-launch host overhead — the trustworthy signal for per-lever deltas.
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
        do_launch(stream, false)?;
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
    let kernel_tflops = (2.0 * total as f64 * n as f64 * k as f64) / kernel_s / 1e12;
    println!(
        "kernel-only: {:.4} ms/iter  ~{kernel_tflops:.2} TFLOP/s (CUDA events)",
        kernel_s * 1e3
    );

    if perf_only {
        println!("RESULT: PERF-ONLY (no correctness check)");
        Ok(())
    } else if cosine >= COSINE_GATE && cosine.is_finite() {
        println!("RESULT: PASS (cosine {cosine:.6} >= {COSINE_GATE})");
        Ok(())
    } else {
        eprintln!(
            "RESULT: FAIL (cosine {cosine:.6} < {COSINE_GATE}) — routing/layout/dequant/accum mismatch"
        );
        std::process::exit(1);
    }
}
