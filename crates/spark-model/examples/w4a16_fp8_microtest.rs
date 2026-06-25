// SPDX-License-Identifier: AGPL-3.0-only

//! Cosine-sanity oracle for `w4a16_gemm_t_m128` (FP8 E4M3 k32 prefill) vs the
//! base `w4a16_gemm` (BF16). UNLIKE the BF16-TC kernel (which is ~bit-identical
//! to base), `w4a16_gemm_t_m128` on NVIDIA crushes the dequanted weights AND the
//! BF16 activations to FP8 E4M3 and runs `mma.sync.m16n8k32.f32.e4m3.e4m3.f32`
//! (k32 → ~2× the BF16-TC k16 MMA throughput). That is LOSSY, so this is a
//! "not garbage" sanity gate, not a losslessness proof: cosine must be high
//! (~0.99, gate >= 0.98) and outputs must not be zero/NaN.
//!
//! Both kernels consume the SAME logical NVFP4 weight (packed E2M1 nibbles +
//! per-group E4M3 block scales + per-tensor `scale2`). They differ in the
//! transposed weight layout AND in the MMA precision (base = BF16, t_m128 = FP8).
//!
//! Layouts (mirrors weight_map/quantized.rs `transpose_for_gemm`, the SSOT):
//!   - base `w4a16_gemm`:    B_packed [N, K/2],   B_scale [N, K/16]
//!   - `w4a16_gemm_t_m128`:  B_packed [K/2, N],   B_scale [K/16, N]
//!
//! Usage:
//!   cargo run --release -p spark-model --example w4a16_fp8_microtest -- [seed]
//! Runs the prefill + edge shape sweep. Exit 0 = all PASS, 1 = any FAIL.

use anyhow::{Result, bail};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

/// NVFP4 group size along K (matches GROUP_SIZE in w4a16_gemm.cu).
const GROUP_SIZE: usize = 16;

/// Cosine gate. FP8 E4M3 (4-bit mantissa-ish) on BOTH operands loses precision
/// vs the BF16 base, so a perfect 1.0 is NOT expected. A sane, non-garbage FP8
/// GEMM lands ~0.99; the perf-gate bar is >= 0.98 (anything below signals
/// corruption / a layout bug rather than mere FP8 rounding).
const COSINE_GATE: f64 = 0.98;

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

// ───────────────────────── bf16 helpers ─────────────────────────
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
/// f32 → BF16 bits, round-to-nearest-even — matches CUDA `__float2bfloat16`.
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}
fn u16s_to_le(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// E2M1 LUT — independent re-derivation of `E2M1_LUT` in w4a16_gemm.cu (an
/// oracle must not import the artifact it validates).
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// OCP E4M3 (e4m3fn) encode for a benign group-scale magnitude. Pick from a
/// small representable set so the byte round-trips exactly through both kernels'
/// `(float)e4m3` cast.
/// Bytes: exp in [5..9] (2^-2 .. 2^2), mantissa 0 → exact values {0.25,0.5,1,2,4}.
fn e4m3_scale_byte(sel: u32) -> u8 {
    let e = 5 + (sel % 5);
    ((e as u8) << 3) & 0x7F // sign=0, mant=0
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

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

/// Generated NVFP4 weight in BOTH layouts, plus scale2.
struct Nvfp4Weight {
    packed_nt: Vec<u8>, // [N, K/2]
    scale_nt: Vec<u8>,  // [N, K/16]
    packed_t: Vec<u8>,  // [K/2, N]
    scale_t: Vec<u8>,   // [K/16, N]
    scale2: f32,
}

/// Build a random NVFP4 weight [N, K] directly in packed form, then derive the
/// transposed layout with the EXACT loop from `transpose_for_gemm`.
fn gen_weight(rng: &mut Rng, n: usize, k: usize) -> Nvfp4Weight {
    assert!(k % GROUP_SIZE == 0, "K must be a multiple of {GROUP_SIZE}");
    let half_k = k / 2;
    let num_groups = k / GROUP_SIZE;
    let mut packed_nt = vec![0u8; n * half_k];
    let mut scale_nt = vec![0u8; n * num_groups];

    for i in 0..n {
        for g in 0..num_groups {
            scale_nt[i * num_groups + g] = e4m3_scale_byte(rng.next_u64() as u32);
        }
        for j in 0..half_k {
            // low nibble = even k (2j), high nibble = odd k (2j+1)
            let lo = (rng.next_u64() % 16) as u8;
            let hi = (rng.next_u64() % 16) as u8;
            packed_nt[i * half_k + j] = (hi << 4) | lo;
        }
    }

    // Transpose: B_packed [N,K/2]→[K/2,N], B_scale [N,K/16]→[K/16,N].
    let mut packed_t = vec![0u8; n * half_k];
    for i in 0..n {
        for j in 0..half_k {
            packed_t[j * n + i] = packed_nt[i * half_k + j];
        }
    }
    let mut scale_t = vec![0u8; n * num_groups];
    for i in 0..n {
        for g in 0..num_groups {
            scale_t[g * n + i] = scale_nt[i * num_groups + g];
        }
    }

    Nvfp4Weight {
        packed_nt,
        scale_nt,
        packed_t,
        scale_t,
        // Per-tensor scale. Keep modest so dequanted magnitudes stay well within
        // FP8 E4M3 range (max ~448) and don't saturate.
        scale2: 0.5,
    }
}

struct Stats {
    cosine: f64,
    max_abs: f64,
    max_rel: f64,
    frac_bit_identical: f64,
}

fn compare(a: &[u16], b: &[u16]) -> Stats {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    let (mut max_abs, mut max_rel) = (0f64, 0f64);
    let mut bit_eq = 0usize;
    for i in 0..a.len() {
        if a[i] == b[i] {
            bit_eq += 1;
        }
        let x = bf16_bits_to_f32(a[i]) as f64;
        let y = bf16_bits_to_f32(b[i]) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
        let d = (x - y).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = x.abs().max(y.abs());
        if denom > 1e-6 {
            let r = d / denom;
            if r > max_rel {
                max_rel = r;
            }
        }
    }
    let cosine = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        1.0
    };
    Stats {
        cosine,
        max_abs,
        max_rel,
        frac_bit_identical: bit_eq as f64 / a.len() as f64,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    base_h: spark_runtime::gpu::KernelHandle,
    fp8_h: spark_runtime::gpu::KernelHandle,
    seed: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Stats> {
    let mut rng = Rng(seed ^ ((m as u64) << 32) ^ ((n as u64) << 16) ^ (k as u64));

    // A [M, K] BF16, realistic post-norm magnitudes.
    let a_bf16: Vec<u16> = (0..m * k)
        .map(|_| f32_to_bf16_bits(rng.uniform(-1.0, 1.0)))
        .collect();
    let a_ptr = upload(gpu, &u16s_to_le(&a_bf16))?;

    let w = gen_weight(&mut rng, n, k);

    let packed_nt = upload(gpu, &w.packed_nt)?;
    let scale_nt = upload(gpu, &w.scale_nt)?;
    // Negative control (ATLAS_MICROTEST_NEGCTL=1): feed the fp8 kernel the WRONG
    // (non-transposed) packed layout. A discriminating test MUST then FAIL even
    // the loose 0.98 gate — proves the high cosine is real layout agreement.
    let neg_ctl = std::env::var_os("ATLAS_MICROTEST_NEGCTL").is_some();
    let packed_t = if neg_ctl {
        upload(gpu, &w.packed_nt)?
    } else {
        upload(gpu, &w.packed_t)?
    };
    let scale_t = if neg_ctl {
        upload(gpu, &w.scale_nt)?
    } else {
        upload(gpu, &w.scale_t)?
    };

    let c_base = gpu.alloc(m * n * 2)?;
    let c_fp8 = gpu.alloc(m * n * 2)?;

    // ── base w4a16_gemm: grid (ceil(N/64), ceil(M/64), 1), block (128,1,1) ──
    KernelLaunch::new(gpu, base_h)
        .grid([n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed_nt)
        .arg_ptr(scale_nt)
        .arg_f32(w.scale2)
        .arg_ptr(c_base)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    // ── w4a16_gemm_t_m128 (FP8): grid (ceil(N/128), ceil(M/128), 1), block (128,1,1) ──
    KernelLaunch::new(gpu, fp8_h)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(a_ptr)
        .arg_ptr(packed_t)
        .arg_ptr(scale_t)
        .arg_f32(w.scale2)
        .arg_ptr(c_fp8)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;

    gpu.synchronize(stream)?;

    let mut raw_base = vec![0u8; m * n * 2];
    let mut raw_fp8 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c_base, &mut raw_base)?;
    gpu.copy_d2h(c_fp8, &mut raw_fp8)?;
    let out_base: Vec<u16> = raw_base
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let out_fp8: Vec<u16> = raw_fp8
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Sanity: outputs must not be all-zero (would mask a dead kernel) and the
    // FP8 output must not be riddled with NaN (FP8 saturation/garbage).
    let base_nz = out_base.iter().filter(|&&x| x != 0).count();
    let fp8_nz = out_fp8.iter().filter(|&&x| x != 0).count();
    let fp8_nan = out_fp8
        .iter()
        .filter(|&&x| bf16_bits_to_f32(x).is_nan())
        .count();
    if base_nz == 0 || fp8_nz == 0 {
        bail!("dead output: base_nonzero={base_nz} fp8_nonzero={fp8_nz} (M={m} N={n} K={k})");
    }
    if fp8_nan > 0 {
        bail!("fp8 garbage: {fp8_nan} NaN outputs (M={m} N={n} K={k})");
    }

    for p in [a_ptr, packed_nt, scale_nt, packed_t, scale_t, c_base, c_fp8] {
        let _ = gpu.free(p);
    }

    Ok(compare(&out_base, &out_fp8))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });

    debug_assert_eq!(E2M1_LUT[7], 6.0);
    debug_assert!((e4m3_to_f32(e4m3_scale_byte(2)) - 1.0).abs() < 1e-6); // sel 2 → e=7 → 1.0

    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let stream = gpu.create_stream()?;

    let base_h = gpu.kernel("w4a16", "w4a16_gemm")?;
    let fp8_h = gpu.kernel("w4a16", "w4a16_gemm_t_m128")?;

    // (label, M, N, K). The two prefill shapes the task names (gate/up: N=17408
    // K=5120; down: N=5120 K=17408) at M=4096, plus M-tile + K-tail edges.
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("gate/up   ", 4096, 17408, 5120),
        ("down      ", 4096, 5120, 17408),
        ("M=33  edge", 33, 5120, 5120),
        ("M=128 edge", 128, 5120, 5120),
        ("M=1015edge", 1015, 5120, 5120),
        // K not a multiple of 32 (K-tail predicate): 5104 = 319*16, %32 != 0.
        ("K-tail    ", 256, 4096, 5104),
    ];

    println!(
        "=== w4a16_fp8 cosine-sanity microtest (base w4a16_gemm vs w4a16_gemm_t_m128 FP8 k32) seed=0x{seed:X} ===\n"
    );
    println!(
        "{:<12} {:>6} {:>6} {:>6} | {:>10} {:>10} {:>10} {:>10}  result",
        "shape", "M", "N", "K", "cosine", "max_abs", "max_rel", "bit_id%"
    );
    println!("{}", "-".repeat(92));

    let mut all_pass = true;
    for &(label, m, n, k) in shapes {
        let s = run_shape(gpu, stream, base_h, fp8_h, seed, m, n, k)?;
        let pass = s.cosine >= COSINE_GATE && s.cosine.is_finite();
        all_pass &= pass;
        println!(
            "{label:<12} {m:>6} {n:>6} {k:>6} | {:>10.6} {:>10.3e} {:>10.3e} {:>9.3}%  {}",
            s.cosine,
            s.max_abs,
            s.max_rel,
            s.frac_bit_identical * 100.0,
            if pass { "PASS" } else { "FAIL" },
        );
    }

    println!("{}", "-".repeat(92));
    if all_pass {
        println!(
            "RESULT: PASS — FP8 k32 prefill is cosine-sane vs base (cosine >= {COSINE_GATE} on all shapes, no garbage/NaN)"
        );
        Ok(())
    } else {
        println!(
            "RESULT: FAIL — at least one shape below cosine {COSINE_GATE} (corruption / layout bug)"
        );
        std::process::exit(1);
    }
}
