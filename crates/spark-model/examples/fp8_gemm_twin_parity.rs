// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the `fp8_gemm_t_row_scaled` twins.
//!
//! `from_weights.rs` selects the DFlash drafter's FP8 GEMM by preference —
//! `_k64` (deep-K, 64 B per weight row per K step) → `_p4` (4-stage cp.async
//! ring) → the 2-stage original — on the claim that all three are
//! byte-identical (each twin's kernel header: same K-slab visit order, same
//! MMA sequence, so every accumulator sees the identical rounding; "verified
//! by sha on the record bench"). That claim is what lets `ATLAS_DFLASH_FP8_
//! GEMM_P2/P4` pin a twin for A/B without a quality leg. This oracle makes
//! the claim standing and re-runnable: every SELECTABLE twin is graded
//! bit-equal against the original, at production drafter shapes, several
//! seeds, and the M shapes the drafter hands the M_TILE=64 kernel (partial
//! tile, exact tile, multi-block + partial).
//!
//! All three kernels share one signature and launch geometry (mirrors
//! `ops::fp8_gemm_n128_row_scaled`):
//!   `f(A_bf16, B_fp8, row_scale_f32, C_bf16, M, N, K)`
//!   Grid (ceil(N/128), ceil(M/64), 1), Block (128, 1, 1).
//!
//! Exit: 0 every present twin is byte-identical everywhere, 1 any twin
//! differs (or the negative control fails to fire), 2 the base kernel or
//! both twins are absent from this target's module set.
//!
//! Run (GPU host):
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 \
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example fp8_gemm_twin_parity

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// The distinct (N, K) pairs of the Qwen3.6-27B drafter projections the FP8
/// drafter path (`DflashQuantization::Fp8Weights`) routes through
/// `fp8_gemm_n128_row_scaled` (h 5120, nq 24, nkv 4, head_dim 256, inter
/// 17408; k_proj and v_proj share a shape). All K are multiples of 64, the
/// only K shape production hands the `_k64` twin.
const SHAPES: [(&str, usize, usize); 5] = [
    ("drafter q_proj   [12288 x  5120]", 12288, 5120),
    ("drafter k/v_proj [ 1024 x  5120]", 1024, 5120),
    ("drafter o_proj   [ 5120 x  6144]", 5120, 6144),
    ("drafter ffn_g/u  [17408 x  5120]", 17408, 5120),
    ("drafter ffn_down [ 5120 x 17408]", 5120, 17408),
];

/// M_TILE=64 coverage: partial single tile, exact tile, multi-block + partial
/// (grid.y = ceil(M/64), so 96 exercises the bounds-checked second block).
const MS: [usize; 3] = [16, 64, 96];
const MAX_M: usize = 96;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    /// A raw E4M3 weight byte. Any code is legal input except the two NaN
    /// encodings (0x7F/0xFF), which the kernels decode to 0 — mapped to 0x00
    /// here so a twin that mishandles NaN-to-zero differently cannot hide a
    /// real mismatch behind an all-NaN column, and outputs stay finite.
    fn e4m3(&mut self) -> u8 {
        self.f();
        let b = (self.0 >> 24) as u8;
        if b & 0x7F == 0x7F { 0x00 } else { b }
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// diff element count, worst |delta|, and worst relative delta over BF16 pairs.
fn worst_delta(a: &[u8], b: &[u8]) -> (usize, f32, f32) {
    let (mut n_diff, mut worst, mut rel) = (0usize, 0f32, 0f32);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if x != y {
            n_diff += 1;
            let fx = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let fy = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            worst = worst.max((fx - fy).abs());
            let denom = fx.abs().max(fy.abs()).max(1e-6);
            rel = rel.max((fx - fy).abs() / denom);
        }
    }
    (n_diff, worst, rel)
}

/// `fp8_gemm_t_row_scaled[*](A, B_fp8, row_scale, C, M, N, K)` — grid
/// (ceil(N/128), ceil(M/64), 1), block (128, 1, 1). Mirrors
/// `ops::fp8_gemm_n128_row_scaled`; identical geometry for all three twins
/// ("Grid/Block unchanged" in each twin's kernel header).
#[allow(clippy::too_many_arguments)]
fn rs_gemm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(scale)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let Ok(base_k) = g.kernel("w4a16", "fp8_gemm_t_row_scaled") else {
        println!("fp8_gemm_t_row_scaled absent from this target set — SKIP");
        std::process::exit(2);
    };
    // Grade every twin the `fp8_gemm_n128_row_scaled` selector can pick on
    // this target; a twin a target predates is reported, not failed.
    let twins: Vec<(&str, KernelHandle)> =
        ["fp8_gemm_t_row_scaled_k64", "fp8_gemm_t_row_scaled_p4"]
            .into_iter()
            .filter_map(|name| match g.kernel("w4a16", name) {
                Ok(kh) => Some((name, kh)),
                Err(_) => {
                    println!("{name} absent from this target set — twin not graded");
                    None
                }
            })
            .collect();
    if twins.is_empty() {
        println!("no row-scaled FP8 twin present — nothing to grade — SKIP");
        std::process::exit(2);
    }

    let mut twins_clean = true;
    let mut control_ok = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k) in SHAPES {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xF8F8);
            let a_bytes: Vec<u8> = (0..MAX_M * k)
                .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
                .collect();
            let b_bytes: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
            let scale_bytes: Vec<u8> = (0..n).flat_map(|_| rng.r(0.5, 2.0).to_le_bytes()).collect();

            let a_d = up(g, &a_bytes)?;
            let b_d = up(g, &b_bytes)?;
            let s_d = up(g, &scale_bytes)?;
            let c_base = g.alloc(MAX_M * n * 2)?;
            let c_twin = g.alloc(MAX_M * n * 2)?;

            for m in MS {
                g.memset(c_base, 0, MAX_M * n * 2)?;
                rs_gemm(
                    g, base_k, a_d, b_d, s_d, c_base, m as u32, n as u32, k as u32,
                )?;
                g.synchronize(0)?;
                let cb = down(g, c_base, m * n * 2)?;

                for &(name, kh) in &twins {
                    g.memset(c_twin, 0, MAX_M * n * 2)?;
                    rs_gemm(g, kh, a_d, b_d, s_d, c_twin, m as u32, n as u32, k as u32)?;
                    g.synchronize(0)?;
                    let ct = down(g, c_twin, m * n * 2)?;
                    let identical = ct == cb;
                    let (n_diff, worst, rel) = worst_delta(&ct, &cb);
                    twins_clean &= identical;
                    let pct = 100.0 * n_diff as f32 / (m * n) as f32;
                    println!(
                        "seed {seed:>5}  {label}  {name:<26} M={m:<3} byte-identical={identical:<5} \
                         diff_elems={n_diff:<7} ({pct:5.2}%) max|delta|={worst:.6} max_rel={rel:.6}"
                    );
                }
            }

            // ── Negative control: a 1-ULP activation perturbation fed to the
            // base kernel MUST break parity against an unperturbed twin, so a
            // "byte-identical" verdict is never an artefact of comparing two
            // blank (or aliased) buffers. Run against a GRADED twin.
            let m = MS[0];
            let mut pert = a_bytes.clone();
            // Perturb an EXPONENT bit, not the low mantissa bit. bf16 is
            // little-endian [lo, hi]; hi carries sign + 7 exponent bits, so
            // flipping hi's bit 0 moves the value by a factor of two. The
            // original control flipped `pert[2 * (k + 7)] ^= 1` — the low
            // mantissa bit — and that delta, accumulated over K=5120 and
            // rounded back to bf16, vanished: the control reported
            // `detected=false` and the harness declared itself VACUOUS
            // (observed on GB10, 2026-09-05). A control must survive the
            // arithmetic it is policing.
            pert[2 * (k + 7) + 1] ^= 1;
            let a_pert = up(g, &pert)?;
            g.memset(c_base, 0, MAX_M * n * 2)?;
            g.memset(c_twin, 0, MAX_M * n * 2)?;
            rs_gemm(
                g, base_k, a_pert, b_d, s_d, c_base, m as u32, n as u32, k as u32,
            )?;
            rs_gemm(
                g, twins[0].1, a_d, b_d, s_d, c_twin, m as u32, n as u32, k as u32,
            )?;
            g.synchronize(0)?;
            let differs = down(g, c_base, m * n * 2)? != down(g, c_twin, m * n * 2)?;
            control_ok &= differs;
            println!("seed {seed:>5}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            g.free(a_pert).ok();

            for p in [a_d, b_d, s_d, c_base, c_twin] {
                g.free(p).ok();
            }
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not mismatch; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if twins_clean {
        println!(
            "PASS — every present fp8_gemm_t_row_scaled twin is byte-identical to the \
             original at every drafter shape, M shape, and seed."
        );
        Ok(())
    } else {
        println!(
            "FAIL — a fp8_gemm_t_row_scaled twin is NOT byte-identical to the original. \
             The selector in `dflash_head/from_weights.rs` picks twins on exactly that \
             claim (and ATLAS_DFLASH_FP8_GEMM_P2/P4 A/Bs rely on it); do NOT loosen this \
             comparison — fix the twin or demote it in the preference order."
        );
        std::process::exit(1);
    }
}
