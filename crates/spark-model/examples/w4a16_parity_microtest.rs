// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-kernel parity oracle for the W4A16 (NVFP4) GEMM family at DFlash
//! verify shapes (M=17) — convicts or clears the small-M dispatch kernels.
//!
//! The 2026-07-04 small-M routing (`dense_ffn::w4a16_prefill_gemm`,
//! `wide_verify_gemm`) put `w4a16_gemm_t` and — for the first time in
//! production — `w4a16_gemm_t_k64` on the verify hot path. This oracle runs
//! all transposed kernels against the base `w4a16_gemm` (battle-tested,
//! months in production) on IDENTICAL random NVFP4 data:
//!
//!   C_base = A · dequant(B)        (base, non-transposed [N, K/2] layout)
//!   C_t    = A · dequant(B_t)      (w4a16_gemm_t,      transposed [K/2, N])
//!   C_k64  = A · dequant(B_t)      (w4a16_gemm_t_k64)
//!   C_m128 = A · dequant(B_t)      (w4a16_gemm_t_m128)
//!
//! B_t is built by the SAME byte-transpose as `QuantizedWeight::
//! transpose_for_gemm`, so a mismatch here is a KERNEL bug, not a layout bug.
//! GATE per kernel: cosine vs base ≥ 0.999 AND max |Δ| within BF16
//! reorder noise. A real defect (wrong scale index, dropped K-step, bad
//! fragment) collapses cosine to ~0.
//!
//!   cargo run -p spark-model --release --example w4a16_parity_microtest \
//!       --features cuda,gpu-examples
//!
//! Exit 0 = all kernels agree with base; 1 = at least one mismatch (named).

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const M: usize = 17;
const GROUP: usize = 16;
const PASS_COS: f64 = 0.999;

/// (label, N, K) — the exact Qwen3.6-27B verify shapes the routing touches.
const SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate/up N=17408 K=5120 ", 17408, 5120),
    ("ffn_down    N=5120  K=17408", 5120, 17408),
    ("attn_q      N=12288 K=5120 ", 12288, 5120),
    ("attn_o      N=5120  K=6144 ", 5120, 6144),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    grid: [u32; 3],
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid(grid)
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(0.01) // scale2: keeps outputs in a sane BF16 range
        .arg_ptr(c)
        .arg_u32(M as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let base_k = g.kernel("w4a16", "w4a16_gemm")?;
    // (name, handle, grid geometry as fn of (n, k... unused), uses transposed B)
    let t_kernels: Vec<(&str, KernelHandle)> = [
        ("w4a16_gemm_t     ", "w4a16_gemm_t"),
        ("w4a16_gemm_t_k64 ", "w4a16_gemm_t_k64"),
        ("w4a16_gemm_t_m128", "w4a16_gemm_t_m128"),
    ]
    .into_iter()
    .filter_map(|(name, f)| g.kernel("w4a16", f).ok().map(|h| (name, h)))
    .collect();

    let mut all_ok = true;
    for &(label, n, k) in SHAPES {
        let mut r = Rng(0x517A_C0DE ^ (n as u64) << 20 ^ k as u64);
        let half_k = k / 2;
        let groups = k / GROUP;

        // A: [M, K] BF16, small values.
        let a_host: Vec<u8> = (0..M * k)
            .flat_map(|_| {
                bf16::from_f32((r.unit() - 0.5) * 0.5)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        // B packed E2M1: [N, K/2], any nibble pattern is a valid E2M1 pair.
        let b_host: Vec<u8> = (0..n * half_k).map(|_| r.byte()).collect();
        // B scales FP8 E4M3: [N, K/16]. Constrain exponent bits to a benign
        // range (0x28..0x48 ≈ 2^-6..2^1) — no NaN/Inf, no overflow.
        let bs_host: Vec<u8> = (0..n * groups).map(|_| 0x28 + (r.byte() & 0x1F)).collect();

        // Transposed copies — EXACT byte loops from transpose_for_gemm.
        let mut bt_host = vec![0u8; n * half_k];
        for i in 0..n {
            for j in 0..half_k {
                bt_host[j * n + i] = b_host[i * half_k + j];
            }
        }
        let mut bst_host = vec![0u8; n * groups];
        for i in 0..n {
            for j in 0..groups {
                bst_host[j * n + i] = bs_host[i * groups + j];
            }
        }

        let a = up(g, &a_host)?;
        let b = up(g, &b_host)?;
        let bs = up(g, &bs_host)?;
        let bt = up(g, &bt_host)?;
        let bst = up(g, &bst_host)?;
        let c_base = g.alloc(M * n * 2)?;
        let c_test = g.alloc(M * n * 2)?;

        // Base reference: grid (N/64, M/64).
        launch(
            g,
            base_k,
            [div_ceil(n as u32, 64), div_ceil(M as u32, 64), 1],
            a,
            b,
            bs,
            c_base,
            n,
            k,
        )?;
        g.synchronize(0)?;
        let base_out = dn_bf16(g, c_base, M * n)?;

        for &(name, kh) in &t_kernels {
            // _t and _t_k64: grid (N/128, M/64). _t_m128: grid (N/128, M/128).
            let grid = if name.trim_end() == "w4a16_gemm_t_m128" {
                [div_ceil(n as u32, 128), div_ceil(M as u32, 128), 1]
            } else {
                [div_ceil(n as u32, 128), div_ceil(M as u32, 64), 1]
            };
            g.memset(c_test, 0, M * n * 2)?;
            launch(g, kh, grid, a, bt, bst, c_test, n, k)?;
            g.synchronize(0)?;
            let out = dn_bf16(g, c_test, M * n)?;

            let c = cos(&out, &base_out);
            let max_abs_base = base_out.iter().fold(0f32, |m, v| m.max(v.abs()));
            let max_d = out
                .iter()
                .zip(&base_out)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let ok = c >= PASS_COS;
            all_ok &= ok;
            eprintln!(
                "{label}  {name}  cos={c:.7}  max|Δ|={max_d:.5} (base max|C|={max_abs_base:.3})  {}",
                if ok {
                    "PASS"
                } else {
                    "FAIL ← kernel disagrees with base"
                }
            );
        }
        eprintln!();
        for p in [a, b, bs, bt, bst, c_base, c_test] {
            let _ = g.free(p);
        }
    }

    eprintln!(
        "W4A16 parity GATE (all transposed kernels vs base, cos≥{PASS_COS}): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
