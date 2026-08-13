// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the FIFTH NVFP4 batched family: the fused
//! deinterleaving GEMVs `w4a16_gemv_qg_batch2/3` and `w4a16_gemv_dual_batch2/3`.
//!
//! These are NOT instantiations of `w4a16_gemv_batchm_impl` and their reference
//! is NOT `w4a16_gemv`. They inherit `w4a16_gemv_qg`'s own shape: `k8 = lane`
//! chunking (8 K-values per chunk, not 16) with the FP8 group scale
//! pre-multiplied into each unpacked weight. So the only sound reference is
//! `w4a16_gemv_qg` itself, run M times — and against it these four diverged in
//! exactly one way, the fused `acc += x*w_lo + y*w_hi` that PR #474 fixed in
//! `w8a16_gemv_batch4` and `moe_w4a16_grouped_gemm`.
//!
//! `dual_batch2/3` write a PLAIN `C[n]` rather than a deinterleaved index, so
//! they are gated against `w4a16_gemv_qg` driven with `num_heads = 1,
//! head_dim = N`, which makes its deinterleave the identity map (`idx = n <
//! head_dim` ⇒ `out_idx = n`) and leaves the arithmetic untouched.
//!
//! 3 seeds x 3 shapes x 4 kernels = 36 legs, plus 9 negative controls.
//!
//! Exit: 0 all legs byte-identical, 1 any leg differs,
//! 2 kernels absent from this target's module set.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=nemotron-3-nano-30b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example w4a16_qg_batch_bitparity_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const GROUP_SIZE: usize = 16;
const MAX_M: usize = 3;
const SCALE2: f32 = 0.0123_f32;

/// `(label, N, K, num_heads, head_dim)` with `N == num_heads * head_dim * 2`,
/// the Q+Gate widths the SSM path actually dispatches.
const SHAPES: [(&str, usize, usize, u32, u32); 3] = [
    ("qg [4096 x 2688] h16 d128", 4096, 2688, 16, 128),
    ("qg [8192 x 4096] h32 d128", 8192, 4096, 32, 128),
    ("qg [2048 x 4096] h8  d128", 2048, 4096, 8, 128),
];

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

fn worst_delta(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n_diff = 0usize;
    let mut worst = 0f32;
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if x != y {
            n_diff += 1;
            let fx = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let fy = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            worst = worst.max((fx - fy).abs());
        }
    }
    (n_diff, worst)
}

struct Args {
    a: DevicePtr,
    w: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
}

/// `w4a16_gemv_qg(A, B_packed, B_scale, scale2, C, N, K, num_heads, head_dim)`
/// — also the shape of `qg_batch2/3`, whose only extra state is the row count
/// baked into the kernel name.
fn launch_qg(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    p: &Args,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(p.n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(p.a)
        .arg_ptr(p.w)
        .arg_ptr(p.ws)
        .arg_f32(SCALE2)
        .arg_ptr(p.c)
        .arg_u32(p.n)
        .arg_u32(p.k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(0)
}

/// `w4a16_gemv_dual_batchM(A, B0, B0s, s2, C0, B1, B1s, s2, C1, N, K)`, grid z=2.
/// Both projections are pointed at the SAME weights and the same output, so the
/// z=1 blocks recompute z=0's values bit-for-bit and the comparison stays a
/// clean test of the z=0 arithmetic.
fn launch_dual(g: &dyn GpuBackend, kh: KernelHandle, p: &Args) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(p.n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(p.a)
        .arg_ptr(p.w)
        .arg_ptr(p.ws)
        .arg_f32(SCALE2)
        .arg_ptr(p.c)
        .arg_ptr(p.w)
        .arg_ptr(p.ws)
        .arg_f32(SCALE2)
        .arg_ptr(p.c)
        .arg_u32(p.n)
        .arg_u32(p.k)
        .launch(0)
}

/// M x single-row `w4a16_gemv_qg` — the reference this family must match.
fn reference(
    g: &dyn GpuBackend,
    qg: KernelHandle,
    p: &Args,
    m: usize,
    num_heads: u32,
    head_dim: u32,
) -> Result<()> {
    for t in 0..m {
        let row = Args {
            a: p.a.offset(t * p.k as usize * 2),
            c: p.c.offset(t * p.n as usize * 2),
            ..*p
        };
        launch_qg(g, qg, &row, num_heads, head_dim)?;
    }
    Ok(())
}

struct Inputs {
    a: Vec<u8>,
    w: Vec<u8>,
    ws: Vec<u8>,
}

/// Same operand recipe as `w4a16_batch_bitparity_microtest`: block-scale bytes
/// held in finite positive E4M3 (0x30..=0x47) so no leg is blanked by a zero or
/// NaN scale, which would hide a reordering.
fn gen_inputs(seed: u64, n: usize, k: usize) -> Inputs {
    let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB4B4);
    let a = (0..MAX_M * k)
        .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
        .collect();
    let w = (0..n * k / 2).map(|_| rng.r(0.0, 256.0) as u8).collect();
    let ws = (0..n * k / GROUP_SIZE)
        .map(|_| 0x30u8 + (rng.r(0.0, 24.0) as u8))
        .collect();
    Inputs { a, w, ws }
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let kernels: Option<Vec<KernelHandle>> = [
        "w4a16_gemv_qg",
        "w4a16_gemv_qg_batch2",
        "w4a16_gemv_qg_batch3",
        "w4a16_gemv_dual_batch2",
        "w4a16_gemv_dual_batch3",
    ]
    .iter()
    .map(|k| g.kernel("w4a16_gemv", k).ok())
    .collect();
    let Some(kh) = kernels else {
        println!("w4a16 qg/dual GEMV kernels absent from this target set — SKIP");
        std::process::exit(2);
    };
    let (qg, qg2, qg3, dual2, dual3) = (kh[0], kh[1], kh[2], kh[3], kh[4]);

    let mut clean = true;
    let mut control_ok = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k, num_heads, head_dim) in SHAPES {
            let inp = gen_inputs(seed, n, k);
            let a_d = up(g, &inp.a)?;
            let w_d = up(g, &inp.w)?;
            let ws_d = up(g, &inp.ws)?;
            let c_batch = g.alloc(MAX_M * n * 2)?;
            let c_ref = g.alloc(MAX_M * n * 2)?;
            let mk = |c: DevicePtr| Args {
                a: a_d,
                w: w_d,
                ws: ws_d,
                c,
                n: n as u32,
                k: k as u32,
            };

            // `dual` writes a plain C[n]; num_heads=1/head_dim=N makes the qg
            // reference's deinterleave the identity map.
            let cases: [(&str, KernelHandle, usize, bool, u32, u32); 4] = [
                ("qg_batch2", qg2, 2, false, num_heads, head_dim),
                ("qg_batch3", qg3, 3, false, num_heads, head_dim),
                ("dual_batch2", dual2, 2, true, 1, n as u32),
                ("dual_batch3", dual3, 3, true, 1, n as u32),
            ];
            for (tier, tier_kh, m, is_dual, nh, hd) in cases {
                g.memset(c_batch, 0, MAX_M * n * 2)?;
                g.memset(c_ref, 0, MAX_M * n * 2)?;
                if is_dual {
                    launch_dual(g, tier_kh, &mk(c_batch))?;
                } else {
                    launch_qg(g, tier_kh, &mk(c_batch), nh, hd)?;
                }
                reference(g, qg, &mk(c_ref), m, nh, hd)?;
                g.synchronize(0)?;
                let cb = down(g, c_batch, m * n * 2)?;
                let cr = down(g, c_ref, m * n * 2)?;
                let identical = cb == cr;
                let (n_diff, worst) = worst_delta(&cb, &cr);
                clean &= identical;
                println!(
                    "seed {seed:>5}  {label}  {tier:<12} M={m:<3} \
                     byte-identical={identical:<5} diff_elems={n_diff:<7} \
                     max|delta|={worst:.6}"
                );
            }

            // ── Negative control: a 1-ULP activation flip on row 1 MUST show up.
            let mut pert = inp.a.clone();
            pert[2 * (k + 7)] ^= 1;
            let a_pert = up(g, &pert)?;
            g.memset(c_batch, 0, MAX_M * n * 2)?;
            g.memset(c_ref, 0, MAX_M * n * 2)?;
            launch_qg(g, qg2, &mk(c_batch), num_heads, head_dim)?;
            let mut refp = mk(c_ref);
            refp.a = a_pert;
            reference(g, qg, &refp, 2, num_heads, head_dim)?;
            g.synchronize(0)?;
            let differs = down(g, c_batch, 2 * n * 2)? != down(g, c_ref, 2 * n * 2)?;
            control_ok &= differs;
            println!("seed {seed:>5}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            g.free(a_pert).ok();

            for p in [a_d, w_d, ws_d, c_batch, c_ref] {
                g.free(p).ok();
            }
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not mismatch; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if clean {
        println!(
            "PASS — w4a16_gemv_qg_batch2/3 and w4a16_gemv_dual_batch2/3 are byte-identical \
             to M x w4a16_gemv_qg at every Q/Gate shape they serve."
        );
        Ok(())
    } else {
        println!(
            "FAIL — a fused deinterleaving batched tier is NOT byte-identical, so NVFP4 \
             SSM Q/Gate output depends on how many sequences share the step."
        );
        std::process::exit(1);
    }
}
