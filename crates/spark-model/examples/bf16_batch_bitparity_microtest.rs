// SPDX-License-Identifier: AGPL-3.0-only
//! BYTE-parity gate for the native-BF16 batched projections.
//!
//! Two legs against the same reference (`m x dense_gemv_bf16`, the kernel every
//! SINGLE-sequence path runs), at production projection shapes, several seeds,
//! and every M a batched arm can be handed at rungs 2..16:
//!
//! * **`dense_gemv_bf16_batchm`** — the batched-GEMV arm. ONE pass over the
//!   weight matrix, M independent accumulators, same per-row K-iteration order
//!   and reduction tree as the M=1 kernel (`--fmad=false` on the dir is what
//!   makes that an identity and not an approximation). MUST be byte-identical
//!   for M in 2..=8; **this leg alone decides the exit code**. Above M=8 the
//!   kernel's compile-time `MAX_M` clamps silently, so it is not exercised
//!   there — the Rust wrapper refuses those M.
//! * **`dense_gemm_bf16_pipelined`** — the tile-GEMM arm. REPORTED, never
//!   graded: it is a different ALGORITHM (m16n8k16 tensor-core MMA, one FP32
//!   accumulator marched sequentially over K in 32-wide steps), so it is
//!   expected to differ and no reassociation recovers parity. Its numbers are
//!   kept as standing evidence of that known gap. Do not "fix" it by loosening
//!   the comparison, and do not fold it into the verdict.
//!
//! The `batchm` leg is why the drafter's M=2..8 propose
//! (`layers::mtp_head::row_dispatch`) can move OFF the pipelined GEMM and be
//! bit-exact to the per-sequence propose rather than merely close to it. It is
//! also the "bit-exact batched BF16 GEMV" an earlier revision of this file
//! called for as unbuilt — it was already in `kernels/gb10/common/`, just not
//! wired into every caller.
//!
//! Exit: 0 the batchm leg is byte-identical everywhere it is defined, 1 any
//! batchm leg differs (or the negative control fails to fire), 2 kernels absent
//! from this target's module set.
//!
//! Run:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=nemotron-3-nano-30b-a3b \
//!   ATLAS_TARGET_QUANT=nvfp4 cargo run -p spark-model --release \
//!     --features cuda,gpu-examples --example bf16_batch_bitparity_microtest

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MAX_M: usize = 16;

/// The two `nano` rows are the EXACT shapes Nano-30B-A3B / Lightning-30B-A3B
/// dispatch (hidden 2688, d_inner 4096, in_proj_size 10304). The two `super`
/// rows cover the hidden-4096 Super/Puzzle backbone at the natural doubling
/// (d_inner 8192); they are shape COVERAGE for a larger N and a deeper K, not
/// a claim about that checkpoint's exact `in_proj_size`.
const SHAPES: [(&str, usize, usize); 10] = [
    ("nano  in_proj    [10304 x  2688]", 10304, 2688),
    ("nano  out_proj   [ 2688 x  4096]", 2688, 4096),
    ("super in_proj    [18560 x  4096]", 18560, 4096),
    ("super out_proj   [ 4096 x  8192]", 4096, 8192),
    // The six DISTINCT (N, K) pairs of one Qwen3.6-27B drafter draft position
    // (h 5120, nq 24, nkv 4, head_dim 256, inter 17408). k_proj and v_proj
    // share a shape, so eight projections are six rows. These are the shapes
    // `mtp_head::row_dispatch` moves onto the batchm arm at M in 2..=8.
    ("27B mtp fc       [ 5120 x 10240]", 5120, 10240),
    ("27B mtp q_proj   [12288 x  5120]", 12288, 5120),
    ("27B mtp k/v_proj [ 1024 x  5120]", 1024, 5120),
    ("27B mtp o_proj   [ 5120 x  6144]", 5120, 6144),
    ("27B mtp ffn_g/u  [17408 x  5120]", 17408, 5120),
    ("27B mtp ffn_down [ 5120 x 17408]", 5120, 17408),
];

/// Compile-time `MAX_M` of `dense_gemv_bf16_batchm`. The kernel CLAMPS above
/// it rather than erroring, so the batchm leg must not be run past it — mirror
/// of `ops::DENSE_GEMV_BATCHM_MAX_M`.
const BATCHM_MAX_M: usize = 8;

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

/// diff element count, worst |delta|, and worst relative delta.
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

/// `dense_gemv_bf16(A, B, C, N, K)` — grid (ceil(N/4),1,1), block 256.
fn gemv(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(c)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

/// `dense_gemm_bf16_pipelined(A, B, C, M, N, K)` — grid
/// (ceil(N/128), ceil(M/128), 1), block 256. Mirrors
/// `ops::dense_gemm_bf16_pipelined`.
#[allow(clippy::too_many_arguments)]
fn gemm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

/// `dense_gemv_bf16_batchm(A, B, C, M, N, K, out_stride)` — grid
/// (ceil(N/4),1,1), block 256. Mirrors `ops::dense_gemv_batchm`; `out_stride`
/// is N here, matching the contiguous `[m, n]` output the drafter uses.
#[allow(clippy::too_many_arguments)]
fn batchm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(n)
        .launch(0)
}

fn reference(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    w: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    for t in 0..m {
        gemv(
            g,
            kh,
            a.offset(t * k * 2),
            w,
            c.offset(t * n * 2),
            n as u32,
            k as u32,
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let (gemm_k, gemv_k, batchm_k) = match (
        g.kernel("gemm", "dense_gemm_bf16_pipelined"),
        g.kernel("gemv", "dense_gemv_bf16"),
        g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
    ) {
        (Ok(a), Ok(b), Ok(c)) => (a, b, c),
        _ => {
            println!("dense BF16 GEMM/GEMV/batchm kernels absent from this target set — SKIP");
            std::process::exit(2);
        }
    };

    // GRADED: the batchm arm, which is what the drafter's M=2..8 propose and
    // the multi-seq decode projections dispatch.
    let mut batchm_clean = true;
    // REPORTED ONLY: the tile-GEMM arm's known non-parity.
    let mut gemm_clean = true;
    let mut control_ok = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k) in SHAPES {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB4B4);
            let a_bytes: Vec<u8> = (0..MAX_M * k)
                .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
                .collect();
            let w_bytes: Vec<u8> = (0..n * k)
                .flat_map(|_| bf16::from_f32(rng.r(-0.08, 0.08)).to_bits().to_le_bytes())
                .collect();

            let a_d = up(g, &a_bytes)?;
            let w_d = up(g, &w_bytes)?;
            let c_batch = g.alloc(MAX_M * n * 2)?;
            let c_ref = g.alloc(MAX_M * n * 2)?;

            for m in [2usize, 3, 4, 6, 8, 12, 16] {
                g.memset(c_ref, 0, MAX_M * n * 2)?;
                reference(g, gemv_k, a_d, w_d, c_ref, m, n, k)?;
                g.synchronize(0)?;
                let cr = down(g, c_ref, m * n * 2)?;

                // ── GRADED leg: batched GEMV, only where the kernel defines it.
                if m <= BATCHM_MAX_M {
                    g.memset(c_batch, 0, MAX_M * n * 2)?;
                    batchm(g, batchm_k, a_d, w_d, c_batch, m as u32, n as u32, k as u32)?;
                    g.synchronize(0)?;
                    let cb = down(g, c_batch, m * n * 2)?;
                    let identical = cb == cr;
                    let (n_diff, worst, rel) = worst_delta(&cb, &cr);
                    batchm_clean &= identical;
                    let pct = 100.0 * n_diff as f32 / (m * n) as f32;
                    println!(
                        "seed {seed:>5}  {label}  batchm    M={m:<3} byte-identical={identical:<5} \
                         diff_elems={n_diff:<7} ({pct:5.2}%) max|delta|={worst:.6} max_rel={rel:.6}"
                    );
                } else {
                    println!(
                        "seed {seed:>5}  {label}  batchm    M={m:<3} \
                         n/a (kernel MAX_M={BATCHM_MAX_M}; the wrapper refuses this M)"
                    );
                }

                // ── REPORTED leg: tile GEMM. Known non-parity; not graded.
                g.memset(c_batch, 0, MAX_M * n * 2)?;
                gemm(g, gemm_k, a_d, w_d, c_batch, m as u32, n as u32, k as u32)?;
                g.synchronize(0)?;
                let cb = down(g, c_batch, m * n * 2)?;
                let identical = cb == cr;
                let (n_diff, worst, rel) = worst_delta(&cb, &cr);
                gemm_clean &= identical;
                let pct = 100.0 * n_diff as f32 / (m * n) as f32;
                println!(
                    "seed {seed:>5}  {label}  pipelined M={m:<3} byte-identical={identical:<5} \
                     diff_elems={n_diff:<7} ({pct:5.2}%) max|delta|={worst:.6} max_rel={rel:.6}"
                );
            }

            // ── Negative control: the harness MUST see a 1-ULP activation
            // perturbation on row 1, so a "byte-identical" verdict is never
            // an artefact of comparing two blank buffers.
            // Run against the GRADED (batchm) leg: a negative control that only
            // exercises the ungraded arm proves nothing about the verdict.
            let m = 4usize;
            let mut pert = a_bytes.clone();
            pert[2 * (k + 7)] ^= 1;
            let a_pert = up(g, &pert)?;
            g.memset(c_batch, 0, MAX_M * n * 2)?;
            g.memset(c_ref, 0, MAX_M * n * 2)?;
            batchm(g, batchm_k, a_d, w_d, c_batch, m as u32, n as u32, k as u32)?;
            reference(g, gemv_k, a_pert, w_d, c_ref, m, n, k)?;
            g.synchronize(0)?;
            let differs = down(g, c_batch, m * n * 2)? != down(g, c_ref, m * n * 2)?;
            control_ok &= differs;
            println!("seed {seed:>5}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            g.free(a_pert).ok();

            for p in [a_d, w_d, c_batch, c_ref] {
                g.free(p).ok();
            }
        }
    }

    println!(
        "\nREPORTED (ungraded) — dense_gemm_bf16_pipelined byte-identical to M x \
         dense_gemv_bf16 across all legs: {gemm_clean}. `false` is the KNOWN BF16 gap \
         documented at the top of this file, not a regression: the tile GEMM is a \
         different algorithm. Callers needing batch-invariant BF16 output must take the \
         batchm arm (M <= {BATCHM_MAX_M}) or keep their batched-projection threshold \
         above the rungs the tile GEMM serves."
    );

    if !control_ok {
        println!("FAIL — negative control did not mismatch; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if batchm_clean {
        println!(
            "PASS — dense_gemv_bf16_batchm is byte-identical to M x dense_gemv_bf16 at \
             every projection shape and every M in 2..={BATCHM_MAX_M}."
        );
        Ok(())
    } else {
        println!(
            "FAIL — dense_gemv_bf16_batchm is NOT byte-identical to M x dense_gemv_bf16. \
             Every caller of the batched arm (multi-seq decode projections, the MTP \
             drafter's M=2..8 propose) claims that identity in its doc comment; one of \
             them is now lying. Do NOT loosen this comparison."
        );
        std::process::exit(1);
    }
}
