// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the sm_89 batch-M FP8 verify GEMVs
//! (`kernels/ada/common/fp8_gemv_ada_batchm.cu`, port row
//! `ffn_fp8_verify_gemv`) against M launches of the M=1 kernels they stand in
//! for (`fp8_gemv_ada.cu`), at the Holo 3.1 4B and 0.8B dense-FFN shapes.
//!
//! WHAT THIS PINS (each row prints PASS/FAIL; any FAIL exits non-zero):
//!  * **batch-M == M x M=1**, every output byte, for M in 2..=8, for the plain,
//!    dual (gate+up) and SiLU-input entries and their `_l2ef` twins, over
//!    three seeds per shape, plus an edge shape (N and K not multiples of 128);
//!  * **== host**: rows 0 and M-1 of the plain entry at the edge shape equal
//!    `fp8_blockscale_ref::gemv_w8a16_exact` rounded to BF16 (an oracle that
//!    is not another kernel);
//!  * **the extent**: every output carries 64 sentinel bytes either side;
//!  * **a planted control**: one activation of row 1 changed in the batched
//!    input must move row 1 and only row 1, so a comparator that cannot fail
//!    does not read as a pass.
//!
//! TIMINGS (informational): warm (back-to-back) and cold-L2 (64 MB rewritten
//! between reps) microseconds of one batch-M launch against M launches of the
//! M=1 kernel, and the weight-stream GB/s of the batch-M launch against the
//! card's 288 GB/s.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example fp8_ada_batchm_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_model::weight_map::fp8_blockscale_ref::{BlockScaledFp8, gemv_w8a16_exact, round_bf16};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[allow(dead_code)]
#[path = "common/l15_fp8.rs"]
mod l15;
use l15::*;

/// `(label, N, K)`: the dense FFN's gate/up and down at both Holo sizes, and
/// an edge shape with a partial scale block in both directions.
const SHAPES: &[(&str, u32, u32)] = &[
    ("4b gate/up", 9216, 2560),
    ("4b down", 2560, 9216),
    ("0.8b gate/up", 3584, 1024),
    ("0.8b down", 1024, 3584),
    ("edge", 260, 272),
];
const SEEDS: &[u64] = &[0x15_2026_0001, 0x15_2026_0002, 0x15_2026_0003];
const MAX_M: usize = 8;

struct M1 {
    plain: KernelHandle,
    dual: KernelHandle,
    silu: KernelHandle,
}

/// `[plain, dual, silu]` for one M, plain and twin.
struct BatchM {
    k: [KernelHandle; 3],
    l2ef: [KernelHandle; 3],
}

fn kernels(gpu: &dyn GpuBackend) -> Result<(M1, Vec<BatchM>)> {
    let m1 = M1 {
        plain: gpu
            .kernel("fp8_gemv_ada", "fp8_gemv_ada")
            .context("build with AVAROK_TARGET_HW=ada")?,
        dual: gpu.kernel("fp8_gemv_ada", "fp8_gemv_ada_dual")?,
        silu: gpu.kernel("fp8_gemv_ada", "fp8_gemv_ada_silu_input")?,
    };
    let mut by_m = Vec::new();
    for m in 2..=MAX_M {
        let get = |l2ef: bool| -> Result<[KernelHandle; 3]> {
            let [a, b, c] = batchm_names(m, l2ef);
            Ok([
                gpu.kernel("fp8_gemv_ada_batchm", &a)?,
                gpu.kernel("fp8_gemv_ada_batchm", &b)?,
                gpu.kernel("fp8_gemv_ada_batchm", &c)?,
            ])
        };
        by_m.push(BatchM {
            k: get(false)?,
            l2ef: get(true)?,
        });
    }
    Ok((m1, by_m))
}

/// One shape's device inputs: two weights (gate and up, or down twice), and
/// MAX_M rows each of the activation, the gate output and the up output.
struct Inputs {
    w: [(DevicePtr, DevicePtr); 2],
    host_w: BlockScaledFp8,
    a: DevicePtr,
    g: DevicePtr,
    u: DevicePtr,
    host_a: Vec<f32>,
}

fn inputs(gpu: &dyn GpuBackend, rng: &mut Lcg, n: u32, k: u32) -> Result<Inputs> {
    let (nu, ku) = (n as usize, k as usize);
    let host_w = BlockScaledFp8 {
        n: nu,
        k: ku,
        bytes: rng.e4m3_bytes(nu * ku),
        scales: rng.block_scales(nu, ku),
    };
    let w0 = (
        upload(gpu, &host_w.bytes)?,
        upload(gpu, &f32_bytes(&host_w.scales))?,
    );
    let w1 = (
        upload(gpu, &rng.e4m3_bytes(nu * ku))?,
        upload(gpu, &f32_bytes(&rng.block_scales(nu, ku)))?,
    );
    let host_a = rng.bf16s(MAX_M * ku, 1.0);
    Ok(Inputs {
        w: [w0, w1],
        host_w,
        a: upload(gpu, &bf16_bytes(&host_a))?,
        g: upload(gpu, &bf16_bytes(&rng.bf16s(MAX_M * ku, 2.0)))?,
        u: upload(gpu, &bf16_bytes(&rng.bf16s(MAX_M * ku, 1.0)))?,
        host_a,
    })
}

/// Rows `0..m` of A, as M separate M=1 launches into `c` (and `c2` for dual).
#[allow(clippy::too_many_arguments)]
fn reference(
    gpu: &dyn GpuBackend,
    m1: &M1,
    role: usize,
    x: &Inputs,
    m: usize,
    n: u32,
    k: u32,
    c: [DevicePtr; 2],
) -> Result<()> {
    let (row_in, row_out) = (k as usize * 2, n as usize * 2);
    let [(w0, s0), (w1, s1)] = x.w;
    for t in 0..m {
        let (a, g, u) = (
            x.a.offset(t * row_in),
            x.g.offset(t * row_in),
            x.u.offset(t * row_in),
        );
        let (c0, c1) = (c[0].offset(t * row_out), c[1].offset(t * row_out));
        match role {
            0 => gemv(gpu, m1.plain, a, w0, s0, c0, n, k)?,
            1 => gemv_dual(gpu, m1.dual, a, w0, s0, c0, w1, s1, c1, n, k)?,
            _ => gemv_silu(gpu, m1.silu, g, u, w0, s0, c0, n, k)?,
        }
    }
    Ok(())
}

/// One batch-M launch of `role` into `c` (and `c2` for dual).
#[allow(clippy::too_many_arguments)]
fn batched(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    role: usize,
    x: &Inputs,
    a: DevicePtr,
    n: u32,
    k: u32,
    c: [DevicePtr; 2],
) -> Result<()> {
    let [(w0, s0), (w1, s1)] = x.w;
    match role {
        0 => gemv(gpu, h, a, w0, s0, c[0], n, k),
        1 => gemv_dual(gpu, h, a, w0, s0, c[0], w1, s1, c[1], n, k),
        _ => gemv_silu(gpu, h, x.g, x.u, w0, s0, c[0], n, k),
    }
}

const ROLES: [&str; 3] = ["plain", "dual", "silu_input"];

fn guarded_pair(gpu: &dyn GpuBackend, bytes: usize) -> Result<[Guarded; 2]> {
    Ok([Guarded::new(gpu, bytes)?, Guarded::new(gpu, bytes)?])
}

fn read_pair(gpu: &dyn GpuBackend, g: &[Guarded; 2]) -> Result<Vec<u8>> {
    let mut v = g[0].read(gpu)?;
    v.extend(g[1].read(gpu)?);
    Ok(v)
}

/// Byte parity of one (shape, seed, M, role): batch-M and its twin vs M x M=1.
#[allow(clippy::too_many_arguments)]
fn parity(
    gpu: &dyn GpuBackend,
    m1: &M1,
    bm: &BatchM,
    x: &Inputs,
    m: usize,
    n: u32,
    k: u32,
    role: usize,
) -> Result<usize> {
    // Fault localization (an earlier GPU run on this card, 2026-11-13: the run died with
    // CUDA_ERROR_ILLEGAL_ADDRESS after the M=2 rows, and a row that passes
    // prints nothing between M=2 and M=8). `L15_TRACE` drains the stream after
    // each of the three launch groups and names it on stderr, so a fault is
    // pinned to one entry point; `L15_SKIP_L2EF` leaves the evict-first twin
    // out, which separates a fault in the plain batch-M kernels (rows
    // `ffn_fp8_verify_gemv`, `ffn_fp8_decode_gemv`) from one in the hinted
    // twins (row `l2_evict_first_weights`).
    let trace = std::env::var_os("L15_TRACE").is_some();
    let skip_twin = std::env::var_os("L15_SKIP_L2EF").is_some();
    let step = |what: &str| -> Result<()> {
        if trace {
            gpu.synchronize(0)?;
            eprintln!("  trace N={n} K={k} M={m} {}: {what} ok", ROLES[role]);
        }
        Ok(())
    };
    let bytes = m * n as usize * 2;
    let (r, b, t) = (
        guarded_pair(gpu, bytes)?,
        guarded_pair(gpu, bytes)?,
        guarded_pair(gpu, bytes)?,
    );
    reference(gpu, m1, role, x, m, n, k, [r[0].ptr(), r[1].ptr()])?;
    step("M x M=1 reference")?;
    batched(
        gpu,
        bm.k[role],
        role,
        x,
        x.a,
        n,
        k,
        [b[0].ptr(), b[1].ptr()],
    )?;
    step("batch-M")?;
    if !skip_twin {
        batched(
            gpu,
            bm.l2ef[role],
            role,
            x,
            x.a,
            n,
            k,
            [t[0].ptr(), t[1].ptr()],
        )?;
        step("l2ef twin")?;
    }
    let (rv, bv, tv) = (
        read_pair(gpu, &r)?,
        read_pair(gpu, &b)?,
        read_pair(gpu, &t)?,
    );
    // A skipped twin left its buffer at the sentinel; it is not compared.
    let bad_t = if skip_twin {
        0
    } else {
        unequal_bytes(&tv, &rv)
    };
    let bad_b = unequal_bytes(&bv, &rv);
    let mut host_bad = 0;
    // The exact host model, on the plain role at the edge shape.
    if role == 0 && n < 512 {
        for row in [0, m - 1] {
            let ku = k as usize;
            let want = gemv_w8a16_exact(&x.host_a[row * ku..(row + 1) * ku], &x.host_w);
            let got = bf16_to_f32s(&bv[row * n as usize * 2..(row + 1) * n as usize * 2]);
            host_bad += want
                .iter()
                .zip(&got)
                .filter(|(w, g)| round_bf16(**w).to_bits() != g.to_bits())
                .count();
        }
    }
    let fails = bad_b + bad_t + host_bad;
    if fails != 0 || m == 2 || m == MAX_M {
        println!(
            "  [{}] M={m} {:<10}: batch-M vs M x M=1 bytes={bad_b} | l2ef twin bytes={bad_t} | vs host={host_bad}",
            verdict(fails),
            ROLES[role]
        );
    }
    for g in r.into_iter().chain(b).chain(t) {
        g.free(gpu)?;
    }
    Ok(usize::from(fails != 0))
}

/// The comparator must be able to fail: a changed activation in row 1 moves
/// row 1 of the batched output and leaves every other row equal.
fn control(
    gpu: &dyn GpuBackend,
    m1: &M1,
    bm: &BatchM,
    x: &Inputs,
    n: u32,
    k: u32,
) -> Result<usize> {
    let m = 4usize;
    let mut planted = x.host_a[..m * k as usize].to_vec();
    planted[k as usize + 3] = round_bf16(planted[k as usize + 3] + 1.0);
    let a2 = upload(gpu, &bf16_bytes(&planted))?;
    let bytes = m * n as usize * 2;
    let (r, b) = (guarded_pair(gpu, bytes)?, guarded_pair(gpu, bytes)?);
    reference(gpu, m1, 0, x, m, n, k, [r[0].ptr(), r[1].ptr()])?;
    batched(gpu, bm.k[0], 0, x, a2, n, k, [b[0].ptr(), b[1].ptr()])?;
    let (rv, bv) = (r[0].read(gpu)?, b[0].read(gpu)?);
    let row = |v: &[u8], t: usize| v[t * n as usize * 2..(t + 1) * n as usize * 2].to_vec();
    let moved = unequal_bytes(&row(&bv, 1), &row(&rv, 1));
    let others = [0, 2, 3]
        .iter()
        .map(|&t| unequal_bytes(&row(&bv, t), &row(&rv, t)))
        .sum::<usize>();
    let ok = moved > 0 && others == 0;
    println!(
        "  [{}] control: planted row moved {moved} bytes (> 0), other rows {others} (== 0)",
        verdict(usize::from(!ok))
    );
    for g in r.into_iter().chain(b) {
        g.free(gpu)?;
    }
    gpu.free(a2)?;
    Ok(usize::from(!ok))
}

/// Warm and cold timings of the plain role at M in {2, 4, 8}.
fn timings(
    gpu: &dyn GpuBackend,
    m1: &M1,
    by_m: &[BatchM],
    x: &Inputs,
    n: u32,
    k: u32,
    flush: DevicePtr,
) -> Result<()> {
    for m in [2usize, 4, 8] {
        let bm = &by_m[m - 2];
        let out = guarded_pair(gpu, m * n as usize * 2)?;
        let c = [out[0].ptr(), out[1].ptr()];
        let run_ref = || reference(gpu, m1, 0, x, m, n, k, c);
        let run_bm = || batched(gpu, bm.k[0], 0, x, x.a, n, k, c);
        let (ref_w, bm_w) = (time_warm_us(gpu, run_ref)?, time_warm_us(gpu, run_bm)?);
        let (ref_c, bm_c) = (
            time_cold_us(gpu, flush, run_ref)?,
            time_cold_us(gpu, flush, run_bm)?,
        );
        let wbytes = n as f64 * k as f64;
        println!(
            "  M={m}: warm M x M=1 {ref_w:.1} us | batch-M {bm_w:.1} us ({:.2}x) || cold M x M=1 {ref_c:.1} us | \
             batch-M {bm_c:.1} us ({:.2}x, {:.0} GB/s of weights, {:.0}% of 288)",
            ref_w / bm_w,
            ref_c / bm_c,
            gbs(wbytes, bm_c),
            100.0 * gbs(wbytes, bm_c) / 288.0
        );
        for g in out {
            g.free(gpu)?;
        }
    }
    Ok(())
}

fn free_inputs(gpu: &dyn GpuBackend, x: Inputs) -> Result<()> {
    for (w, s) in x.w {
        gpu.free(w)?;
        gpu.free(s)?;
    }
    for p in [x.a, x.g, x.u] {
        gpu.free(p)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let (m1, by_m) = kernels(&gpu)?;
    let flush = gpu.alloc(L2_FLUSH_BYTES)?;
    let mut fails = 0;
    for &(label, n, k) in SHAPES {
        println!("{label} [N={n} K={k}]:");
        for (i, &seed) in SEEDS.iter().enumerate() {
            let mut rng = Lcg(seed ^ (u64::from(n) << 20) ^ u64::from(k));
            let x = inputs(&gpu, &mut rng, n, k)?;
            for m in 2..=MAX_M {
                for role in 0..ROLES.len() {
                    fails += parity(&gpu, &m1, &by_m[m - 2], &x, m, n, k, role)?;
                }
            }
            if i == 0 {
                fails += control(&gpu, &m1, &by_m[2], &x, n, k)?;
                if n >= 1024 {
                    timings(&gpu, &m1, &by_m, &x, n, k, flush)?;
                }
            }
            free_inputs(&gpu, x)?;
        }
    }
    gpu.free(flush)?;
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
