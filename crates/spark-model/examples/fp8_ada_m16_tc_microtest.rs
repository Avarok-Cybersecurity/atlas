// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the sm_89 16-row tensor-core FP8 FFN GEMM
//! (`kernels/ada/common/fp8_gemm_ada_m16.cu`, port row `ffn_fp8_m16_tc`)
//! against the decode GEMV it stands in for (`fp8_gemv_ada`) and a host FP64
//! reference, at the Holo 3.1 4B and 0.8B dense-FFN shapes and an edge shape.
//!
//! WHAT THIS PINS (each row prints PASS/FAIL; any FAIL exits non-zero):
//!  * **vs the GEMV, every M in 1..=16**: the kernel reassociates the K
//!    reduction (16 products per MMA, one scale fold per 128-wide block), so
//!    the rule is the M16 tensor-core contract
//!    (`layers::dense_ffn::m16_tc::oracle`): every element within 2 BF16 ULP
//!    of `fp8_gemv_ada`, or under the FP32 accumulation floor
//!    `8 * 2^-24 * sqrt(K) * row_rms` for a cancelled output, and the block's
//!    relative RMS error under 1e-3;
//!  * **vs the host, every M in 1..=16** (first seed): the same rule against
//!    FP64 dot products folded per scale block and rounded to BF16, an oracle
//!    that is not another kernel;
//!  * **row invariance, bytes**: an M-row launch equals the first M rows of
//!    the 16-row launch, and 16 one-row launches equal the 16-row launch;
//!  * **the twins, bytes**: both halves of the dual (gate + up) entry equal
//!    the plain entry on each weight, and the `ahead1` / `ahead4` timing arms
//!    equal the plain entry;
//!  * **the extent**: 64 sentinel bytes either side of every output, and rows
//!    at and past M left untouched;
//!  * **two planted controls**: one activation of row 1 changed must move
//!    row 1 and only row 1, and an element moved 3 BF16 ordinals off the GEMV
//!    must be rejected, so a comparator that cannot fail does not read as a
//!    pass.
//!
//! TIMINGS (informational, `FM16_SKIP_TIMINGS=1` leaves them out, as a
//! memcheck run should; `fp8_ada_m16_tc_microtest/timing.rs`): per M in
//! 1..=16, microseconds of one launch over a ring of weight copies larger
//! than three L2s (DRAM-true, no per-launch sync), its weight-stream GB/s
//! against the card's measured 260 GB/s sustained read
//! (`hbm_bandwidth_probe`, 259.6), the flush-cold and warm figures the other
//! L15 microtests print, and beside them the batch-M GEMV at M = 2..8 and the
//! M = 1 GEMV.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example fp8_ada_m16_tc_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use avarok_core::numeric::fp8_e4m3_to_f32;
use spark_model::layers::dense_ffn::m16_tc::oracle::{M16_TC_MAX_ULP, compare_m16_tc_block};
use spark_model::weight_map::fp8_blockscale_ref::{BlockScaledFp8, round_bf16};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

#[allow(dead_code)]
#[path = "common/l15_fp8.rs"]
mod l15;
use l15::*;

#[path = "fp8_ada_m16_tc_microtest/timing.rs"]
mod timing;

/// `(label, N, K)`: the dense FFN's gate/up and down at both Holo sizes, and
/// an edge shape whose N is odd and ends inside a warp's 8-column tile.
const SHAPES: &[(&str, u32, u32)] = &[
    ("4b gate/up", 9216, 2560),
    ("4b down", 2560, 9216),
    ("0.8b gate/up", 3584, 1024),
    ("0.8b down", 1024, 3584),
    ("edge", 261, 384),
];
const SEEDS: &[u64] = &[0x16_2026_0001, 0x16_2026_0002, 0x16_2026_0003];
const MAX_M: usize = 16;
const MODULE: &str = "fp8_gemm_ada_m16";
/// Sustained read the card measured (`hbm_bandwidth_probe`: 259.6 GB/s).
const PEAK_GBS: f64 = 260.0;
/// Block-level relative-RMS gate, as `examples/common/m16_tc_compare.rs`.
const REL_RMS_GATE: f64 = 1e-3;

struct Kernels {
    gemv: KernelHandle,
    /// `fp8_gemv_ada_batchm_<M>`, M = 2..=8 (timing context only).
    batchm: Vec<KernelHandle>,
    tc: KernelHandle,
    dual: KernelHandle,
    ahead1: KernelHandle,
    ahead4: KernelHandle,
}

fn kernels(gpu: &dyn GpuBackend) -> Result<Kernels> {
    let mut batchm = Vec::new();
    for m in 2..=8 {
        batchm.push(gpu.kernel("fp8_gemv_ada_batchm", &batchm_names(m, false)[0])?);
    }
    Ok(Kernels {
        gemv: gpu
            .kernel("fp8_gemv_ada", "fp8_gemv_ada")
            .context("build with AVAROK_TARGET_HW=ada")?,
        batchm,
        tc: gpu.kernel(MODULE, "fp8_gemm_ada_m16")?,
        dual: gpu.kernel(MODULE, "fp8_gemm_ada_m16_dual")?,
        ahead1: gpu.kernel(MODULE, "fp8_gemm_ada_m16_ahead1")?,
        ahead4: gpu.kernel(MODULE, "fp8_gemm_ada_m16_ahead4")?,
    })
}

/// `fp8_gemm_ada_m16{,_ahead1,_ahead4}`: A, B, scale, C, M, N, K.
#[allow(clippy::too_many_arguments)]
fn tc(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    w: (DevicePtr, DevicePtr),
    c: DevicePtr,
    m: usize,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, 32), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.0)
        .arg_ptr(w.1)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

/// `fp8_gemm_ada_m16_dual`: A, B1, S1, C1, B2, S2, C2, M, N, K; grid z = 2.
#[allow(clippy::too_many_arguments)]
fn tc_dual(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    w1: (DevicePtr, DevicePtr),
    c1: DevicePtr,
    w2: (DevicePtr, DevicePtr),
    c2: DevicePtr,
    m: usize,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(gpu, h)
        .grid([div_ceil(n, 32), 1, 2])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w1.0)
        .arg_ptr(w1.1)
        .arg_ptr(c1)
        .arg_ptr(w2.0)
        .arg_ptr(w2.1)
        .arg_ptr(c2)
        .arg_u32(m as u32)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

/// One shape's device inputs: two weights (gate and up) and 16 activation rows.
struct Inputs {
    w: [(DevicePtr, DevicePtr); 2],
    host_w: BlockScaledFp8,
    a: DevicePtr,
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
        host_a,
    })
}

fn free_inputs(gpu: &dyn GpuBackend, x: Inputs) -> Result<()> {
    for (w, s) in x.w {
        gpu.free(w)?;
        gpu.free(s)?;
    }
    gpu.free(x.a)
}

/// The 16 rows through 16 launches of the M=1 decode GEMV, `[16, N]` BF16.
fn gemv_rows(gpu: &dyn GpuBackend, k: &Kernels, x: &Inputs, n: u32, kk: u32) -> Result<Vec<u8>> {
    let out = Guarded::new(gpu, MAX_M * n as usize * 2)?;
    for t in 0..MAX_M {
        let a = x.a.offset(t * kk as usize * 2);
        let c = out.ptr().offset(t * n as usize * 2);
        gemv(gpu, k.gemv, a, x.w[0].0, x.w[0].1, c, n, kk)?;
    }
    let v = out.read(gpu)?;
    out.free(gpu)?;
    Ok(v)
}

/// One launch of `h` over `m` rows into a sentinel-filled `[16, N]` buffer.
fn tc_rows(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    w: (DevicePtr, DevicePtr),
    m: usize,
    n: u32,
    kk: u32,
) -> Result<Vec<u8>> {
    let out = Guarded::new(gpu, MAX_M * n as usize * 2)?;
    tc(gpu, h, a, w, out.ptr(), m, n, kk)?;
    let v = out.read(gpu)?;
    out.free(gpu)?;
    Ok(v)
}

/// The 16 rows on the host: FP64 dot products over each 128-wide scale block,
/// folded with the block scale in FP64, rounded once to BF16.
fn host_rows(x: &Inputs) -> Vec<u8> {
    let (n, k) = (x.host_w.n, x.host_w.k);
    let kb = x.host_w.k_blocks();
    let mut vals = vec![0f32; MAX_M * n];
    let mut wrow = vec![0f64; k];
    for col in 0..n {
        for (j, w) in wrow.iter_mut().enumerate() {
            *w = f64::from(fp8_e4m3_to_f32(x.host_w.bytes[col * k + j]));
        }
        let scales = &x.host_w.scales[(col / 128) * kb..(col / 128 + 1) * kb];
        for t in 0..MAX_M {
            let a = &x.host_a[t * k..(t + 1) * k];
            let mut total = 0f64;
            for (g, s) in scales.iter().enumerate() {
                let (lo, hi) = (g * 128, (g * 128 + 128).min(k));
                let inner: f64 = a[lo..hi]
                    .iter()
                    .zip(&wrow[lo..hi])
                    .map(|(a, w)| f64::from(*a) * w)
                    .sum();
                total += inner * f64::from(*s);
            }
            vals[t * n + col] = total as f32;
        }
    }
    bf16_bytes(&vals)
}

/// `(fail, max_ulp, over_ulp_only)` of one block under the M16 contract.
fn graded(actual: &[u8], reference: &[u8], n: usize, k: usize) -> (bool, i32, usize) {
    let d = compare_m16_tc_block(actual, reference, n, k);
    for o in d.over_budget.iter().take(4) {
        println!(
            "      over budget: row {} col {} ref {:e} got {:e} ulp {} row_rms {:.3}",
            o.row, o.col, o.reference, o.actual, o.ulp, o.row_rms
        );
    }
    (
        !d.over_budget.is_empty() || d.rel_rms > REL_RMS_GATE,
        d.max_ulp,
        d.over_ulp_only,
    )
}

/// Every check of one (shape, seed). `first` adds the host reference, the
/// controls and a line per M; other seeds print one summary line.
fn check_seed(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    x: &Inputs,
    n: u32,
    kk: u32,
    first: bool,
) -> Result<usize> {
    let (nu, ku, row) = (n as usize, kk as usize, n as usize * 2);
    let mut fails = 0usize;
    let gemv16 = gemv_rows(gpu, k, x, n, kk)?;
    let tc16 = tc_rows(gpu, k.tc, x.a, x.w[0], MAX_M, n, kk)?;
    let host16 = first.then(|| host_rows(x));

    // Every M: against the GEMV and the host under the contract, against the
    // 16-row launch in bytes, and rows at and past M untouched.
    let (mut worst_ulp, mut worst_over) = (0i32, 0usize);
    for m in 1..=MAX_M {
        let got = tc_rows(gpu, k.tc, x.a, x.w[0], m, n, kk)?;
        let live = &got[..m * row];
        let same = unequal_bytes(live, &tc16[..m * row]);
        let spill = got[m * row..].iter().filter(|b| **b != SENTINEL).count();
        let (bad_g, ulp_g, over_g) = graded(live, &gemv16[..m * row], nu, ku);
        let (bad_h, ulp_h) = match &host16 {
            Some(h) => {
                let (b, u, _) = graded(live, &h[..m * row], nu, ku);
                (b, u)
            }
            None => (false, 0),
        };
        let bad = usize::from(bad_g || bad_h || same != 0 || spill != 0);
        fails += bad;
        worst_ulp = worst_ulp.max(ulp_g);
        worst_over = worst_over.max(over_g);
        if first || bad != 0 {
            println!(
                "  [{}] M={m:>2}: vs gemv max_ulp={ulp_g} (over {M16_TC_MAX_ULP} ulp, under the floor: {over_g}) | \
                 vs host max_ulp={ulp_h} | vs 16-row launch bytes={same} | rows >= M written={spill}",
                verdict(bad)
            );
        }
    }
    if !first {
        println!(
            "  [{}] M=1..16: vs gemv worst max_ulp={worst_ulp}, most elements over {M16_TC_MAX_ULP} ulp but under the \
             floor in one M: {worst_over}",
            verdict(fails)
        );
    }

    // Row position: 16 one-row launches against the 16-row launch.
    let mut moved = 0usize;
    for t in 0..MAX_M {
        let got = tc_rows(gpu, k.tc, x.a.offset(t * ku * 2), x.w[0], 1, n, kk)?;
        moved += unequal_bytes(&got[..row], &tc16[t * row..(t + 1) * row]);
    }
    println!(
        "  [{}] 16 one-row launches vs the 16-row launch: bytes={moved}",
        verdict(moved)
    );
    fails += usize::from(moved != 0);

    // The dual entry and the two timing arms, in bytes.
    let up16 = tc_rows(gpu, k.tc, x.a, x.w[1], MAX_M, n, kk)?;
    for m in [1usize, 8, 16] {
        let (c1, c2) = (
            Guarded::new(gpu, MAX_M * row)?,
            Guarded::new(gpu, MAX_M * row)?,
        );
        tc_dual(
            gpu,
            k.dual,
            x.a,
            x.w[0],
            c1.ptr(),
            x.w[1],
            c2.ptr(),
            m,
            n,
            kk,
        )?;
        let (v1, v2) = (c1.read(gpu)?, c2.read(gpu)?);
        let bad = unequal_bytes(&v1[..m * row], &tc16[..m * row])
            + unequal_bytes(&v2[..m * row], &up16[..m * row]);
        println!(
            "  [{}] dual M={m:>2}: gate and up vs the plain entry bytes={bad}",
            verdict(bad)
        );
        fails += usize::from(bad != 0);
        c1.free(gpu)?;
        c2.free(gpu)?;
    }
    for (name, h) in [("ahead1", k.ahead1), ("ahead4", k.ahead4)] {
        let bad = unequal_bytes(&tc_rows(gpu, h, x.a, x.w[0], MAX_M, n, kk)?, &tc16);
        println!(
            "  [{}] {name} M=16 vs the plain entry bytes={bad}",
            verdict(bad)
        );
        fails += usize::from(bad != 0);
    }
    if first {
        fails += controls(gpu, k, x, n, kk, &tc16, &gemv16)?;
    }
    Ok(fails)
}

/// The two comparators must be able to fail.
fn controls(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    x: &Inputs,
    n: u32,
    kk: u32,
    tc16: &[u8],
    gemv16: &[u8],
) -> Result<usize> {
    let (nu, ku, row, m) = (n as usize, kk as usize, n as usize * 2, 4usize);
    // 1. One activation of row 1 changed: row 1 moves, rows 0, 2, 3 do not.
    let mut planted = x.host_a[..m * ku].to_vec();
    planted[ku + 3] = round_bf16(planted[ku + 3] + 1.0);
    let a2 = upload(gpu, &bf16_bytes(&planted))?;
    let got = tc_rows(gpu, k.tc, a2, x.w[0], m, n, kk)?;
    gpu.free(a2)?;
    let diff =
        |t: usize| unequal_bytes(&got[t * row..(t + 1) * row], &tc16[t * row..(t + 1) * row]);
    let (moved, others) = (diff(1), diff(0) + diff(2) + diff(3));
    let ok1 = moved > 0 && others == 0;
    println!(
        "  [{}] control: planted row moved {moved} bytes (> 0), other rows {others} (== 0)",
        verdict(usize::from(!ok1))
    );
    // 2. One element set 3 ordinals off the GEMV where |value| > 1: rejected.
    let at = gemv16.chunks_exact(2).position(|b| {
        half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))
            .to_f32()
            .abs()
            > 1.0
    });
    let ok2 = match at {
        Some(i) => {
            let mut mutant = tc16.to_vec();
            let bits = u16::from_le_bytes([gemv16[2 * i], gemv16[2 * i + 1]]) + 3;
            mutant[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
            !compare_m16_tc_block(&mutant, gemv16, nu, ku)
                .over_budget
                .is_empty()
        }
        None => false,
    };
    println!(
        "  [{}] control: an element 3 ordinals off the GEMV is rejected",
        verdict(usize::from(!ok2))
    );
    Ok(usize::from(!ok1) + usize::from(!ok2))
}

fn main() -> Result<()> {
    let gpu = AvarokCudaBackend::new(0, &avarok_kernels::ptx_modules())?;
    let k = kernels(&gpu)?;
    let flush = gpu.alloc(L2_FLUSH_BYTES)?;
    let skip_timings = std::env::var_os("FM16_SKIP_TIMINGS").is_some();
    let mut fails = 0;
    for &(label, n, kk) in SHAPES {
        println!("{label} [N={n} K={kk}]:");
        for (i, &seed) in SEEDS.iter().enumerate() {
            let mut rng = Lcg(seed ^ (u64::from(n) << 20) ^ u64::from(kk));
            let x = inputs(&gpu, &mut rng, n, kk)?;
            fails += check_seed(&gpu, &k, &x, n, kk, i == 0)?;
            if i == 0 && n >= 1024 && !skip_timings {
                timing::timings(&gpu, &k, &x, n, kk, flush)?;
            }
            free_inputs(&gpu, x)?;
        }
    }
    gpu.free(flush)?;
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
