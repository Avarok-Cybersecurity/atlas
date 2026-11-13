// SPDX-License-Identifier: AGPL-3.0-only
//! GPU oracle + timings for the sm_89 FP8 GEMVs' evict-first L2 twins (port
//! row `l2_evict_first_weights`): `fp8_gemv_ada_l2ef{,_dual,_silu_input}` in
//! `kernels/ada/common/fp8_gemv_ada.cu` and the `_l2ef_` batch-M entries in
//! `fp8_gemv_ada_batchm.cu`.
//!
//! RUN THE SASS CHECK FIRST (L15 bench snippet `pass_L15_sass`): Cinference
//! documents a CUDA 13.1 ptxas fault on hinted loads, and Atlas's CUDA 13.3 is
//! unvalidated. A twin whose cache-policy operand is not written fails at run
//! time or silently loads with a garbage policy.
//!
//! WHAT THIS PINS (PASS/FAIL; any FAIL exits non-zero):
//!  * **hint on == hint off**, every output byte, for the three M=1 twins and
//!    the batch-M twins at M in 2..=8, at the Holo 4B and 0.8B FFN shapes over
//!    three seeds;
//!  * **the extent**: sentinel guards either side of every output;
//!  * **a control**: the twin on a different weight must differ.
//!
//! TIMINGS (informational, ABBA order, the A/B the row's receipt starts from):
//!  * warm and cold-L2 microseconds of each M=1 entry, hinted vs not;
//!  * a RESIDENCY PROBE: a 2 MiB buffer (the size of one 4B GDN layer's
//!    recurrent state) is read, then one 4B gate/up GEMV streams 47 MB of
//!    weights, then the buffer is read again; the second read's time is
//!    reported for both twins. The hint is meant to make the second read an
//!    L2 hit; this is the mechanism, the end-to-end number is bench5's.
//!
//! Run on the Ada card, in a build of the ada target:
//!     AVAROK_TARGET_HW=ada cargo run --release -p spark-model \
//!       --example fp8_ada_l2hint_microtest --features cuda,gpu-examples
use anyhow::{Context, Result, ensure};
use spark_runtime::cuda_backend::AvarokCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::time::Instant;

#[allow(dead_code)]
#[path = "common/l15_fp8.rs"]
mod l15;
use l15::*;

const SHAPES: &[(&str, u32, u32)] = &[
    ("4b gate/up", 9216, 2560),
    ("4b down", 2560, 9216),
    ("0.8b gate/up", 3584, 1024),
    ("0.8b down", 1024, 3584),
];
const SEEDS: &[u64] = &[0x15_12EF_0001, 0x15_12EF_0002, 0x15_12EF_0003];
const MAX_M: usize = 8;
/// One 4B GDN layer's recurrent state: 32 value heads x 128 x 128 FP32.
const PROBE_BYTES: usize = 2 << 20;
const ROLES: [&str; 3] = ["plain", "dual", "silu_input"];

/// `[plain, dual, silu]`: `.0` streaming loads, `.1` evict-first.
type Pair = ([KernelHandle; 3], [KernelHandle; 3]);

fn m1_kernels(gpu: &dyn GpuBackend) -> Result<Pair> {
    let get = |e: &str| gpu.kernel("fp8_gemv_ada", e);
    Ok((
        [
            get("fp8_gemv_ada").context("build with AVAROK_TARGET_HW=ada")?,
            get("fp8_gemv_ada_dual")?,
            get("fp8_gemv_ada_silu_input")?,
        ],
        [
            get("fp8_gemv_ada_l2ef")?,
            get("fp8_gemv_ada_l2ef_dual")?,
            get("fp8_gemv_ada_l2ef_silu_input")?,
        ],
    ))
}

fn batchm_kernels(gpu: &dyn GpuBackend, m: usize) -> Result<Pair> {
    let get = |l2ef: bool| -> Result<[KernelHandle; 3]> {
        let [a, b, c] = batchm_names(m, l2ef);
        Ok([
            gpu.kernel("fp8_gemv_ada_batchm", &a)?,
            gpu.kernel("fp8_gemv_ada_batchm", &b)?,
            gpu.kernel("fp8_gemv_ada_batchm", &c)?,
        ])
    };
    Ok((get(false)?, get(true)?))
}

struct Inputs {
    w: [(DevicePtr, DevicePtr); 2],
    w_other: (DevicePtr, DevicePtr),
    a: DevicePtr,
    g: DevicePtr,
    u: DevicePtr,
}

fn inputs(gpu: &dyn GpuBackend, rng: &mut Lcg, n: u32, k: u32) -> Result<Inputs> {
    let (nu, ku) = (n as usize, k as usize);
    let mut weight = || -> Result<(DevicePtr, DevicePtr)> {
        Ok((
            upload(gpu, &rng.e4m3_bytes(nu * ku))?,
            upload(gpu, &f32_bytes(&rng.block_scales(nu, ku)))?,
        ))
    };
    let w = [weight()?, weight()?];
    let w_other = weight()?;
    Ok(Inputs {
        w,
        w_other,
        a: upload(gpu, &bf16_bytes(&rng.bf16s(MAX_M * ku, 1.0)))?,
        g: upload(gpu, &bf16_bytes(&rng.bf16s(MAX_M * ku, 2.0)))?,
        u: upload(gpu, &bf16_bytes(&rng.bf16s(MAX_M * ku, 1.0)))?,
    })
}

/// One launch of `role` with handle `h` over weights `w` into `c` (`c[1]`
/// for the dual's second output).
#[allow(clippy::too_many_arguments)]
fn run(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    role: usize,
    x: &Inputs,
    w: [(DevicePtr, DevicePtr); 2],
    n: u32,
    k: u32,
    c: [DevicePtr; 2],
) -> Result<()> {
    let [(w0, s0), (w1, s1)] = w;
    match role {
        0 => gemv(gpu, h, x.a, w0, s0, c[0], n, k),
        1 => gemv_dual(gpu, h, x.a, w0, s0, c[0], w1, s1, c[1], n, k),
        _ => gemv_silu(gpu, h, x.g, x.u, w0, s0, c[0], n, k),
    }
}

fn outputs(gpu: &dyn GpuBackend, bytes: usize) -> Result<[Guarded; 2]> {
    Ok([Guarded::new(gpu, bytes)?, Guarded::new(gpu, bytes)?])
}

fn read(gpu: &dyn GpuBackend, o: &[Guarded; 2]) -> Result<Vec<u8>> {
    let mut v = o[0].read(gpu)?;
    v.extend(o[1].read(gpu)?);
    Ok(v)
}

/// Hinted vs not for one `(handles, rows)`; `label` names the tier.
#[allow(clippy::too_many_arguments)]
fn parity(
    gpu: &dyn GpuBackend,
    k: &Pair,
    rows: usize,
    x: &Inputs,
    n: u32,
    kk: u32,
    label: &str,
    print: bool,
) -> Result<usize> {
    let mut fails = 0;
    for role in 0..ROLES.len() {
        let bytes = rows * n as usize * 2;
        let (plain, hinted) = (outputs(gpu, bytes)?, outputs(gpu, bytes)?);
        run(
            gpu,
            k.0[role],
            role,
            x,
            x.w,
            n,
            kk,
            [plain[0].ptr(), plain[1].ptr()],
        )?;
        run(
            gpu,
            k.1[role],
            role,
            x,
            x.w,
            n,
            kk,
            [hinted[0].ptr(), hinted[1].ptr()],
        )?;
        let bad = unequal_bytes(&read(gpu, &hinted)?, &read(gpu, &plain)?);
        if bad != 0 || print {
            println!(
                "  [{}] {label} {:<10}: l2ef vs plain bytes={bad}",
                verdict(bad),
                ROLES[role]
            );
        }
        fails += usize::from(bad != 0);
        for o in plain.into_iter().chain(hinted) {
            o.free(gpu)?;
        }
    }
    Ok(fails)
}

/// The twin on another weight must differ from the plain kernel on this one.
fn control(gpu: &dyn GpuBackend, k: &Pair, x: &Inputs, n: u32, kk: u32) -> Result<usize> {
    let bytes = n as usize * 2;
    let (plain, other) = (outputs(gpu, bytes)?, outputs(gpu, bytes)?);
    run(
        gpu,
        k.0[0],
        0,
        x,
        x.w,
        n,
        kk,
        [plain[0].ptr(), plain[1].ptr()],
    )?;
    run(
        gpu,
        k.1[0],
        0,
        x,
        [x.w_other, x.w[1]],
        n,
        kk,
        [other[0].ptr(), other[1].ptr()],
    )?;
    let differ = unequal_bytes(&other[0].read(gpu)?, &plain[0].read(gpu)?);
    let ok = differ > 0;
    println!(
        "  [{}] control (twin on another weight): {differ} bytes differ, must be > 0",
        verdict(usize::from(!ok))
    );
    for o in plain.into_iter().chain(other) {
        o.free(gpu)?;
    }
    Ok(usize::from(!ok))
}

/// Warm and cold, ABBA: plain, hinted, hinted, plain; each side's mean of
/// its two measurements.
fn timings(
    gpu: &dyn GpuBackend,
    k: &Pair,
    x: &Inputs,
    n: u32,
    kk: u32,
    flush: DevicePtr,
) -> Result<()> {
    let out = outputs(gpu, n as usize * 2)?;
    let c = [out[0].ptr(), out[1].ptr()];
    for role in 0..ROLES.len() {
        let side = |h: KernelHandle| move || run(gpu, h, role, x, x.w, n, kk, c);
        let (a, b) = (side(k.0[role]), side(k.1[role]));
        let warm = [
            time_warm_us(gpu, a)?,
            time_warm_us(gpu, b)?,
            time_warm_us(gpu, b)?,
            time_warm_us(gpu, a)?,
        ];
        let cold = [
            time_cold_us(gpu, flush, a)?,
            time_cold_us(gpu, flush, b)?,
            time_cold_us(gpu, flush, b)?,
            time_cold_us(gpu, flush, a)?,
        ];
        let (pw, hw) = ((warm[0] + warm[3]) / 2.0, (warm[1] + warm[2]) / 2.0);
        let (pc, hc) = ((cold[0] + cold[3]) / 2.0, (cold[1] + cold[2]) / 2.0);
        let wbytes = n as f64 * kk as f64 * if role == 1 { 2.0 } else { 1.0 };
        println!(
            "  {:<10} warm plain {pw:.1} us / l2ef {hw:.1} us ({:+.1}%) | cold plain {pc:.1} us ({:.0} GB/s) / \
             l2ef {hc:.1} us ({:.0} GB/s, {:+.1}%)",
            ROLES[role],
            100.0 * (pw - hw) / pw,
            gbs(wbytes, pc),
            gbs(wbytes, hc),
            100.0 * (pc - hc) / pc
        );
    }
    for o in out {
        o.free(gpu)?;
    }
    Ok(())
}

/// Median time of the probe's second read after one GEMV of `h` (dual role,
/// 4B gate/up: the largest per-layer weight stream).
fn probe_us(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    x: &Inputs,
    n: u32,
    kk: u32,
    flush: DevicePtr,
    probe: [DevicePtr; 2],
) -> Result<f64> {
    let out = outputs(gpu, n as usize * 2)?;
    let mut samples = Vec::new();
    for rep in 0..WARMUP + REPS {
        gpu.memset_async(flush, rep as u8, L2_FLUSH_BYTES, 0)?;
        gpu.copy_d2d(probe[0], probe[1], PROBE_BYTES)?;
        run(gpu, h, 1, x, x.w, n, kk, [out[0].ptr(), out[1].ptr()])?;
        gpu.synchronize(0)?;
        let t0 = Instant::now();
        gpu.copy_d2d(probe[0], probe[1], PROBE_BYTES)?;
        gpu.synchronize(0)?;
        if rep >= WARMUP {
            samples.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    for o in out {
        o.free(gpu)?;
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[samples.len() / 2])
}

fn free_inputs(gpu: &dyn GpuBackend, x: Inputs) -> Result<()> {
    for (w, s) in x.w.into_iter().chain([x.w_other]) {
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
    let m1 = m1_kernels(&gpu)?;
    let by_m: Vec<Pair> = (2..=MAX_M)
        .map(|m| batchm_kernels(&gpu, m))
        .collect::<Result<_>>()?;
    let flush = gpu.alloc(L2_FLUSH_BYTES)?;
    let probe = [gpu.alloc(PROBE_BYTES)?, gpu.alloc(PROBE_BYTES)?];
    let mut fails = 0;
    for &(label, n, k) in SHAPES {
        println!("{label} [N={n} K={k}]:");
        for (i, &seed) in SEEDS.iter().enumerate() {
            let mut rng = Lcg(seed ^ (u64::from(n) << 20) ^ u64::from(k));
            let x = inputs(&gpu, &mut rng, n, k)?;
            fails += parity(&gpu, &m1, 1, &x, n, k, "M=1", i == 0)?;
            for (j, bm) in by_m.iter().enumerate() {
                let m = j + 2;
                fails += parity(
                    &gpu,
                    bm,
                    m,
                    &x,
                    n,
                    k,
                    &format!("M={m}"),
                    i == 0 && m == MAX_M,
                )?;
            }
            if i == 0 {
                fails += control(&gpu, &m1, &x, n, k)?;
                timings(&gpu, &m1, &x, n, k, flush)?;
                if label == "4b gate/up" {
                    let plain = probe_us(&gpu, m1.0[1], &x, n, k, flush, probe)?;
                    let hinted = probe_us(&gpu, m1.1[1], &x, n, k, flush, probe)?;
                    println!(
                        "  residency probe (2 MiB re-read after a 4B gate/up GEMV): plain {plain:.1} us, \
                         l2ef {hinted:.1} us ({:+.1}%; info)",
                        100.0 * (plain - hinted) / plain
                    );
                }
            }
            free_inputs(&gpu, x)?;
        }
    }
    for p in probe {
        gpu.free(p)?;
    }
    gpu.free(flush)?;
    ensure!(fails == 0, "{fails} row(s) failed");
    println!("all rows PASS");
    Ok(())
}
