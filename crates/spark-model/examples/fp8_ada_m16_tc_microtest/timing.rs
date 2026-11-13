// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the timing half of `fp8_ada_m16_tc_microtest`, split out
//! for the size cap. Three figures per launch:
//!
//! * **ring**: back-to-back launches that rotate over copies of the weight
//!   totalling more than three L2s, one stream sync for the whole run. Every
//!   weight byte comes from DRAM and no launch pays a sync, so this is the
//!   figure held against the card's measured sustained read;
//! * **flush-cold** and **warm**: the two figures `fp8_ada_batchm_microtest`
//!   prints (`common/l15_fp8.rs`), kept so the microtests read side by side.
//!   Flush-cold includes one sync's host latency per launch.
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::Instant;

use crate::l15::{Guarded, gbs, gemv, time_cold_us, time_warm_us};
use crate::{Inputs, Kernels, MAX_M, PEAK_GBS, tc, tc_dual};

/// Device bytes the weight ring must exceed: three times the card's 32 MB L2.
const RING_BYTES: usize = 96 << 20;
/// Back-to-back launches per ring sample, and at least two passes of the ring.
const RING_REPS: usize = 40;

/// Copies of weight 0's bytes (the scales are shared) totalling over
/// [`RING_BYTES`], at least 4 and at most 64.
fn weight_ring(gpu: &dyn GpuBackend, x: &Inputs, n: u32, kk: u32) -> Result<Vec<DevicePtr>> {
    let bytes = n as usize * kk as usize;
    let copies = (RING_BYTES / bytes + 1).clamp(4, 64);
    let mut ring = Vec::with_capacity(copies);
    for _ in 0..copies {
        let p = gpu.alloc(bytes)?;
        gpu.copy_d2d(x.w[0].0, p, bytes)?;
        ring.push(p);
    }
    Ok(ring)
}

/// Mean microseconds of one launch over the ring. `launch(w, w2)` enqueues one
/// launch on copy `w`; `w2` is the copy half a ring away, for an entry that
/// reads two weights.
fn ring_us(
    gpu: &dyn GpuBackend,
    ring: &[DevicePtr],
    mut launch: impl FnMut(DevicePtr, DevicePtr) -> Result<()>,
) -> Result<f64> {
    let len = ring.len();
    let pick = |i: usize| (ring[i % len], ring[(i + len / 2) % len]);
    for i in 0..len {
        let (w, w2) = pick(i);
        launch(w, w2)?;
    }
    gpu.synchronize(0)?;
    let reps = RING_REPS.max(2 * len);
    let t0 = Instant::now();
    for i in 0..reps {
        let (w, w2) = pick(i);
        launch(w, w2)?;
    }
    gpu.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

/// Timings of the plain entry at every M, with the GEMVs beside, then the two
/// timing arms and the dual entry at M = 8 and 16.
pub fn timings(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    x: &Inputs,
    n: u32,
    kk: u32,
    flush: DevicePtr,
) -> Result<()> {
    let bytes = MAX_M * n as usize * 2;
    let (out, out2) = (Guarded::new(gpu, bytes)?, Guarded::new(gpu, bytes)?);
    let (c, c2) = (out.ptr(), out2.ptr());
    let ring = weight_ring(gpu, x, n, kk)?;
    let s = x.w[0].1;
    let wbytes = f64::from(n) * f64::from(kk);
    let floor = wbytes / (PEAK_GBS * 1e9) * 1e6;
    let pct = |b: f64, us: f64| 100.0 * gbs(b, us) / PEAK_GBS;

    let g_ring = ring_us(gpu, &ring, |w, _| gemv(gpu, k.gemv, x.a, w, s, c, n, kk))?;
    let g_cold = time_cold_us(gpu, flush, || gemv(gpu, k.gemv, x.a, x.w[0].0, s, c, n, kk))?;
    println!(
        "  floor at {PEAK_GBS:.0} GB/s: {floor:.1} us | ring of {} weight copies | fp8_gemv_ada M=1: ring {g_ring:.1} us \
         ({:.0} GB/s, {:.0}%) | flush-cold {g_cold:.1} us",
        ring.len(),
        gbs(wbytes, g_ring),
        pct(wbytes, g_ring)
    );
    for m in 1..=MAX_M {
        let r = ring_us(gpu, &ring, |w, _| tc(gpu, k.tc, x.a, (w, s), c, m, n, kk))?;
        let run = || tc(gpu, k.tc, x.a, x.w[0], c, m, n, kk);
        let (warm, cold) = (time_warm_us(gpu, run)?, time_cold_us(gpu, flush, run)?);
        let beside = if (2..=8).contains(&m) {
            let h = k.batchm[m - 2];
            let bm = ring_us(gpu, &ring, |w, _| gemv(gpu, h, x.a, w, s, c, n, kk))?;
            format!(" | batch-M gemv ring {bm:.1} us ({:.2}x)", bm / r)
        } else {
            String::new()
        };
        println!(
            "  M={m:>2}: tc ring {r:.1} us ({:.0} GB/s of weights, {:.0}% of {PEAK_GBS:.0}, {:.2}x the floor) | \
             flush-cold {cold:.1} us | warm {warm:.1} us{beside}",
            gbs(wbytes, r),
            pct(wbytes, r),
            r / floor
        );
    }
    for m in [8usize, 16] {
        let a1 = ring_us(gpu, &ring, |w, _| {
            tc(gpu, k.ahead1, x.a, (w, s), c, m, n, kk)
        })?;
        let a4 = ring_us(gpu, &ring, |w, _| {
            tc(gpu, k.ahead4, x.a, (w, s), c, m, n, kk)
        })?;
        // Gate and up on two different ring copies, so both weights are cold.
        let dual = ring_us(gpu, &ring, |w, w2| {
            tc_dual(gpu, k.dual, x.a, (w, s), c, (w2, s), c2, m, n, kk)
        })?;
        println!(
            "  M={m:>2}: ahead1 ring {a1:.1} us | ahead4 ring {a4:.1} us | dual (2 weights) ring {dual:.1} us \
             ({:.0} GB/s, {:.0}% of {PEAK_GBS:.0})",
            gbs(2.0 * wbytes, dual),
            pct(2.0 * wbytes, dual)
        );
    }
    for p in ring {
        gpu.free(p)?;
    }
    out.free(gpu)?;
    out2.free(gpu)
}
