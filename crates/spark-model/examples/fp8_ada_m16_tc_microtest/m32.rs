// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the 17..=32 row half of `fp8_ada_m16_tc_microtest`: the
//! oracle for the 32-row entries `fp8_gemm_ada_m32` and
//! `fp8_gemm_ada_m32_dual` (`kernels/ada/common/fp8_gemm_ada_m16.cu`), split
//! out of the main file for the size cap. Each line prints PASS/FAIL and any
//! FAIL fails the run:
//!
//!  * **every M in 1..=32**, `fp8_gemm_ada_m32` against the M=1 decode GEMV
//!    and (first seed) the host FP64 reference under the M16 tensor-core
//!    contract (at most 2 BF16 ULP, or under the FP32 accumulation floor, and
//!    the block's relative RMS under 1e-3), against the 32-row launch in
//!    bytes, and rows at and past M untouched;
//!  * **rows 0..=15 of the 32-row launch equal the 16-row kernel's rows in
//!    bytes** (tile 0 runs the same MMA sequence), and every M in 1..=16 on
//!    the m32 entry equals the 16-row kernel's launch of that M;
//!  * **row position**: 32 one-row launches equal the 32-row launch in bytes;
//!  * **the dual entry** at M = 17, 24 and 32: both halves equal the plain
//!    entry on each weight, in bytes;
//!  * **two planted controls** (first seed), as the 16-row half has.
use anyhow::Result;
use spark_model::layers::dense_ffn::m16_tc::oracle::{M16_TC_MAX_ULP, compare_m16_tc_block};
use spark_model::weight_map::fp8_blockscale_ref::round_bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::l15::{Guarded, SENTINEL, bf16_bytes, gemv, unequal_bytes, upload, verdict};
use crate::{Inputs, Kernels, MAX_M, MAX_M32, graded, host_rows, tc, tc_dual};

/// One launch of `h` over `m` rows into a sentinel-filled `[32, N]` buffer.
fn launch_rows(
    gpu: &dyn GpuBackend,
    h: KernelHandle,
    a: DevicePtr,
    w: (DevicePtr, DevicePtr),
    m: usize,
    n: u32,
    kk: u32,
) -> Result<Vec<u8>> {
    let out = Guarded::new(gpu, MAX_M32 * n as usize * 2)?;
    tc(gpu, h, a, w, out.ptr(), m, n, kk)?;
    let v = out.read(gpu)?;
    out.free(gpu)?;
    Ok(v)
}

/// The 32 rows through 32 launches of the M=1 decode GEMV, `[32, N]` BF16.
fn gemv_rows(gpu: &dyn GpuBackend, k: &Kernels, x: &Inputs, n: u32, kk: u32) -> Result<Vec<u8>> {
    let out = Guarded::new(gpu, MAX_M32 * n as usize * 2)?;
    for t in 0..MAX_M32 {
        let a = x.a.offset(t * kk as usize * 2);
        let c = out.ptr().offset(t * n as usize * 2);
        gemv(gpu, k.gemv, a, x.w[0].0, x.w[0].1, c, n, kk)?;
    }
    let v = out.read(gpu)?;
    out.free(gpu)?;
    Ok(v)
}

/// Every check of one (shape, seed) for the 32-row entries. `first` adds the
/// host reference, a line per M and the controls.
pub fn check(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    x: &Inputs,
    n: u32,
    kk: u32,
    first: bool,
) -> Result<usize> {
    let (nu, ku, row) = (n as usize, kk as usize, n as usize * 2);
    let mut fails = 0usize;
    let gemv32 = gemv_rows(gpu, k, x, n, kk)?;
    let full = launch_rows(gpu, k.tc32, x.a, x.w[0], MAX_M32, n, kk)?;
    let tc16 = launch_rows(gpu, k.tc, x.a, x.w[0], MAX_M, n, kk)?;
    let host = first.then(|| host_rows(x, MAX_M32));

    // Tile 0 is the 16-row kernel's own arithmetic.
    let tile0 = unequal_bytes(&full[..MAX_M * row], &tc16[..MAX_M * row]);
    println!(
        "  [{}] m32 rows 0..15 vs the 16-row kernel: bytes={tile0}",
        verdict(tile0)
    );
    fails += usize::from(tile0 != 0);

    let before = fails;
    let (mut worst_ulp, mut worst_over) = (0i32, 0usize);
    for m in 1..=MAX_M32 {
        let got = launch_rows(gpu, k.tc32, x.a, x.w[0], m, n, kk)?;
        let live = &got[..m * row];
        let same = unequal_bytes(live, &full[..m * row]);
        // At M <= 16 the m32 entry must also equal the 16-row kernel's launch.
        let same16 = if m <= MAX_M {
            unequal_bytes(live, &tc16[..m * row])
        } else {
            0
        };
        let spill = got[m * row..].iter().filter(|b| **b != SENTINEL).count();
        let (bad_g, ulp_g, over_g) = graded(live, &gemv32[..m * row], nu, ku);
        let (bad_h, ulp_h) = match &host {
            Some(h) => {
                let (b, u, _) = graded(live, &h[..m * row], nu, ku);
                (b, u)
            }
            None => (false, 0),
        };
        let bad = usize::from(bad_g || bad_h || same != 0 || same16 != 0 || spill != 0);
        fails += bad;
        worst_ulp = worst_ulp.max(ulp_g);
        worst_over = worst_over.max(over_g);
        if first || bad != 0 {
            println!(
                "  [{}] m32 M={m:>2}: vs gemv max_ulp={ulp_g} (over {M16_TC_MAX_ULP} ulp, under the floor: {over_g}) | \
                 vs host max_ulp={ulp_h} | vs 32-row launch bytes={same} | vs 16-row kernel bytes={same16} | \
                 rows >= M written={spill}",
                verdict(bad)
            );
        }
    }
    if !first {
        println!(
            "  [{}] m32 M=1..32: vs gemv worst max_ulp={worst_ulp}, most elements over {M16_TC_MAX_ULP} ulp but under \
             the floor in one M: {worst_over}",
            verdict(fails - before)
        );
    }

    // Row position: 32 one-row launches against the 32-row launch.
    let mut moved = 0usize;
    for t in 0..MAX_M32 {
        let got = launch_rows(gpu, k.tc32, x.a.offset(t * ku * 2), x.w[0], 1, n, kk)?;
        moved += unequal_bytes(&got[..row], &full[t * row..(t + 1) * row]);
    }
    println!(
        "  [{}] m32: 32 one-row launches vs the 32-row launch: bytes={moved}",
        verdict(moved)
    );
    fails += usize::from(moved != 0);

    // The dual entry, in bytes.
    let up32 = launch_rows(gpu, k.tc32, x.a, x.w[1], MAX_M32, n, kk)?;
    for m in [17usize, 24, 32] {
        let (c1, c2) = (
            Guarded::new(gpu, MAX_M32 * row)?,
            Guarded::new(gpu, MAX_M32 * row)?,
        );
        tc_dual(
            gpu,
            k.dual32,
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
        let bad = unequal_bytes(&v1[..m * row], &full[..m * row])
            + unequal_bytes(&v2[..m * row], &up32[..m * row]);
        println!(
            "  [{}] m32 dual M={m:>2}: gate and up vs the plain entry bytes={bad}",
            verdict(bad)
        );
        fails += usize::from(bad != 0);
        c1.free(gpu)?;
        c2.free(gpu)?;
    }
    if first {
        fails += controls(gpu, k, x, n, kk, &full, &gemv32)?;
    }
    Ok(fails)
}

/// The comparators must be able to fail, on rows of the second tile.
fn controls(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    x: &Inputs,
    n: u32,
    kk: u32,
    full: &[u8],
    gemv32: &[u8],
) -> Result<usize> {
    let (nu, ku, row, m) = (n as usize, kk as usize, n as usize * 2, 20usize);
    // 1. One activation of row 17 changed: row 17 moves, no other row does.
    let mut planted = x.host_a[..m * ku].to_vec();
    planted[17 * ku + 3] = round_bf16(planted[17 * ku + 3] + 1.0);
    let a2 = upload(gpu, &bf16_bytes(&planted))?;
    let got = launch_rows(gpu, k.tc32, a2, x.w[0], m, n, kk)?;
    gpu.free(a2)?;
    let diff =
        |t: usize| unequal_bytes(&got[t * row..(t + 1) * row], &full[t * row..(t + 1) * row]);
    let moved = diff(17);
    let others: usize = (0..m).filter(|t| *t != 17).map(diff).sum();
    let ok1 = moved > 0 && others == 0;
    println!(
        "  [{}] m32 control: planted row 17 moved {moved} bytes (> 0), the other {} rows {others} (== 0)",
        verdict(usize::from(!ok1)),
        m - 1
    );
    // 2. One element of row 20 (tile 1) set 3 ordinals off the GEMV where
    //    |value| > 1: rejected.
    let at = gemv32[20 * row..21 * row].chunks_exact(2).position(|b| {
        half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))
            .to_f32()
            .abs()
            > 1.0
    });
    let ok2 = match at {
        Some(i) => {
            let i = 20 * nu + i;
            let mut mutant = full.to_vec();
            let bits = u16::from_le_bytes([gemv32[2 * i], gemv32[2 * i + 1]]) + 3;
            mutant[2 * i..2 * i + 2].copy_from_slice(&bits.to_le_bytes());
            !compare_m16_tc_block(&mutant, gemv32, nu, ku)
                .over_budget
                .is_empty()
        }
        None => false,
    };
    println!(
        "  [{}] m32 control: an element of row 20 moved 3 ordinals off the GEMV is rejected",
        verdict(usize::from(!ok2))
    );
    Ok(usize::from(!ok1) + usize::from(!ok2))
}
