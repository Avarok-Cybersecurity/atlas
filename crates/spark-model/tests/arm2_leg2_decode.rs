// SPDX-License-Identifier: AGPL-3.0-only
//
//! ARM-2 Phase-K Leg-2 — native-MXFP4 (E8M0) numeric gate, **Family A (decode)**.
//!
//! Regression net for the E8M0 decode kernels (`moe_shared_expert_fused_t`):
//!   - CHECK 1 — Family A decode e8m0 routed vs host f32 GEMV (full-range E8M0, <=1 ULP).
//!   - CHECK 2 — RIDER A3: NVFP4 shared branch bit-identical (baseline vs e8m0 wrapper).
//!   - CHECK 3 — RIDER A4: mixed launch (routed-E8M0 + shared-NVFP4) no cross-contamination.
//!
//! `#[ignore]`d: requires a GB10 GPU. CI builds+links this against the libcuda
//! stubs (catches kernel-signature drift) but never runs it. On a GB10 host:
//!   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=deepseek-v4-flash ATLAS_TARGET_QUANT=nvfp4 \
//!     cargo test -p spark-model --test arm2_leg2_decode -- --ignored --nocapture

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

#[path = "arm2_common/support.rs"]
mod support;
use support::*;

const DMOD: &str = "moe_shared_expert_fused_t";
const SEED: u64 = 0x_ADA2_1E62_5EED_0002;

// ONE `#[test]` per binary: the CUDA context lives on the AtlasRegistry
// singleton and is current only on the thread that first initialized it. cargo
// runs each `#[test]` on its own thread, so 5 separate backend-init tests would
// break (only the first thread has a current context). Mirror the original
// single-`main` harness: init the backend once, run all checks on this thread.
#[test]
#[ignore] // Requires GB10 GPU (native-MXFP4 E8M0 decode kernels)
fn leg2_family_a_decode() -> Result<()> {
    let (backend, st) = setup()?;
    let gpu: &dyn GpuBackend = &backend;
    check1_family_a_decode_vs_host_gemv(gpu, st)?;
    check2_rider_a3_shared_nvfp4_bit_identical(gpu, st)?;
    check3_rider_a4_mixed_no_cross_contamination(gpu, st)?;
    Ok(())
}

// ══════════ CHECK 1 — Family A decode, e8m0 routed vs host f32 GEMV (full-range E8M0) ══════════
fn check1_family_a_decode_vs_host_gemv(gpu: &dyn GpuBackend, st: u64) -> Result<()> {
    let mut rng = Rng(SEED);
    let null = DevicePtr(0);
    let k_dec_e8m0 = gpu.kernel(DMOD, "moe_expert_gate_up_shared_t_e8m0")?;

    let (k, n, top_k) = (512usize, 256usize, 1u32);
    let a: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect();
    let gw = gen_wt_fullrange(&mut rng, k, n);
    let uw = gen_wt_fullrange(&mut rng, k, n);
    let a_p = up_u16(gpu, &a)?;
    let gwp = up_u8(gpu, &gw.packed)?;
    let gws = up_u8(gpu, &gw.s_e8m0)?;
    let uwp = up_u8(gpu, &uw.packed)?;
    let uws = up_u8(gpu, &uw.s_e8m0)?;
    let gpt = up_u64(gpu, &[gwp.0])?;
    let gst = up_u64(gpu, &[gws.0])?;
    let upt = up_u64(gpu, &[uwp.0])?;
    let ust = up_u64(gpu, &[uws.0])?;
    let s2 = up_f32(gpu, &[1.0])?;
    let eidx = up_u32(gpu, &[0u32])?;
    let gate_out = gpu.alloc(n * 2)?;
    let up_out = gpu.alloc(n * 2)?;
    let sh_g_out = gpu.alloc(n * 2)?;
    let sh_u_out = gpu.alloc(n * 2)?;
    launch_decode_gate_up(
        gpu, k_dec_e8m0, a_p, gpt, gst, s2, gate_out, upt, ust, s2, up_out, eidx, null, null, 0.0,
        sh_g_out, null, null, 0.0, sh_u_out, n as u32, k as u32, top_k, st,
    )?;
    gpu.synchronize(st)?;
    let kg = rd_u16(gpu, gate_out, n)?;
    let ku = rd_u16(gpu, up_out, n)?;
    let hg = host_gemv(&a, &gw, k, n);
    let hu = host_gemv(&a, &uw, k, n);
    let (pg, eg, ug, _) = cmp_tol(&kg, &hg);
    let (pu, eu, uu, _) = cmp_tol(&ku, &hu);
    let pass = pg && pu;
    println!("CHECK 1  Family A decode e8m0 vs host f32 GEMV (K={k} N={n}, full-range E8M0):");
    println!(
        "         gate: exact {eg}/{n} maxULP {ug} | up: exact {eu}/{n} maxULP {uu}  => {}",
        if pass { "PASS (<=1 ULP)" } else { "FAIL" }
    );
    for p in [
        a_p, gwp, gws, uwp, uws, gpt, gst, upt, ust, s2, eidx, gate_out, up_out, sh_g_out, sh_u_out,
    ] {
        gpu.free(p).ok();
    }
    assert!(
        pass,
        "CHECK 1 FAIL: gate exact {eg}/{n} maxULP {ug}, up exact {eu}/{n} maxULP {uu} (>1 ULP)"
    );
    Ok(())
}

// ══════════ CHECK 2 — RIDER A3: NVFP4 shared branch bit-identical (baseline vs e8m0 wrapper) ══════════
// Shared expert is NVFP4 in BOTH wrappers (<GROUP_SIZE,false>). Routed ptr
// table = [0] so the routed slot writes 0 (no deref). Compare sh_*_out.
fn check2_rider_a3_shared_nvfp4_bit_identical(gpu: &dyn GpuBackend, st: u64) -> Result<()> {
    let mut rng = Rng(SEED);
    let k_dec_base = gpu.kernel(DMOD, "moe_expert_gate_up_shared_t")?;
    let k_dec_e8m0 = gpu.kernel(DMOD, "moe_expert_gate_up_shared_t_e8m0")?;

    let (k, n, top_k) = (512usize, 256usize, 1u32);
    let a: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect();
    // NVFP4 shared weight: nibbles + random valid E4M3 scale bytes (avoid NaN 0x7F/0xFF).
    let g16 = k / 16;
    let mk_nvfp4 = |rng: &mut Rng| -> (Vec<u8>, Vec<u8>) {
        let mut nib = vec![0u8; k * n];
        for x in nib.iter_mut() {
            *x = rng.nibble();
        }
        let mut packed = vec![0u8; k / 2 * n];
        for kh in 0..k / 2 {
            for col in 0..n {
                packed[kh * n + col] =
                    (nib[(2 * kh) * n + col] & 0xF) | ((nib[(2 * kh + 1) * n + col] & 0xF) << 4);
            }
        }
        let mut sc = vec![0u8; g16 * n];
        for x in sc.iter_mut() {
            let mut b = (rng.next_u64() & 0xFF) as u8;
            if b == 0x7F || b == 0xFF {
                b = 0x38; // 1.0
            }
            *x = b;
        }
        (packed, sc)
    };
    let (sgp, sgs) = mk_nvfp4(&mut rng);
    let (sup, sus) = mk_nvfp4(&mut rng);
    let a_p = up_u16(gpu, &a)?;
    let sgp_p = up_u8(gpu, &sgp)?;
    let sgs_p = up_u8(gpu, &sgs)?;
    let sup_p = up_u8(gpu, &sup)?;
    let sus_p = up_u8(gpu, &sus)?;
    // routed ptr tables = [0] (null routed weight -> slot writes 0).
    let z_tbl = up_u64(gpu, &[0u64])?;
    let s2 = up_f32(gpu, &[1.0])?;
    let eidx = up_u32(gpu, &[0u32])?;
    let (sh_g2, sh_u2) = (0.75f32, 1.25f32);
    let run = |kern: KernelHandle| -> Result<(Vec<u16>, Vec<u16>)> {
        let gate_out = gpu.alloc(n * 2)?;
        let up_out = gpu.alloc(n * 2)?;
        let sh_g_out = gpu.alloc(n * 2)?;
        let sh_u_out = gpu.alloc(n * 2)?;
        launch_decode_gate_up(
            gpu, kern, a_p, z_tbl, z_tbl, s2, gate_out, z_tbl, z_tbl, s2, up_out, eidx, sgp_p,
            sgs_p, sh_g2, sh_g_out, sup_p, sus_p, sh_u2, sh_u_out, n as u32, k as u32, top_k, st,
        )?;
        gpu.synchronize(st)?;
        let g = rd_u16(gpu, sh_g_out, n)?;
        let u = rd_u16(gpu, sh_u_out, n)?;
        for p in [gate_out, up_out, sh_g_out, sh_u_out] {
            gpu.free(p).ok();
        }
        Ok((g, u))
    };
    let (bg, bu) = run(k_dec_base)?;
    let (eg, eu) = run(k_dec_e8m0)?;
    let (p1, d1, _) = cmp_bits(&bg, &eg);
    let (p2, d2, _) = cmp_bits(&bu, &eu);
    let pass = p1 && p2;
    println!("CHECK 2  RIDER A3 NVFP4 shared branch baseline vs e8m0-wrapper (K={k} N={n}):");
    println!(
        "         sh_gate diffs {d1}/{n} | sh_up diffs {d2}/{n}  => {}",
        if pass { "PASS (bit-identical)" } else { "FAIL" }
    );
    for p in [a_p, sgp_p, sgs_p, sup_p, sus_p, z_tbl, s2, eidx] {
        gpu.free(p).ok();
    }
    assert!(
        pass,
        "CHECK 2 FAIL: sh_gate diffs {d1}/{n}, sh_up diffs {d2}/{n} (not bit-identical)"
    );
    Ok(())
}

// ══════════ CHECK 3 — RIDER A4: mixed launch (routed-E8M0 + shared-NVFP4) no cross-contamination ══════════
fn check3_rider_a4_mixed_no_cross_contamination(gpu: &dyn GpuBackend, st: u64) -> Result<()> {
    let mut rng = Rng(SEED);
    let null = DevicePtr(0);
    let k_dec_e8m0 = gpu.kernel(DMOD, "moe_expert_gate_up_shared_t_e8m0")?;

    let (k, n, top_k) = (512usize, 256usize, 1u32);
    let a: Vec<u16> = (0..k)
        .map(|_| f32_to_bf16_bits(rng.unit() * 2.0 - 1.0))
        .collect();
    let gw = gen_wt_fullrange(&mut rng, k, n); // routed E8M0
    let uw = gen_wt_fullrange(&mut rng, k, n);
    let g16 = k / 16;
    let mut snib = vec![0u8; k * n];
    for x in snib.iter_mut() {
        *x = rng.nibble();
    }
    let mut spacked = vec![0u8; k / 2 * n];
    for kh in 0..k / 2 {
        for col in 0..n {
            spacked[kh * n + col] =
                (snib[(2 * kh) * n + col] & 0xF) | ((snib[(2 * kh + 1) * n + col] & 0xF) << 4);
        }
    }
    let mut sscale = vec![0u8; g16 * n];
    for x in sscale.iter_mut() {
        let mut b = (rng.next_u64() & 0xFF) as u8;
        if b == 0x7F || b == 0xFF {
            b = 0x38;
        }
        *x = b;
    }
    let a_p = up_u16(gpu, &a)?;
    let gwp = up_u8(gpu, &gw.packed)?;
    let gws = up_u8(gpu, &gw.s_e8m0)?;
    let uwp = up_u8(gpu, &uw.packed)?;
    let uws = up_u8(gpu, &uw.s_e8m0)?;
    let gpt = up_u64(gpu, &[gwp.0])?;
    let gst = up_u64(gpu, &[gws.0])?;
    let upt = up_u64(gpu, &[uwp.0])?;
    let ust = up_u64(gpu, &[uws.0])?;
    let sgp_p = up_u8(gpu, &spacked)?;
    let sgs_p = up_u8(gpu, &sscale)?;
    let s2 = up_f32(gpu, &[1.0])?;
    let eidx = up_u32(gpu, &[0u32])?;
    let sh_g2 = 0.9f32;

    // (a) routed reference (shared passed by caller).
    let run_routed = |sh_p: DevicePtr, sh_s: DevicePtr| -> Result<(Vec<u16>, Vec<u16>)> {
        let gate_out = gpu.alloc(n * 2)?;
        let up_out = gpu.alloc(n * 2)?;
        let sh_g_out = gpu.alloc(n * 2)?;
        let sh_u_out = gpu.alloc(n * 2)?;
        launch_decode_gate_up(
            gpu, k_dec_e8m0, a_p, gpt, gst, s2, gate_out, upt, ust, s2, up_out, eidx, sh_p, sh_s,
            sh_g2, sh_g_out, sh_p, sh_s, sh_g2, sh_u_out, n as u32, k as u32, top_k, st,
        )?;
        gpu.synchronize(st)?;
        let go = rd_u16(gpu, gate_out, n)?;
        let sgo = rd_u16(gpu, sh_g_out, n)?;
        for p in [gate_out, up_out, sh_g_out, sh_u_out] {
            gpu.free(p).ok();
        }
        Ok((go, sgo))
    };
    // routed-only: shared null -> routed gate output = pure routed.
    let (routed_only_gate, _) = run_routed(null, null)?;
    // shared-only: null the routed via a z ptr-table so shared-NVFP4 runs alone.
    let ztbl = up_u64(gpu, &[0u64])?;
    let run_shared_only = || -> Result<Vec<u16>> {
        let gate_out = gpu.alloc(n * 2)?;
        let up_out = gpu.alloc(n * 2)?;
        let sh_g_out = gpu.alloc(n * 2)?;
        let sh_u_out = gpu.alloc(n * 2)?;
        launch_decode_gate_up(
            gpu, k_dec_e8m0, a_p, ztbl, ztbl, s2, gate_out, ztbl, ztbl, s2, up_out, eidx, sgp_p,
            sgs_p, sh_g2, sh_g_out, sgp_p, sgs_p, sh_g2, sh_u_out, n as u32, k as u32, top_k, st,
        )?;
        gpu.synchronize(st)?;
        let sgo = rd_u16(gpu, sh_g_out, n)?;
        for p in [gate_out, up_out, sh_g_out, sh_u_out] {
            gpu.free(p).ok();
        }
        Ok(sgo)
    };
    let shared_only_gate = run_shared_only()?;
    // (c) mixed: routed-E8M0 + shared-NVFP4 in ONE launch.
    let (mixed_routed, mixed_shared) = run_routed(sgp_p, sgs_p)?;
    let (p1, d1, _) = cmp_bits(&routed_only_gate, &mixed_routed);
    let (p2, d2, _) = cmp_bits(&shared_only_gate, &mixed_shared);
    let pass = p1 && p2;
    println!(
        "CHECK 3  RIDER A4 mixed-fusion (routed-E8M0 + shared-NVFP4) no cross-contamination (K={k} N={n}):"
    );
    println!(
        "         routed(mixed vs alone) diffs {d1}/{n} | shared(mixed vs alone) diffs {d2}/{n}  => {}",
        if pass { "PASS (bit-identical)" } else { "FAIL" }
    );
    for p in [
        a_p, gwp, gws, uwp, uws, gpt, gst, upt, ust, sgp_p, sgs_p, s2, eidx, ztbl,
    ] {
        gpu.free(p).ok();
    }
    assert!(
        pass,
        "CHECK 3 FAIL: routed diffs {d1}/{n}, shared diffs {d2}/{n} (cross-contamination)"
    );
    Ok(())
}
