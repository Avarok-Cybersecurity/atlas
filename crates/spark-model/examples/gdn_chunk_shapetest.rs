// SPDX-License-Identifier: AGPL-3.0-only
//! Standalone SHAPE TEST + perf A/B for the wmma + DV-block-split GDN state-spine
//! `gated_delta_rule_chunk_delta_h_tc_vblock` vs the SCALAR production reference
//! `gated_delta_rule_chunk_delta_h_ksplit`.
//!
//! The new kernel grafts our shelved `chunk_delta_h_tc`'s Phase-A wmma (`mma_gram`,
//! W·S on tensor cores) onto croll83's DV-block split (DV 128 → 2×64). The DV split
//! halves every DV-dimensioned smem buffer, so the wmma Sᵀ + ws buffers AND a double
//! buffer fit under 99KB (81KB / 82952 B used) — the thing the shelved single-buffered
//! TC (96KB) couldn't. Drop-in ABI == ksplit (SPEC C): same 21 args, same
//! S_out/uc_out/h_state layout. The new kernel adds a DV-block grid axis folded into
//! `blockIdx.y` (`grid.y = NUM_DV_BLK(2) * batch`), so it runs 2× the CTAs.
//!
//! WHY cosine + norm-ratio (NOT bit-parity like the vblock microtest): the wmma path
//! reorders the FMA accumulation and quantizes its operands (Sᵀ snapshot, W, K) through
//! bf16 fragments, so it is NOT bit-identical to the scalar f32 spine. The standard
//! posture (f32 accumulation everywhere it matters) keeps drift small, gated at
//! cos>=0.99. The scale-sensitive norm-ratio (|‖new‖/‖ref‖ − 1| < 0.05) catches a wrong
//! per-v-head decay scale or a dropped `edl·S_c` carry that cosine (scale-invariant)
//! would miss.
//!
//! Pipeline per (t, batch):
//!   1. recompute_wu → real W,U,gc (faithful isolated test of the spine, not synthetic).
//!   2. ksplit (reference): grid [NV, batch, 1], block 256, smem 99336.
//!   3. tc_vblock (under test): grid [NV, NUM_DV_BLK*batch, 1], block 256, smem 82952.
//!   4. compare {S_c, uc, S_final} new-vs-ref by cosine + norm-ratio dev.
//!   5. perf A/B (CUDA-event, 8 warmup + 50 iters) → speedup = t_ref / t_new.
//!
//! Sweep t ∈ {128,256,512} (nt=2,4,8) × batch ∈ {1,2,4}. t=512 (8 serial chunks) is the
//! load-bearing case where wmma drift would compound and a state-carry/DV-slice bug
//! would diverge in S_final while an early chunk's S_c might still pass.
//!
//! Run on a GB10 host:
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example gdn_chunk_shapetest
//! Exit 0 = all (cos>=0.99 && nrdev<0.05) gates pass (scriptable).

use anyhow::{Result, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const C: usize = 64;

// DV-block split (V_DIM 128 → 2 blocks of 64). The new kernel folds this into
// blockIdx.y: grid.y = NUM_DV_BLK * batch.
const DV_BLK: usize = 64;
const NUM_DV_BLK: usize = VD / DV_BLK; // 2

// ksplit reference smem (99336 B): 2×{W,K,U} db (bf16) + 2×gc + 2×decay(CHUNK+1).
const KSPLIT_SMEM: u32 = (2 * (C * (2 * KD + VD) * 2) + 2 * C * 4 + 2 * (C + 1) * 4) as u32;
// tc_vblock smem (82952 B): St[DV_BLK*KD] bf16 (Kb aliased onto it) + ws[CHUNK*DV_BLK]
// f32 + buf[2][CHUNK*KD + CHUNK*DV_BLK] bf16 + gcb[2][CHUNK] f32 + decb[2][CHUNK+1] f32.
const TC_VBLOCK_SMEM: u32 = (DV_BLK * KD * 2          // St (bf16)
    + C * DV_BLK * 4                                  // ws (f32)
    + 2 * (C * KD + C * DV_BLK) * 2                   // buf db (bf16)
    + 2 * C * 4                                       // gcb (f32)
    + 2 * (C + 1) * 4) as u32; // decb (f32)

// Acceptance gates (loose for the wmma bf16-operand drift, per design §4.5).
const COS_GATE: f64 = 0.99;
const NRDEV_GATE: f64 = 0.05;

// CUDA driver event API for kernel-only timing (mirrors gdn_cdh_vblock_microtest.rs).
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

// ───────────────────────── compare helpers ─────────────────────────
fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Decode a raw little-endian bf16 byte buffer to f32.
fn dn_bf16(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
        .collect()
}
/// Decode a raw little-endian f32 byte buffer to f64.
fn dn_f32(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

/// cosine = dot / (‖a‖‖b‖) — direction only (scale-invariant).
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}
/// norm-ratio deviation = |‖a‖/‖b‖ − 1| — SCALE-SENSITIVE (catches a wrong decay
/// scale / dropped edl·S_c carry that cosine misses).
fn norm_ratio_dev(a: &[f64], b: &[f64]) -> f64 {
    let (mut na, mut nb) = (0f64, 0f64);
    for i in 0..a.len() {
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if nb > 0.0 {
        (na.sqrt() / nb.sqrt() - 1.0).abs()
    } else {
        0.0
    }
}

struct Case {
    t: usize,
    nt: usize,
    batch: usize,
    key: Vec<bf16>,
    val: Vec<bf16>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    h0: Vec<f32>, // [batch*NV*KD*VD]
}

fn gen_case(t: usize, batch: usize) -> Case {
    let nt = t.div_ceil(C);
    let (mut key, mut val, mut gate, mut beta, mut h0) = (vec![], vec![], vec![], vec![], vec![]);
    for bi in 0..batch {
        let mut r = Lcg(0xDE17A ^ ((t as u64) ^ (bi as u64).wrapping_mul(0x9E3779B9)));
        for _ in 0..t * NK * KD {
            key.push(bf16::from_f64(r.r(-0.5, 0.5)));
        }
        for _ in 0..t * NV * VD {
            val.push(bf16::from_f64(r.r(-0.5, 0.5)));
        }
        for _ in 0..t * NV {
            gate.push(r.r(0.80, 0.999) as f32);
        }
        for _ in 0..t * NV {
            beta.push(r.r(0.0, 1.0) as f32);
        }
        for _ in 0..NV * KD * VD {
            h0.push(r.r(-0.1, 0.1) as f32);
        }
    }
    Case {
        t,
        nt,
        batch,
        key,
        val,
        gate,
        beta,
        h0,
    }
}

// recompute_wu → W,U (bf16) + gc_out (f32 cumulative log-gate, consumed by the scan
// as gc_in). Grid [nt, NV, batch]. ABI matches ssm_gdn_a.rs:609.
#[allow(clippy::too_many_arguments)]
fn run_wu(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    c: &Case,
    kp: DevicePtr,
    vp: DevicePtr,
    gp: DevicePtr,
    bp: DevicePtr,
    wp: DevicePtr,
    up: DevicePtr,
    gcp: DevicePtr,
) -> Result<()> {
    let smem1 = (C * KD * 2 + C * C * 4 + C * C * 4 + C * 4) as u32;
    KernelLaunch::new(g, k_wu)
        .grid([c.nt as u32, NV as u32, c.batch as u32])
        .block([128, 1, 1])
        .shared_mem(smem1)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(wp)
        .arg_ptr(up)
        .arg_ptr(gcp)
        .arg_u32(c.batch as u32)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32((NV * VD) as u32)
        .arg_u32(NV as u32)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(0)?;
    Ok(())
}

// chunk_delta_h ksplit (reference) OR tc_vblock (under test). Drop-in ABI: SAME 21 args.
//   tc=false → ksplit:    grid [NV, batch, 1],              smem 99336.
//   tc=true  → tc_vblock: grid [NV, NUM_DV_BLK*batch, 1],   smem 82952.
#[allow(clippy::too_many_arguments)]
fn launch_scan(
    g: &dyn GpuBackend,
    k: KernelHandle,
    c: &Case,
    hp: DevicePtr,
    wp: DevicePtr,
    up: DevicePtr,
    kp: DevicePtr,
    gp: DevicePtr,
    gcp: DevicePtr,
    scp: DevicePtr,
    ucp: DevicePtr,
    tc: bool,
    stream: u64,
) -> Result<()> {
    let (grid, smem) = if tc {
        (
            [NV as u32, (NUM_DV_BLK * c.batch) as u32, 1],
            TC_VBLOCK_SMEM,
        )
    } else {
        ([NV as u32, c.batch as u32, 1], KSPLIT_SMEM)
    };
    KernelLaunch::new(g, k)
        .grid(grid)
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(hp)
        .arg_ptr(wp)
        .arg_ptr(up)
        .arg_ptr(kp)
        .arg_ptr(gp)
        .arg_ptr(gcp)
        .arg_ptr(scp)
        .arg_ptr(ucp)
        .arg_u32(c.batch as u32)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32(NV as u32)
        .arg_u32(0) // h_state_is_table
        .arg_ptr(DevicePtr::NULL) // cu_seqlens
        .arg_ptr(DevicePtr::NULL) // cu_chunks
        .arg_u32(0) // is_varlen
        .launch(stream)?;
    Ok(())
}

// One full run → (S_c bytes, uc bytes, S_final bytes). Fresh h0 each call
// (chunk_delta_h mutates h_state in place). W/U/gc come from recompute_wu so this is
// a faithful isolated test of the spine.
fn run_full(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    tc: bool,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let kp = up_bf16(g, &c.key)?;
    let vp = up_bf16(g, &c.val)?;
    let gp = up_f32(g, &c.gate)?;
    let bp = up_f32(g, &c.beta)?;
    let gcp = g.alloc(c.batch * c.nt * NV * C * 4)?; // gc_out, filled by recompute_wu
    let wp = g.alloc(c.batch * c.nt * NV * C * KD * 2)?;
    let up = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    run_wu(g, k_wu, c, kp, vp, gp, bp, wp, up, gcp)?;
    let hp = up_f32(g, &c.h0)?; // mutated → final S
    let scp = g.alloc(c.batch * c.nt * NV * KD * VD * 2)?; // S_out bf16
    let ucp = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, tc, 0)?;
    g.synchronize(0)?;
    let sc = dn(g, scp, c.batch * c.nt * NV * KD * VD * 2)?;
    let uc = dn(g, ucp, c.batch * c.nt * NV * C * VD * 2)?;
    let sf = dn(g, hp, c.batch * NV * KD * VD * 4)?;
    for p in [kp, vp, gp, bp, gcp, wp, up, hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok((sc, uc, sf))
}

// Kernel-only timing of the scan (W/U pre-staged once; we don't read S back so state
// corruption across iters is fine for pure timing). 8 warmup + `iters` timed.
fn time_scan(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    tc: bool,
    iters: u32,
) -> Result<f64> {
    let kp = up_bf16(g, &c.key)?;
    let vp = up_bf16(g, &c.val)?;
    let gp = up_f32(g, &c.gate)?;
    let bp = up_f32(g, &c.beta)?;
    let gcp = g.alloc(c.batch * c.nt * NV * C * 4)?;
    let wp = g.alloc(c.batch * c.nt * NV * C * KD * 2)?;
    let up = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    run_wu(g, k_wu, c, kp, vp, gp, bp, wp, up, gcp)?;
    let hp = up_f32(g, &c.h0)?;
    let scp = g.alloc(c.batch * c.nt * NV * KD * VD * 2)?;
    let ucp = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    let s = g.create_stream()?;
    for _ in 0..8 {
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, tc, s)?;
    }
    g.synchronize(s)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    let mut ms: f32 = 0.0;
    unsafe {
        if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
            bail!("cuEventCreate");
        }
        if cuEventRecord(e0, s) != 0 {
            bail!("record start");
        }
    }
    for _ in 0..iters {
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, tc, s)?;
    }
    unsafe {
        if cuEventRecord(e1, s) != 0 {
            bail!("record end");
        }
        if cuEventSynchronize(e1) != 0 {
            bail!("sync");
        }
        if cuEventElapsedTime(&mut ms, e0, e1) != 0 {
            bail!("elapsed");
        }
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    for p in [kp, vp, gp, bp, gcp, wp, up, hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok(ms as f64 / iters as f64)
}

/// (cos, nrdev) for a bf16 output (S_c, uc) decoded from raw bytes.
fn cmp_bf16(new: &[u8], reference: &[u8]) -> (f64, f64) {
    let a = dn_bf16(new);
    let b = dn_bf16(reference);
    (cosine(&a, &b), norm_ratio_dev(&a, &b))
}
/// (cos, nrdev) for an f32 output (S_final) decoded from raw bytes.
fn cmp_f32(new: &[u8], reference: &[u8]) -> (f64, f64) {
    let a = dn_f32(new);
    let b = dn_f32(reference);
    (cosine(&a, &b), norm_ratio_dev(&a, &b))
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_ref = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_ksplit",
    )?;
    let k_tc = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_tc_vblock",
    )?;

    let iters = 50u32;
    let mut all_ok = true;

    println!(
        "=== GDN chunk_delta_h tc_vblock SHAPE TEST (cos>={COS_GATE} && nrdev<{NRDEV_GATE}) ==="
    );
    println!(
        "Holo GDN: KD={KD} VD={VD} NK={NK} NV={NV} C={C}; DV_BLK={DV_BLK} NUM_DV_BLK={NUM_DV_BLK}"
    );
    println!("ksplit smem={KSPLIT_SMEM}  tc_vblock smem={TC_VBLOCK_SMEM}");
    println!(
        "{:>5} {:>5} | {:>16} | {:>16} | {:>16} | {:>10} | {}",
        "t", "batch", "S_c: cos nrdev", "uc: cos nrdev", "S_final: cos nrdev", "speedup", "result"
    );
    println!("{}", "-".repeat(96));

    // Default sweep keeps the batch-{1,2,4} + short-length coverage (correctness
    // across batch axis + partial-chunk tails). Pass `--long` for only the
    // long-length batch-1 perf A/B (the wmma DV-block speedup focus).
    let long_only = std::env::args().any(|a| a == "--long");
    let (lens, batches): (&[usize], &[usize]) = if long_only {
        (&[2048, 8192, 16384], &[1])
    } else {
        (&[256, 512, 2048, 8192, 16384], &[1, 2, 4])
    };
    for &t in lens {
        for &batch in batches {
            let case = gen_case(t, batch);

            // reference (scalar ksplit) + new kernel (wmma tc_vblock), fresh h0 each.
            let (sc0, uc0, sf0) = run_full(g, k_wu, k_ref, &case, false)?;
            let (sc1, uc1, sf1) = run_full(g, k_wu, k_tc, &case, true)?;

            let (sc_cos, sc_nr) = cmp_bf16(&sc1, &sc0);
            let (uc_cos, uc_nr) = cmp_bf16(&uc1, &uc0);
            let (sf_cos, sf_nr) = cmp_f32(&sf1, &sf0);

            let pass = sc_cos >= COS_GATE
                && sc_nr < NRDEV_GATE
                && uc_cos >= COS_GATE
                && uc_nr < NRDEV_GATE
                && sf_cos >= COS_GATE
                && sf_nr < NRDEV_GATE;
            all_ok &= pass;

            // perf A/B (kernel-only CUDA-event timing).
            let t_ref = time_scan(g, k_wu, k_ref, &case, false, iters)?;
            let t_new = time_scan(g, k_wu, k_tc, &case, true, iters)?;
            let speedup = if t_new > 0.0 { t_ref / t_new } else { 0.0 };

            println!(
                "{t:>5} {batch:>5} | ksplit {t_ref:>8.4}ms  tc_vblock {t_new:>8.4}ms | {speedup:>6.2}x | Sf_cos {sf_cos:>7.4} | {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }

    println!("{}", "-".repeat(96));
    if all_ok {
        println!("GDN-CHUNK RESULT: PASS (all cos>={COS_GATE} && nrdev<{NRDEV_GATE})");
        Ok(())
    } else {
        eprintln!("GDN-CHUNK RESULT: FAIL (some cos<{COS_GATE} or nrdev>={NRDEV_GATE})");
        std::process::exit(1);
    }
}
