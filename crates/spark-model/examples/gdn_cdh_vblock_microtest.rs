// SPDX-License-Identifier: AGPL-3.0-only
//! MICROTEST + perf A/B for the V-block occupancy lever on FLA kernel #2.
//!
//! A/Bs the CURRENT production scan `gated_delta_rule_chunk_delta_h_ksplit`
//! (grid [nv, batch], 32 CTAs at batch=1 → occupancy-starved on GB10's ~48 SMs)
//! vs the NEW `gated_delta_rule_chunk_delta_h_ksplit_vblock{2,4,8}` which adds a
//! VALUE-dim grid axis (VTILES tiles, each CTA owns V_DIM/VTILES v-columns →
//! nv·VTILES·batch CTAs).
//!
//! The vblock kernel is a byte-for-byte copy of cdh_ksplit_core with only the
//! grid/v-indexing changed (each v-column's state is independent), so the gate
//! is STRICT BIT-PARITY vs ksplit on identical inputs (S_c, uc, S_final), plus
//! kernel-only CUDA-event timing of both → speedup per (t, batch, VTILES).
//!
//! Run on a GB10 host:
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example gdn_cdh_vblock_microtest
//! Exit 0 = all bit-parity gates pass (scriptable). Prints per-case ms/iter.

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

// CUDA driver event API for kernel-only timing (mirrors w8a16_microtest.rs).
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

// VTILES → (kernel entry name, block_x = (VD/VTILES)*SPLIT, SPLIT=2).
fn vblock_kernel(vt: u32) -> (&'static str, u32) {
    match vt {
        2 => (
            "gated_delta_rule_chunk_delta_h_ksplit_vblock2",
            (VD as u32 / 2) * 2,
        ),
        4 => (
            "gated_delta_rule_chunk_delta_h_ksplit_vblock4",
            (VD as u32 / 4) * 2,
        ),
        8 => (
            "gated_delta_rule_chunk_delta_h_ksplit_vblock8",
            (VD as u32 / 8) * 2,
        ),
        _ => panic!("vtiles must be 2/4/8"),
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
        let _ = bi;
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

// recompute_wu → W,U (bf16) + gc_out (f32 cumulative log-gate, consumed by the
// scan as gc_in). Grid [nt, NV, batch]. ABI matches ssm_gdn_a.rs:609.
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

// chunk_delta_h ksplit OR vblock. `vt`=0 → current ksplit (grid [NV,batch,1],
// block 256); else vblock variant (grid [NV,vt,batch], block (VD/vt)*2).
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
    vt: u32,
    stream: u64,
) -> Result<()> {
    // FULL ksplit smem: 2×{W,K,U} db (bf16) + 2×gc + 2×decay(CHUNK+1).
    let smem = (2 * (C * (2 * KD + VD) * 2) + 2 * C * 4 + 2 * (C + 1) * 4) as u32;
    let (grid, block_x) = if vt == 0 {
        ([NV as u32, c.batch as u32, 1], 256u32)
    } else {
        ([NV as u32, vt, c.batch as u32], (VD as u32 / vt) * 2)
    };
    KernelLaunch::new(g, k)
        .grid(grid)
        .block([block_x, 1, 1])
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
// (chunk_delta_h mutates h_state in place).
fn run_full(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    vt: u32,
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
    launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, vt, 0)?;
    g.synchronize(0)?;
    let sc = dn(g, scp, c.batch * c.nt * NV * KD * VD * 2)?;
    let uc = dn(g, ucp, c.batch * c.nt * NV * C * VD * 2)?;
    let sf = dn(g, hp, c.batch * NV * KD * VD * 4)?;
    for p in [kp, vp, gp, bp, gcp, wp, up, hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok((sc, uc, sf))
}

// Kernel-only timing of the scan (W/U pre-staged once; re-upload h0 each iter is
// skipped — we don't read S back, state corruption is fine for pure timing).
#[allow(clippy::too_many_arguments)]
fn time_scan(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    vt: u32,
    iters: u32,
) -> Result<f64> {
    let kp = up_bf16(g, &c.key)?;
    let vp = up_bf16(g, &c.val)?;
    let gp = up_f32(g, &c.gate)?;
    let bp = up_f32(g, &c.beta)?;
    let gcp = g.alloc(c.batch * c.nt * NV * C * 4)?; // gc_out, filled by recompute_wu
    let wp = g.alloc(c.batch * c.nt * NV * C * KD * 2)?;
    let up = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    run_wu(g, k_wu, c, kp, vp, gp, bp, wp, up, gcp)?;
    let hp = up_f32(g, &c.h0)?;
    let scp = g.alloc(c.batch * c.nt * NV * KD * VD * 2)?;
    let ucp = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    let s = g.create_stream()?;
    for _ in 0..8 {
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, vt, s)?;
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
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, vt, s)?;
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

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_ksplit = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_ksplit",
    )?;

    let iters = 50u32;
    let mut all_ok = true;
    println!("=== GDN chunk_delta_h V-block A/B (bit-parity vs ksplit + perf) ===");
    for &t in &[128usize, 256] {
        for &batch in &[1usize, 2, 4] {
            let case = gen_case(t, batch);
            // reference: current ksplit
            let (sc0, uc0, sf0) = run_full(g, k_wu, k_ksplit, &case, 0)?;
            let t_cur = time_scan(g, k_wu, k_ksplit, &case, 0, iters)?;
            for &vt in &[2u32, 4, 8] {
                let (kname, _) = vblock_kernel(vt);
                let k_new = g.kernel("gated_delta_rule_fla", kname)?;
                let (sc1, uc1, sf1) = run_full(g, k_wu, k_new, &case, vt)?;
                let parity = sc0 == sc1 && uc0 == uc1 && sf0 == sf1;
                let t_new = time_scan(g, k_wu, k_new, &case, vt, iters)?;
                all_ok &= parity;
                println!(
                    "t={t:4} batch={batch} VT={vt}: bit-parity={}  current={t_cur:.4}ms  vblock={t_new:.4}ms  speedup={:.2}x",
                    if parity { "PASS" } else { "FAIL ❌" },
                    t_cur / t_new
                );
            }
        }
    }
    println!(
        "\n{}",
        if all_ok {
            "ALL BIT-PARITY GATES PASS ✅"
        } else {
            "BIT-PARITY FAILED ❌"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
