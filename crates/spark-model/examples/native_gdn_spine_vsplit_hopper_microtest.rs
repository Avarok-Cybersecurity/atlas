// SPDX-License-Identifier: AGPL-3.0-only
//! ORACLE for the VALUE-SPLIT GDN chunked-prefill state spine (#928).
//!
//! A/Bs `gated_delta_rule_chunk_delta_h_tcfuse_x2` — the arm
//! `[defaults] gdn_prefill_tc` ships — against its Hopper twins
//! `..._vsplit2_hopper` and `..._vsplit4_hopper` on IDENTICAL inputs at
//! T in {256, 1193, 4593}, the microtest shapes of
//! `GDN-PREFILL-ATTRIBUTION.md`.
//!
//! CONTRACT (gated, exit 1 on failure): **BYTE EQUALITY** of all three
//! outputs — `h` (the f32 recurrent state), `uc` and `S_c` (bf16) — at every
//! T and every split. Not a tolerance, and the reason is structural rather
//! than empirical: neither phase of the recurrence contracts over the value
//! dimension, so column block j of the state depends only on column block j of
//! the operands plus the SHARED k-space `W`, `K` and decay row, which are
//! re-read and not reduced. The split therefore re-partitions independent MMA
//! accumulators without reassociating any one of them. Anything short of byte
//! equality is a map defect, not a numerics trade, and a rel_rms gate would
//! swallow exactly that. (The companion CPU test
//! `ops::ssm_gdn_vsplit_tests::the_value_split_is_bit_identical_to_the_unsplit_spine`
//! makes the same comparison on the index maps alone, with no GPU.)
//!
//! A KNOWN_BAD mutation proves the comparison can fail.
//!
//! HOPPER ONLY — the twin entry points exist in no other image, so this
//! example refuses rather than silently measuring one arm twice.
//!
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example native_gdn_spine_vsplit_hopper_microtest

use anyhow::{Result, bail};
use half::bf16;
use spark_model::layers::ops::{
    GDN_SPINE_VSPLIT_MODULE, GDN_SPINE_VSPLIT2_ENTRY, GDN_SPINE_VSPLIT4_ENTRY, GDN_TC_SMEM,
    gdn_spine_vsplit_smem,
};
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// Qwen3.8-27B GDN geometry (config parser: 16 / 48 / 128 / 128).
const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 48;
const C: usize = 64;
/// The parent's module and entry — the arm `[defaults] gdn_prefill_tc` ships.
const TC_MODULE: &str = "gated_delta_rule_chunk_tc";
const TC_ENTRY: &str = "gated_delta_rule_chunk_delta_h_tcfuse_x2";

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
/// RAW bytes, because the gate is byte equality: a decode to f32 would make
/// two different bf16 NaN payloads or -0.0/+0.0 compare equal.
fn dn_raw(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// Bytes that differ, and the index of the first — an operator reading a
/// failure needs to know whether it is one column or all of them.
fn first_diff(a: &[u8], b: &[u8]) -> (usize, Option<usize>) {
    let n = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    let at = a.iter().zip(b.iter()).position(|(x, y)| x != y);
    (n, at)
}

struct Case {
    t: usize,
    nt: usize,
    key: Vec<bf16>,
    val: Vec<bf16>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    h0: Vec<f32>,
}

/// The sibling GDN microtests' fixture recipe: a fixed LCG, gates in
/// [0.80, 0.999], beta in [0, 1]. Identical to
/// `native_gdn_chunk_prefill_microtest` so the two oracles exercise one shape.
fn gen_case(t: usize) -> Case {
    let nt = t.div_ceil(C);
    let mut r = Lcg(0x9D8E_2026 ^ (t as u64));
    let bf = |r: &mut Lcg| bf16::from_f64(r.r(-0.5, 0.5));
    Case {
        t,
        nt,
        key: (0..t * NK * KD).map(|_| bf(&mut r)).collect(),
        val: (0..t * NV * VD).map(|_| bf(&mut r)).collect(),
        gate: (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect(),
        beta: (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect(),
        h0: (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect(),
    }
}

struct Bufs {
    kp: DevicePtr,
    vp: DevicePtr,
    gp: DevicePtr,
    bp: DevicePtr,
    wp: DevicePtr,
    up: DevicePtr,
    gcp: DevicePtr,
}

/// recompute_wu: K,V,gate,beta -> W,U (bf16) + gc (f32). Shared by every arm,
/// so the A/B isolates the spine and nothing else.
fn run_wu(g: &dyn GpuBackend, k_wu: KernelHandle, c: &Case) -> Result<Bufs> {
    let b = Bufs {
        kp: up_bf16(g, &c.key)?,
        vp: up_bf16(g, &c.val)?,
        gp: up_f32(g, &c.gate)?,
        bp: up_f32(g, &c.beta)?,
        wp: g.alloc(c.nt * NV * C * KD * 2)?,
        up: g.alloc(c.nt * NV * C * VD * 2)?,
        gcp: g.alloc(c.nt * NV * C * 4)?,
    };
    KernelLaunch::new(g, k_wu)
        .grid([c.nt as u32, NV as u32, 1])
        .block([256, 1, 1])
        .shared_mem((C * KD * 2 + C * C * 4 + C * 4) as u32)
        .arg_ptr(b.kp)
        .arg_ptr(b.vp)
        .arg_ptr(b.gp)
        .arg_ptr(b.bp)
        .arg_ptr(b.wp)
        .arg_ptr(b.up)
        .arg_ptr(b.gcp)
        .arg_u32(1)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32) // qk_stride: K is a standalone [t][NK*KD] tensor
        .arg_u32((NV * VD) as u32) // v_stride
        .arg_u32(NV as u32) // gb_stride
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(0)?;
    Ok(b)
}

/// One spine launch. The ONLY thing a split changes is `grid.y` and `smem` —
/// the 21-arg ABI and block 256 are the parent's, which is what makes the
/// comparison below a comparison of geometries and not of interfaces.
#[allow(clippy::too_many_arguments)]
fn launch_spine(
    g: &dyn GpuBackend,
    k: KernelHandle,
    split: u32,
    smem: u32,
    c: &Case,
    b: &Bufs,
    hp: DevicePtr,
    scp: DevicePtr,
    ucp: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, split, 1])
        .block([256, 1, 1])
        .shared_mem(smem)
        .arg_ptr(hp)
        .arg_ptr(b.wp)
        .arg_ptr(b.up)
        .arg_ptr(b.kp)
        .arg_ptr(b.gp)
        .arg_ptr(b.gcp)
        .arg_ptr(scp)
        .arg_ptr(ucp)
        .arg_u32(1)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32) // qk_stride (multiple of 8: the K stage vectorises)
        .arg_u32(NV as u32) // gb_stride
        .arg_u32(0) // h_state_is_table
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(stream)?;
    Ok(())
}

struct Arm {
    sc: Vec<u8>,
    uc: Vec<u8>,
    hf: Vec<u8>,
    ms: f64,
}

fn run_arm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    split: u32,
    smem: u32,
    c: &Case,
    b: &Bufs,
    h0: &[f32],
    iters: u32,
) -> Result<Arm> {
    let hp = up_f32(g, h0)?;
    let scp = g.alloc(c.nt * NV * KD * VD * 2)?;
    let ucp = g.alloc(c.nt * NV * C * VD * 2)?;
    // `S_out`/`uc_out` are POISONED before the run: a split that left a column
    // block unwritten would otherwise inherit the previous arm's bytes through
    // a fresh allocation the driver happened to reuse, and read as equal.
    g.copy_h2d(&vec![0xA5u8; c.nt * NV * KD * VD * 2], scp)?;
    g.copy_h2d(&vec![0x5Au8; c.nt * NV * C * VD * 2], ucp)?;
    launch_spine(g, k, split, smem, c, b, hp, scp, ucp, 0)?;
    g.synchronize(0)?;
    let sc = dn_raw(g, scp, c.nt * NV * KD * VD * 2)?;
    let uc = dn_raw(g, ucp, c.nt * NV * C * VD * 2)?;
    let hf = dn_raw(g, hp, NV * KD * VD * 4)?;

    let s = g.create_stream()?; // h is corrupted by the repeats; already read back
    for _ in 0..3 {
        launch_spine(g, k, split, smem, c, b, hp, scp, ucp, s)?;
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
        launch_spine(g, k, split, smem, c, b, hp, scp, ucp, s)?;
    }
    unsafe {
        if cuEventRecord(e1, s) != 0 || cuEventSynchronize(e1) != 0 {
            bail!("record/sync end");
        }
        if cuEventElapsedTime(&mut ms, e0, e1) != 0 {
            bail!("elapsed");
        }
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    for p in [hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok(Arm {
        sc,
        uc,
        hf,
        ms: ms as f64 / iters as f64,
    })
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_par = g.kernel(TC_MODULE, TC_ENTRY)?;
    // HOPPER ONLY. `kernel` and not `try_kernel`: an image without the twins
    // must refuse here rather than report a comparison of one arm with itself.
    let k_v2 = g.kernel(GDN_SPINE_VSPLIT_MODULE, GDN_SPINE_VSPLIT2_ENTRY)?;
    let k_v4 = g.kernel(GDN_SPINE_VSPLIT_MODULE, GDN_SPINE_VSPLIT4_ENTRY)?;

    println!("=== GDN prefill spine: value split across CTAs (#928) ===");
    println!(
        "nk={NK} nv={NV} kd={KD} vd={VD} chunk={C}  smem: split1={GDN_TC_SMEM}B \
         split2={}B split4={}B",
        gdn_spine_vsplit_smem(2),
        gdn_spine_vsplit_smem(4)
    );
    println!(
        "grid=[nv,{{1,2,4}}] block=256 -> {} / {} / {} CTAs of a 132-SM H100",
        NV,
        NV * 2,
        NV * 4
    );
    println!("gate: h, uc and S_c BYTE-IDENTICAL to {TC_ENTRY} at every split\n");

    let mut all_ok = true;
    for &t in &[256usize, 1193, 4593] {
        let case = gen_case(t);
        let b = run_wu(g, k_wu, &case)?;
        g.synchronize(0)?;

        let iters = if t > 2048 { 10 } else { 30 };
        // 4 MAC-pairs per (chunk, head): W.S (C*KD*VD) + K^T.duc (KD*C*VD).
        let flops = (case.nt * NV * 4 * C * KD * VD) as f64;
        println!("T={t} chunks={}", case.nt);
        let base = run_arm(g, k_par, 1, GDN_TC_SMEM, &case, &b, &case.h0, iters)?;
        println!(
            "  {:<30} {:.4} ms / {:6.2} TFLOP/s / {:>3} CTAs / 1.00x",
            "tcfuse_x2 (parent, 1 CTA/vh)",
            base.ms,
            flops / (base.ms * 1e9),
            NV
        );

        for (name, k, split) in [
            ("vsplit2_hopper (2 CTAs/vh)", k_v2, 2u32),
            ("vsplit4_hopper (4 CTAs/vh)", k_v4, 4u32),
        ] {
            let smem = gdn_spine_vsplit_smem(split);
            let a = run_arm(g, k, split, smem, &case, &b, &case.h0, iters)?;
            println!(
                "  {name:<30} {:.4} ms / {:6.2} TFLOP/s / {:>3} CTAs / {:.2}x",
                a.ms,
                flops / (a.ms * 1e9),
                NV as u32 * split,
                base.ms / a.ms
            );
            let mut ok = true;
            for (tag, got, want) in [
                ("h (f32 state)", &a.hf, &base.hf),
                ("uc (bf16 out)", &a.uc, &base.uc),
                ("S_c (bf16 out)", &a.sc, &base.sc),
            ] {
                let (n, at) = first_diff(got, want);
                println!(
                    "    {tag:<20} bytes_differing={n} of {} first_at={}",
                    want.len(),
                    at.map(|i| i.to_string()).unwrap_or_else(|| "-".into())
                );
                ok &= n == 0;
            }
            println!(
                "  VERDICT T={t} ({name}): {}",
                if ok {
                    "BYTE-IDENTICAL"
                } else {
                    "DIFFERS -> FAIL"
                }
            );
            all_ok &= ok;
        }

        // KNOWN_BAD: a harness that has never rejected is not evidence. Launch
        // the 2-way twin with the PARENT's grid.y, so half the state's columns
        // are never written and keep the poison pattern. Every shape and bound
        // stays legal; only the comparison may notice.
        {
            let hp = up_f32(g, &case.h0)?;
            let scp = g.alloc(case.nt * NV * KD * VD * 2)?;
            let ucp = g.alloc(case.nt * NV * C * VD * 2)?;
            g.copy_h2d(&vec![0xA5u8; case.nt * NV * KD * VD * 2], scp)?;
            g.copy_h2d(&vec![0x5Au8; case.nt * NV * C * VD * 2], ucp)?;
            launch_spine(
                g,
                k_v2,
                1,
                gdn_spine_vsplit_smem(2),
                &case,
                &b,
                hp,
                scp,
                ucp,
                0,
            )?;
            g.synchronize(0)?;
            let bad = dn_raw(g, scp, case.nt * NV * KD * VD * 2)?;
            let (n, _) = first_diff(&bad, &base.sc);
            if n == 0 {
                println!("  KNOWN_BAD control DID NOT trip (half the columns unwritten)");
                all_ok = false;
            } else {
                println!("  KNOWN_BAD control refused: {n} bytes of S_c differ\n");
            }
            for p in [hp, scp, ucp] {
                let _ = g.free(p);
            }
        }

        for p in [b.kp, b.vp, b.gp, b.bp, b.wp, b.up, b.gcp] {
            let _ = g.free(p);
        }
    }

    println!(
        "{}",
        if all_ok {
            "ALL GATES PASS"
        } else {
            "GATES FAILED"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
