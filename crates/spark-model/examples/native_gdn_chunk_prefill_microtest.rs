// SPDX-License-Identifier: AGPL-3.0-only
//! ORACLE for the tensor-core GDN chunked-prefill state spine (#928).
//!
//! A/Bs the shipped scalar spine `gated_delta_rule_chunk_delta_h_vfused` against
//! `..._tcfuse` and `..._tcfuse_x2` (`ATLAS_GDN_PREFILL_TC`) on IDENTICAL inputs
//! at T in {256, 1193, 4593} — the two nsys shapes of GDN-PREFILL-ATTRIBUTION.md
//! plus a short one. All three are scored against an f64 CPU reference, not each
//! other: the new arms round S_c and duc to bf16 as MMA operands, so "matches
//! the old kernel" is the wrong question and "how far from exact, and does it
//! compound" is the right one.
//!
//! CONTRACT (gated, exit 1 on failure) — note WHICH quantity each half binds:
//!   * `h`, the FP32 recurrent state, is the only output whose dtype can carry a
//!     1e-3 claim: rel_rms(arm, f64 reference) <= 1.0e-3.
//!   * `uc` and `S_c` are bf16 TENSORS. The scalar spine itself measures 1.65e-3
//!     on both — the bf16 STORAGE floor, so a 1e-3 gate there is unsatisfiable
//!     by construction. Gated relative to the spine: <= 1.25x its own rel_rms.
//!   * the per-chunk entry-state rel_rms must not GROW without bound across the
//!     serial chunk chain (last <= 3x the median).
//! Only the shipped arm (`_x2`) is gated; the plain one is measured beside it to
//! show what the second limb buys. A KNOWN_BAD mutation proves it can fail.
//!
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!       --example native_gdn_chunk_prefill_microtest

use anyhow::{Result, bail};
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

// Qwen3.8-27B GDN geometry (config parser: 16 / 48 / 128 / 128).
const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 48;
const C: usize = 64;
/// Heads the f64 reference actually recomputes. The recurrence is independent
/// per head, so 2 of 48 is a complete check of the math at 1/24 of the CPU cost
/// (the full set is ~14.5 GFLOP of f64 at T=4593).
const REF_HEADS: usize = 2;
/// SSOT mirror of `TCF_SMEM` in kernels/hopper/common/gated_delta_rule_chunk_tc.cu.
const TCF_SMEM: u32 = (VD * 136 * 2 + 2 * (C * 136 * 2) + VD * 72 * 2 + (C + 1) * 4) as u32;
/// The shipped fused spine's footprint (W + K + U single-buffered + decay row).
const VFUSED_SMEM: u32 = (C * KD * 2 + C * KD * 2 + C * VD * 2 + (C + 1) * 4) as u32;

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
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    let f = |c: &[u8]| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
    Ok(b.chunks_exact(2).map(f).collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    let f = |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    Ok(b.chunks_exact(4).map(f).collect())
}

/// max_abs, rel_rms = ||a-r||/||r||, cosine. Reference in f64.
fn metrics(a: &[f32], r: &[f64]) -> (f64, f64, f64) {
    let (mut mx, mut se, mut sr, mut dot, mut sa) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(r.iter()) {
        let (x, y) = (*x as f64, *y);
        let d = (x - y).abs();
        mx = mx.max(d);
        (se, sr, dot, sa) = (se + d * d, sr + y * y, dot + x * y, sa + x * x);
    }
    let rel = if sr > 0.0 { (se / sr).sqrt() } else { 0.0 };
    let cos = if sa > 0.0 && sr > 0.0 {
        dot / (sa.sqrt() * sr.sqrt())
    } else {
        1.0
    };
    (mx, rel, cos)
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
/// [0.80, 0.999] (never near the GATE_FLOOR the chunked log form needs),
/// beta in [0, 1].
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

/// recompute_wu: K,V,gate,beta -> W,U (bf16) + gc (f32). Shared by both arms
/// AND by the reference, so the A/B isolates the spine and nothing else.
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

#[allow(clippy::too_many_arguments)]
fn launch_spine(
    g: &dyn GpuBackend,
    k: KernelHandle,
    smem: u32,
    c: &Case,
    b: &Bufs,
    hp: DevicePtr,
    scp: DevicePtr,
    ucp: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
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
        .arg_u32((NK * KD) as u32) // qk_stride (multiple of 8: the TC K stage vectorises)
        .arg_u32(NV as u32) // gb_stride
        .arg_u32(0) // h_state_is_table
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(stream)?;
    Ok(())
}

struct Arm {
    sc: Vec<f32>,
    uc: Vec<f32>,
    hf: Vec<f32>,
    ms: f64,
}

fn run_arm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    smem: u32,
    c: &Case,
    b: &Bufs,
    h0: &[f32],
    iters: u32,
) -> Result<Arm> {
    let hp = up_f32(g, h0)?;
    let scp = g.alloc(c.nt * NV * KD * VD * 2)?;
    let ucp = g.alloc(c.nt * NV * C * VD * 2)?;
    launch_spine(g, k, smem, c, b, hp, scp, ucp, 0)?;
    g.synchronize(0)?;
    let sc = dn_bf16(g, scp, c.nt * NV * KD * VD)?;
    let uc = dn_bf16(g, ucp, c.nt * NV * C * VD)?;
    let hf = dn_f32(g, hp, NV * KD * VD)?;

    let s = g.create_stream()?; // h is corrupted by the repeats; already read back
    for _ in 0..3 {
        launch_spine(g, k, smem, c, b, hp, scp, ucp, s)?;
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
        launch_spine(g, k, smem, c, b, hp, scp, ucp, s)?;
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

/// f64 reference for heads [0, REF_HEADS): the same recurrence the kernels run,
/// laid out for the reference heads only in the kernels' own index order.
struct Ref {
    sc: Vec<f64>,
    uc: Vec<f64>,
    hf: Vec<f64>,
}

fn ref_chunk_delta_h(c: &Case, w: &[f32], u: &[f32], gc: &[f32], h0: &[f32]) -> Ref {
    let hr = NV / NK;
    let (n1, n2) = (c.nt * REF_HEADS * KD * VD, c.nt * REF_HEADS * C * VD);
    let mut r = Ref {
        sc: vec![0.0; n1],
        uc: vec![0.0; n2],
        hf: vec![0.0; REF_HEADS * KD * VD],
    };
    for vh in 0..REF_HEADS {
        let kh = vh / hr;
        let mut s = vec![0.0f64; KD * VD];
        for (i, v) in s.iter_mut().enumerate() {
            *v = h0[vh * KD * VD + i] as f64;
        }
        for ch in 0..c.nt {
            let cs = ch * C;
            let ce = (c.t - cs).min(C);
            let base = ch * NV + vh;
            let rbase = ch * REF_HEADS + vh;
            r.sc[rbase * KD * VD..(rbase + 1) * KD * VD].copy_from_slice(&s);
            let gl = gc[base * C + ce - 1] as f64;
            let mut duc = vec![0.0f64; C * VD];
            for i in 0..ce {
                let dc = (gl - gc[base * C + i] as f64).exp();
                for v in 0..VD {
                    let mut ws = 0.0f64;
                    for k in 0..KD {
                        ws += w[base * C * KD + i * KD + k] as f64 * s[k * VD + v];
                    }
                    let uci = u[base * C * VD + i * VD + v] as f64 - ws;
                    r.uc[rbase * C * VD + i * VD + v] = uci;
                    duc[i * VD + v] = dc * uci;
                }
            }
            let edl = gl.exp();
            for k in 0..KD {
                for v in 0..VD {
                    let mut acc = edl * s[k * VD + v];
                    for i in 0..ce {
                        acc += duc[i * VD + v] * c.key[(cs + i) * NK * KD + kh * KD + k].to_f64();
                    }
                    s[k * VD + v] = acc;
                }
            }
        }
        r.hf[vh * KD * VD..(vh + 1) * KD * VD].copy_from_slice(&s);
    }
    r
}

/// Gather the reference heads' slice out of a full-NV kernel output.
fn take_heads(full: &[f32], nt: usize, per_head: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(nt * REF_HEADS * per_head);
    for ch in 0..nt {
        for vh in 0..REF_HEADS {
            let b = (ch * NV + vh) * per_head;
            out.extend_from_slice(&full[b..b + per_head]);
        }
    }
    out
}

fn report(tag: &str, a: &[f32], r: &[f64]) -> (f64, f64, f64) {
    let (mx, rel, cos) = metrics(a, r);
    println!("    {tag:<26} max_abs={mx:.6e}  rel_rms={rel:.4e}  cosine={cos:.9}");
    (mx, rel, cos)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_old = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_vfused",
    )?;
    // The spine is declared in `kernels/hopper`'s `[kernels] overrides`, so it
    // is absent from every other image and this oracle SKIPS there rather than
    // failing. It was developed and validated on GB10 and is arch-neutral; it
    // sits in the Hopper tree only because rule S1 refuses new cross-hardware
    // symlinks (`kernels/hopper/HARDWARE.toml`). A GB10 receipt moves the file
    // back down and this skip stops firing there.
    let (Ok(k_new), Ok(k_x2)) = (
        g.kernel(
            "gated_delta_rule_chunk_tc",
            "gated_delta_rule_chunk_delta_h_tcfuse",
        ),
        g.kernel(
            "gated_delta_rule_chunk_tc",
            "gated_delta_rule_chunk_delta_h_tcfuse_x2",
        ),
    ) else {
        println!(
            "SKIPPED: the tensor-core GDN prefill spine is not in this image. It \
             is declared under kernels/hopper (sm_90a); build with that hardware \
             set to run this oracle."
        );
        return Ok(());
    };

    println!("=== GDN chunked-prefill spine: scalar vfused vs tensor-core tcfuse (#928) ===");
    println!(
        "nk={NK} nv={NV} kd={KD} vd={VD} chunk={C}  smem: vfused={VFUSED_SMEM}B tcfuse={TCF_SMEM}B"
    );
    println!("gate (_x2 arm only): h rel_rms <= 1.0e-3 vs f64; uc/S_c <= 1.25x the spine\n");

    let mut all_ok = true;
    for &t in &[256usize, 1193, 4593] {
        let case = gen_case(t);
        let b = run_wu(g, k_wu, &case)?;
        g.synchronize(0)?;
        let w = dn_bf16(g, b.wp, case.nt * NV * C * KD)?;
        let u = dn_bf16(g, b.up, case.nt * NV * C * VD)?;
        let gc = dn_f32(g, b.gcp, case.nt * NV * C)?;
        let rf = ref_chunk_delta_h(&case, &w, &u, &gc, &case.h0);

        let iters = if t > 2048 { 10 } else { 30 };
        // 4 MAC-pairs per (chunk, head): W.S (C*KD*VD) + K^T.duc (KD*C*VD).
        let flops = (case.nt * NV * 4 * C * KD * VD) as f64;
        let per = REF_HEADS * KD * VD;
        let mut base = (0.0f64, 0.0f64, 0.0f64); // scalar spine's (uc, S_c, ms)
        println!("T={t} chunks={}", case.nt);
        for (name, k, smem, gated) in [
            ("vfused (scalar, default)", k_old, VFUSED_SMEM, false),
            ("tcfuse  (1 bf16 limb)", k_new, TCF_SMEM, false),
            ("tcfuse_x2 (2 bf16 limbs)", k_x2, TCF_SMEM, true),
        ] {
            let a = run_arm(g, k, smem, &case, &b, &case.h0, iters)?;
            let uc = take_heads(&a.uc, case.nt, C * VD);
            let sc = take_heads(&a.sc, case.nt, KD * VD);
            let hf: Vec<f32> = a.hf[..per].to_vec();
            println!(
                "  {name:<25} {:.4} ms / {:.2} TFLOP/s / {:.2}x",
                a.ms,
                flops / (a.ms * 1e9),
                if base.2 > 0.0 { base.2 / a.ms } else { 1.0 }
            );
            let (_, rel_uc, _) = report("uc (bf16 out)", &uc, &rf.uc);
            let (_, rel_sc, _) = report("S_c (bf16 out)", &sc, &rf.sc);
            let (_, rel_h, _) = report("h (f32 state)", &hf, &rf.hf);
            let mut growth = Vec::with_capacity(case.nt);
            for ch in 0..case.nt {
                let (_, r, _) = metrics(
                    &sc[ch * per..(ch + 1) * per],
                    &rf.sc[ch * per..(ch + 1) * per],
                );
                growth.push(r);
            }
            let mut sorted = growth.clone();
            sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let median = sorted[sorted.len() / 2];
            let last = *growth.last().unwrap();
            println!(
                "    per-chunk S_c rel_rms: first={:.3e} median={median:.3e} last={last:.3e} \
                 growth={:.2}x",
                growth[0],
                if median > 0.0 { last / median } else { 0.0 }
            );
            if base.2 == 0.0 {
                base = (rel_uc, rel_sc, a.ms); // the scalar spine IS the floor
            }
            if !gated {
                continue; // the plain arm is measured, not gated
            }
            let v = |c: bool| if c { "PASS" } else { "FAIL" };
            let bounded = median <= 0.0 || last <= 3.0 * median;
            let ok = rel_h <= 1e-3 && rel_uc <= 1.25 * base.0 && rel_sc <= 1.25 * base.1 && bounded;
            println!(
                "  VERDICT T={t} ({name}): h {} uc {} S_c {} drift {} => {}",
                v(rel_h <= 1e-3),
                v(rel_uc <= 1.25 * base.0),
                v(rel_sc <= 1.25 * base.1),
                if bounded { "BOUNDED" } else { "GROWING" },
                v(ok)
            );
            all_ok &= ok;
        }
        let hf_n: Vec<f32> = run_arm(g, k_x2, TCF_SMEM, &case, &b, &case.h0, 1)?.hf[..per].to_vec();

        // KNOWN_BAD: a harness that has never rejected is not evidence. Perturb
        // one reference element by 10% of the tensor rms and require max_abs to
        // move (the norm gate is deliberately blind to a single element).
        let mut bad = rf.hf.clone();
        let rms = (rf.hf.iter().map(|x| x * x).sum::<f64>() / rf.hf.len() as f64).sqrt();
        bad[0] += 0.1 * rms;
        let (mx_bad, _, _) = metrics(&hf_n, &bad);
        let (mx_ok, _, _) = metrics(&hf_n, &rf.hf);
        if mx_bad <= mx_ok {
            println!("  KNOWN_BAD control DID NOT trip (max_abs {mx_bad:.3e} <= {mx_ok:.3e})");
            all_ok = false;
        } else {
            println!("  KNOWN_BAD control refused: max_abs {mx_bad:.3e} > clean {mx_ok:.3e}\n");
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
