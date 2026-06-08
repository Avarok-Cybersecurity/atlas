// SPDX-License-Identifier: AGPL-3.0-only
//! END-TO-END GATE B for the FLA decomposition: chain #1 recompute_w_u →
//! #2 chunk_delta_h → #3 chunk_fwd_o on GPU, compare the output O + final state
//! to the recurrent SSOT (per-token reference). cos≈1.0 expected (bf16 pipeline).
//!   cargo run -p spark-model --release --example gdn_fla_e2e_gateb

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HR: usize = NV / NK;
const C: usize = 64;

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
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn cmp(a: &[f32], b: &[f32]) -> (f32, f64) {
    let (mut md, mut dot, mut na, mut nb) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        md = md.max((x - y).abs());
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (md, dot / (na.sqrt() * nb.sqrt() + 1e-12))
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_config("qwen3_6_moe", 2048).expect("no ptx set");
    let backend = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &backend;
    let k_wu: KernelHandle = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_dh: KernelHandle = g.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_delta_h")?;
    let k_dh_tc: KernelHandle =
        g.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_delta_h_tc")?;
    let k_dh_ks: KernelHandle = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_ksplit",
    )?;
    let k_fo: KernelHandle = g.kernel("gated_delta_rule_fla", "gated_delta_rule_chunk_fwd_o")?;

    let mut all_ok = true;
    for &t in &[64usize, 100, 128, 200] {
        let nt = t.div_ceil(C);
        let mut r = Lcg(0xE2E ^ t as u64);
        let q: Vec<bf16> = (0..t * NK * KD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let key: Vec<bf16> = (0..t * NK * KD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let val: Vec<bf16> = (0..t * NV * VD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let gate: Vec<f32> = (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect();
        let beta: Vec<f32> = (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect();
        let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();

        // ── recurrent SSOT ──
        let scale = (KD as f64).powf(-0.5);
        let mut o_ref = vec![0.0f32; t * NV * VD];
        let mut s = h0.iter().map(|&x| x as f64).collect::<Vec<_>>();
        for vh in 0..NV {
            let kh = vh / HR;
            for ti in 0..t {
                let gg = gate[ti * NV + vh] as f64;
                let bt = beta[ti * NV + vh] as f64;
                for v in 0..VD {
                    let mut hk = 0.0;
                    for k in 0..KD {
                        hk += s[(vh * KD + k) * VD + v] * key[(ti * NK + kh) * KD + k].to_f64();
                    }
                    let vnew = (val[(ti * NV + vh) * VD + v].to_f64() - gg * hk) * bt;
                    let mut qd = 0.0;
                    for k in 0..KD {
                        let idx = (vh * KD + k) * VD + v;
                        let hn = gg * s[idx] + key[(ti * NK + kh) * KD + k].to_f64() * vnew;
                        s[idx] = hn;
                        qd += hn * q[(ti * NK + kh) * KD + k].to_f64();
                    }
                    o_ref[(ti * NV + vh) * VD + v] = (qd * scale) as f32;
                }
            }
        }

        // ── GPU: #1 → #2 → #3 ──
        let qp = up_bf16(g, &q)?;
        let kp = up_bf16(g, &key)?;
        let vp = up_bf16(g, &val)?;
        let gp = up_f32(g, &gate)?;
        let bp = up_f32(g, &beta)?;
        let wp = g.alloc(nt * NV * C * KD * 2)?;
        let up = g.alloc(nt * NV * C * VD * 2)?;
        let smem1 = (C * KD * 2 + C * C * 4 + C * C * 4 + C * 4) as u32;
        KernelLaunch::new(g, k_wu)
            .grid([nt as u32, NV as u32, 1])
            .block([128, 1, 1])
            .shared_mem(smem1)
            .arg_ptr(kp)
            .arg_ptr(vp)
            .arg_ptr(gp)
            .arg_ptr(bp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32((NV * VD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        let hp = up_f32(g, &h0)?;
        let scp = g.alloc(nt * NV * KD * VD * 4)?;
        let ucp = g.alloc(nt * NV * C * VD * 2)?;
        let smem2 = (2 * (C * (2 * KD + VD) * 2) + 2 * C * 4) as u32; // 2×{W,K,U} double-buffer + 2×gc
        KernelLaunch::new(g, k_dh)
            .grid([NV as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(smem2)
            .arg_ptr(hp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        let op = g.alloc(t * NV * VD * 2)?;
        let smem3 = (C * KD * 2 + C * KD * 2 + C * C * 4 + C * VD * 2 + KD * VD * 2 + C * 4) as u32;
        KernelLaunch::new(g, k_fo)
            .grid([nt as u32, NV as u32, 1])
            .block([128, 1, 1])
            .shared_mem(smem3)
            .arg_ptr(qp)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_ptr(op)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        g.synchronize(0)?;
        let o_gpu = dn_bf16(g, op, t * NV * VD)?;
        let _ = dn_f32(g, hp, 1)?; // (final S available in hp if needed)
        for p in [qp, kp, vp, gp, bp, wp, up, hp, scp, ucp, op] {
            let _ = g.free(p);
        }

        let (md, cos) = cmp(&o_gpu, &o_ref);
        let ok = cos >= 0.999;
        all_ok &= ok;
        eprintln!(
            "t={t:4} nt={nt}  O vs recurrent-SSOT: max={md:.4} cos={cos:.6}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }
    eprintln!(
        "\nFLA end-to-end GATE B (#1→#2→#3 vs SSOT): {}",
        if all_ok { "PASS ✅" } else { "FAIL ❌" }
    );

    // ── SPEED: FLA 3-kernel pipeline vs wy4 (the payoff measurement) ──
    let kw: KernelHandle = g.kernel(
        "gated_delta_rule_persistent",
        "gated_delta_rule_prefill_persistent_wy4",
    )?;
    let t = 16384usize;
    let nt = t.div_ceil(C);
    let mut r = Lcg(0x5beed);
    let q: Vec<bf16> = (0..t * NK * KD)
        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
        .collect();
    let key: Vec<bf16> = (0..t * NK * KD)
        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
        .collect();
    let val: Vec<bf16> = (0..t * NV * VD)
        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
        .collect();
    let gate: Vec<f32> = (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect();
    let beta: Vec<f32> = (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect();
    let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();
    let qp = up_bf16(g, &q)?;
    let kp = up_bf16(g, &key)?;
    let vp = up_bf16(g, &val)?;
    let gp = up_f32(g, &gate)?;
    let bp = up_f32(g, &beta)?;
    let hp = up_f32(g, &h0)?;
    let wp = g.alloc(nt * NV * C * KD * 2)?;
    let up = g.alloc(nt * NV * C * VD * 2)?;
    let scp = g.alloc(nt * NV * KD * VD * 4)?;
    let ucp = g.alloc(nt * NV * C * VD * 2)?;
    let op = g.alloc(t * NV * VD * 2)?;
    let s1 = (C * KD * 2 + C * C * 4 + C * C * 4 + C * 4) as u32;
    let s2 = (2 * (C * (2 * KD + VD) * 2) + 2 * C * 4) as u32; // 2×{W,K,U} double-buffer + 2×gc
    let s3 = (C * KD * 2 + C * KD * 2 + C * C * 4 + C * VD * 2 + KD * VD * 2 + C * 4) as u32;
    let fla = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_wu)
            .grid([nt as u32, NV as u32, 1])
            .block([128, 1, 1])
            .shared_mem(s1)
            .arg_ptr(kp)
            .arg_ptr(vp)
            .arg_ptr(gp)
            .arg_ptr(bp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32((NV * VD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        KernelLaunch::new(g, k_dh)
            .grid([NV as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(s2)
            .arg_ptr(hp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        KernelLaunch::new(g, k_fo)
            .grid([nt as u32, NV as u32, 1])
            .block([128, 1, 1])
            .shared_mem(s3)
            .arg_ptr(qp)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_ptr(op)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        Ok(())
    };
    let wy4 = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, kw)
            .grid([NV as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(69688)
            .arg_ptr(hp)
            .arg_ptr(qp)
            .arg_ptr(kp)
            .arg_ptr(vp)
            .arg_ptr(gp)
            .arg_ptr(bp)
            .arg_ptr(op)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32((NV * VD) as u32)
            .arg_u32(NV as u32)
            .launch(0)?;
        Ok(())
    };
    // per-kernel launch closures (reuse buffers)
    let k1 = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_wu)
            .grid([nt as u32, NV as u32, 1])
            .block([128, 1, 1])
            .shared_mem(s1)
            .arg_ptr(kp)
            .arg_ptr(vp)
            .arg_ptr(gp)
            .arg_ptr(bp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32((NV * VD) as u32)
            .arg_u32(NV as u32)
            .launch(0)
    };
    let k2 = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_dh)
            .grid([NV as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(s2)
            .arg_ptr(hp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)
    };
    let k3 = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_fo)
            .grid([nt as u32, NV as u32, 1])
            .block([128, 1, 1])
            .shared_mem(s3)
            .arg_ptr(qp)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_ptr(op)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)
    };
    // TC chunk_delta_h variant (A/B candidate). smem: St(32K)+W(16K)+ws(32K f32)+U(16K)+gc.
    let s2tc = (VD * KD * 2 + C * KD * 2 + C * VD * 4 + C * VD * 2 + C * 4) as u32; // 98560
    let k2_tc = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_dh_tc)
            .grid([NV as u32, 1, 1])
            .block([128, 1, 1])
            .shared_mem(s2tc)
            .arg_ptr(hp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)
    };
    let fla_tc = |g: &dyn GpuBackend| -> Result<()> {
        k1(g)?;
        k2_tc(g)?;
        k3(g)
    };
    // k-split chunk_delta_h: 256-thread block (2 threads/v-column), same double-buffer smem.
    let k2_ks = |g: &dyn GpuBackend| -> Result<()> {
        KernelLaunch::new(g, k_dh_ks)
            .grid([NV as u32, 1, 1])
            .block([256, 1, 1])
            .shared_mem(s2)
            .arg_ptr(hp)
            .arg_ptr(wp)
            .arg_ptr(up)
            .arg_ptr(kp)
            .arg_ptr(gp)
            .arg_ptr(scp)
            .arg_ptr(ucp)
            .arg_u32(1)
            .arg_u32(t as u32)
            .arg_u32(nt as u32)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32(NV as u32)
            .launch(0)
    };
    let fla_ks = |g: &dyn GpuBackend| -> Result<()> {
        k1(g)?;
        k2_ks(g)?;
        k3(g)
    };
    fla(g)?;
    g.synchronize(0)?;
    wy4(g)?;
    g.synchronize(0)?;
    fla_tc(g)?;
    g.synchronize(0)?;
    fla_ks(g)?;
    g.synchronize(0)?; // warmup
    let iters = 5;
    let timeit = |f: &dyn Fn(&dyn GpuBackend) -> Result<()>| -> Result<f64> {
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            f(g)?;
        }
        g.synchronize(0)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3 / iters as f64)
    };
    let (m1, m2, m3) = (timeit(&k1)?, timeit(&k2)?, timeit(&k3)?);
    let m2tc = timeit(&k2_tc)?;
    let m2ks = timeit(&k2_ks)?;
    let fla_ms = timeit(&fla)?;
    let fla_ks_ms = timeit(&fla_ks)?;
    let wy4_ms = timeit(&wy4)?;
    eprintln!(
        "SPEED @t={t}: FLA total={fla_ms:.2}ms (recompute_wu={m1:.2} chunk_delta_h={m2:.2} chunk_fwd_o={m3:.2})  wy4={wy4_ms:.2}ms  speedup={:.2}x",
        wy4_ms / fla_ms
    );
    eprintln!("  chunk_delta_h_tc={m2tc:.2}ms (TC-staging, occupancy-bound → no win)",);
    eprintln!(
        "  chunk_delta_h_ksplit={m2ks:.2}ms (8 warps) → FLA total={fla_ks_ms:.2}ms  speedup={:.2}x  (scalar dh={m2:.2})  ← WINNER",
        wy4_ms / fla_ks_ms
    );

    // A/B correctness: alt-dh pipeline O vs scalar-dh pipeline O (scalar already SSOT-validated above).
    fla(g)?;
    g.synchronize(0)?;
    let o_scalar = dn_bf16(g, op, t * NV * VD)?;
    fla_tc(g)?;
    g.synchronize(0)?;
    let o_tc = dn_bf16(g, op, t * NV * VD)?;
    fla_ks(g)?;
    g.synchronize(0)?;
    let o_ks = dn_bf16(g, op, t * NV * VD)?;
    let (md_tc, cos_tc) = cmp(&o_scalar, &o_tc);
    let (md_ks, cos_ks) = cmp(&o_scalar, &o_ks);
    eprintln!(
        "  A/B vs scalar-dh:  TC max={md_tc:.5} cos={cos_tc:.6} {}  |  KSPLIT max={md_ks:.5} cos={cos_ks:.6} {}",
        if cos_tc >= 0.9999 { "PASS" } else { "FAIL" },
        if cos_ks >= 0.9999 { "PASS" } else { "FAIL" }
    );

    for p in [qp, kp, vp, gp, bp, hp, wp, up, scp, ucp, op] {
        let _ = g.free(p);
    }

    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
