// SPDX-License-Identifier: AGPL-3.0-only
//! Oracle for gated_delta_rule_prefill_split4 (atlas_scale-forced GDN prefill,
//! never cross-validated — gb10 uses wy32). Reuses the gdn_fla_e2e_gateb
//! recurrent SSOT (split4 takes the SAME gate=exp(g)/beta inputs). cos<0.999 =
//! split4 is the cascade-seeding bug.
//!   cargo run -p spark-model --release --example gdn_split4_microtest --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;
const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HR: usize = NV / NK;
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
fn cmp(a: &[f32], b: &[f32]) -> (f32, f64) {
    let (mut md, mut dot, mut na, mut nb) = (0f32, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        md = md.max((x - y).abs());
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (md, dot / (na.sqrt() * nb.sqrt() + 1e-12))
}
fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let k4 = g.kernel("gated_delta_rule", "gated_delta_rule_prefill_split4")?;
    let mut all_ok = true;
    for &t in &[64usize, 100, 128, 200] {
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
        let scale = (KD as f64).powf(-0.5);
        let mut o_ref = vec![0f32; t * NV * VD];
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
        let qp = up_bf16(g, &q)?;
        let kp = up_bf16(g, &key)?;
        let vp = up_bf16(g, &val)?;
        let gp = up_f32(g, &gate)?;
        let bp = up_f32(g, &beta)?;
        let hp = up_f32(g, &h0)?;
        let op = g.alloc(t * NV * VD * 2)?;
        KernelLaunch::new(g, k4)
            .grid([(NV * 4) as u32, 1, 1])
            .block([32, 1, 1])
            .shared_mem((4 * KD * 4) as u32)
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
        g.synchronize(0)?;
        let o_gpu = dn_bf16(g, op, t * NV * VD)?;
        for p in [qp, kp, vp, gp, bp, hp, op] {
            let _ = g.free(p);
        }
        let (md, cos) = cmp(&o_gpu, &o_ref);
        let ok = cos >= 0.999;
        all_ok &= ok;
        eprintln!(
            "t={t:4}  split4 O vs recurrent-SSOT: max={md:.4} cos={cos:.6}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }
    eprintln!("\nsplit4 GATE: {}", if all_ok { "PASS" } else { "FAIL" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
