// SPDX-License-Identifier: AGPL-3.0-only
//! GATE B for FLA-decomposition kernel #2: `chunk_delta_h` (chained with #1).
//! GPU: recompute_w_u → W/U(bf16) → chunk_delta_h → {S_c entry states, uc, final S}.
//! Compares to a CPU reference (W/U forward-sub + S recurrence). cos≈1.0 expected
//! (bf16 W/U + bf16 uc noise; S is f32). Run:
//!   cargo run -p spark-model --release --example gdn_chunk_delta_h_gateb

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
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
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

    let mut all_ok = true;
    for &t in &[64usize, 128, 200] {
        let nt = t.div_ceil(C);
        let mut r = Lcg(0xDE17A ^ t as u64);
        let key: Vec<bf16> = (0..t * NK * KD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let val: Vec<bf16> = (0..t * NV * VD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let gate: Vec<f32> = (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect();
        let beta: Vec<f32> = (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect();
        let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();

        // ── CPU reference: W/U forward-sub + S recurrence (per head) ──
        let mut sc_ref = vec![0.0f32; nt * NV * KD * VD]; // entry S_c
        let mut uc_ref = vec![0.0f32; nt * NV * C * VD];
        let mut sf_ref = vec![0.0f32; NV * KD * VD]; // final S
        for vh in 0..NV {
            let kh = vh / HR;
            let mut s = vec![0.0f64; KD * VD];
            for kv in 0..KD * VD {
                s[kv] = h0[vh * KD * VD + kv] as f64;
            }
            for c in 0..nt {
                let cs = c * C;
                let ce = C.min(t - cs);
                let base = (c * NV + vh) * KD * VD;
                for kv in 0..KD * VD {
                    sc_ref[base + kv] = s[kv] as f32;
                }
                let mut gc = vec![0.0f64; ce];
                let mut acc = 0.0;
                for i in 0..ce {
                    acc += (gate[(cs + i) * NV + vh] as f64).ln();
                    gc[i] = acc;
                }
                // W/U forward-sub
                let mut uu = vec![0.0f64; ce * VD];
                let mut ww = vec![0.0f64; ce * KD];
                for i in 0..ce {
                    let bi = beta[(cs + i) * NV + vh] as f64;
                    for v in 0..VD {
                        uu[i * VD + v] = bi * val[((cs + i) * NV + vh) * VD + v].to_f64();
                    }
                    let egci = gc[i].exp();
                    for kk in 0..KD {
                        ww[i * KD + kk] = bi * egci * key[((cs + i) * NK + kh) * KD + kk].to_f64();
                    }
                    for l in 0..i {
                        let mut kkd = 0.0f64;
                        for j in 0..KD {
                            kkd += key[((cs + l) * NK + kh) * KD + j].to_f64()
                                * key[((cs + i) * NK + kh) * KD + j].to_f64();
                        }
                        let lil = bi * (gc[i] - gc[l]).exp() * kkd;
                        for v in 0..VD {
                            uu[i * VD + v] -= lil * uu[l * VD + v];
                        }
                        for kk in 0..KD {
                            ww[i * KD + kk] -= lil * ww[l * KD + kk];
                        }
                    }
                }
                // uc = U - W·S   (entry S)
                let mut uf = vec![0.0f64; ce * VD];
                let ucb = (c * NV + vh) * C;
                for i in 0..ce {
                    for v in 0..VD {
                        let mut x = uu[i * VD + v];
                        for kk in 0..KD {
                            x -= ww[i * KD + kk] * s[kk * VD + v];
                        }
                        uf[i * VD + v] = x;
                        uc_ref[(ucb + i) * VD + v] = x as f32;
                    }
                }
                // S update
                let dl = gc[ce - 1];
                let mut sn = vec![0.0f64; KD * VD];
                for kk in 0..KD {
                    for v in 0..VD {
                        let mut hv = dl.exp() * s[kk * VD + v];
                        for i in 0..ce {
                            hv += (dl - gc[i]).exp()
                                * uf[i * VD + v]
                                * key[((cs + i) * NK + kh) * KD + kk].to_f64();
                        }
                        sn[kk * VD + v] = hv;
                    }
                }
                s = sn;
            }
            for kv in 0..KD * VD {
                sf_ref[vh * KD * VD + kv] = s[kv] as f32;
            }
        }

        // ── GPU: recompute_wu → chunk_delta_h ──
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
        let hp = up_f32(g, &h0)?; // chunk_delta_h mutates this → final S
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
        g.synchronize(0)?;
        let sc_gpu = dn_f32(g, scp, nt * NV * KD * VD)?;
        let uc_gpu = dn_bf16(g, ucp, nt * NV * C * VD)?;
        let sf_gpu = dn_f32(g, hp, NV * KD * VD)?;
        for p in [kp, vp, gp, bp, wp, up, hp, scp, ucp] {
            let _ = g.free(p);
        }

        let (md_s, cos_s) = cmp(&sc_gpu, &sc_ref);
        let (md_u, cos_u) = cmp(&uc_gpu, &uc_ref);
        let (md_f, cos_f) = cmp(&sf_gpu, &sf_ref);
        let ok = cos_s >= 0.999 && cos_u >= 0.999 && cos_f >= 0.999;
        all_ok &= ok;
        eprintln!(
            "t={t:4} nt={nt}  S_c: max={md_s:.4} cos={cos_s:.6} | uc: max={md_u:.4} cos={cos_u:.6} | S_final: max={md_f:.4} cos={cos_f:.6}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }
    eprintln!(
        "\nchunk_delta_h GATE B (chained #1→#2): {}",
        if all_ok { "PASS ✅" } else { "FAIL ❌" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
