// SPDX-License-Identifier: AGPL-3.0-only
//! Equivalence oracle for the K∈{2,3,4} WY speculative-verify GDN kernels
//! (`gated_delta_rule_wy2/wy3/wy4`) against an N-step sequential single-token
//! GDN recurrence reference.
//!
//! This is the losslessness foundation for the in-place SSM verify-commit
//! change (item #2 of the MTP hybrid spec-decode fix): the in-place commit
//! removes the dual-buffer copy machinery but does NOT change kernel
//! numerics, so committed state must equal what the wy kernel already
//! produces today. This oracle proves the wy kernels match the
//! token-by-token recurrence so that "commit surviving state, no rollback"
//! is byte-for-byte equivalent to the current copy-based commit.
//!
//! Per SSM layer with random H_0/q/k/v/gate/beta, for each K:
//!   (1) run the K-step sequential single-token GDN recurrence (f64 SSOT),
//!       capturing per-token output and the committed H after each token;
//!   (2) run the matching `gated_delta_rule_wy{K}` once;
//!   (3) assert per-token output cos ≥ 0.99999 AND committed-state cos
//!       ≥ 0.99999 (final H for full-accept, intermediate[n-1] for prefix-n).
//!
//!   cargo run -p spark-model --release --example gdn_wy_verify_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

// Qwen3-Next GDN head config (matches gdn_split4_microtest / production).
const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HR: usize = NV / NK;

const PASS_COS: f64 = 0.99999;

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
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
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
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Sequential single-token recurrent reference (f64 SSOT, mirrors the exact
/// per-token GDN recurrence the WY kernels implement: bf16 inputs cast to
/// f32, gate clamped to (1e-6, 1-1e-6), output scaled by 1/sqrt(k_dim)).
///
/// This is the override-independent oracle — the production decode kernel
/// is model-specialized (the qwen3-next override takes FP32 q/k/v), so the
/// SSOT must be the math itself, exactly as `gdn_split4_microtest` validates
/// the prefill kernel against a pure recurrent reference. Returns
/// (per-token outputs `[k][NV*VD]`, committed H after each token
/// `[k][NV*KD*VD]` in `[vh][kd][vd]` layout).
fn sequential_ref(
    h0: &[f32],
    q: &[Vec<bf16>],
    key: &[Vec<bf16>],
    val: &[Vec<bf16>],
    gate: &[Vec<f32>],
    beta: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let scale = (KD as f64).powf(-0.5);
    // State per (vh): H[vh][kd][vd], seeded from h0.
    let mut s: Vec<f64> = h0.iter().map(|&x| x as f64).collect();
    let mut outs = Vec::with_capacity(k);
    let mut h_after = Vec::with_capacity(k);
    for t in 0..k {
        let mut o_t = vec![0f32; NV * VD];
        for vh in 0..NV {
            let kh = vh / HR;
            let gg = (gate[t][vh] as f64).clamp(1e-6, 1.0 - 1e-6);
            let bt = beta[t][vh] as f64;
            for v in 0..VD {
                // hk = sum_j H[j][v] * k[j]
                let mut hk = 0.0;
                for kk in 0..KD {
                    hk += s[(vh * KD + kk) * VD + v] * key[t][kh * KD + kk].to_f64();
                }
                let vnew = (val[t][vh * VD + v].to_f64() - gg * hk) * bt;
                let mut qd = 0.0;
                for kk in 0..KD {
                    let idx = (vh * KD + kk) * VD + v;
                    let hn = gg * s[idx] + key[t][kh * KD + kk].to_f64() * vnew;
                    s[idx] = hn;
                    qd += hn * q[t][kh * KD + kk].to_f64();
                }
                o_t[vh * VD + v] = (qd * scale) as f32;
            }
        }
        outs.push(o_t);
        h_after.push(s.iter().map(|&x| x as f32).collect());
    }
    (outs, h_after)
}

/// Run the WY{K} kernel once. Tokens are strided: qk_stride=NK*KD,
/// v_stride=NV*VD, gb_stride=NV (one row per token), matching the
/// production `decode_batched` layout. Returns (interleaved output,
/// intermediates[0..K-1], final H).
fn run_wy(
    g: &dyn GpuBackend,
    kernel: spark_runtime::gpu::KernelHandle,
    h0: &[f32],
    q: &[Vec<bf16>],
    key: &[Vec<bf16>],
    val: &[Vec<bf16>],
    gate: &[Vec<f32>],
    beta: &[Vec<f32>],
    k: usize,
) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
    use spark_runtime::kernel_args::KernelLaunch;
    // Pack tokens contiguously with per-token stride.
    let mut q_flat = Vec::with_capacity(k * NK * KD);
    let mut k_flat = Vec::with_capacity(k * NK * KD);
    let mut v_flat = Vec::with_capacity(k * NV * VD);
    let mut g_flat = Vec::with_capacity(k * NV);
    let mut b_flat = Vec::with_capacity(k * NV);
    for t in 0..k {
        q_flat.extend_from_slice(&q[t]);
        k_flat.extend_from_slice(&key[t]);
        v_flat.extend_from_slice(&val[t]);
        g_flat.extend_from_slice(&gate[t]);
        b_flat.extend_from_slice(&beta[t]);
    }
    let hp = up_f32(g, h0)?;
    let qp = up_bf16(g, &q_flat)?;
    let kp = up_bf16(g, &k_flat)?;
    let vp = up_bf16(g, &v_flat)?;
    let gp = up_f32(g, &g_flat)?;
    let bp = up_f32(g, &b_flat)?;
    let op = g.alloc(k * NV * VD * 2)?;
    // (K-1) intermediates: inter[i] = H after token i.
    let inters: Vec<DevicePtr> = (0..k - 1)
        .map(|_| g.alloc(NV * KD * VD * 4))
        .collect::<Result<_>>()?;

    let mut launch = KernelLaunch::new(g, kernel)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(hp)
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(op);
    for &ip in &inters {
        launch = launch.arg_ptr(ip);
    }
    launch
        .arg_u32(1) // batch_size
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32) // qk_stride
        .arg_u32((NV * VD) as u32) // v_stride
        .arg_u32(NV as u32) // gb_stride
        .launch(0)?;
    g.synchronize(0)?;

    let out = dn_bf16(g, op, k * NV * VD)?;
    let mut inter_h = Vec::with_capacity(k - 1);
    for &ip in &inters {
        inter_h.push(dn_f32(g, ip, NV * KD * VD)?);
    }
    let final_h = dn_f32(g, hp, NV * KD * VD)?;
    for p in [hp, qp, kp, vp, gp, bp, op] {
        let _ = g.free(p);
    }
    for ip in inters {
        let _ = g.free(ip);
    }
    Ok((out, inter_h, final_h))
}

fn main() -> Result<()> {
    let g0 = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let wy = [
        (
            2usize,
            g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
        ),
        (
            3usize,
            g.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
        ),
        (
            4usize,
            g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
        ),
    ];

    let mut all_ok = true;
    // Multiple random layers (seeds) per K to exercise diverse H_0/inputs.
    for &k in &[2usize, 3, 4] {
        let wk = wy.iter().find(|(kk, _)| *kk == k).unwrap().1;
        for layer in 0..6u64 {
            let mut r = Lcg(0xD17A ^ (k as u64) << 8 ^ layer);
            let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();
            let q: Vec<Vec<bf16>> = (0..k)
                .map(|_| {
                    (0..NK * KD)
                        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
                        .collect()
                })
                .collect();
            let key: Vec<Vec<bf16>> = (0..k)
                .map(|_| {
                    (0..NK * KD)
                        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
                        .collect()
                })
                .collect();
            let val: Vec<Vec<bf16>> = (0..k)
                .map(|_| {
                    (0..NV * VD)
                        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
                        .collect()
                })
                .collect();
            let gate: Vec<Vec<f32>> = (0..k)
                .map(|_| (0..NV).map(|_| r.r(0.80, 0.999) as f32).collect())
                .collect();
            let beta: Vec<Vec<f32>> = (0..k)
                .map(|_| (0..NV).map(|_| r.r(0.0, 1.0) as f32).collect())
                .collect();

            let (ref_out, ref_h) = sequential_ref(&h0, &q, &key, &val, &gate, &beta, k);
            let (wy_out, wy_inter, wy_final) = run_wy(g, wk, &h0, &q, &key, &val, &gate, &beta, k)?;

            // Per-token output cosine: wy_out is interleaved [token, vh, vd].
            let mut min_out_cos = 1.0f64;
            for t in 0..k {
                let wy_t = &wy_out[t * NV * VD..(t + 1) * NV * VD];
                min_out_cos = min_out_cos.min(cos(wy_t, &ref_out[t]));
            }

            // Committed-state cosine: prefix-n committed = H after token n-1.
            //   prefix n=k (full accept) → final H == ref_h[k-1]
            //   prefix n<k             → inter[n-1] == ref_h[n-1]
            let mut min_state_cos = cos(&wy_final, &ref_h[k - 1]);
            for n in 1..k {
                min_state_cos = min_state_cos.min(cos(&wy_inter[n - 1], &ref_h[n - 1]));
            }

            let ok = min_out_cos >= PASS_COS && min_state_cos >= PASS_COS;
            all_ok &= ok;
            eprintln!(
                "K={k} layer={layer}  out_cos={min_out_cos:.7} state_cos={min_state_cos:.7}  {}",
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    eprintln!(
        "\nWY-verify equivalence GATE: {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
