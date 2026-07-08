// SPDX-License-Identifier: AGPL-3.0-only
//! Equivalence test for `gated_delta_rule_decode_f32_strided_norm`.
//!
//! Compares the new fused strided GDN+gated-RMS kernel against:
//!   gated_delta_rule_decode_f32_strided -> gated_rms_norm_f32_input
//!
//! Uses Holo/Qwen3.6 GDN dimensions with synthetic batch=4 inputs.

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

const B: usize = 4;
const NK: usize = 16;
const NV: usize = 32;
const KD: usize = 128;
const VD: usize = 128;
const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM;
const QKVZ_SIZE: usize = CONV_DIM + VALUE_DIM;
const GB_STRIDE: usize = NV * 2;
const RMS_EPS: f32 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }

    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
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

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[allow(clippy::too_many_arguments)]
fn launch_strided(
    g: &dyn GpuBackend,
    k: KernelHandle,
    h: DevicePtr,
    conv: DevicePtr,
    gates: DevicePtr,
    gdn_out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, B as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(conv)
        .arg_ptr(conv.offset(KEY_DIM * 4))
        .arg_ptr(conv.offset(KEY_DIM * 2 * 4))
        .arg_ptr(gates)
        .arg_ptr(gates.offset(NV * 4))
        .arg_ptr(gdn_out)
        .arg_u32(B as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(GB_STRIDE as u32)
        .arg_u32(VALUE_DIM as u32)
        .launch(0)
}

fn launch_norm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    input: DevicePtr,
    z_gate: DevicePtr,
    norm_w: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([VD as u32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(z_gate)
        .arg_ptr(norm_w)
        .arg_ptr(out)
        .arg_u32(VD as u32)
        .arg_f32(RMS_EPS)
        .arg_u32(VD as u32)
        .arg_u32(VD as u32)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn launch_fused(
    g: &dyn GpuBackend,
    k: KernelHandle,
    h: DevicePtr,
    conv: DevicePtr,
    gates: DevicePtr,
    z: DevicePtr,
    norm_w: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, B as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(conv)
        .arg_ptr(conv.offset(KEY_DIM * 4))
        .arg_ptr(conv.offset(KEY_DIM * 2 * 4))
        .arg_ptr(gates)
        .arg_ptr(gates.offset(NV * 4))
        .arg_ptr(z)
        .arg_ptr(norm_w)
        .arg_ptr(out)
        .arg_u32(B as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(GB_STRIDE as u32)
        .arg_u32(QKVZ_SIZE as u32)
        .arg_u32(VALUE_DIM as u32)
        .arg_f32(RMS_EPS)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let old_k = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided")?;
    let fused_k = g.kernel(
        "gated_delta_rule",
        "gated_delta_rule_decode_f32_strided_norm",
    )?;
    let norm_k = g.kernel("norm", "gated_rms_norm_f32_input")?;

    let mut rng = Lcg(0x5155_4d5f_c4f0_0001);
    let h0: Vec<f32> = (0..B * NV * KD * VD).map(|_| rng.r(-0.05, 0.05)).collect();
    let conv: Vec<f32> = (0..B * CONV_DIM).map(|_| rng.r(-0.5, 0.5)).collect();
    let mut gates = Vec::with_capacity(B * GB_STRIDE);
    for _ in 0..B {
        for _ in 0..NV {
            gates.push(rng.r(0.80, 0.999));
        }
        for _ in 0..NV {
            gates.push(rng.r(0.0, 1.0));
        }
    }
    let z: Vec<bf16> = (0..B * QKVZ_SIZE)
        .map(|_| bf16::from_f32(rng.r(-0.5, 0.5)))
        .collect();
    let norm_w: Vec<bf16> = (0..VD).map(|_| bf16::from_f32(rng.r(0.5, 1.5))).collect();

    let h_ref = up_f32(g, &h0)?;
    let h_fused = up_f32(g, &h0)?;
    let conv_d = up_f32(g, &conv)?;
    let gates_d = up_f32(g, &gates)?;
    let z_d = up_bf16(g, &z)?;
    let norm_w_d = up_bf16(g, &norm_w)?;
    let gdn_ref = g.alloc(B * VALUE_DIM * 4)?;
    let out_ref = g.alloc(B * VALUE_DIM * 2)?;
    let out_fused = g.alloc(B * VALUE_DIM * 2)?;

    launch_strided(g, old_k, h_ref, conv_d, gates_d, gdn_ref)?;
    for b in 0..B {
        launch_norm(
            g,
            norm_k,
            gdn_ref.offset(b * VALUE_DIM * 4),
            z_d.offset(b * QKVZ_SIZE * 2 + CONV_DIM * 2),
            norm_w_d,
            out_ref.offset(b * VALUE_DIM * 2),
        )?;
    }
    launch_fused(
        g,
        fused_k,
        h_fused,
        conv_d,
        gates_d,
        z_d.offset(CONV_DIM * 2),
        norm_w_d,
        out_fused,
    )?;
    g.synchronize(0)?;

    let out_a = dn_bf16(g, out_ref, B * VALUE_DIM)?;
    let out_b = dn_bf16(g, out_fused, B * VALUE_DIM)?;
    let h_a = dn_f32(g, h_ref, B * NV * KD * VD)?;
    let h_b = dn_f32(g, h_fused, B * NV * KD * VD)?;

    let out_cos = cos(&out_a, &out_b);
    let h_cos = cos(&h_a, &h_b);
    let out_max = max_abs(&out_a, &out_b);
    let h_max = max_abs(&h_a, &h_b);
    eprintln!("gdn_strided_norm_microtest batch={B}");
    eprintln!("  output cos={out_cos:.9} max_abs={out_max:.8}");
    eprintln!("  h_state cos={h_cos:.9} max_abs={h_max:.8}");

    anyhow::ensure!(out_cos >= 0.99999, "output cosine below threshold");
    anyhow::ensure!(h_cos >= 0.999999, "h_state cosine below threshold");
    Ok(())
}
