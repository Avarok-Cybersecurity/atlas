// SPDX-License-Identifier: AGPL-3.0-only
//! #435 route-(a) COST microbenchmark — conv+GDN+norm phase of the MTP verify
//! step, real production kernels, real Qwen3.6-27B GDN shape. Untracked
//! scratch example.
//!
//! Per (K, n) it times the whole phase as launched in production:
//!
//!   FUSED (today's verify):
//!     n=1:  K x conv_bf16 + (K-1) x d2d(conv snapshot) + 1 x wy{K}
//!           + 1 x gated_rms_norm over K rows
//!     n>1:  1 x gdn_verify_fused_conv_kn_batched + 1 x wy{K}(batch=n)
//!           + 1 x gated_rms_norm over n*K rows
//!   EXACT (route a — decode numerics, per token):
//!     n=1:  K x conv_f32 + K x gated_delta_rule_decode_f32_norm
//!           + (K-1) x d2d(conv) + (K-1) x d2d(h_state, 3 MiB)
//!     n>1:  K x conv_f32_strided(batch=n)
//!           + K x gated_delta_rule_decode_f32_strided_norm(batch=n)
//!           + (K-1) x d2d(n*conv) + (K-1) x d2d(n*h)
//!
//! Components are also timed in isolation (conv arms, GDN arms, d2d) so the
//! "seq-exact resident kernel" variant of route (a) (intermediates written
//! inline, no d2d) can be bounded without building it first.
//!
//!   cargo run -p spark-model --release --example x435_ssm_cost \
//!       --features cuda,gpu-examples

use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use std::time::Instant;

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const KEY_DIM: usize = NK * KD; // 2048
const VALUE_DIM: usize = NV * VD; // 6144
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM; // 10240
const QKVZ: usize = CONV_DIM + VALUE_DIM; // 16384 (production deint row)
const D_CONV: usize = 4;
const H_NUMEL: usize = NV * KD * VD; // 786432 (3 MiB)
const CONV_ST: usize = CONV_DIM * D_CONV; // 40960 f32 (160 KiB)
const WARMUP: usize = 30;
const ITERS: usize = 300;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn alloc_fill_bf16(g: &dyn GpuBackend, n: usize, r: &mut Lcg) -> Result<DevicePtr> {
    let b: Vec<u8> = (0..n)
        .flat_map(|_| bf16::from_f64(r.f() * 2.0 - 1.0).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn alloc_fill_f32(
    g: &dyn GpuBackend,
    n: usize,
    lo: f32,
    hi: f32,
    r: &mut Lcg,
) -> Result<DevicePtr> {
    let b: Vec<u8> = (0..n)
        .flat_map(|_| (lo + (hi - lo) * r.f() as f32).to_le_bytes())
        .collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

struct Kit {
    conv_b: KernelHandle,
    conv_f: KernelHandle,
    conv_f_str: KernelHandle,
    conv_kn_batched: KernelHandle,
    gdn_f32_norm: KernelHandle,
    gdn_f32_str_norm: KernelHandle,
    wy2: KernelHandle,
    wy4: KernelHandle,
    grms: KernelHandle,
    gdn_snap_str: KernelHandle,
}

#[allow(clippy::too_many_arguments)]
struct Bufs {
    n: usize,
    k: usize,
    conv_state: DevicePtr, // n * CONV_ST f32 (contiguous slots)
    input: DevicePtr,      // n * k * CONV_DIM bf16 (rows CONV_DIM apart, seq-major)
    wconv: DevicePtr,
    conv_out_b: DevicePtr, // n * k * CONV_DIM bf16
    conv_out_f: DevicePtr, // n * k * CONV_DIM f32
    conv_inter: DevicePtr, // n * k * CONV_ST f32
    h: DevicePtr,          // n * H_NUMEL f32
    inter: [DevicePtr; 3], // n * H_NUMEL f32 each (oversized, timing only)
    gates: DevicePtr,      // n * k * 2*NV f32
    z: DevicePtr,          // n * k * VALUE_DIM bf16
    normw: DevicePtr,      // VALUE_DIM bf16
    gdn_out: DevicePtr,    // n * k * VALUE_DIM bf16
    normed: DevicePtr,     // n * k * VALUE_DIM bf16
}

fn fused_arm(g: &dyn GpuBackend, kit: &Kit, b: &Bufs) -> Result<()> {
    let (n, k) = (b.n, b.k);
    if n == 1 {
        for t in 0..k {
            KernelLaunch::new(g, kit.conv_b)
                .grid([div_ceil(CONV_DIM as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.conv_state)
                .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                .arg_ptr(b.wconv)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(b.conv_out_b.offset(t * CONV_DIM * 2))
                .arg_u32(1)
                .arg_u32(CONV_DIM as u32)
                .arg_u32(D_CONV as u32)
                .arg_u32((2 * KEY_DIM) as u32)
                .arg_u32(KD as u32)
                .arg_f32(1e-6)
                .launch(0)?;
            if t + 1 < k {
                g.copy_d2d_async(
                    b.conv_state,
                    b.conv_inter.offset(t * CONV_ST * 4),
                    CONV_ST * 4,
                    0,
                )?;
            }
        }
    } else {
        KernelLaunch::new(g, kit.conv_kn_batched)
            .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(b.conv_state)
            .arg_ptr(b.input)
            .arg_ptr(b.wconv)
            .arg_ptr(b.conv_out_b)
            .arg_ptr(b.conv_inter)
            .arg_u32(k as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(D_CONV as u32)
            .arg_u32((2 * KEY_DIM) as u32)
            .arg_u32(KD as u32)
            .arg_u32(CONV_DIM as u32) // input stride (tokens)
            .arg_u32(CONV_DIM as u32) // output stride (tokens)
            .arg_u32(CONV_ST as u32) // inter stride (tokens)
            .arg_f32(1e-6)
            .arg_u32(CONV_ST as u32) // conv_state seq stride
            .arg_u32((k * CONV_DIM) as u32) // input seq stride
            .arg_u32((k * CONV_DIM) as u32) // output seq stride
            .arg_u32((k * CONV_ST) as u32) // inter seq stride
            .launch(0)?;
    }
    // wy{K}: rows (seq-major) — (b*K+T)*qk_stride indexing matches layout.
    let wy = if k == 2 { kit.wy2 } else { kit.wy4 };
    let mut l = KernelLaunch::new(g, wy)
        .grid([NV as u32, n as u32, 1])
        .block([128, 1, 1])
        .arg_ptr(b.h)
        .arg_ptr(b.conv_out_b)
        .arg_ptr(b.conv_out_b.offset(KEY_DIM * 2))
        .arg_ptr(b.conv_out_b.offset(2 * KEY_DIM * 2))
        .arg_ptr(b.gates)
        .arg_ptr(b.gates.offset(NV * 4))
        .arg_ptr(b.gdn_out);
    for i in 0..(k - 1) {
        l = l.arg_ptr(b.inter[i]);
    }
    l.arg_u32(n as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32((2 * NV) as u32)
        .arg_u32(0)
        .launch(0)?;
    // Phase-8 norm over n*k rows.
    KernelLaunch::new(g, kit.grms)
        .grid([(n * k) as u32, 1, 1])
        .block([1024, 1, 1])
        .arg_ptr(b.gdn_out)
        .arg_ptr(b.z)
        .arg_ptr(b.normw)
        .arg_ptr(b.normed)
        .arg_u32(VALUE_DIM as u32)
        .arg_f32(1e-6)
        .arg_u32(VALUE_DIM as u32)
        .arg_u32(0)
        .launch(0)?;
    Ok(())
}

fn exact_arm(g: &dyn GpuBackend, kit: &Kit, b: &Bufs, with_d2d: bool) -> Result<()> {
    let (n, k) = (b.n, b.k);
    for t in 0..k {
        if n == 1 {
            KernelLaunch::new(g, kit.conv_f)
                .grid([div_ceil(CONV_DIM as u32, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.conv_state)
                .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                .arg_ptr(b.wconv)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(b.conv_out_f.offset(t * CONV_DIM * 4))
                .arg_u32(1)
                .arg_u32(CONV_DIM as u32)
                .arg_u32(D_CONV as u32)
                .arg_u32((2 * KEY_DIM) as u32)
                .arg_u32(KD as u32)
                .arg_f32(1e-6)
                .launch(0)?;
            let ot = b.conv_out_f.offset(t * CONV_DIM * 4);
            KernelLaunch::new(g, kit.gdn_f32_norm)
                .grid([NV as u32, 1, 1])
                .block([128, 1, 1])
                .arg_ptr(b.h)
                .arg_ptr(ot)
                .arg_ptr(ot.offset(KEY_DIM * 4))
                .arg_ptr(ot.offset(2 * KEY_DIM * 4))
                .arg_ptr(b.gates.offset(t * 2 * NV * 4))
                .arg_ptr(b.gates.offset((t * 2 * NV + NV) * 4))
                .arg_ptr(b.z.offset(t * VALUE_DIM * 2))
                .arg_ptr(b.normw)
                .arg_ptr(b.normed.offset(t * VALUE_DIM * 2))
                .arg_u32(1)
                .arg_u32(NK as u32)
                .arg_u32(NV as u32)
                .arg_u32(KD as u32)
                .arg_u32(VD as u32)
                .arg_f32(1e-6)
                .launch(0)?;
        } else {
            // Strided twins: one launch per token covering all n seqs.
            // Input rows for token t of seq s live at (s*k + t)*CONV_DIM.
            KernelLaunch::new(g, kit.conv_f_str)
                .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(b.conv_state)
                .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                .arg_ptr(b.wconv)
                .arg_ptr(DevicePtr::NULL)
                .arg_ptr(b.conv_out_f.offset(t * CONV_DIM * 4))
                .arg_u32(n as u32)
                .arg_u32(CONV_DIM as u32)
                .arg_u32(D_CONV as u32)
                .arg_u32((2 * KEY_DIM) as u32)
                .arg_u32(KD as u32)
                .arg_f32(1e-6)
                .arg_u32((k * CONV_DIM) as u32) // input seq stride
                .arg_u32((k * CONV_DIM) as u32) // output seq stride
                .launch(0)?;
            let ot = b.conv_out_f.offset(t * CONV_DIM * 4);
            KernelLaunch::new(g, kit.gdn_f32_str_norm)
                .grid([NV as u32, n as u32, 1])
                .block([128, 1, 1])
                .arg_ptr(b.h)
                .arg_ptr(ot)
                .arg_ptr(ot.offset(KEY_DIM * 4))
                .arg_ptr(ot.offset(2 * KEY_DIM * 4))
                .arg_ptr(b.gates.offset(t * 2 * NV * 4))
                .arg_ptr(b.gates.offset((t * 2 * NV + NV) * 4))
                .arg_ptr(b.z.offset(t * VALUE_DIM * 2))
                .arg_ptr(b.normw)
                .arg_ptr(b.normed.offset(t * VALUE_DIM * 2))
                .arg_u32(n as u32)
                .arg_u32(NK as u32)
                .arg_u32(NV as u32)
                .arg_u32(KD as u32)
                .arg_u32(VD as u32)
                .arg_u32((k * CONV_DIM) as u32) // qk seq stride
                .arg_u32((k * CONV_DIM) as u32) // v seq stride
                .arg_u32((k * 2 * NV) as u32) // gb seq stride
                .arg_u32((k * VALUE_DIM) as u32) // z seq stride
                .arg_u32((k * VALUE_DIM) as u32) // out seq stride
                .arg_f32(1e-6)
                .launch(0)?;
        }
        if with_d2d && t + 1 < k {
            g.copy_d2d_async(
                b.conv_state,
                b.conv_inter.offset(t * n * CONV_ST * 4),
                n * CONV_ST * 4,
                0,
            )?;
            g.copy_d2d_async(b.h, b.inter[t.min(2)], n * H_NUMEL * 4, 0)?;
        }
    }
    Ok(())
}

fn time_arm<F: FnMut() -> Result<()>>(g: &dyn GpuBackend, mut f: F) -> Result<f64> {
    for _ in 0..WARMUP {
        f()?;
    }
    g.synchronize(0)?;
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}

fn main() -> Result<()> {
    let set = atlas_kernels::ptx_for_model("qwen3.6-27b")
        .or_else(|| {
            atlas_kernels::ptx_for_shape(atlas_kernels::ModelShape {
                model_type: "qwen3_5_text",
                hidden_size: 5120,
                mtp_layers: 0,
            })
        })
        .expect("no qwen3.6-27b ptx set");
    eprintln!("kernel set: {}", set.target.model);
    let g0 = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &g0;
    let kit = Kit {
        conv_b: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
        conv_f: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")?,
        conv_f_str: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_strided")?,
        conv_kn_batched: g.kernel(
            "gdn_verify_fused_conv_kn",
            "gdn_verify_fused_conv_kn_batched",
        )?,
        gdn_f32_norm: g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_norm")?,
        gdn_f32_str_norm: g.kernel(
            "gated_delta_rule",
            "gated_delta_rule_decode_f32_strided_norm",
        )?,
        wy2: g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
        wy4: g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
        grms: g.kernel("norm", "gated_rms_norm")?,
        gdn_snap_str: g.kernel(
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_strided_norm_snap",
        )?,
    };
    let mut r = Lcg(0xC057);

    println!(
        "K, n, fused_us, exact_us, exact_nod2d_us, snap_shipped_us, snap_delta_pct, exact_delta_pct, conv_b_us, conv_f_us, wy_us, gdnseq_us, d2d_us"
    );
    for &k in &[2usize, 4] {
        for &n in &[1usize, 4, 8, 16, 32] {
            let b = Bufs {
                n,
                k,
                conv_state: alloc_fill_f32(g, n * CONV_ST, -1.0, 1.0, &mut r)?,
                input: alloc_fill_bf16(g, n * k * CONV_DIM, &mut r)?,
                wconv: alloc_fill_bf16(g, CONV_DIM * D_CONV, &mut r)?,
                conv_out_b: g.alloc(n * k * CONV_DIM * 2)?,
                conv_out_f: g.alloc(n * k * CONV_DIM * 4)?,
                conv_inter: g.alloc(n * k * CONV_ST * 4)?,
                h: alloc_fill_f32(g, n * H_NUMEL, -0.1, 0.1, &mut r)?,
                inter: [
                    alloc_fill_f32(g, n * H_NUMEL, 0.0, 0.0, &mut r)?,
                    alloc_fill_f32(g, n * H_NUMEL, 0.0, 0.0, &mut r)?,
                    alloc_fill_f32(g, n * H_NUMEL, 0.0, 0.0, &mut r)?,
                ],
                gates: alloc_fill_f32(g, n * k * 2 * NV, 0.5, 0.999, &mut r)?,
                z: alloc_fill_bf16(g, n * k * VALUE_DIM, &mut r)?,
                normw: alloc_fill_bf16(g, VALUE_DIM, &mut r)?,
                gdn_out: g.alloc(n * k * VALUE_DIM * 2)?,
                normed: g.alloc(n * k * VALUE_DIM * 2)?,
            };
            let fused = time_arm(g, || fused_arm(g, &kit, &b))?;
            let exact = time_arm(g, || exact_arm(g, &kit, &b, true))?;
            let exact_nod2d = time_arm(g, || exact_arm(g, &kit, &b, false))?;
            // EXACT-with-inline-snap: the SHIPPED #435 strided arm, launched
            // exactly as in verify_exact_microtest/strided.rs — per token one
            // conv_f32_strided + one strided_norm_snap (inline h snapshot for
            // t<K-1), plus (K-1) conv-state d2d snapshots.
            let snap_shipped = if n > 1 {
                let deint = alloc_fill_bf16(g, n * k * QKVZ, &mut r)?;
                let conv_scratch = g.alloc(n * QKVZ * 4)?;
                let inter_seq_stride = (k - 1) * H_NUMEL * 4;
                let h_inters = g.alloc(n * inter_seq_stride)?;
                Some(time_arm(g, || {
                    for t in 0..k {
                        KernelLaunch::new(g, kit.conv_f_str)
                            .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
                            .block([256, 1, 1])
                            .arg_ptr(b.conv_state)
                            .arg_ptr(deint.offset(t * QKVZ * 2))
                            .arg_ptr(b.wconv)
                            .arg_ptr(DevicePtr::NULL)
                            .arg_ptr(conv_scratch)
                            .arg_u32(n as u32)
                            .arg_u32(CONV_DIM as u32)
                            .arg_u32(D_CONV as u32)
                            .arg_u32((2 * KEY_DIM) as u32)
                            .arg_u32(KD as u32)
                            .arg_f32(1e-6)
                            .arg_u32((k * QKVZ) as u32)
                            .arg_u32(QKVZ as u32)
                            .launch(0)?;
                        let snapshot = t + 1 < k;
                        let (hi, stride) = if snapshot {
                            (
                                h_inters.offset(t * H_NUMEL * 4),
                                (inter_seq_stride / 4) as u64,
                            )
                        } else {
                            (DevicePtr::NULL, 0)
                        };
                        KernelLaunch::new(g, kit.gdn_snap_str)
                            .grid([NV as u32, n as u32, 1])
                            .block([128, 1, 1])
                            .arg_ptr(b.h)
                            .arg_ptr(conv_scratch)
                            .arg_ptr(conv_scratch.offset(KEY_DIM * 4))
                            .arg_ptr(conv_scratch.offset(2 * KEY_DIM * 4))
                            .arg_ptr(b.gates.offset(t * 2 * NV * 4))
                            .arg_ptr(b.gates.offset((t * 2 * NV + NV) * 4))
                            .arg_ptr(deint.offset((t * QKVZ + CONV_DIM) * 2))
                            .arg_ptr(b.normw)
                            .arg_ptr(b.normed.offset(t * VALUE_DIM * 2))
                            .arg_u32(n as u32)
                            .arg_u32(NK as u32)
                            .arg_u32(NV as u32)
                            .arg_u32(KD as u32)
                            .arg_u32(VD as u32)
                            .arg_u32(QKVZ as u32)
                            .arg_u32(QKVZ as u32)
                            .arg_u32((k * NV * 2) as u32)
                            .arg_u32((k * QKVZ) as u32)
                            .arg_u32((k * VALUE_DIM) as u32)
                            .arg_f32(1e-6)
                            .arg_ptr(hi)
                            .arg_u64(stride)
                            .launch(0)?;
                        if snapshot {
                            g.copy_d2d_async(
                                b.conv_state,
                                b.conv_inter.offset(t * n * CONV_ST * 4),
                                n * CONV_ST * 4,
                                0,
                            )?;
                        }
                    }
                    Ok(())
                })?)
            } else {
                None
            };
            // Components.
            let conv_b_t = time_arm(g, || {
                for t in 0..k {
                    KernelLaunch::new(g, kit.conv_b)
                        .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(b.conv_state)
                        .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                        .arg_ptr(b.wconv)
                        .arg_ptr(DevicePtr::NULL)
                        .arg_ptr(b.conv_out_b.offset(t * CONV_DIM * 2))
                        .arg_u32(n as u32)
                        .arg_u32(CONV_DIM as u32)
                        .arg_u32(D_CONV as u32)
                        .arg_u32((2 * KEY_DIM) as u32)
                        .arg_u32(KD as u32)
                        .arg_f32(1e-6)
                        .launch(0)?;
                }
                Ok(())
            })?;
            let conv_f_t = time_arm(g, || {
                for t in 0..k {
                    KernelLaunch::new(g, kit.conv_f)
                        .grid([div_ceil(CONV_DIM as u32, 256), n as u32, 1])
                        .block([256, 1, 1])
                        .arg_ptr(b.conv_state)
                        .arg_ptr(b.input.offset(t * CONV_DIM * 2))
                        .arg_ptr(b.wconv)
                        .arg_ptr(DevicePtr::NULL)
                        .arg_ptr(b.conv_out_f.offset(t * CONV_DIM * 4))
                        .arg_u32(n as u32)
                        .arg_u32(CONV_DIM as u32)
                        .arg_u32(D_CONV as u32)
                        .arg_u32((2 * KEY_DIM) as u32)
                        .arg_u32(KD as u32)
                        .arg_f32(1e-6)
                        .launch(0)?;
                }
                Ok(())
            })?;
            let wy_t = time_arm(g, || {
                let wy = if k == 2 { kit.wy2 } else { kit.wy4 };
                let mut l = KernelLaunch::new(g, wy)
                    .grid([NV as u32, n as u32, 1])
                    .block([128, 1, 1])
                    .arg_ptr(b.h)
                    .arg_ptr(b.conv_out_b)
                    .arg_ptr(b.conv_out_b.offset(KEY_DIM * 2))
                    .arg_ptr(b.conv_out_b.offset(2 * KEY_DIM * 2))
                    .arg_ptr(b.gates)
                    .arg_ptr(b.gates.offset(NV * 4))
                    .arg_ptr(b.gdn_out);
                for i in 0..(k - 1) {
                    l = l.arg_ptr(b.inter[i]);
                }
                l.arg_u32(n as u32)
                    .arg_u32(NK as u32)
                    .arg_u32(NV as u32)
                    .arg_u32(KD as u32)
                    .arg_u32(VD as u32)
                    .arg_u32(CONV_DIM as u32)
                    .arg_u32(CONV_DIM as u32)
                    .arg_u32((2 * NV) as u32)
                    .arg_u32(0)
                    .launch(0)
            })?;
            let gdnseq_t = time_arm(g, || {
                for t in 0..k {
                    let ot = b.conv_out_f.offset(t * CONV_DIM * 4);
                    let kern = if n == 1 {
                        kit.gdn_f32_norm
                    } else {
                        kit.gdn_f32_str_norm
                    };
                    let mut l = KernelLaunch::new(g, kern)
                        .grid([NV as u32, n as u32, 1])
                        .block([128, 1, 1])
                        .arg_ptr(b.h)
                        .arg_ptr(ot)
                        .arg_ptr(ot.offset(KEY_DIM * 4))
                        .arg_ptr(ot.offset(2 * KEY_DIM * 4))
                        .arg_ptr(b.gates.offset(t * 2 * NV * 4))
                        .arg_ptr(b.gates.offset((t * 2 * NV + NV) * 4))
                        .arg_ptr(b.z.offset(t * VALUE_DIM * 2))
                        .arg_ptr(b.normw)
                        .arg_ptr(b.normed.offset(t * VALUE_DIM * 2))
                        .arg_u32(n as u32)
                        .arg_u32(NK as u32)
                        .arg_u32(NV as u32)
                        .arg_u32(KD as u32)
                        .arg_u32(VD as u32);
                    if n > 1 {
                        l = l
                            .arg_u32((k * CONV_DIM) as u32)
                            .arg_u32((k * CONV_DIM) as u32)
                            .arg_u32((k * 2 * NV) as u32)
                            .arg_u32((k * VALUE_DIM) as u32)
                            .arg_u32((k * VALUE_DIM) as u32);
                    }
                    l.arg_f32(1e-6).launch(0)?;
                }
                Ok(())
            })?;
            let d2d_t = time_arm(g, || {
                for t in 0..(k - 1) {
                    g.copy_d2d_async(
                        b.conv_state,
                        b.conv_inter.offset(t * n * CONV_ST * 4),
                        n * CONV_ST * 4,
                        0,
                    )?;
                    g.copy_d2d_async(b.h, b.inter[t.min(2)], n * H_NUMEL * 4, 0)?;
                }
                Ok(())
            })?;
            let (snap_s, snap_d) = match snap_shipped {
                Some(s) => (
                    format!("{s:.1}"),
                    format!("{:+.1}%", (s - fused) / fused * 100.0),
                ),
                None => ("-".into(), "-".into()),
            };
            println!(
                "{k}, {n}, {fused:.1}, {exact:.1}, {exact_nod2d:.1}, {snap_s}, {snap_d}, {:+.1}%, {conv_b_t:.1}, {conv_f_t:.1}, {wy_t:.1}, {gdnseq_t:.1}, {d2d_t:.1}",
                (exact - fused) / fused * 100.0
            );
        }
    }
    Ok(())
}
