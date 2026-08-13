// SPDX-License-Identifier: AGPL-3.0-only
//! Bitwise gate for the exact MTP-verify arm (issue #435 route (a)).
//!
//! GOLDEN is the sequential single-token decode chain — per token,
//! `causal_conv1d_update_l2norm_f32` then `gated_delta_rule_decode_f32_norm`
//! — with the h/conv state captured after every token. The legs assert:
//!
//! 1. SNAP: per-token conv_f32 + `gated_delta_rule_decode_f32_norm_snap`
//!    (the exact arm's kernel) — final H, final conv_state, normed outputs
//!    AND every inline h snapshot must be BYTE-IDENTICAL to golden.
//! 2. FUSED CONV F32: one `gdn_verify_fused_conv_kn_f32` launch (conv rows +
//!    inline conv snapshots) + per-token snap GDN — byte-identical to golden.
//! 3. STRIDED (batch=n): per token, one `causal_conv1d_update_l2norm_f32_strided`
//!    + one `gated_delta_rule_decode_f32_strided_norm_snap` over n sequences —
//!    each sequence byte-identical to its own golden run.
//! 4. NEGATIVE CONTROL (the pre-#435 verify — the DEFAULT WY arms; the exact
//!    chain under test here is the `--exact-verify` OPT-IN):
//!    BF16 conv (`causal_conv1d_update_l2norm`) + `gated_delta_rule_wy4`
//!    — the final H MUST DIFFER from golden (measured ~8.6e-4 relL2 on these
//!    shapes). If it matches, this gate is not exercising the defect and the
//!    positive legs prove nothing.
//!
//! GPU REQUIRED — this example cannot run in a CPU-only session. Build with
//! the qwen3.6-27b kernels present (`ATLAS_TARGET_MODEL="*"` or the 27b
//! target), then:
//!   cargo run -p spark-model --release --example verify_exact_microtest \
//!       --features cuda,gpu-examples
//! Exit codes: 0 = pass, 1 = FAIL, 2 = skipped (snap kernels not in this
//! build's PTX set — the gate did NOT run).
use anyhow::Result;
use half::bf16;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

mod strided;
use strided::run_strided;

// Qwen3.6-27B GDN shapes (issue #435 measurement config).
pub(crate) const KD: usize = 128;
pub(crate) const NK: usize = 16;
pub(crate) const NV: usize = 48;
pub(crate) const VD: usize = 128;
pub(crate) const D_CONV: usize = 4;
pub(crate) const KEY_DIM: usize = NK * KD; // 2048
pub(crate) const VALUE_DIM: usize = NV * VD; // 6144
pub(crate) const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM; // 10240
pub(crate) const QK_CH: usize = KEY_DIM * 2; // 4096
pub(crate) const QKVZ: usize = CONV_DIM + VALUE_DIM; // 16384
pub(crate) const K: usize = 4; // verify width (num_drafts=3 rung)
pub(crate) const N: usize = 3; // strided-leg batch
pub(crate) const H_ELEMS: usize = NV * KD * VD;
pub(crate) const CONV_ELEMS: usize = CONV_DIM * D_CONV;
pub(crate) const EPS: f32 = 1e-6;

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

pub(crate) fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn bf16v(r: &mut Lcg, n: usize, lo: f64, hi: f64) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16::from_f64(r.r(lo, hi)).to_bits().to_le_bytes())
        .collect()
}
fn f32v(r: &mut Lcg, n: usize, lo: f64, hi: f64) -> Vec<u8> {
    (0..n)
        .flat_map(|_| (r.r(lo, hi) as f32).to_le_bytes())
        .collect()
}
pub(crate) fn dn(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn rel_l2(a: &[u8], b: &[u8]) -> f64 {
    let f = |x: &[u8]| -> Vec<f64> {
        x.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect()
    };
    let (a, b) = (f(a), f(b));
    let d: f64 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
    let n: f64 = a.iter().map(|x| x * x).sum();
    (d / (n + 1e-30)).sqrt()
}

/// Per-sequence inputs; gates hold [gate(NV) | beta(NV)] FP32 per token.
pub(crate) struct Inputs {
    pub(crate) deint: Vec<u8>,  // [K, QKVZ] BF16
    pub(crate) conv0: Vec<u8>,  // [CONV_ELEMS] FP32
    pub(crate) h0: Vec<u8>,     // [H_ELEMS] FP32
    pub(crate) gates: Vec<u8>,  // [K, 2*NV] FP32
    pub(crate) weight: Vec<u8>, // conv weight [CONV_DIM, D_CONV] BF16
    pub(crate) norm_w: Vec<u8>, // [VD] BF16
}
fn gen_inputs(seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    Inputs {
        deint: bf16v(&mut r, K * QKVZ, -0.5, 0.5),
        conv0: f32v(&mut r, CONV_ELEMS, -0.3, 0.3),
        h0: f32v(&mut r, H_ELEMS, -0.2, 0.2),
        gates: f32v(&mut r, K * 2 * NV, 0.05, 0.95),
        weight: bf16v(&mut r, CONV_ELEMS, -0.3, 0.3),
        norm_w: bf16v(&mut r, VD, 0.5, 1.5),
    }
}

pub(crate) struct Kernels {
    pub(crate) conv_f32: KernelHandle,
    pub(crate) conv_bf16: KernelHandle,
    pub(crate) conv_f32_strided: KernelHandle,
    pub(crate) f32_norm: KernelHandle,
    pub(crate) snap: KernelHandle,
    pub(crate) snap_strided: KernelHandle,
    pub(crate) fused_conv_f32: KernelHandle,
    pub(crate) wy4: KernelHandle,
}

#[allow(clippy::too_many_arguments)]
fn conv_launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([CONV_DIM.div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_f32(EPS)
        .launch(0)
}

/// f32_norm / snap launch (batch=1). `h_inter` NULL for the parent kernel or
/// the last token.
#[allow(clippy::too_many_arguments)]
fn gdn_launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    snap: bool,
    h: DevicePtr,
    conv_row: DevicePtr,
    gate: DevicePtr,
    z: DevicePtr,
    norm_w: DevicePtr,
    out: DevicePtr,
    h_inter: DevicePtr,
) -> Result<()> {
    let mut l = KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(conv_row)
        .arg_ptr(conv_row.offset(KEY_DIM * 4))
        .arg_ptr(conv_row.offset(KEY_DIM * 2 * 4))
        .arg_ptr(gate)
        .arg_ptr(gate.offset(NV * 4))
        .arg_ptr(z)
        .arg_ptr(norm_w)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_f32(EPS);
    if snap {
        l = l.arg_ptr(h_inter);
    }
    l.launch(0)
}

struct Golden {
    h_after: Vec<Vec<u8>>,    // K snapshots of H (post-token)
    conv_after: Vec<Vec<u8>>, // K snapshots of conv_state
    normed: Vec<u8>,          // [K, VALUE_DIM] BF16
}

/// The sequential decode reference: per-token conv_f32 + f32_norm, states
/// captured after every token (device sync + d2h — slow, and irrelevant:
/// this is an oracle, not a benchmark).
fn run_golden(g: &dyn GpuBackend, ks: &Kernels, inp: &Inputs) -> Result<Golden> {
    let state = up(g, &inp.conv0)?;
    let h = up(g, &inp.h0)?;
    let deint = up(g, &inp.deint)?;
    let gates = up(g, &inp.gates)?;
    let (w, nw) = (up(g, &inp.weight)?, up(g, &inp.norm_w)?);
    let conv_rows = g.alloc(K * QKVZ * 4)?;
    let normed = g.alloc(K * VALUE_DIM * 2)?;
    let (mut h_after, mut conv_after) = (Vec::new(), Vec::new());
    for t in 0..K {
        let row = conv_rows.offset(t * QKVZ * 4);
        conv_launch(g, ks.conv_f32, state, deint.offset(t * QKVZ * 2), w, row)?;
        gdn_launch(
            g,
            ks.f32_norm,
            false,
            h,
            row,
            gates.offset(t * 2 * NV * 4),
            deint.offset((t * QKVZ + CONV_DIM) * 2),
            nw,
            normed.offset(t * VALUE_DIM * 2),
            DevicePtr::NULL,
        )?;
        g.synchronize(0)?;
        h_after.push(dn(g, h, H_ELEMS * 4)?);
        conv_after.push(dn(g, state, CONV_ELEMS * 4)?);
    }
    Ok(Golden {
        h_after,
        conv_after,
        normed: dn(g, normed, K * VALUE_DIM * 2)?,
    })
}

/// The exact verify arm: `fused_conv` selects the one-launch FP32 conv with
/// inline snapshots vs the per-token conv + d2d-free capture; GDN is always
/// the snap twin. Returns (final h, final conv, normed, h_inters, conv snaps).
pub(crate) type LegOut = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>);
fn run_exact(g: &dyn GpuBackend, ks: &Kernels, inp: &Inputs, fused_conv: bool) -> Result<LegOut> {
    let state = up(g, &inp.conv0)?;
    let h = up(g, &inp.h0)?;
    let deint = up(g, &inp.deint)?;
    let gates = up(g, &inp.gates)?;
    let (w, nw) = (up(g, &inp.weight)?, up(g, &inp.norm_w)?);
    let conv_rows = g.alloc(K * QKVZ * 4)?;
    let normed = g.alloc(K * VALUE_DIM * 2)?;
    let h_inters: Vec<DevicePtr> = (0..K - 1)
        .map(|_| g.alloc(H_ELEMS * 4))
        .collect::<Result<_>>()?;
    let conv_inters = g.alloc(K * CONV_ELEMS * 4)?; // contiguous, kn layout
    if fused_conv {
        KernelLaunch::new(g, ks.fused_conv_f32)
            .grid([CONV_DIM.div_ceil(256) as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(state)
            .arg_ptr(deint)
            .arg_ptr(w)
            .arg_ptr(conv_rows)
            .arg_ptr(conv_inters)
            .arg_u32(K as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(D_CONV as u32)
            .arg_u32(QK_CH as u32)
            .arg_u32(KD as u32)
            .arg_u32(QKVZ as u32)
            .arg_u32(QKVZ as u32)
            .arg_u32(CONV_ELEMS as u32)
            .arg_f32(EPS)
            .launch(0)?;
    }
    let mut conv_snaps = Vec::new();
    for t in 0..K {
        let row = conv_rows.offset(t * QKVZ * 4);
        if !fused_conv {
            conv_launch(g, ks.conv_f32, state, deint.offset(t * QKVZ * 2), w, row)?;
            if t + 1 < K {
                g.copy_d2d_async(
                    state,
                    conv_inters.offset(t * CONV_ELEMS * 4),
                    CONV_ELEMS * 4,
                    0,
                )?;
            }
        }
        let hi = if t + 1 < K {
            h_inters[t]
        } else {
            DevicePtr::NULL
        };
        gdn_launch(
            g,
            ks.snap,
            true,
            h,
            row,
            gates.offset(t * 2 * NV * 4),
            deint.offset((t * QKVZ + CONV_DIM) * 2),
            nw,
            normed.offset(t * VALUE_DIM * 2),
            hi,
        )?;
    }
    g.synchronize(0)?;
    let mut his = Vec::new();
    for p in &h_inters {
        his.push(dn(g, *p, H_ELEMS * 4)?);
    }
    for t in 0..K - 1 {
        conv_snaps.push(dn(
            g,
            conv_inters.offset(t * CONV_ELEMS * 4),
            CONV_ELEMS * 4,
        )?);
    }
    Ok((
        dn(g, h, H_ELEMS * 4)?,
        dn(g, state, CONV_ELEMS * 4)?,
        dn(g, normed, K * VALUE_DIM * 2)?,
        his,
        conv_snaps,
    ))
}

/// NEGATIVE CONTROL: the pre-#435 K=4 verify (BF16 conv + wy4). Returns the
/// final H — which must NOT match golden's.
fn run_legacy_wy4(g: &dyn GpuBackend, ks: &Kernels, inp: &Inputs) -> Result<Vec<u8>> {
    let state = up(g, &inp.conv0)?;
    let h = up(g, &inp.h0)?;
    let deint = up(g, &inp.deint)?;
    let gates = up(g, &inp.gates)?;
    let w = up(g, &inp.weight)?;
    let conv_rows = g.alloc(K * CONV_DIM * 2)?; // BF16, conv_dim stride
    let gdn_out = g.alloc(K * VALUE_DIM * 2)?;
    let his: Vec<DevicePtr> = (0..3)
        .map(|_| g.alloc(H_ELEMS * 4))
        .collect::<Result<_>>()?;
    for t in 0..K {
        conv_launch(
            g,
            ks.conv_bf16,
            state,
            deint.offset(t * QKVZ * 2),
            w,
            conv_rows.offset(t * CONV_DIM * 2),
        )?;
    }
    // gate/beta rows are [gate|beta] at 2*NV stride; wy kernels take base
    // pointers + gb_stride exactly like production (gb_stride = 2*NV).
    KernelLaunch::new(g, ks.wy4)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(conv_rows)
        .arg_ptr(conv_rows.offset(KEY_DIM * 2))
        .arg_ptr(conv_rows.offset(KEY_DIM * 2 * 2))
        .arg_ptr(gates)
        .arg_ptr(gates.offset(NV * 4))
        .arg_ptr(gdn_out)
        .arg_ptr(his[0])
        .arg_ptr(his[1])
        .arg_ptr(his[2])
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32((NV * 2) as u32)
        .arg_u32(0) // state_is_table = contiguous
        .launch(0)?;
    g.synchronize(0)?;
    dn(g, h, H_ELEMS * 4)
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
    let gpu = AtlasCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let opt = |m: &str, f: &str| g.kernel(m, f).unwrap_or(KernelHandle(0));
    let ks = Kernels {
        conv_f32: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")?,
        conv_bf16: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?,
        conv_f32_strided: g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_strided")?,
        f32_norm: g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_norm")?,
        snap: opt(
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_norm_snap",
        ),
        snap_strided: opt(
            "gated_delta_rule_snap",
            "gated_delta_rule_decode_f32_strided_norm_snap",
        ),
        fused_conv_f32: opt(
            "gdn_verify_fused_conv_kn_f32",
            "gdn_verify_fused_conv_kn_f32",
        ),
        wy4: g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
    };
    if ks.snap.0 == 0 || ks.snap_strided.0 == 0 || ks.fused_conv_f32.0 == 0 {
        println!(
            "SKIPPED: snap/fused-f32 kernels not in this build's PTX set — build with \
             the qwen3.6-27b target (ATLAS_TARGET_MODEL=\"*\"). The bitwise gate did NOT run."
        );
        std::process::exit(2);
    }

    let mut all_ok = true;
    for seed in [7u64, 4242] {
        let inp = gen_inputs(seed);
        let gold = run_golden(g, &ks, &inp)?;

        for (name, fused) in [("snap", false), ("fused_conv_f32", true)] {
            let (hf, cf, nf, his, csnaps) = run_exact(g, &ks, &inp, fused)?;
            let mut ok = hf == gold.h_after[K - 1] && cf == gold.conv_after[K - 1];
            ok &= nf == gold.normed;
            for t in 0..K - 1 {
                ok &= his[t] == gold.h_after[t];
                ok &= csnaps[t] == gold.conv_after[t];
            }
            println!(
                "seed {seed:>5} leg {name:<14}: bitwise={ok}  h relL2={:.3e}",
                rel_l2(&hf, &gold.h_after[K - 1])
            );
            all_ok &= ok;
        }

        // Strided leg: N sequences, sequence 0 shares `inp`'s weights.
        let mut inps = vec![inp];
        for s in 0..N - 1 {
            let mut i2 = gen_inputs(1000 + s as u64);
            i2.weight = inps[0].weight.clone();
            i2.norm_w = inps[0].norm_w.clone();
            inps.push(i2);
        }
        let strided = run_strided(g, &ks, &inps)?;
        for (i, (hf, cf, nf, his, _)) in strided.iter().enumerate() {
            let gold_i = run_golden(g, &ks, &inps[i])?;
            let mut ok = *hf == gold_i.h_after[K - 1] && *cf == gold_i.conv_after[K - 1];
            ok &= *nf == gold_i.normed;
            for t in 0..K - 1 {
                ok &= his[t] == gold_i.h_after[t];
            }
            println!("seed {seed:>5} leg strided seq {i}:  bitwise={ok}");
            all_ok &= ok;
        }

        // NEGATIVE: legacy BF16-conv + wy4 (the DEFAULT verify path — exact
        // is the --exact-verify opt-in) must differ.
        let h_legacy = run_legacy_wy4(g, &ks, &inps[0])?;
        let differs = h_legacy != gold.h_after[K - 1];
        println!(
            "seed {seed:>5} NEGATIVE legacy wy4: differs={differs}  h relL2={:.3e} \
             (expected ~1e-3; bitwise match here would VOID the gate)",
            rel_l2(&h_legacy, &gold.h_after[K - 1])
        );
        all_ok &= differs;
    }

    println!(
        "\n{}",
        if all_ok {
            "PASS — exact verify chain is byte-identical to sequential decode; \
             the legacy WY chain provably is not."
        } else {
            "FAIL"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
