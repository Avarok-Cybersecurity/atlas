// SPDX-License-Identifier: AGPL-3.0-only

//! Oracle for the Hopper GDN decode twins (#927/#928).
//!
//! THREE ARMS since round 17: the gb10 parent, the COLUMN-TILED twin
//! (`gdn_decode_hopper.cu`, default off) and the ONE-READ strided twin
//! (`gdn_decode_strided_hopper.cu`, default on for n >= 4). The third runs on
//! the strided legs only — it has no contiguous sibling, because the arm it
//! was written for is the batched decode step.
//!
//! THE GATE IS BITWISE. The twins in `kernels/hopper/common/gdn_decode_hopper.cu`
//! change launch geometry and unroll depth and nothing else, so "close enough"
//! is the wrong answer: any differing byte in the recurrent state or the
//! readout means an expression or a reduction order moved, and a decode that
//! diverges one bit per layer per step diverges visibly by the end of a
//! conversation. `max_abs` is printed so a failure is diagnosable, not because
//! a non-zero value would be accepted.
//!
//! It runs BOTH state scales:
//!   * `hs = 0.05`, where the Frobenius norm stays far below
//!     SSM_STATE_MAX_NORM and the strided parent's clamp never fires;
//!   * `hs = 20.0`, where it does. Without this leg the clamp — the only part
//!     of the strided kernel that is a whole-head reduction, and the reason
//!     the strided twin may NOT tile its columns — would be untested.
//!
//! A KNOWN_BAD control runs first: the same comparison against a state that
//! has had ONE float perturbed by one ULP. It must report a non-zero
//! `state_diff`. A bitwise gate whose failure path has never executed is not
//! evidence — the whole suite would pass just as loudly if `diff` returned
//! `(0, 0.0)` unconditionally.
//!
//! Guard bytes bracket every device buffer the kernels write, because the
//! twins compute their own grid and a tile-width bug writes PAST the head
//! rather than producing wrong numbers inside it.
//!
//! Timing is reported per layer and as GB/s of state traffic (one read plus
//! one write of the live state), the roofline the attribution doc argues
//! against. It is a single-harness observation on whatever GPU runs it — on a
//! non-Hopper device the twins still RUN, they simply have no underfill to
//! fix, so treat the numbers as engagement evidence, not as the H100 A/B.
//!
//! Run: `cargo run -p spark-model --features cuda,gpu-examples \
//!        --example native_gdn_decode_hopper_microtest`

use anyhow::Result;
use spark_model::layers::ops;
use spark_runtime::cuda_backend::AtlasCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

// Qwen3.8-27B GDN shapes. num_v_heads is DERIVED from the nsys receipt:
// 3.16 MB of FP32 state per layer per sequence / (128*128*4 B) = 48.
const NK: usize = 48;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const GUARD: usize = 64;
const SENTINEL: u8 = 0x5a;
const WARMUP: u32 = 5;
const REPS: u32 = 30;

/// Deterministic LCG — the same inputs must reach both kernels, and a
/// thread-rng would make a failure unreproducible.
struct Lcg(u32);
impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((self.0 >> 8) & 0xFF_FFFF) as f32 / 8_388_608.0 - 1.0
    }
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// Allocate `bytes` of payload with `GUARD` sentinel bytes on each side and
/// return (base, payload).
fn guarded(g: &dyn GpuBackend, bytes: usize) -> Result<(DevicePtr, DevicePtr)> {
    let base = g.alloc(bytes + 2 * GUARD)?;
    g.copy_h2d(&vec![SENTINEL; bytes + 2 * GUARD], base)?;
    Ok((base, base.offset(GUARD)))
}

fn guards_intact(g: &dyn GpuBackend, base: DevicePtr, bytes: usize) -> Result<bool> {
    let mut raw = vec![0u8; bytes + 2 * GUARD];
    g.copy_d2h(base, &mut raw)?;
    Ok(raw[..GUARD].iter().all(|b| *b == SENTINEL)
        && raw[GUARD + bytes..].iter().all(|b| *b == SENTINEL))
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// (differing 4-byte lanes, largest absolute difference).
fn diff(a: &[u8], b: &[u8]) -> (usize, f32) {
    let mut n = 0usize;
    let mut worst = 0.0f32;
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if x != y {
            n += 1;
            let (xv, yv) = (
                f32::from_le_bytes(x.try_into().unwrap()),
                f32::from_le_bytes(y.try_into().unwrap()),
            );
            worst = worst.max((xv - yv).abs());
        }
    }
    (n, worst)
}

fn time_us(g: &dyn GpuBackend, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP {
        run()?;
    }
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..REPS {
        run()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / f64::from(REPS))
}

struct Inputs {
    h: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
}

fn inputs(n: usize, h_scale: f32) -> Inputs {
    let mut r = Lcg(12_345);
    let mk =
        |r: &mut Lcg, len: usize, s: f32| (0..len).map(|_| r.next_f32() * s).collect::<Vec<_>>();
    let h = mk(&mut r, n * NV * KD * VD, h_scale);
    let q = mk(&mut r, n * NK * KD, 1.0);
    let k = mk(&mut r, n * NK * KD, 1.0);
    let v = mk(&mut r, n * NV * VD, 1.0);
    // The gate is clamped into (1e-6, 1-1e-6) by both kernels; keep it in the
    // decaying range a real step produces rather than exercising the clamp.
    let gate = (0..n * NV).map(|_| 0.5 + 0.4 * r.next_f32()).collect();
    let beta = mk(&mut r, n * NV, 1.0);
    Inputs {
        h,
        q,
        k,
        v,
        gate,
        beta,
    }
}

/// Which twin a leg compares against the parent.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// `gated_delta_rule_decode_f32{,_strided}_hopper` — the column-tiled twin.
    Tiled,
    /// `gated_delta_rule_decode_f32_strided_hopper_smem` — the one-read twin.
    Smem,
}

#[allow(clippy::too_many_arguments)]
fn leg(
    g: &dyn GpuBackend,
    parent: KernelHandle,
    twin: KernelHandle,
    arm: Arm,
    strided: bool,
    n: usize,
    h_scale: f32,
) -> Result<bool> {
    let inp = inputs(n, h_scale);
    let h_bytes = inp.h.len() * 4;
    let out_len = n * NV * VD;
    let out_bytes = out_len * 4;

    let (dq, dk, dv) = (up_f32(g, &inp.q)?, up_f32(g, &inp.k)?, up_f32(g, &inp.v)?);
    let (dg, db) = (up_f32(g, &inp.gate)?, up_f32(g, &inp.beta)?);
    let (h_ref_base, h_ref) = guarded(g, h_bytes)?;
    let (h_new_base, h_new) = guarded(g, h_bytes)?;
    let (o_ref_base, o_ref) = guarded(g, out_bytes)?;
    let (o_new_base, o_new) = guarded(g, out_bytes)?;

    let reset = |p: DevicePtr| -> Result<()> {
        let b: Vec<u8> = inp.h.iter().flat_map(|x| x.to_le_bytes()).collect();
        g.copy_h2d(&b, p)
    };
    // Strides for the strided pair: the contiguous packing its parent assumes.
    let (qk_s, v_s, gb_s, out_s) = (
        (NK * KD) as u32,
        (NV * VD) as u32,
        NV as u32,
        (NV * VD) as u32,
    );
    let n32 = n as u32;
    let run_parent = |dst_h: DevicePtr, dst_o: DevicePtr| -> Result<()> {
        if strided {
            ops::gdn_decode_f32_strided(
                g, parent, dst_h, dq, dk, dv, dg, db, dst_o, n32, NK as u32, NV as u32, KD as u32,
                VD as u32, qk_s, v_s, gb_s, out_s, 0,
            )
        } else {
            ops::gdn_decode(
                g, parent, dst_h, dq, dk, dv, dg, db, dst_o, n32, NK as u32, NV as u32, KD as u32,
                VD as u32, 0,
            )
        }
    };
    let run_twin = |dst_h: DevicePtr, dst_o: DevicePtr| -> Result<()> {
        match (arm, strided) {
            (Arm::Smem, _) => ops::gdn_decode_f32_strided_hopper_smem(
                g, twin, dst_h, dq, dk, dv, dg, db, dst_o, n32, NK as u32, NV as u32, KD as u32,
                VD as u32, qk_s, v_s, gb_s, out_s, 0,
            ),
            (Arm::Tiled, true) => ops::gdn_decode_f32_strided_hopper(
                g, twin, dst_h, dq, dk, dv, dg, db, dst_o, n32, NK as u32, NV as u32, KD as u32,
                VD as u32, qk_s, v_s, gb_s, out_s, 0,
            ),
            (Arm::Tiled, false) => ops::gdn_decode_f32_hopper(
                g, twin, dst_h, dq, dk, dv, dg, db, dst_o, n32, NK as u32, NV as u32, KD as u32,
                VD as u32, 0,
            ),
        }
    };

    reset(h_ref)?;
    reset(h_new)?;
    run_parent(h_ref, o_ref)?;
    run_twin(h_new, o_new)?;
    g.synchronize(0)?;

    let (sn, sm) = diff(&dn(g, h_ref, h_bytes)?, &dn(g, h_new, h_bytes)?);
    let (on, om) = diff(&dn(g, o_ref, out_bytes)?, &dn(g, o_new, out_bytes)?);
    let guards = guards_intact(g, h_new_base, h_bytes)?
        && guards_intact(g, o_new_base, out_bytes)?
        && guards_intact(g, h_ref_base, h_bytes)?
        && guards_intact(g, o_ref_base, out_bytes)?;

    let t_parent = time_us(g, || run_parent(h_ref, o_ref))?;
    let t_twin = time_us(g, || run_twin(h_new, o_new))?;
    // One read plus one write of the live state — the compulsory traffic.
    let bytes = 2.0 * h_bytes as f64;
    let gbps = |us: f64| bytes / (us * 1e-6) / 1e9;

    let kind = match (arm, strided) {
        (Arm::Smem, _) => "smem   ",
        (Arm::Tiled, true) => "strided",
        (Arm::Tiled, false) => "contig ",
    };
    eprintln!(
        "  {kind} n={n:<2} hs={h_scale:<5} state_diff={sn} out_diff={on} \
         max_abs={:.3e} guards={} | parent {t_parent:8.2} us {:6.0} GB/s | \
         twin {t_twin:8.2} us {:6.0} GB/s | {:.2}x",
        sm.max(om),
        if guards { "ok" } else { "CLOBBERED" },
        gbps(t_parent),
        gbps(t_twin),
        t_parent / t_twin,
    );

    for p in [
        dq, dk, dv, dg, db, h_ref_base, h_new_base, o_ref_base, o_new_base,
    ] {
        g.free(p).ok();
    }
    anyhow::ensure!(
        guards,
        "{kind} n={n}: a twin wrote outside its output buffer"
    );
    anyhow::ensure!(
        sn == 0 && on == 0,
        "{kind} n={n} hs={h_scale}: {sn} state lanes and {on} output lanes differ from the \
         gb10 parent (max_abs {:.3e}); the twins must be BIT-identical",
        sm.max(om)
    );
    Ok(true)
}

fn main() -> Result<()> {
    let backend = AtlasCudaBackend::new(0, &atlas_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let parent_c = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32")?;
    let parent_s = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32_strided")?;
    let twin_c = g.kernel("gdn_decode_hopper", "gated_delta_rule_decode_f32_hopper")?;
    let twin_s = g.kernel(
        "gdn_decode_hopper",
        "gated_delta_rule_decode_f32_strided_hopper",
    )?;

    let twin_smem = g.kernel(ops::GDN_STRIDED_SMEM_MODULE, ops::GDN_STRIDED_SMEM_ENTRY)?;

    let sms = ops::gdn_hopper_sm_count(g);
    eprintln!("native_gdn_decode_hopper_microtest: nv={NV} kd={KD} vd={VD} sm_count={sms}");
    eprintln!(
        "  C=1 tile choice: {} columns/CTA ({} CTAs)",
        ops::gdn_hopper_cols_per_cta(VD as u32, NV as u32, sms),
        NV as u32 * (VD as u32).div_ceil(ops::gdn_hopper_cols_per_cta(VD as u32, NV as u32, sms)),
    );
    for n in [1u32, 4, 16] {
        eprintln!(
            "  {}",
            ops::gdn_decode_strided_smem_route_line(
                ops::gdn_decode_strided_smem_accept(n, NV as u32, sms),
                NV as u32,
                n,
                sms,
            )
        );
    }

    known_bad(g, parent_s)?;

    for hs in [0.05f32, 20.0] {
        for n in [1usize, 4, 16] {
            leg(g, parent_c, twin_c, Arm::Tiled, false, n, hs)?;
            // The strided pair is the n >= 2 batched arm; n=1 is run anyway
            // because its geometry is the degenerate case of the same grid.
            leg(g, parent_s, twin_s, Arm::Tiled, true, n, hs)?;
            // The one-read twin (#927). The launcher declines it below n=4,
            // but the KERNEL is correct at every n, and a gate that only ran
            // it where the launcher runs it could not tell "declined" from
            // "broken". So it is compared at all three widths.
            leg(g, parent_s, twin_smem, Arm::Smem, true, n, hs)?;
        }
    }
    eprintln!("  ALL LEGS BIT-IDENTICAL");
    Ok(())
}

/// KNOWN_BAD — the control that proves the gate can fail.
///
/// Runs the parent against ITSELF with one f32 of the second state perturbed
/// by one ULP, and requires the comparison to report it. Without this, a
/// `diff` that returned `(0, 0.0)` unconditionally — or a `leg` that compared
/// a buffer against itself — would make every assertion below vacuous, and the
/// suite would print `ALL LEGS BIT-IDENTICAL` just as loudly.
fn known_bad(g: &dyn GpuBackend, parent: KernelHandle) -> Result<()> {
    let inp = inputs(1, 0.05);
    let h_bytes = inp.h.len() * 4;
    let out_bytes = NV * VD * 4;
    let (dq, dk, dv) = (up_f32(g, &inp.q)?, up_f32(g, &inp.k)?, up_f32(g, &inp.v)?);
    let (dg, db) = (up_f32(g, &inp.gate)?, up_f32(g, &inp.beta)?);
    let (a_base, a) = guarded(g, h_bytes)?;
    let (b_base, b) = guarded(g, h_bytes)?;
    let (o_base, o) = guarded(g, out_bytes)?;

    let mut poisoned = inp.h.clone();
    let victim = poisoned.len() / 2;
    poisoned[victim] = f32::from_bits(poisoned[victim].to_bits() ^ 1);
    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    g.copy_h2d(&bytes(&inp.h), a)?;
    g.copy_h2d(&bytes(&poisoned), b)?;

    let (n32, nk, nv, kd, vd) = (1u32, NK as u32, NV as u32, KD as u32, VD as u32);
    let (qk_s, v_s, gb_s, out_s) = (
        (NK * KD) as u32,
        (NV * VD) as u32,
        NV as u32,
        (NV * VD) as u32,
    );
    for dst in [a, b] {
        ops::gdn_decode_f32_strided(
            g, parent, dst, dq, dk, dv, dg, db, o, n32, nk, nv, kd, vd, qk_s, v_s, gb_s, out_s, 0,
        )?;
    }
    g.synchronize(0)?;
    let (n, worst) = diff(&dn(g, a, h_bytes)?, &dn(g, b, h_bytes)?);
    eprintln!("  KNOWN_BAD one-ULP state perturbation: state_diff={n} max_abs={worst:.3e}");
    for p in [dq, dk, dv, dg, db, a_base, b_base, o_base] {
        g.free(p).ok();
    }
    anyhow::ensure!(
        n > 0,
        "the KNOWN_BAD control reported 0 differing lanes — the bitwise gate \
         below cannot fail, so none of its passes mean anything"
    );
    Ok(())
}
