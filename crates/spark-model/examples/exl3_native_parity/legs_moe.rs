// SPDX-License-Identifier: AGPL-3.0-only
//! Decode-MoE leg: the PRODUCTION 3x-mgemm routed-expert pipeline
//! (`ops::exl3_moe_decode_routed` — staging kernels + gate/up/silu/down with
//! the routing probs folded into the down call's fp32 grouped reduction)
//! against a host reference, plus an EP sub-leg asserting exact remote-slot
//! exclusion.
//!
//! Synthetic layer: 8 experts at the real qwen4_exp projection geometry
//! (gate/up [2560 -> 640], down [640 -> 2560]), K=4 MUL1 (the checkpoint's
//! hot template), top_k = 3, T in {1, 4, 8}, random routing + probs.
//!
//! Reference: reconstruct each expert (`decode_what_f64`) -> f64 rotation
//! matmul (`truth_matmul`) for gate and up -> f64 silu(gate)*up, rounded to
//! f16 at the down call's input boundary (the pipeline hands the down mgemm
//! an f16 A; the rounding is part of the serving numerics, everything else
//! stays f64) -> f64 down matmul scaled by the f16-rounded routing weight ->
//! sum over the token's slots.
//!
//! Tolerance derivation (like the existing legs): the pipeline chains two
//! f16-C mgemms (f16-C gemm measured ~6e-4 rel_rms, gated loose at
//! GEMV_REL_RMS = 8e-3), a half-precision product + f16 boundary rounding
//! (relative errors of the two factors ADD), and one fp32-C mgemm
//! (GEMM_REL_RMS = 2.5e-3). Measured on GB10: rel_rms ~1.9e-3, max_z
//! ~2.8e-2 across T in {1,4,8} + the EP arm — gates set at ~4x/3x that
//! (rel 8e-3, the GEMV-leg level; z 8e-2). A negative control (EP-masked
//! output vs the FULL-routing reference) must exceed the gate or the leg
//! is vacuous.

use anyhow::Result;
use half::f16;
use spark_model::layers::ops::{Exl3MoeProj, Exl3MoeScratch, exl3_moe_decode_routed};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::truth::{cb_enum, decode_what_f64, truth_matmul};
use crate::util::{Ctx, DevWeight, Lcg, as_bytes, gate_leg, metrics, up};

const MOE_REL_RMS: f64 = 8e-3;
const MOE_MAX_Z: f64 = 8e-2;

const H: usize = 2560;
const I: usize = 640;
const E: usize = 8;
const TOP_K: usize = 3;
const K_BITS: u32 = 4;
const CB: u32 = 2;

pub struct ProjSet {
    pub weights: Vec<(Vec<u16>, Vec<u16>, Vec<u16>)>, // (trellis, suh, svh) per expert
    pub whats: Vec<Vec<f64>>,                         // f64 reconstruction per expert
    pub dev: Vec<DevWeight>,
    /// Trellis bits/weight of every expert in the set (MUL1 codebook).
    pub k_bits: u32,
}

impl ProjSet {
    /// `e` experts of one projection at `[k -> n]` (K=4 MUL1 template).
    pub fn generate(ctx: &Ctx, rng: &mut Lcg, e: usize, k: usize, n: usize) -> Result<Self> {
        Self::generate_k(ctx, rng, e, k, n, K_BITS)
    }

    /// `e` experts of one projection at `[k -> n]`, `k_bits` MUL1.
    pub fn generate_k(
        ctx: &Ctx,
        rng: &mut Lcg,
        e: usize,
        k: usize,
        n: usize,
        k_bits: u32,
    ) -> Result<Self> {
        let mut weights = Vec::with_capacity(e);
        for _ in 0..e {
            let trellis: Vec<u16> = (0..(k / 16) * (n / 16) * 16 * k_bits as usize)
                .map(|_| rng.u16())
                .collect();
            let suh: Vec<u16> = (0..k).map(|_| rng.scale_f16()).collect();
            let svh: Vec<u16> = (0..n).map(|_| rng.scale_f16()).collect();
            weights.push((trellis, suh, svh));
        }
        let whats = weights
            .iter()
            .map(|(t, _, _)| decode_what_f64(t, k, n, k_bits, cb_enum(CB)))
            .collect();
        let dev = weights
            .iter()
            .map(|(t, su, sv)| DevWeight::upload(ctx.g, t, su, sv))
            .collect::<Result<_>>()?;
        Ok(Self {
            weights,
            whats,
            dev,
            k_bits,
        })
    }

    /// Dense device pointer table over experts `[first, len)`.
    pub fn table(&self, ctx: &Ctx, first: usize) -> Result<(Exl3MoeProj, [DevicePtr; 3])> {
        let bytes = |f: &dyn Fn(&DevWeight) -> u64| -> Vec<u8> {
            self.dev[first..]
                .iter()
                .flat_map(|d| f(d).to_le_bytes())
                .collect()
        };
        let trellis_ptrs = up(ctx.g, &bytes(&|d| d.trellis.0))?;
        let suh_ptrs = up(ctx.g, &bytes(&|d| d.suh.0))?;
        let svh_ptrs = up(ctx.g, &bytes(&|d| d.svh.0))?;
        Ok((
            Exl3MoeProj {
                trellis_ptrs,
                suh_ptrs,
                svh_ptrs,
                k_bits: self.k_bits,
                cb: CB,
            },
            [trellis_ptrs, suh_ptrs, svh_ptrs],
        ))
    }

    pub fn free(&self, g: &dyn GpuBackend) {
        for d in &self.dev {
            d.free(g);
        }
    }
}

pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// f64 reference for one token through the routed pipeline; slots outside
/// `[local_start, local_start + num_local)` are excluded (EP `-1` slots).
#[allow(clippy::too_many_arguments)]
pub fn ref_token(
    x_f16: &[u16], // [H] the token's activation as f16 bits (post-ingress)
    ids: &[u32],
    probs: &[f32],
    gate: &ProjSet,
    upp: &ProjSet,
    down: &ProjSet,
    local_start: usize,
    num_local: usize,
) -> Vec<f64> {
    let mut y = vec![0f64; H];
    for (&gid, &p) in ids.iter().zip(probs.iter()) {
        let e = gid as usize;
        if e < local_start || e >= local_start + num_local {
            continue;
        }
        let (_, g_su, g_sv) = &gate.weights[e];
        let (_, u_su, u_sv) = &upp.weights[e];
        let (_, d_su, d_sv) = &down.weights[e];
        let yg = truth_matmul(x_f16, g_su, g_sv, &gate.whats[e], 1, H, I, 1.0);
        let yu = truth_matmul(x_f16, u_su, u_sv, &upp.whats[e], 1, H, I, 1.0);
        // silu(gate)*up in f64, rounded once at the down input boundary
        // (the pipeline hands the down mgemm an f16 A).
        let act_f16: Vec<u16> = yg
            .iter()
            .zip(yu.iter())
            .map(|(&g, &u)| f16::from_f64(silu(g) * u).to_bits())
            .collect();
        // Weight applied the way the kernel does: f16-rounded, folded into
        // the down epilogue scale, summed in fp32 (f64 here).
        let w = f16::from_f32(p).to_f64();
        let yd = truth_matmul(&act_f16, d_su, d_sv, &down.whats[e], 1, I, H, w);
        for (acc, v) in y.iter_mut().zip(yd.iter()) {
            *acc += v;
        }
    }
    y
}

pub struct Slabs {
    pub(super) scratch: Exl3MoeScratch,
    pub owned: Vec<DevicePtr>,
    pub(super) out: DevicePtr,
    pub(super) input: DevicePtr,
    pub(super) indices: DevicePtr,
    pub(super) probs: DevicePtr,
}

pub fn alloc_slabs(ctx: &Ctx, s_cap: usize, t_max: usize) -> Result<Slabs> {
    let g = ctx.g;
    let a_f16 = g.alloc(s_cap * H * 2)?;
    let a_had_f16 = g.alloc(s_cap * H * 2)?;
    let c_gate = g.alloc(s_cap * I * 2)?;
    let c_up = g.alloc(s_cap * I * 2)?;
    let inter = g.alloc(s_cap * I * 2)?;
    let c_down = g.alloc(s_cap * H * 4)?;
    let b_indices = g.alloc(s_cap * 8)?;
    let b_weights = g.alloc(s_cap * 2)?;
    let out = g.alloc(t_max * H * 2)?;
    let input = g.alloc(t_max * H * 2)?;
    let indices = g.alloc(s_cap * 4)?;
    let probs = g.alloc(s_cap * 4)?;
    Ok(Slabs {
        scratch: Exl3MoeScratch {
            a_f16,
            a_had_f16,
            a_had_capacity_elems: s_cap * H,
            c_gate_f16: c_gate,
            c_up_f16: c_up,
            inter_f16: inter,
            c_down_f32: c_down,
            b_indices,
            b_weights,
            s_cap,
        },
        owned: vec![
            a_f16, a_had_f16, c_gate, c_up, inter, c_down, b_indices, b_weights, out, input,
            indices, probs,
        ],
        out,
        input,
        indices,
        probs,
    })
}

/// Run the production pipeline for T tokens over the given table/EP range;
/// returns (out_f64, out_bf16_bits).
#[allow(clippy::too_many_arguments)]
pub fn run_native(
    ctx: &Ctx,
    sl: &Slabs,
    tables: &[Exl3MoeProj; 3],
    input_bf16: &[u16],
    ids: &[u32],
    probs: &[f32],
    t: usize,
    local_start: usize,
    num_local: usize,
) -> Result<(Vec<f64>, Vec<u16>)> {
    let g = ctx.g;
    let stream = g.default_stream();
    g.copy_h2d(&as_bytes(input_bf16), sl.input)?;
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&id_bytes, sl.indices)?;
    let p_bytes: Vec<u8> = probs.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&p_bytes, sl.probs)?;

    exl3_moe_decode_routed(
        g,
        sl.input,
        sl.indices,
        sl.probs,
        sl.out,
        tables,
        &sl.scratch,
        ctx.locks,
        t,
        TOP_K,
        H,
        I,
        local_start,
        num_local,
        0.0,
        false,
        ctx.sms,
        stream,
    )?;
    g.synchronize(stream)?;

    let mut bytes = vec![0u8; t * H * 2];
    g.copy_d2h(sl.out, &mut bytes)?;
    let bits: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let y = bits
        .iter()
        .map(|&b| half::bf16::from_bits(b).to_f64())
        .collect();
    Ok((y, bits))
}

pub fn leg_moe_decode(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let mut ok = true;

    let gate = ProjSet::generate(ctx, rng, E, H, I)?;
    let upp = ProjSet::generate(ctx, rng, E, H, I)?;
    let down = ProjSet::generate(ctx, rng, E, I, H)?;

    let (gate_t, gate_own) = gate.table(ctx, 0)?;
    let (up_t, up_own) = upp.table(ctx, 0)?;
    let (down_t, down_own) = down.table(ctx, 0)?;
    let full = [gate_t, up_t, down_t];

    let s_cap = 8 * TOP_K;
    let sl = alloc_slabs(ctx, s_cap, 8)?;

    for t in [1usize, 4, 8] {
        let s = t * TOP_K;
        // Activations: host-rounded bf16 (the layer input dtype); the
        // reference sees the same values through the ingress f16 rounding.
        let input_bf16: Vec<u16> = (0..t * H)
            .map(|_| half::bf16::from_f32(rng.gauss()).to_bits())
            .collect();
        let input_f16: Vec<u16> = input_bf16
            .iter()
            .map(|&b| f16::from_f32(half::bf16::from_bits(b).to_f32()).to_bits())
            .collect();
        // Random routing: distinct-ish ids, positive normalized probs.
        let ids: Vec<u32> = (0..s).map(|_| (rng.next() % E as u64) as u32).collect();
        let probs: Vec<f32> = {
            let mut p: Vec<f32> = (0..s).map(|_| 0.05 + rng.f()).collect();
            for chunk in p.chunks_mut(TOP_K) {
                let sum: f32 = chunk.iter().sum();
                for v in chunk {
                    *v /= sum;
                }
            }
            p
        };

        let (y_gpu, _) = run_native(ctx, &sl, &full, &input_bf16, &ids, &probs, t, 0, E)?;
        let mut y64 = Vec::with_capacity(t * H);
        for tok in 0..t {
            y64.extend(ref_token(
                &input_f16[tok * H..(tok + 1) * H],
                &ids[tok * TOP_K..(tok + 1) * TOP_K],
                &probs[tok * TOP_K..(tok + 1) * TOP_K],
                &gate,
                &upp,
                &down,
                0,
                E,
            ));
        }
        ok &= gate_leg(
            &format!("moe-decode 3x-mgemm [{H}x{I}] E={E} top_k={TOP_K} T={t}"),
            &y_gpu,
            &y64,
            MOE_REL_RMS,
            MOE_MAX_Z,
        );

        // ── EP sub-leg (T=8 arm): experts [4, 8) local, table dense over
        // them, ids stay GLOBAL; token 0 forced all-remote. ──
        if t == 8 {
            let (gate_ep, gate_ep_own) = gate.table(ctx, 4)?;
            let (up_ep, up_ep_own) = upp.table(ctx, 4)?;
            let (down_ep, down_ep_own) = down.table(ctx, 4)?;
            let ep = [gate_ep, up_ep, down_ep];
            let mut ep_ids = ids.clone();
            for v in ep_ids.iter_mut().take(TOP_K) {
                *v %= 4; // token 0: every expert remote
            }
            let (y_ep, bits_ep) = run_native(ctx, &sl, &ep, &input_bf16, &ep_ids, &probs, t, 4, 4)?;
            let mut y64_ep = Vec::with_capacity(t * H);
            for tok in 0..t {
                y64_ep.extend(ref_token(
                    &input_f16[tok * H..(tok + 1) * H],
                    &ep_ids[tok * TOP_K..(tok + 1) * TOP_K],
                    &probs[tok * TOP_K..(tok + 1) * TOP_K],
                    &gate,
                    &upp,
                    &down,
                    4,
                    4,
                ));
            }
            ok &= gate_leg(
                "moe-decode EP local=[4,8) remote=-1 T=8",
                &y_ep,
                &y64_ep,
                MOE_REL_RMS,
                MOE_MAX_Z,
            );
            // Exact exclusion: the all-remote token's row is EXACTLY zero
            // (the fp32 reduction sums no slots and writes 0.0 -> bf16 0).
            let row0_zero = bits_ep[..H].iter().all(|&b| b == 0 || b == 0x8000);
            println!("moe-decode EP all-remote token row is exact zero = {row0_zero}");
            ok &= row0_zero;
            // Negative control: the masked output must NOT match the
            // full-routing reference (proves the -1 exclusion did work and
            // the gates are non-vacuous). Token 0's remote slots guarantee a
            // difference.
            let mut y64_full = Vec::with_capacity(t * H);
            for tok in 0..t {
                y64_full.extend(ref_token(
                    &input_f16[tok * H..(tok + 1) * H],
                    &ep_ids[tok * TOP_K..(tok + 1) * TOP_K],
                    &probs[tok * TOP_K..(tok + 1) * TOP_K],
                    &gate,
                    &upp,
                    &down,
                    0,
                    E,
                ));
            }
            let (rr, _) = metrics(&y_ep, &y64_full);
            let moved = rr > MOE_REL_RMS;
            println!("moe-decode CONTROL masked-vs-full rel_rms={rr:.3e} exceeds gate = {moved}");
            if !moved {
                println!("FAIL — EP control stayed under the gate; leg is VACUOUS.");
                ok = false;
            }
            for p in gate_ep_own.into_iter().chain(up_ep_own).chain(down_ep_own) {
                g.free(p).ok();
            }
        }
    }

    for p in sl.owned.iter() {
        g.free(*p).ok();
    }
    for p in gate_own.into_iter().chain(up_own).chain(down_own) {
        g.free(p).ok();
    }
    gate.free(g);
    upp.free(g);
    down.free(g);
    Ok(ok)
}
