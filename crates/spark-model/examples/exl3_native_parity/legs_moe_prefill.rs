// SPDX-License-Identifier: AGPL-3.0-only
//! Prefill-MoE leg: the PRODUCTION sort-by-expert tier
//! (`ops::exl3_moe_prefill_routed` — Atlas `moe_sort_by_expert` counting
//! sort + `exl3_moe_stage_sorted` + the fused persistent `exl3_moe` kernel +
//! the chunked exl3_gemm overflow tier) against the same
//! reconstruct→f64→silu→weighted-sum reference the decode leg uses.
//!
//! Synthetic layer: 16 experts at the real qwen4_exp projection geometry
//! (gate/up [2560 -> 640], down [640 -> 2560]), K=4 MUL1, top_k = 4, and the
//! LEGACY 128-row per-expert cap (`ROWS_PER_EXPERT`) so the tier boundary is
//! exercised at this harness's scale. Sub-legs:
//!  * T=3  — the T*top_k <= 128 NO-SYNC shortcut (num_active = -1, no D2H).
//!  * T=64 — the host-sync fused tier, every expert 0 < count <= 128.
//!  * T=64 EP — experts [4, 16) local (dense table over 12), GLOBAL ids,
//!    token 0 forced all-remote: gated parity + exact-zero row + a negative
//!    control (masked output vs the full-routing reference must exceed the
//!    gate or the leg is vacuous).
//!  * T=192 skewed — ~60% of all slots routed to expert 0 (count ~460 >
//!    128), forcing the OVERFLOW path (asserted via the returned stats)
//!    through multiple ov_chunk=256 GEMM chunks, mixed with fused experts.
//!
//! Tolerances: the fused tier stages/computes in f16 with an fp32
//! accumulator — the decode-leg boundary model applies, gates reused
//! (rel 8e-3 / z 8e-2). The overflow tier decodes the same trellis through
//! the cooperative exl3_gemm at the same f16 activation precision, so the
//! skewed sub-leg gates at the SAME tolerance — a looser gate here would
//! hide a tier-dependent numerics seam (the same expert answering
//! differently across the 128-row boundary).

use anyhow::Result;
use half::f16;
use spark_model::layers::ops::{
    Exl3MoeOverflowCtx, Exl3MoePrefillScratch, exl3_moe_prefill_routed, moe_sort_by_expert,
};
use spark_runtime::gpu::DevicePtr;

use crate::legs_moe::{ProjSet, ref_token};
use crate::util::{Ctx, Lcg, as_bytes, gate_leg, metrics};

pub const MOE_REL_RMS: f64 = 8e-3;
pub const MOE_MAX_Z: f64 = 8e-2;
const OV_REL_RMS: f64 = MOE_REL_RMS;
const OV_MAX_Z: f64 = MOE_MAX_Z;

pub const H: usize = 2560;
pub const I: usize = 640;
const E: usize = 16;
pub const TOP_K: usize = 4;
const T_MAX: usize = 192;
const OV_CHUNK: usize = 256;
/// The LEGACY per-expert row cap, pinned here on purpose: the skewed sub-leg
/// below forces the overflow tier at ~460 rows, which the serving default
/// (1024) would keep fused. Slabs are sized at it and the scratch carries it.
const ROWS_PER_EXPERT: usize = 128;

pub struct Slabs {
    pub scratch: Exl3MoePrefillScratch,
    pub owned: Vec<DevicePtr>,
    pub input: DevicePtr,
    pub out: DevicePtr,
    pub indices: DevicePtr,
    pub probs: DevicePtr,
    pub sorted_token_ids: DevicePtr,
    pub sorted_expert_ids: DevicePtr,
    pub expert_offsets: DevicePtr,
    pub token_to_perm: DevicePtr,
}

pub fn alloc_slabs(ctx: &Ctx) -> Result<Slabs> {
    let g = ctx.g;
    let s_max = T_MAX * TOP_K;
    let c = ((ctx.sms as usize) / 8).clamp(1, 64);
    let mut owned = Vec::new();
    let mut a = |bytes: usize| -> Result<DevicePtr> {
        let p = g.alloc(bytes)?;
        owned.push(p);
        Ok(p)
    };
    let hidden_f16 = a(T_MAX * H * 2)?;
    let out_f32 = a(T_MAX * H * 4)?;
    // Deterministic epilogue's per-sorted-slot rows (the serving default), so
    // every sub-leg below gates the arm production actually runs.
    let slot_f32 = a(s_max * H * 4)?;
    let temp_state_g = a(c * ROWS_PER_EXPERT * H * 2)?;
    let temp_state_u = a(c * ROWS_PER_EXPERT * H * 2)?;
    let temp_inter_g = a(c * ROWS_PER_EXPERT * I * 2)?;
    let temp_inter_u = a(c * ROWS_PER_EXPERT * I * 2)?;
    let token_sorted = a(s_max * 8)?;
    let weight_sorted = a(s_max * 2)?;
    let expert_count = a((E + 1) * 8)?;
    let ov_a_f16 = a(OV_CHUNK * H * 2)?;
    let ov_a_had_f16 = a(OV_CHUNK * H * 2)?;
    let ov_gate_f16 = a(OV_CHUNK * I * 2)?;
    let ov_up_f16 = a(OV_CHUNK * I * 2)?;
    let ov_down_f32 = a(OV_CHUNK * H * 4)?;
    let input = a(T_MAX * H * 2)?;
    let out = a(T_MAX * H * 2)?;
    let indices = a(s_max * 4)?;
    let probs = a(s_max * 4)?;
    let sorted_token_ids = a(s_max * 4)?;
    let sorted_expert_ids = a(s_max * 4)?;
    let expert_offsets = a((E + 1) * 4)?;
    let token_to_perm = a(s_max * 4)?;
    Ok(Slabs {
        scratch: Exl3MoePrefillScratch {
            hidden_f16,
            out_f32,
            slot_f32: Some(slot_f32),
            temp_state_g,
            temp_state_u,
            temp_inter_g,
            temp_inter_u,
            token_sorted,
            weight_sorted,
            expert_count,
            ov_a_f16,
            ov_a_had_f16,
            ov_gate_f16,
            ov_up_f16,
            ov_down_f32,
            t_cap: T_MAX,
            slot_cap: s_max,
            e_cap: E,
            concurrency: c,
            rows_per_expert: ROWS_PER_EXPERT,
            ov_chunk: OV_CHUNK,
        },
        owned,
        input,
        out,
        indices,
        probs,
        sorted_token_ids,
        sorted_expert_ids,
        expert_offsets,
        token_to_perm,
    })
}

/// Run the production prefill pipeline (sort + staged fused kernel +
/// overflow) over the given table/EP range; returns (out_f64, out_bf16_bits,
/// num_active, overflow_experts).
#[allow(clippy::too_many_arguments)]
pub fn run_native(
    ctx: &Ctx,
    sl: &Slabs,
    tables: &[spark_model::layers::ops::Exl3MoeProj; 3],
    ov: &Exl3MoeOverflowCtx,
    input_bf16: &[u16],
    ids: &[u32],
    probs: &[f32],
    t: usize,
    local_start: usize,
    num_local: usize,
) -> Result<(Vec<f64>, Vec<u16>, i64, usize)> {
    let g = ctx.g;
    let stream = g.default_stream();
    g.copy_h2d(&as_bytes(input_bf16), sl.input)?;
    let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&id_bytes, sl.indices)?;
    let p_bytes: Vec<u8> = probs.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&p_bytes, sl.probs)?;

    let sort_k = g.kernel("moe", "moe_sort_by_expert")?;
    moe_sort_by_expert(
        g,
        sort_k,
        sl.indices,
        sl.sorted_token_ids,
        sl.sorted_expert_ids,
        sl.expert_offsets,
        sl.token_to_perm,
        (t * TOP_K) as u32,
        E as u32,
        TOP_K as u32,
        stream,
    )?;
    let stats = exl3_moe_prefill_routed(
        g,
        sl.input,
        sl.probs,
        sl.expert_offsets,
        sl.token_to_perm,
        sl.out,
        tables,
        ov,
        &sl.scratch,
        ctx.locks,
        t,
        TOP_K,
        H,
        I,
        local_start,
        num_local,
        0.0,
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
    Ok((y, bits, stats.num_active, stats.overflow_experts))
}

#[allow(clippy::too_many_arguments)]
pub fn ref_all(
    input_f16: &[u16],
    ids: &[u32],
    probs: &[f32],
    gate: &ProjSet,
    upp: &ProjSet,
    down: &ProjSet,
    t: usize,
    local_start: usize,
    num_local: usize,
) -> Vec<f64> {
    let mut y = Vec::with_capacity(t * H);
    for tok in 0..t {
        y.extend(ref_token(
            &input_f16[tok * H..(tok + 1) * H],
            &ids[tok * TOP_K..(tok + 1) * TOP_K],
            &probs[tok * TOP_K..(tok + 1) * TOP_K],
            gate,
            upp,
            down,
            local_start,
            num_local,
        ));
    }
    y
}

pub fn gen_inputs(rng: &mut Lcg, t: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
    // 0.25x the decode leg's activation scale: the FUSED kernel's down GEMM
    // stages its pre-svh C in f16 (the decode tier's down mgemm C is fp32),
    // and unit-gauss activations against the synthetic unit-scale trellis
    // drive that intermediate to ~|75K| > f16 max 65504 -> inf/NaN. Real
    // checkpoints sit orders of magnitude below the boundary (their MoE
    // outputs are O(1-10); the synthetic ref at unit scale was O(30K)); the
    // 0.25 factor (~16x smaller products) restores that headroom while
    // keeping every stage well above rounding noise.
    let input_bf16: Vec<u16> = (0..t * H)
        .map(|_| half::bf16::from_f32(0.25 * rng.gauss()).to_bits())
        .collect();
    let input_f16: Vec<u16> = input_bf16
        .iter()
        .map(|&b| f16::from_f32(half::bf16::from_bits(b).to_f32()).to_bits())
        .collect();
    let probs: Vec<f32> = {
        let mut p: Vec<f32> = (0..t * TOP_K).map(|_| 0.05 + rng.f()).collect();
        for chunk in p.chunks_mut(TOP_K) {
            let sum: f32 = chunk.iter().sum();
            for v in chunk {
                *v /= sum;
            }
        }
        p
    };
    (input_bf16, input_f16, probs)
}

pub fn leg_moe_prefill(ctx: &Ctx, rng: &mut Lcg) -> Result<bool> {
    let g = ctx.g;
    let mut ok = true;

    let gate = ProjSet::generate(ctx, rng, E, H, I)?;
    let upp = ProjSet::generate(ctx, rng, E, H, I)?;
    let down = ProjSet::generate(ctx, rng, E, I, H)?;

    let (gate_t, gate_own) = gate.table(ctx, 0)?;
    let (up_t, up_own) = upp.table(ctx, 0)?;
    let (down_t, down_own) = down.table(ctx, 0)?;
    let full = [gate_t, up_t, down_t];

    let host = |p: &ProjSet, first: usize| -> Vec<[u64; 3]> {
        p.dev[first..]
            .iter()
            .map(|d| [d.trellis.0, d.suh.0, d.svh.0])
            .collect()
    };
    let (gh, uh, dh) = (host(&gate, 0), host(&upp, 0), host(&down, 0));
    let ov_full = Exl3MoeOverflowCtx {
        gate_host: &gh,
        up_host: &uh,
        down_host: &dh,
    };

    let sl = alloc_slabs(ctx)?;

    // ── Sub-leg 1+2: uniform routing at T=3 (no-sync shortcut) and T=64
    // (host-sync fused tier, no overflow) ──
    for t in [3usize, 64] {
        let s = t * TOP_K;
        let (input_bf16, input_f16, probs) = gen_inputs(rng, t);
        let ids: Vec<u32> = (0..s).map(|_| (rng.next() % E as u64) as u32).collect();
        let (y_gpu, _, num_active, n_ov) = run_native(
            ctx,
            &sl,
            &full,
            &ov_full,
            &input_bf16,
            &ids,
            &probs,
            t,
            0,
            E,
        )?;
        let y64 = ref_all(&input_f16, &ids, &probs, &gate, &upp, &down, t, 0, E);
        ok &= gate_leg(
            &format!("moe-prefill fused [{H}x{I}] E={E} top_k={TOP_K} T={t}"),
            &y_gpu,
            &y64,
            MOE_REL_RMS,
            MOE_MAX_Z,
        );
        if t == 3 {
            // T*top_k = 12 <= 128 must take the no-sync shortcut.
            let shortcut = num_active == -1 && n_ov == 0;
            println!("moe-prefill T=3 no-sync shortcut taken (num_active=-1) = {shortcut}");
            ok &= shortcut;
        } else {
            // Host-sync tier, every count <= 64 < 128: no overflow.
            let fused_only = num_active > 0 && n_ov == 0;
            println!(
                "moe-prefill T=64 host-sync tier fused-only \
                 (num_active={num_active}, overflow={n_ov}) = {fused_only}"
            );
            ok &= fused_only;
        }

        // ── EP sub-leg on the T=64 arm: experts [4, 16) local, table dense
        // over the 12, ids stay GLOBAL; token 0 forced all-remote. ──
        if t == 64 {
            let (gate_ep, gate_ep_own) = gate.table(ctx, 4)?;
            let (up_ep, up_ep_own) = upp.table(ctx, 4)?;
            let (down_ep, down_ep_own) = down.table(ctx, 4)?;
            let ep = [gate_ep, up_ep, down_ep];
            let (ghe, uhe, dhe) = (host(&gate, 4), host(&upp, 4), host(&down, 4));
            let ov_ep = Exl3MoeOverflowCtx {
                gate_host: &ghe,
                up_host: &uhe,
                down_host: &dhe,
            };
            let mut ep_ids = ids.clone();
            for v in ep_ids.iter_mut().take(TOP_K) {
                *v %= 4; // token 0: every expert remote
            }
            let (y_ep, bits_ep, _, _) = run_native(
                ctx,
                &sl,
                &ep,
                &ov_ep,
                &input_bf16,
                &ep_ids,
                &probs,
                t,
                4,
                12,
            )?;
            let y64_ep = ref_all(&input_f16, &ep_ids, &probs, &gate, &upp, &down, t, 4, 12);
            ok &= gate_leg(
                "moe-prefill EP local=[4,16) sentinel-tail T=64",
                &y_ep,
                &y64_ep,
                MOE_REL_RMS,
                MOE_MAX_Z,
            );
            let row0_zero = bits_ep[..H].iter().all(|&b| b == 0 || b == 0x8000);
            println!("moe-prefill EP all-remote token row is exact zero = {row0_zero}");
            ok &= row0_zero;
            // Negative control: masked output vs the FULL-routing reference.
            let y64_full = ref_all(&input_f16, &ep_ids, &probs, &gate, &upp, &down, t, 0, E);
            let (rr, _) = metrics(&y_ep, &y64_full);
            let moved = rr > MOE_REL_RMS;
            println!("moe-prefill CONTROL masked-vs-full rel_rms={rr:.3e} exceeds gate = {moved}");
            if !moved {
                println!("FAIL — EP control stayed under the gate; leg is VACUOUS.");
                ok = false;
            }
            for p in gate_ep_own.into_iter().chain(up_ep_own).chain(down_ep_own) {
                g.free(p).ok();
            }
        }
    }

    // ── Sub-leg 3: skewed routing at T=192 — ~60% of slots to expert 0
    // (count ~460 > 128) exercises the chunked exl3_gemm overflow tier
    // across multiple 256-row chunks, mixed with fused experts. ──
    {
        let t = T_MAX;
        let s = t * TOP_K;
        let (input_bf16, input_f16, probs) = gen_inputs(rng, t);
        let ids: Vec<u32> = (0..s)
            .map(|_| {
                if rng.f() < 0.6 {
                    0u32
                } else {
                    1 + (rng.next() % (E as u64 - 1)) as u32
                }
            })
            .collect();
        let hot = ids.iter().filter(|&&v| v == 0).count();
        let (y_gpu, _, num_active, n_ov) = run_native(
            ctx,
            &sl,
            &full,
            &ov_full,
            &input_bf16,
            &ids,
            &probs,
            t,
            0,
            E,
        )?;
        let y64 = ref_all(&input_f16, &ids, &probs, &gate, &upp, &down, t, 0, E);
        ok &= gate_leg(
            &format!("moe-prefill OVERFLOW skew T={t} (expert0 rows={hot})"),
            &y_gpu,
            &y64,
            OV_REL_RMS,
            OV_MAX_Z,
        );
        let exercised = hot > 128 && n_ov >= 1 && num_active > 0;
        println!(
            "moe-prefill OVERFLOW exercised (hot_rows={hot} > 128, \
             overflow_experts={n_ov}, fused num_active={num_active}) = {exercised}"
        );
        ok &= exercised;
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
