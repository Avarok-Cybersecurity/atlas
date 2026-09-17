// SPDX-License-Identifier: AGPL-3.0-only

//! `exl3_moe_prefill_routed`, split out of `moe_prefill.rs` to keep it
//! under the 500-line cap.

use super::*;

/// The full prefill-tier routed-expert pipeline over ONE token batch (header
/// diagram). The caller has already run `moe_sort_by_expert` over this
/// batch's GLOBAL expert ids; `expert_offsets`/`token_to_perm` are its
/// outputs, `probs_f32` the batch's flat `[t*top_k]` routing weights. Writes
/// the per-token WEIGHTED routed sums (probs applied in the fp32
/// accumulator; the caller's blend must NOT re-apply them) as BF16
/// `[t, hidden]` at `out_bf16`. A token whose experts are all remote
/// contributes an exact 0.0 row (EP partial-sum convention).
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_prefill_routed(
    gpu: &dyn GpuBackend,
    input_bf16: DevicePtr,
    probs_f32: DevicePtr,
    expert_offsets: DevicePtr,
    token_to_perm: DevicePtr,
    out_bf16: DevicePtr,
    tables: &[Exl3MoeProj; 3],
    ov: &Exl3MoeOverflowCtx,
    scratch: &Exl3MoePrefillScratch,
    locks: DevicePtr,
    t: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    local_start: usize,
    num_local: usize,
    act_limit: f32,
    sm_count: u32,
    stream: u64,
) -> Result<Exl3MoePrefillStats> {
    let s = t * top_k;
    ensure!(
        t >= 1 && top_k >= 1 && t <= scratch.t_cap,
        "exl3_moe_prefill_routed: {t} tokens exceeds the batch capacity {} — \
         the caller must token-batch",
        scratch.t_cap
    );
    ensure!(
        num_local >= 1 && num_local <= scratch.e_cap,
        "exl3_moe_prefill_routed: {num_local} local experts exceeds the \
         expert_count slab capacity {}",
        scratch.e_cap
    );
    ensure!(
        ov.gate_host.len() >= num_local
            && ov.up_host.len() >= num_local
            && ov.down_host.len() >= num_local,
        "exl3_moe_prefill_routed: host pointer tables shorter than num_local"
    );
    ensure!(
        hidden.is_multiple_of(128) && inter.is_multiple_of(128),
        "exl3_moe_prefill_routed: hidden {hidden} / inter {inter} must be \
         multiples of 128 (trellis tile + Hadamard block)"
    );
    ensure!(
        scratch.slot_f32.is_none() || s <= scratch.slot_cap,
        "exl3_moe_prefill_routed: {t} tokens x top_k {top_k} = {s} slots \
         exceeds the deterministic slot slab's {} rows — an undersized slab \
         is a silent out-of-bounds write, not a slower path",
        scratch.slot_cap
    );

    // 1) Staging (LOCAL-expert order + sentinel tail) and f16 ingress.
    exl3_moe_stage_sorted(
        gpu,
        token_to_perm,
        probs_f32,
        expert_offsets,
        scratch.token_sorted,
        scratch.weight_sorted,
        scratch.expert_count,
        local_start,
        num_local,
        top_k,
        s,
        stream,
    )?;
    super::super::exl3_bf16_to_f16(gpu, input_bf16, scratch.hidden_f16, t * hidden, stream)?;

    // 2) ATOMIC arm only: zero the fp32 accumulator (there both tiers
    //    accumulate into it). The DETERMINISTIC arm writes one row per
    //    sorted slot — every local slot exactly once — and its reduce
    //    (step 6) overwrites the accumulator whole, so neither wants a memset.
    if scratch.slot_f32.is_none() {
        gpu.memset_async(scratch.out_f32, 0, t * hidden * 4, stream)?;
    }

    // 3) Tier select. S <= cap: upstream's no-sync shortcut — every expert
    //    count is <= S <= cap, so the fused kernel covers everything and no
    //    host readback is needed (added upstream because the sync was ~33%
    //    idle at MTP verify shapes). Otherwise ONE stream-sync D2H of the
    //    LOCAL slice of expert_offsets (the host-sync tier).
    let cap = scratch.rows_per_expert;
    let mut num_active: i64 = -1;
    let mut overflow: Vec<(usize, usize, usize)> = Vec::new(); // (e_local, span_start, count)
    if exl3_moe_needs_host_sync(s, cap) {
        let mut raw = vec![0u8; (num_local + 1) * 4];
        gpu.copy_d2h_on_stream(expert_offsets.offset(local_start * 4), &mut raw, stream)?;
        let off: Vec<i32> = raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let lo = off[0];
        let mut active = 0i64;
        for e in 0..num_local {
            let count = (off[e + 1] - off[e]) as usize;
            match exl3_moe_expert_tier(count, cap) {
                Exl3MoeExpertTier::Idle => {}
                Exl3MoeExpertTier::Fused => active += 1,
                Exl3MoeExpertTier::Overflow => {
                    overflow.push((e, (off[e] - lo) as usize, count));
                }
            }
        }
        num_active = active;
        tier_stats_record(num_local, cap, &off, overflow.len());
    } else {
        tier_stats_record_nosync();
    }

    // 4) Fused launch (skipped only when the host-sync tier saw no fusable
    //    expert).
    if num_active != 0 {
        exl3_moe_fused(
            gpu, tables, scratch, t, top_k, hidden, inter, num_local, num_active, act_limit, locks,
            sm_count, stream,
        )?;
    }

    // 5) Overflow experts (count > cap): chunked trellis GEMMs + weighted
    //    scatter-add, stream-ordered behind the fused kernel.
    for &(e_local, span_start, count) in &overflow {
        run_overflow_expert(
            gpu, ov, tables, scratch, e_local, span_start, count, hidden, inter, act_limit, locks,
            sm_count, stream,
        )?;
    }

    // 6) DETERMINISTIC arm: reduce each token's top_k per-slot rows into the
    //    accumulator in FIXED flat-slot order — the step that makes prefill
    //    bit-reproducible whatever order the ticket scheduler ran the experts
    //    in. No-op on the atomic arm; both leave `out_f32` holding the sums.
    super::super::moe_prefill_det::reduce_slots_if_deterministic(
        gpu,
        scratch,
        token_to_perm,
        expert_offsets,
        local_start,
        num_local,
        top_k,
        t,
        hidden,
        stream,
    )?;

    // 7) Egress: fp32 accumulator -> BF16 token-major output.
    super::super::exl3_f32_to_bf16(gpu, scratch.out_f32, out_bf16, t * hidden, stream)?;

    Ok(Exl3MoePrefillStats {
        num_active,
        overflow_experts: overflow.len(),
    })
}
