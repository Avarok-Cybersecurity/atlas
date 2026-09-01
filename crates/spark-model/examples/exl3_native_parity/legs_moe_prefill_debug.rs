// SPDX-License-Identifier: AGPL-3.0-only
//! Bisection probe for the prefill-MoE leg (`EXL3_PF_DEBUG=1`): drives the
//! fused `exl3_moe` kernel with HOST-BUILT staging (bypassing sort +
//! stage_sorted), dumps the device-staged arrays for a real sorted case, and
//! runs a pure-overflow (count > 128, num_active = 0) isolation case with
//! chunk-slab dumps. Split from `legs_moe_prefill.rs` on the 500-LoC cap.
//! Found in anger: the f16 down-GEMM intermediate overflow (unit-scale
//! synthetic data) and the duplicate-token scatter race (now atomicAdd).

use anyhow::Result;
use half::f16;
use spark_model::layers::ops::{Exl3MoeOverflowCtx, exl3_moe_prefill_routed, moe_sort_by_expert};
use spark_runtime::gpu::DevicePtr;

use crate::legs_moe::{ProjSet, ref_token};
use crate::legs_moe_prefill::{H, I, TOP_K, alloc_slabs, gen_inputs};
use crate::util::{Ctx, Lcg, as_bytes, metrics};

/// Bisection probe (`EXL3_PF_DEBUG=1`): drive the fused kernel with
/// HOST-BUILT staging (bypassing sort + stage_sorted) at the smallest
/// possible shape, and dump the device-staged arrays for a real sorted case
/// — separates a fused-kernel defect from a staging defect.
pub fn debug_pf(ctx: &Ctx, rng: &mut Lcg) -> Result<()> {
    use spark_model::layers::ops::{exl3_moe_fused, exl3_moe_stage_sorted};
    let g = ctx.g;
    let stream = g.default_stream();
    let gate = ProjSet::generate(ctx, rng, 2, H, I)?;
    let upp = ProjSet::generate(ctx, rng, 2, H, I)?;
    let down = ProjSet::generate(ctx, rng, 2, I, H)?;
    let (gate_t, _o1) = gate.table(ctx, 0)?;
    let (up_t, _o2) = upp.table(ctx, 0)?;
    let (down_t, _o3) = down.table(ctx, 0)?;
    let tables = [gate_t, up_t, down_t];
    let sl = alloc_slabs(ctx)?;

    // Host-built staging: T=4, top_k=1, all four rows on expert 0.
    let t = 4usize;
    let (input_bf16, input_f16, _) = gen_inputs(rng, t);
    g.copy_h2d(&as_bytes(&input_bf16), sl.input)?;
    let ts: Vec<u8> = (0..t as i64).flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&ts, sl.scratch.token_sorted)?;
    let ws: Vec<u8> = (0..t)
        .flat_map(|_| f16::from_f32(1.0).to_bits().to_le_bytes())
        .collect();
    g.copy_h2d(&ws, sl.scratch.weight_sorted)?;
    let ec: Vec<u8> = [t as i64, 0i64]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    g.copy_h2d(&ec, sl.scratch.expert_count)?;
    spark_model::layers::ops::exl3_bf16_to_f16(g, sl.input, sl.scratch.hidden_f16, t * H, stream)?;
    g.memset_async(sl.scratch.out_f32, 0, t * H * 4, stream)?;
    exl3_moe_fused(
        g,
        &tables,
        &sl.scratch,
        t,
        1,
        H,
        I,
        1,
        1,
        0.0,
        ctx.locks,
        ctx.sms,
        stream,
    )?;
    g.synchronize(stream)?;
    let mut raw = vec![0u8; t * H * 4];
    g.copy_d2h(sl.scratch.out_f32, &mut raw)?;
    let y: Vec<f64> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    let ids = [0u32];
    let probs = [1.0f32];
    let mut y64 = Vec::new();
    for tok in 0..t {
        y64.extend(ref_token(
            &input_f16[tok * H..(tok + 1) * H],
            &ids,
            &probs,
            &gate,
            &upp,
            &down,
            0,
            2,
        ));
    }
    let (rr, mz) = metrics(&y, &y64);
    let nan = y.iter().filter(|v| !v.is_finite()).count();
    println!(
        "PF-DEBUG fused-only T=4 top_k=1 E=1: rel_rms={rr:.3e} max_z={mz:.3e} \
         nonfinite={nan}/{} y[0..4]={:?} ref[0..4]={:?}",
        y.len(),
        &y[..4],
        &y64[..4],
    );
    // Stage bisect: temp_state_u = gathered+rotated up input (stage 1),
    // temp_intermediate_u = up GEMM out (stage 2), temp_intermediate_g =
    // guad silu-product (stage 3). temp_state_g was overwritten by the down
    // GEMM (stage 4).
    let dump_f16 = |ptr: DevicePtr, n: usize, label: &str| -> Result<()> {
        let mut raw = vec![0u8; n * 2];
        g.copy_d2h(ptr, &mut raw)?;
        let v: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let bad = v.iter().filter(|x| !x.is_finite()).count();
        println!("PF-DEBUG {label}: nonfinite={bad}/{n} head={:?}", &v[..6]);
        Ok(())
    };
    dump_f16(sl.scratch.temp_state_u, t * H, "temp_state_u (gather)")?;
    dump_f16(sl.scratch.temp_inter_u, t * I, "temp_inter_u (up gemm)")?;
    dump_f16(sl.scratch.temp_inter_g, t * I, "temp_inter_g (guad)")?;
    dump_f16(sl.scratch.temp_state_g, t * H, "temp_state_g (down gemm)")?;

    // Staged-array dump for a real sorted T=3 top_k=4 case over 2 experts.
    let t = 3usize;
    let s = t * TOP_K;
    let (_, _, probs3) = gen_inputs(rng, t);
    let ids3: Vec<u32> = (0..s).map(|_| (rng.next() % 2) as u32).collect();
    let idb: Vec<u8> = ids3.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&idb, sl.indices)?;
    let pb: Vec<u8> = probs3.iter().flat_map(|v| v.to_le_bytes()).collect();
    g.copy_h2d(&pb, sl.probs)?;
    let sort_k = g.kernel("moe", "moe_sort_by_expert")?;
    moe_sort_by_expert(
        g,
        sort_k,
        sl.indices,
        sl.sorted_token_ids,
        sl.sorted_expert_ids,
        sl.expert_offsets,
        sl.token_to_perm,
        s as u32,
        2,
        TOP_K as u32,
        stream,
    )?;
    exl3_moe_stage_sorted(
        g,
        sl.token_to_perm,
        sl.probs,
        sl.expert_offsets,
        sl.scratch.token_sorted,
        sl.scratch.weight_sorted,
        sl.scratch.expert_count,
        0,
        2,
        TOP_K,
        s,
        stream,
    )?;
    g.synchronize(stream)?;
    let mut tsb = vec![0u8; s * 8];
    g.copy_d2h(sl.scratch.token_sorted, &mut tsb)?;
    let tsv: Vec<i64> = tsb
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let mut wsb = vec![0u8; s * 2];
    g.copy_d2h(sl.scratch.weight_sorted, &mut wsb)?;
    let wsv: Vec<f32> = wsb
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let mut ecb = vec![0u8; 3 * 8];
    g.copy_d2h(sl.scratch.expert_count, &mut ecb)?;
    let ecv: Vec<i64> = ecb
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("PF-DEBUG stage ids={ids3:?}");
    println!("PF-DEBUG stage probs={probs3:?}");
    println!("PF-DEBUG stage token_sorted={tsv:?}");
    println!("PF-DEBUG stage weight_sorted={wsv:?}");
    println!("PF-DEBUG stage expert_count={ecv:?}");

    // Pure-overflow case: top_k=1, T=160, every slot on expert 0 -> count
    // 160 > 128, num_active=0 (no fused launch) — isolates the reconstruct
    // + dense-GEMM + scatter-add path.
    {
        let t = 160usize;
        let (input_bf16, input_f16, _) = gen_inputs(rng, t);
        g.copy_h2d(&as_bytes(&input_bf16), sl.input)?;
        let ids: Vec<u32> = vec![0; t];
        let probs: Vec<f32> = (0..t).map(|_| 0.3 + 0.5 * rng.f()).collect();
        let idb: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&idb, sl.indices)?;
        let pb: Vec<u8> = probs.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&pb, sl.probs)?;
        moe_sort_by_expert(
            g,
            sort_k,
            sl.indices,
            sl.sorted_token_ids,
            sl.sorted_expert_ids,
            sl.expert_offsets,
            sl.token_to_perm,
            t as u32,
            2,
            1,
            stream,
        )?;
        let gh: Vec<[u64; 3]> = gate
            .dev
            .iter()
            .map(|d| [d.trellis.0, d.suh.0, d.svh.0])
            .collect();
        let uh: Vec<[u64; 3]> = upp
            .dev
            .iter()
            .map(|d| [d.trellis.0, d.suh.0, d.svh.0])
            .collect();
        let dh: Vec<[u64; 3]> = down
            .dev
            .iter()
            .map(|d| [d.trellis.0, d.suh.0, d.svh.0])
            .collect();
        let ov = Exl3MoeOverflowCtx {
            gate_host: &gh,
            up_host: &uh,
            down_host: &dh,
        };
        let stats = exl3_moe_prefill_routed(
            g,
            sl.input,
            sl.probs,
            sl.expert_offsets,
            sl.token_to_perm,
            sl.out,
            &tables,
            &ov,
            &sl.scratch,
            ctx.locks,
            t,
            1,
            H,
            I,
            0,
            2,
            0.0,
            ctx.sms,
            stream,
        )?;
        g.synchronize(stream)?;
        println!(
            "PF-DEBUG overflow-only stats: num_active={} overflow={}",
            stats.num_active, stats.overflow_experts
        );
        let mut bytes = vec![0u8; t * H * 2];
        g.copy_d2h(sl.out, &mut bytes)?;
        let y: Vec<f64> = bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
            .collect();
        let ids0 = [0u32];
        let mut y64 = Vec::new();
        for tok in 0..t {
            y64.extend(ref_token(
                &input_f16[tok * H..(tok + 1) * H],
                &ids0,
                &probs[tok..tok + 1],
                &gate,
                &upp,
                &down,
                0,
                2,
            ));
        }
        let (rr, mz) = metrics(&y, &y64);
        println!(
            "PF-DEBUG overflow-only T=160: rel_rms={rr:.3e} max_z={mz:.3e} \
             y[0..4]={:?} ref[0..4]={:?}",
            &y[..4],
            &y64[..4],
        );
        // Chunk slab dumps (single 160-row chunk with OV_CHUNK=256).
        let dump_f16 = |ptr: DevicePtr, n: usize, label: &str| -> Result<()> {
            let mut raw = vec![0u8; n * 2];
            g.copy_d2h(ptr, &mut raw)?;
            let v: Vec<f32> = raw
                .chunks_exact(2)
                .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect();
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            println!("PF-DEBUG {label}: nonfinite={bad}/{n} head={:?}", &v[..6]);
            Ok(())
        };
        let dump_f32 = |ptr: DevicePtr, n: usize, label: &str| -> Result<()> {
            let mut raw = vec![0u8; n * 4];
            g.copy_d2h(ptr, &mut raw)?;
            let v: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            println!("PF-DEBUG {label}: nonfinite={bad}/{n} head={:?}", &v[..6]);
            Ok(())
        };
        dump_f16(sl.scratch.ov_a_f16, 160 * H, "ov_a (gathered)")?;
        dump_f16(sl.scratch.ov_gate_f16, 160 * I, "ov_gate (silu prod)")?;
        dump_f16(sl.scratch.ov_up_f16, 160 * I, "ov_up")?;
        dump_f32(sl.scratch.ov_down_f32, 160 * H, "ov_down")?;
        println!(
            "PF-DEBUG input head={:?}",
            &input_f16[..4]
                .iter()
                .map(|&b| f16::from_bits(b).to_f32())
                .collect::<Vec<_>>()
        );
    }
    for p in sl.owned.iter() {
        g.free(*p).ok();
    }
    Ok(())
}
