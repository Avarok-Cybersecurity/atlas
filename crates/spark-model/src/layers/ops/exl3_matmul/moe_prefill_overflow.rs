// SPDX-License-Identifier: AGPL-3.0-only

//! Overflow tier of the native EXL3 MoE prefill pipeline — an expert routed
//! more sorted rows in one batch than the fused `exl3_moe` kernel's temp-slab
//! height (`Exl3MoePrefillScratch::rows_per_expert`, resolved in
//! `moe_prefill_cap.rs`; default 1024, legacy 128) is served by
//! upstream's `run_single_expert` tier instead: the SAME packed trellis
//! through the cooperative [`exl3_gemm`](super::super::exl3_gemm) — chunked
//! gather of its sorted rows (fp16) -> gate/up trellis GEMM (fp16 C) ->
//! upstream's half-precision SiLU·mul -> down trellis GEMM (fp32 C) ->
//! weighted fp32 scatter-add into the accumulator the fused kernel writes.
//!
//! Every tier therefore decodes the identical trellis at the identical fp16
//! activation precision, so an expert's output does not depend on which tier
//! served it. No weight reconstruction, no device allocation and no host
//! sync on this path — it runs entirely out of the persistent `ov_*` slabs,
//! stream-ordered behind the fused launch. Split from `moe_prefill.rs` on
//! the 500-LoC cap.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::moe_decode::Exl3MoeProj;
use super::Exl3MoePrefillScratch;

/// Host-side context for the overflow (count > cap) path: per-local-expert
/// raw device addresses `[trellis, suh, svh]` for each projection (the same
/// pointers the device tables hold — `Exl3ExpertPtrTable::host_ptrs`).
pub struct Exl3MoeOverflowCtx<'a> {
    /// `[num_local]` gate `[trellis, suh, svh]` device addresses.
    pub gate_host: &'a [[u64; 3]],
    /// `[num_local]` up pointers.
    pub up_host: &'a [[u64; 3]],
    /// `[num_local]` down pointers.
    pub down_host: &'a [[u64; 3]],
}

/// Gather `m` fp16 rows of the RAW activation ingress (`hidden_f16`) by the
/// sorted token index (16-bit copy — the kernel is dtype-agnostic).
fn gather_rows_f16(
    gpu: &dyn GpuBackend,
    hidden_f16: DevicePtr,
    token_sorted_base: DevicePtr,
    out_f16: DevicePtr,
    m: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_gather_rows_h16")?;
    let total = m * hidden;
    let grid = div_ceil(total as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden_f16)
        .arg_ptr(token_sorted_base)
        .arg_ptr(out_f16)
        .arg_u64(hidden as u64)
        .arg_u64(total as u64)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
fn scatter_add_f32(
    gpu: &dyn GpuBackend,
    down_f32: DevicePtr,
    token_sorted_base: DevicePtr,
    weight_sorted_base: DevicePtr,
    out_f32: DevicePtr,
    m: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_scatter_add_f32")?;
    let total = m * hidden;
    let grid = div_ceil(total as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(down_f32)
        .arg_ptr(token_sorted_base)
        .arg_ptr(weight_sorted_base)
        .arg_ptr(out_f32)
        .arg_u64(hidden as u64)
        .arg_u64(total as u64)
        .launch(stream)
}

/// One overflow expert (count > cap rows): chunked native trellis GEMMs over
/// its sorted rows, weighted scatter-add into the fp32 accumulator. The
/// `A_had` slab is dedicated (never aliases the gathered A) because the same
/// gathered rows feed BOTH the gate and the up GEMM, each with its own `suh`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_overflow_expert(
    gpu: &dyn GpuBackend,
    ov: &Exl3MoeOverflowCtx,
    tables: &[Exl3MoeProj; 3],
    scratch: &Exl3MoePrefillScratch,
    e_local: usize,
    span_start: usize, // slot offset in the LOCAL-sorted token/weight arrays
    count: usize,
    hidden: usize,
    inter: usize,
    act_limit: f32,
    locks: DevicePtr,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    let gemm =
        |a: DevicePtr, host: &[[u64; 3]], proj: &Exl3MoeProj, c: DevicePtr, m, k, n, c_fp32| {
            let p = host[e_local];
            super::super::exl3_gemm(
                gpu,
                a,
                DevicePtr(p[0]),
                c,
                m,
                k,
                n,
                proj.k_bits,
                proj.cb,
                c_fp32,
                locks,
                DevicePtr(p[1]),
                scratch.ov_a_had_f16,
                DevicePtr(p[2]),
                None,
                sm_count,
                stream,
            )
        };
    let mut done = 0usize;
    while done < count {
        let m = scratch.ov_chunk.min(count - done);
        let slot = span_start + done;
        let ts = scratch.token_sorted.offset(slot * 8);
        let ws = scratch.weight_sorted.offset(slot * 2);
        gather_rows_f16(
            gpu,
            scratch.hidden_f16,
            ts,
            scratch.ov_a_f16,
            m,
            hidden,
            stream,
        )?;
        gemm(
            scratch.ov_a_f16,
            ov.gate_host,
            &tables[0],
            scratch.ov_gate_f16,
            m,
            hidden,
            inter,
            false,
        )?;
        gemm(
            scratch.ov_a_f16,
            ov.up_host,
            &tables[1],
            scratch.ov_up_f16,
            m,
            hidden,
            inter,
            false,
        )?;
        super::super::exl3_silu_mul_f16(
            gpu,
            scratch.ov_gate_f16,
            scratch.ov_up_f16,
            scratch.ov_gate_f16,
            act_limit,
            m * inter,
            stream,
        )?;
        gemm(
            scratch.ov_gate_f16,
            ov.down_host,
            &tables[2],
            scratch.ov_down_f32,
            m,
            inter,
            hidden,
            true,
        )?;
        // DETERMINISTIC arm: this chunk's weighted rows go to their OWN
        // slots (`slot` is their local-sorted base), to be reduced with the
        // fused tier's in fixed order. The atomic arm below carries the same
        // unordered-fp32-accumulation defect as the fused kernel's epilogue,
        // and this is the tier that fires on LONG prefills — leaving it
        // atomic would keep long-context serving nondeterministic.
        if let Some(slots) = scratch.slot_f32 {
            super::super::moe_prefill_det::exl3_moe_store_slots_f32(
                gpu,
                scratch.ov_down_f32,
                ws,
                slots.offset(slot * hidden * 4),
                m,
                hidden,
                stream,
            )?;
        } else {
            scatter_add_f32(
                gpu,
                scratch.ov_down_f32,
                ts,
                ws,
                scratch.out_f32,
                m,
                hidden,
                stream,
            )?;
        }
        done += m;
    }
    Ok(())
}
