// SPDX-License-Identifier: AGPL-3.0-only

//! The grouped-GEMM launch and the routed-MoE prefill driver that sequences it.
//!
//! Split out of `forward_prefill_gemm.rs` to keep that file under the 500-LoC cap; see
//! `tile.rs` for the tile geometry and env-lever resolution this calls into.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::super::forward::Glm5NextMlpWorkspace;
use super::super::weights::Glm5NextMoeWeights;
use super::super::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use super::tile::{GemmTile, gemm_tile, max_m_tiles_from_offsets, prefill_gemm_exact_tiles};

/// `C[te, n_out] = gather(A)[te, k] @ dequant(expert weights)^T`, all experts, ONE launch.
///
/// 🪤 grid.x, grid.y and the block width ALL come from `tile` — they are properties of the
/// kernel entry point, not constants. Mirrors `ops::moe_w4a16_grouped_gemm_ptrtable`, which
/// is `pub` but lives behind `MoeLayer`'s own dispatch — called directly here to keep GLM
/// off that type.
#[allow(clippy::too_many_arguments)]
fn grouped_gemm(
    gpu: &dyn GpuBackend,
    k: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    t: &super::super::weights::Glm5NextExpertPtrTable,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: usize,
    n_out: usize,
    kk: usize,
    max_m_tiles: u32,
    tile: GemmTile,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            (n_out as u32).div_ceil(tile.n_tile),
            max_m_tiles,
            num_experts as u32,
        ])
        .block([tile.threads, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts as u32)
        .arg_u32(n_out as u32)
        .arg_u32(kk as u32)
        .launch(stream)
}

/// Sort → grouped gate GEMM → grouped up GEMM → clamped SwiGLU → grouped down GEMM.
///
/// Consumes the router's `ws.ids` (already produced by the caller) and leaves the routed
/// expert outputs in `ws.expert_out` **in expert-sorted row order**, addressed by
/// `ws.token_to_perm`. The caller finishes with `glm5next_moe_combine_indexed`.
///
/// 🪤 `ws.expert_out` MUST already be zeroed by the caller: under EP the grouped GEMM
/// writes nothing for a remote expert's rows, exactly as the GEMV path does.
///
/// 🔴 The WHOLE row group goes through one launch per projection. The 8-row
/// `MOE_ROW_BATCH_MAX_ROWS` cap is a property of the union-table GEMV and does not apply
/// here — that is the entire point of the path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_moe_grouped_prefill(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    rows: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    let te = rows * cfg.top_k;
    let mi = cfg.moe_intermediate;
    if te > ws.max_total_expanded() {
        bail!(
            "GLM MoE grouped prefill: {te} routed slots exceed the {} a workspace built for \
             {} rows holds",
            ws.max_total_expanded(),
            ws.max_rows()
        );
    }

    // ── 1. counting sort: ids[rows, top_k] → expert-contiguous rows ──
    // 🪤 `moe_sort_by_expert` indexes `counts[topk_ids[i]]` with NO range guard, so every
    // id must be a real expert. `glm5next_router_topk` only emits `-1` when `top_k` exceeds
    // the expert count, which `Glm5NextMlpConfig::validate` refuses (`top_k <= num_experts`).
    KernelLaunch::new(gpu, k.moe_sort_by_expert)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ws.ids())
        .arg_ptr(ws.sorted_token_ids())
        .arg_ptr(ws.sorted_expert_ids())
        .arg_ptr(ws.expert_offsets())
        .arg_ptr(ws.token_to_perm())
        .arg_u32(te as u32)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        .launch(stream)?;

    // ── 2. grid height from the REAL histogram ──
    // 🪤 `copy_d2h_on_stream` drains the stream inside the call, so the host read below
    // happens-after the sort. It is a host stall, paid once per routed layer per prefill
    // sub-chunk — never on decode or verify, which never reach this path.
    let tile = gemm_tile();
    let worst_case = te.div_ceil(tile.m_tile).max(1) as u32;
    let max_m_tiles = if prefill_gemm_exact_tiles() {
        let mut off_raw = vec![0u8; (cfg.num_experts + 1) * 4];
        gpu.copy_d2h_on_stream(ws.expert_offsets(), &mut off_raw, stream)?;
        let offsets: Vec<i32> = off_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        max_m_tiles_from_offsets(&offsets, worst_case, tile.m_tile)
    } else {
        worst_case
    };

    // ── 3. gate + up: gather x by sorted_token_ids INSIDE the kernel (no permute pass) ──
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        x,
        &w.ptrs.gate,
        ws.a_gate(),
        ws.expert_offsets(),
        ws.sorted_token_ids(),
        cfg.num_experts,
        mi,
        cfg.hidden,
        max_m_tiles,
        tile,
        stream,
    )?;
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        x,
        &w.ptrs.up,
        ws.a_up(),
        ws.expert_offsets(),
        ws.sorted_token_ids(),
        cfg.num_experts,
        mi,
        cfg.hidden,
        max_m_tiles,
        tile,
        stream,
    )?;

    // ── 4. clamped SwiGLU over every sorted row at once ──
    // 🪤 GLM's clamp is ASYMMETRIC and is NOT `moe_silu_mul`. Elementwise, so the sorted
    // layout changes nothing. Rows belonging to a remote expert hold uninitialised values
    // here; the down GEMM skips them on the same null-pointer test, so they are never read.
    super::super::forward::swiglu_rows(
        gpu,
        k.swiglu,
        ws.a_gate(),
        ws.a_up(),
        ws.a_act(),
        te * mi,
        cfg.swiglu_limit,
        stream,
    )?;

    // ── 5. down: A is ALREADY expert-sorted, so the gather map is NULL ──
    // 🪤 `DevicePtr(0)` is the kernel's documented "no gather" sentinel
    // (`sorted_token_ids ? sorted_token_ids[...] : cta_m + row`). Passing the sort map here
    // would gather sorted rows by TOKEN index — silently wrong, same shape, no error.
    grouped_gemm(
        gpu,
        k.moe_grouped_gemm,
        ws.a_act(),
        &w.ptrs.down,
        ws.expert_out(),
        ws.expert_offsets(),
        DevicePtr(0),
        cfg.num_experts,
        cfg.hidden,
        mi,
        max_m_tiles,
        tile,
        stream,
    )?;
    Ok(())
}
