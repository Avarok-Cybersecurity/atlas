// SPDX-License-Identifier: AGPL-3.0-only

//! Device-side MoE expert down_proj LoRA fold launcher (`moe_lora_grouped_down`).
//!
//! Replaces the host-synced per-expert loop (`crate::lora::expert_apply`, which
//! D2H-copies `expert_offsets` and drives a host launch count — both illegal
//! under CUDA-graph capture) with a single two-launch kernel that reads
//! `expert_offsets` DEVICE-side. The grid is a STATIC worst-case bound
//! (`worst_case_m_tiles = ceil(te/64)`, matching the base grouped GEMM), so the
//! launch shape is constant across capture/replay; per-tile early-return on an
//! empty / out-of-range / unadapted expert span keeps it correct without a host
//! value. See `kernels/gb10/common/moe_lora_grouped_down.cu`.
//!
//! The fold math is BYTE-IDENTICAL to `apply_lora_bgmv` / per-row
//! `apply_lora_delta(m=1)` (shrink→BF16 xa, expand→BF16 delta, then
//! `base += scale·fp32(bf16(delta))`), so one kernel serves the nvfp4, bf16, and
//! fp8 grouped prefill paths — all write the same sorted BF16 `expert_down_out`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::lora_delta::LoraKernels;

/// Per-EXPERT routing tables for the grouped down fold — the expert-keyed
/// analogue of the slot-keyed [`super::lora_delta::LoraRoute`]. Built once at
/// adapter install from the layer's `Down` pairs; load-time-fixed device
/// addresses, so they are stable kernel args across capture/replay (adapter
/// identity for a mixed batch flows through the per-row `moe_row_adapter`, not
/// these tables).
///
/// `n_experts` is the TABLE LENGTH = `max adapted expert id + 1` (NOT the
/// layer's full `num_experts`): the grid launches `grid.z = n_experts`, and
/// every adapted expert has index `< n_experts`, so any higher-index expert is
/// unadapted and correctly folds nothing. `expert_offsets[e]` / `[e+1]` are read
/// for `e < n_experts <= num_experts`, always in range of the `[num_experts+1]`
/// prefix sum.
#[derive(Debug, Clone, Copy)]
pub struct MoeExpertRoute {
    /// `[n_experts]` u64 device array of `A_e` addresses (`0` = expert unadapted).
    pub a_table: DevicePtr,
    /// `[n_experts]` u64 device array of `B_e` addresses (`0` = expert unadapted).
    pub b_table: DevicePtr,
    /// `[n_experts]` f32 device array of per-expert `scale_e` (`0.0` where unadapted).
    pub scale_table: DevicePtr,
    /// Table length = max adapted expert id + 1 (== grid.z).
    pub n_experts: u32,
    /// Contraction dim of the shrink stage (`moe_intermediate_size`).
    pub k_in: u32,
    /// Output dim of the expand stage (`hidden_size`).
    pub n_out: u32,
    /// Padded rank (contraction dim of the expand stage; row stride of `B_e`).
    pub max_rank: u32,
}

/// PURE (GPU-free, unit-tested): pack a set of adapted-expert
/// `(expert_id, a_addr, b_addr, scale)` entries into dense `[n_experts]` tables
/// indexed by expert id, with `0` / `0.0` at every unadapted slot.
/// `n_experts = max expert_id + 1`. Returns `None` when `entries` is empty (a
/// router-only adapter installs no expert route). Duplicate expert ids keep the
/// LAST entry (callers pass at most one `Down` pair per expert).
pub fn pack_expert_tables(entries: &[(u16, u64, u64, f32)]) -> Option<ExpertTables> {
    let max_e = entries.iter().map(|(e, ..)| *e).max()?;
    let n = max_e as usize + 1;
    let mut a = vec![0u64; n];
    let mut b = vec![0u64; n];
    let mut scale = vec![0.0f32; n];
    for &(e, a_addr, b_addr, sc) in entries {
        let i = e as usize;
        a[i] = a_addr;
        b[i] = b_addr;
        scale[i] = sc;
    }
    Some(ExpertTables {
        a,
        b,
        scale,
        n_experts: n as u32,
    })
}

/// Host-side packed tables from [`pack_expert_tables`], ready for H2D upload.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertTables {
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    pub scale: Vec<f32>,
    pub n_experts: u32,
}

/// Launch the device-side grouped fold for ONE chunk window `[row_offset,
/// row_end)` of the sorted rows. Down (`x_gather==0`): `x` = post-SiLU sorted
/// activations (`[te, k_in]` BF16), `base_out` = sorted `expert_down_out`.
/// Gate/up (`x_gather==1`): `x` = the TOKEN-MAJOR `expert_input` (`[num_tokens,
/// k_in=hidden]` BF16, gathered per sorted row via `sorted_token_ids`), `base_out`
/// = sorted `expert_gate_out`/`expert_up_out` (`[te, n_out=inter]`). In both,
/// `base_out` is `[te, n_out]` BF16 folded IN PLACE, `expert_offsets` = the device
/// `[num_experts+1]` i32
/// prefix sum, `sorted_token_ids` = the device `[te]` i32 sorted-row→token map,
/// `moe_row_adapter` = `[num_tokens]` i32 device map (`< 0` = base skip) or
/// `DevicePtr::NULL` for the single-active-adapter path, `xa` = the fixed-address
/// `[cap, max_rank]` BF16 shrink scratch indexed at the LOCAL row `r-row_offset`
/// (so the caller only needs `>= (row_end-row_offset)` rows, NOT `>= te`). The
/// hooks loop `[0, te)` in windows of `cap`; a single call at `row_offset=0,
/// row_end=te` (te <= cap) is bit-identical to the pre-chunk kernel.
///
/// ARG ORDER is in lockstep with `moe_lora_grouped_down.cu` (cuLaunchKernel is
/// type-blind; the byte-identity oracle is the only guard — keep both in sync).
/// `row_offset`/`row_end` are appended LAST in both kernels, so existing arg
/// offsets are untouched.
#[allow(clippy::too_many_arguments)]
pub fn moe_lora_grouped_down(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    route: &MoeExpertRoute,
    x: DevicePtr,                // [te, k_in] BF16 (post-SiLU sorted)
    base_out: DevicePtr,         // [te, n_out] BF16, folded in place
    expert_offsets: DevicePtr,   // [num_experts+1] i32 DEVICE
    sorted_token_ids: DevicePtr, // [te] i32 DEVICE
    moe_row_adapter: DevicePtr,  // [num_tokens] i32 DEVICE or NULL
    xa: DevicePtr,               // [cap, max_rank] BF16 scratch (fixed address, LOCAL-row indexed)
    row_offset: u32,             // first ABSOLUTE sorted row of this chunk window
    row_end: u32,                // one-past-last ABSOLUTE row (== min(row_offset+cap, te))
    x_gather: u32, // 0: x row = sorted row r (down); 1: x row = sorted_token_ids[r] (gate/up)
    stream: u64,
) -> Result<()> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    anyhow::ensure!(
        kernels.moe_down_shrink_k.0 != 0 && kernels.moe_down_expand_fold_k.0 != 0,
        "moe_lora_grouped_down kernels unresolved (module `moe_lora_grouped_down` missing \
         from the compiled kernel set — CUDA build required)"
    );
    // Grid.y covers the window (<= cap rows), not the whole te: each expert's span
    // ∩ window has at most `window` rows, and per-expert tiles rebase to the
    // window start device-side.
    let window = row_end.saturating_sub(row_offset);
    let wc = div_ceil(window, MLG_M_TILE).max(1);

    // Kernel 1: shrink — xa[local_row, max_rank] = x @ A_e^T.
    // grid = (ceil(max_rank/4), ceil(window/64), n_experts)  block = (256,1,1).
    KernelLaunch::new(gpu, kernels.moe_down_shrink_k)
        .grid([div_ceil(route.max_rank, 4), wc, route.n_experts])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(moe_row_adapter)
        .arg_ptr(route.a_table)
        .arg_ptr(xa)
        .arg_u32(route.n_experts)
        .arg_u32(route.max_rank)
        .arg_u32(route.k_in)
        .arg_u32(x_gather)
        .arg_u32(row_offset)
        .arg_u32(row_end)
        .launch(stream)?;

    // Kernel 2: expand + fold — base_out[r] += scale_e * (xa[r-row_offset] @ B_e^T).
    // grid = (ceil(n_out/4), ceil(window/64), n_experts)  block = (256,1,1).
    KernelLaunch::new(gpu, kernels.moe_down_expand_fold_k)
        .grid([div_ceil(route.n_out, 4), wc, route.n_experts])
        .block([256, 1, 1])
        .arg_ptr(xa)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(moe_row_adapter)
        .arg_ptr(route.b_table)
        .arg_ptr(route.scale_table)
        .arg_ptr(base_out)
        .arg_u32(route.n_experts)
        .arg_u32(route.n_out)
        .arg_u32(route.max_rank)
        .arg_u32(row_offset)
        .arg_u32(row_end)
        .launch(stream)
}

/// PURE (GPU-free, unit-tested): the shrink/expand grid `grid.y` (m-tile count)
/// for one chunk window `[row_offset, row_end)` — `ceil((row_end-row_offset)/64)`,
/// min 1. Matches the launcher's `wc`; exposed so the chunk-boundary math is
/// verifiable without a GPU. A full-window call (`row_offset=0, row_end=te`)
/// returns exactly the pre-chunk `ceil(te/64)`.
pub fn grouped_down_wc(row_offset: u32, row_end: u32) -> u32 {
    use spark_runtime::kernel_args::div_ceil;
    div_ceil(row_end.saturating_sub(row_offset), MLG_M_TILE).max(1)
}

/// M_TILE the static worst-case grid pairs with — must match the `MLG_M_TILE`
/// `#define` in `moe_lora_grouped_down.cu` AND the base grouped GEMM's
/// `worst_case_m_tiles = ceil(total_expanded/64)` sizing.
pub const MLG_M_TILE: u32 = 64;

/// PURE (GPU-free, unit-tested): the exact `(shrink, expand)` grid triples for
/// the decode gather-fold, given the route dims and the flat row count. Each is
/// `[ceil(out/4), n_slots, 1]` with a `(256,1,1)` block (one 64-lane group per
/// output, `N_PER_BLOCK = 4`). `n_slots` is a host constant per captured graph,
/// so the shape is EXACT (no worst-case tiles) and capture-stable.
pub fn gather_bgmv_grids(max_rank: u32, n_out: u32, n_slots: u32) -> ([u32; 3], [u32; 3]) {
    use spark_runtime::kernel_args::div_ceil;
    (
        [div_ceil(max_rank, 4), n_slots, 1],
        [div_ceil(n_out, 4), n_slots, 1],
    )
}

/// PURE: the owning token index of a flat `(token, slot)` row — mirrors the
/// kernel's `row / top_k` so the per-token `row_adapter` gather is verifiable.
pub fn gather_row_token(row: u32, top_k: u32) -> u32 {
    row / top_k
}

/// SOLID Incr-4: launch the DECODE-path MoE expert down fold. The unsorted,
/// slot-major analogue of [`moe_lora_grouped_down`] — instead of an
/// `expert_offsets` prefix sum over sorted rows, each flat `(token, slot)` row
/// gathers its expert from `indices[row]` (the same `indices_dev` the fused
/// expert GEMV routed on) and its base/adapt decision from
/// `row_adapter[row / top_k]` (`< 0` = base skip, or `DevicePtr::NULL` to fold
/// every row on the single-active-adapter path).
///
/// `x` = the post-swiglu activations (`silu(gate)*up`, produced by the caller's
/// `moe_silu_mul` launch into a packed `[n_slots, k_in]` BF16 scratch — the SAME
/// kernel + BF16 round the prefill fold uses, so the delta is BF16-ULP identical
/// to prefill). `base_out` = the slot-major `expert_down_out` (`[n_slots, n_out]`
/// BF16, folded IN PLACE before `moe_weighted_sum_blend`, so the router weight
/// multiplies base+delta). `xa` = the fixed-address `[n_slots, max_rank]` BF16
/// shrink scratch. The grid is EXACT (`n_slots` is a host constant per captured
/// graph) — no worst-case tiles — and all args are pointer/value-stable, so the
/// launch captures cleanly.
///
/// ARG ORDER is in lockstep with `moe_lora_gather_bgmv.cu` (cuLaunchKernel is
/// type-blind; keep both in sync).
#[allow(clippy::too_many_arguments)]
pub fn moe_lora_gather_bgmv(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    route: &MoeExpertRoute,
    x: DevicePtr,           // [n_slots, k_in] BF16 (silu(gate)*up)
    base_out: DevicePtr,    // [n_slots, n_out] BF16 = expert_down_out, folded in place
    indices: DevicePtr,     // [n_slots] u32 = indices_dev (expert id per flat slot)
    row_adapter: DevicePtr, // [num_tokens] i32 (<0 skip) or DevicePtr::NULL (fold all)
    xa: DevicePtr,          // [n_slots, max_rank] BF16 scratch (fixed address)
    n_slots: u32,           // num_tokens * top_k
    top_k: u32,
    x_gather: u32, // 0: x row = flat slot (down); 1: x row = token = row/top_k (gate/up)
    stream: u64,
) -> Result<()> {
    use spark_runtime::kernel_args::KernelLaunch;

    anyhow::ensure!(
        kernels.moe_gather_shrink_k.0 != 0 && kernels.moe_gather_expand_fold_k.0 != 0,
        "moe_lora_gather_bgmv kernels unresolved (module `moe_lora_gather_bgmv` missing \
         from the compiled kernel set — CUDA build required)"
    );
    let (shrink_grid, expand_grid) = gather_bgmv_grids(route.max_rank, route.n_out, n_slots);

    // Kernel 1: shrink — xa[n_slots, max_rank] = x @ A_e^T.
    // grid = (ceil(max_rank/4), n_slots, 1)  block = (256,1,1).
    KernelLaunch::new(gpu, kernels.moe_gather_shrink_k)
        .grid(shrink_grid)
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(indices)
        .arg_ptr(row_adapter)
        .arg_ptr(route.a_table)
        .arg_ptr(xa)
        .arg_u32(n_slots)
        .arg_u32(top_k)
        .arg_u32(route.n_experts)
        .arg_u32(route.max_rank)
        .arg_u32(route.k_in)
        .arg_u32(x_gather)
        .launch(stream)?;

    // Kernel 2: expand + fold — base_out += scale_e * (xa @ B_e^T).
    // grid = (ceil(n_out/4), n_slots, 1)  block = (256,1,1).
    KernelLaunch::new(gpu, kernels.moe_gather_expand_fold_k)
        .grid(expand_grid)
        .block([256, 1, 1])
        .arg_ptr(xa)
        .arg_ptr(indices)
        .arg_ptr(row_adapter)
        .arg_ptr(route.b_table)
        .arg_ptr(route.scale_table)
        .arg_ptr(base_out)
        .arg_u32(n_slots)
        .arg_u32(top_k)
        .arg_u32(route.n_experts)
        .arg_u32(route.n_out)
        .arg_u32(route.max_rank)
        .launch(stream)
}

#[cfg(test)]
#[path = "moe_lora_grouped_tests.rs"]
mod tests;
