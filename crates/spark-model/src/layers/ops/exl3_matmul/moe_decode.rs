// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-tier native EXL3 MoE pipeline: the routed experts of one MoE layer
//! served DIRECTLY from packed trellis as exactly THREE `exl3_mgemm` calls
//! (gate, up, down) over `S = num_tokens * top_k` (token, expert) slots —
//! upstream ExLlamaV3's `BC_BlockSparseMLP.run_bszN` tier (T <= 8), ported
//! to Atlas's device routing state.
//!
//! Pipeline (all device-side, no D2H, no allocation — 901 playbook):
//!
//! ```text
//!   stage_routing:  u32 ids + f32 probs -> i64 local b_indices (-1 remote)
//!                                          + f16 b_weights
//!   replicate_a:    bf16 [T, H] -> fp16 A [S, H] (slot s copies row s/top_k)
//!   GATE  mgemm:    A [S,1,H]      -> C_gate f16 [S,1,I]   (b_indices only)
//!   UP    mgemm:    A [S,1,H]      -> C_up   f16 [S,1,I]
//!   silu_mul:       f16 silu(gate)*up -> inter [S,1,I]  (upstream numerics)
//!   DOWN  mgemm:    inter [S,1,I]  -> C_down f32 [S,1,H], b_weights folded,
//!                   num_tokens=T -> rows 0..T-1 hold the per-token WEIGHTED
//!                   routed sums (fp32 grouped reduction; -1 slots skipped)
//!   egress:         f32 rows 0..T-1 -> bf16 [T, H]
//! ```
//!
//! Contracts carried from the vendored kernel (`exl3_gemm_kernel.cuh`):
//!  * `b_indices` are LOCAL indices over a DENSE per-projection pointer
//!    table; EP-remote experts are `-1` slots (skipped by compute AND the
//!    reduction). NEVER null table entries — a null entry reachable through
//!    a valid index sums stale C scratch. NEVER `min_index`/`max_index`
//!    filtering (128-slot cap + module-scope `__device__` globals).
//!  * `A_had` must NOT alias `A` (mgemm slabs are unverified for the gemm's
//!    aliasing sanction) and needs `S * k` halves per call.
//!  * gate/up leave their per-slot C UNREDUCED (no `b_weights`); slots with
//!    `-1` indices keep stale gate/up/inter rows, which the down call never
//!    reads (its compute skips the slot and its reduction skips the index).
//!  * Cooperative launches: NOT graph-capturable; the caller must hold the
//!    decode-graph veto (`MoeLayer::exl3_native_active`) and launch on the
//!    primary stream sharing ONE locks buffer per concurrent caller.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::exl3_mgemm;

/// One projection's dense-local expert pointer tables + its mgemm template
/// selection. Mirrors `moe::tables::Exl3ExpertPtrTable`'s device fields (this
/// struct is `pub` so the parity example can drive the exact serving
/// pipeline with synthetic experts).
#[derive(Clone, Copy, Debug)]
pub struct Exl3MoeProj {
    /// `[num_local]` u64 device pointers to each local expert's trellis.
    pub trellis_ptrs: DevicePtr,
    /// `[num_local]` u64 device pointers to each local expert's suh.
    pub suh_ptrs: DevicePtr,
    /// `[num_local]` u64 device pointers to each local expert's svh.
    pub svh_ptrs: DevicePtr,
    /// Trellis bits/weight (one kernel template per launch).
    pub k_bits: u32,
    /// Kernel codebook index: 1 = MCG, 2 = MUL1.
    pub cb: u32,
}

/// Device scratch for the pipeline; capacities are in SLOTS (`s_cap`), see
/// `moe::tables::Exl3MoeState` (whose slabs the serving path passes here).
#[derive(Clone, Copy, Debug)]
pub struct Exl3MoeScratch {
    /// fp16 `[s_cap, hidden]` replicated activation ingress (mgemm A).
    pub a_f16: DevicePtr,
    /// fp16 `[s_cap, hidden]` A_had rotation scratch — never aliases A.
    pub a_had_f16: DevicePtr,
    /// Capacity of `a_had_f16` in halves (>= s_cap*hidden).
    pub a_had_capacity_elems: usize,
    /// f16 `[s_cap, inter]` gate C (stored in the f32-sized slab; only
    /// `s_cap*inter*2` bytes are used).
    pub c_gate_f16: DevicePtr,
    /// f16 `[s_cap, inter]` up C.
    pub c_up_f16: DevicePtr,
    /// f16 `[s_cap, inter]` silu(gate)*up (the down call's A).
    pub inter_f16: DevicePtr,
    /// f32 `[s_cap, hidden]` down C / grouped-reduction scratch.
    pub c_down_f32: DevicePtr,
    /// i64 `[s_cap]` staged local expert indices (-1 = remote).
    pub b_indices: DevicePtr,
    /// f16 `[s_cap]` staged routing weights (slab may be wider; only
    /// `s_cap*2` bytes are used).
    pub b_weights: DevicePtr,
    /// Slot capacity of every slab above.
    pub s_cap: usize,
}

/// Stage Atlas's device routing state into the mgemm `b_indices`/`b_weights`
/// forms (plain launch). `indices_u32`: `[s]` GLOBAL expert ids;
/// `probs_f32`: `[s]` f32 routing weights. Local mapping per
/// `moe::tables::exl3_expert_slot_index`: `gid - local_start` when inside
/// the EP-local range, else `-1`.
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_stage_routing(
    gpu: &dyn GpuBackend,
    indices_u32: DevicePtr,
    probs_f32: DevicePtr,
    b_indices: DevicePtr,
    b_weights: DevicePtr,
    local_start: usize,
    num_local: usize,
    s: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_stage_routing")?;
    let grid = div_ceil(s as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(indices_u32)
        .arg_ptr(probs_f32)
        .arg_ptr(b_indices)
        .arg_ptr(b_weights)
        .arg_i32(local_start as i32)
        .arg_i32(num_local as i32)
        .arg_u64(s as u64)
        .launch(stream)
}

/// BF16 `[T, hidden]` -> fp16 A `[T*top_k, hidden]`, slot `t*top_k + j` a
/// copy of token `t`'s row (plain launch). |v| > 65504 saturates (post-norm
/// activations are safe — the lm_head ingress precedent).
pub fn exl3_moe_replicate_a_bf16(
    gpu: &dyn GpuBackend,
    input_bf16: DevicePtr,
    out_f16: DevicePtr,
    num_tokens: usize,
    top_k: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_replicate_a_bf16")?;
    let total = num_tokens * top_k * hidden;
    let grid = div_ceil(total as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input_bf16)
        .arg_ptr(out_f16)
        .arg_i32(top_k as i32)
        .arg_u64(hidden as u64)
        .arg_u64(total as u64)
        .launch(stream)
}

/// Upstream `ext.silu_mul` twin over fp16 slot buffers (plain launch):
/// `out = silu(gate) * up`, half-precision numerics, optional act clamp
/// (`0.0` = none; qwen4_exp declares none). `numel` must be even.
pub fn exl3_silu_mul_f16(
    gpu: &dyn GpuBackend,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    act_limit: f32,
    numel: usize,
    stream: u64,
) -> Result<()> {
    ensure!(
        numel.is_multiple_of(2),
        "exl3_silu_mul_f16: numel {numel} must be even (half2 lanes)"
    );
    let n2 = numel / 2;
    let h = gpu.kernel("exl3_matmul", "exl3_silu_mul_f16")?;
    let grid = div_ceil(n2 as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_f32(act_limit)
        .arg_u64(n2 as u64)
        .launch(stream)
}

/// The full decode-tier routed-expert pipeline (header diagram). Writes the
/// per-token WEIGHTED routed sums — routing probs folded into the down
/// mgemm's fp32 grouped reduction, so the caller's blend must NOT re-apply
/// them — as BF16 `[num_tokens, hidden]` at `out_bf16`. A token whose
/// experts are ALL remote contributes an exact 0.0 row (EP partial-sum
/// convention; the cross-rank all-reduce completes it).
///
/// `tables` order is `[gate, up, down]`; gate/up decode `hidden -> inter`,
/// down `inter -> hidden`. `k_bits`/`cb` may differ per projection (three
/// separate launches), but every expert within one table shares them.
/// `stable_token_grid` retains single-token expert shape/split-K geometry
/// for all three projections, while processing the full batch in slot waves.
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_decode_routed(
    gpu: &dyn GpuBackend,
    input_bf16: DevicePtr,
    indices_u32: DevicePtr,
    probs_f32: DevicePtr,
    out_bf16: DevicePtr,
    tables: &[Exl3MoeProj; 3],
    scratch: &Exl3MoeScratch,
    locks: DevicePtr,
    num_tokens: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    local_start: usize,
    num_local: usize,
    act_limit: f32,
    stable_token_grid: bool,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    let s = num_tokens * top_k;
    ensure!(
        num_tokens >= 1 && top_k >= 1 && s <= scratch.s_cap,
        "exl3_moe_decode_routed: {num_tokens} tokens x top_k {top_k} = {s} \
         slots exceeds the slab capacity {} — slot-batch the caller",
        scratch.s_cap
    );
    ensure!(
        hidden.is_multiple_of(128) && inter.is_multiple_of(128),
        "exl3_moe_decode_routed: hidden {hidden} / inter {inter} must be \
         multiples of 128 (trellis tile + Hadamard block)"
    );

    // 1) Independent route casts + replicated fp16 ingress, normally fused.
    if super::moe_ingress::fused_ingress_enabled() {
        super::exl3_moe_stage_ingress(
            gpu,
            input_bf16,
            indices_u32,
            probs_f32,
            scratch.b_indices,
            scratch.b_weights,
            scratch.a_f16,
            local_start,
            num_local,
            num_tokens,
            top_k,
            hidden,
            stream,
        )?;
    } else {
        exl3_moe_stage_routing(
            gpu,
            indices_u32,
            probs_f32,
            scratch.b_indices,
            scratch.b_weights,
            local_start,
            num_local,
            s,
            stream,
        )?;
        exl3_moe_replicate_a_bf16(
            gpu,
            input_bf16,
            scratch.a_f16,
            num_tokens,
            top_k,
            hidden,
            stream,
        )?;
    }

    // 2) GATE / UP: per-slot f16 C, indices select the expert, NO weights
    //    (the reduction must not run — per-slot rows feed the activation).
    for (t, c) in [
        (&tables[0], scratch.c_gate_f16),
        (&tables[1], scratch.c_up_f16),
    ] {
        exl3_mgemm(
            gpu,
            scratch.a_f16,
            t.trellis_ptrs,
            c,
            1,
            hidden,
            inter,
            t.k_bits,
            t.cb,
            false,
            locks,
            t.suh_ptrs,
            scratch.a_had_f16,
            scratch.a_had_capacity_elems,
            t.svh_ptrs,
            Some(scratch.b_indices),
            None,
            s,
            s,
            -1,
            -1,
            1,
            None,
            None,
            None,
            stable_token_grid.then_some(top_k),
            sm_count,
            stream,
        )?;
    }

    // 3) Activation on the slot buffers (upstream f16 numerics). Stale rows
    //    of -1 slots may compute garbage here — the down call never reads
    //    them (compute skips the null-B slot, the reduction skips the -1).
    exl3_silu_mul_f16(
        gpu,
        scratch.c_gate_f16,
        scratch.c_up_f16,
        scratch.inter_f16,
        act_limit,
        s * inter,
        stream,
    )?;

    // 4) DOWN with the routing weights folded: fp32 C, grouped per-token
    //    reduction into rows 0..num_tokens-1.
    let down = &tables[2];
    exl3_mgemm(
        gpu,
        scratch.inter_f16,
        down.trellis_ptrs,
        scratch.c_down_f32,
        1,
        inter,
        hidden,
        down.k_bits,
        down.cb,
        true,
        locks,
        down.suh_ptrs,
        scratch.a_had_f16,
        scratch.a_had_capacity_elems,
        down.svh_ptrs,
        Some(scratch.b_indices),
        Some(scratch.b_weights),
        s,
        s,
        -1,
        -1,
        num_tokens,
        None,
        None,
        None,
        stable_token_grid.then_some(top_k),
        sm_count,
        stream,
    )?;

    // 5) Egress: reduced rows 0..T-1 -> BF16 token-major output.
    super::exl3_f32_to_bf16(
        gpu,
        scratch.c_down_f32,
        out_bf16,
        num_tokens * hidden,
        stream,
    )
}
