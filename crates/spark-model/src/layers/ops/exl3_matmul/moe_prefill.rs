// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill-tier native EXL3 MoE pipeline: the routed experts of one MoE
//! layer served DIRECTLY from packed trellis by upstream ExLlamaV3's
//! sort-by-expert tier (`ext.exl3_moe`) — ONE persistent fused launch
//! (module `exl3_moe`) runs gather→Had→gate/up trellis GEMM→SiLU·mul→down
//! trellis GEMM→Had→weighted fp32 scatter-add for EVERY local expert with
//! `0 < token_count <= 128` sorted rows; experts exceeding 128 rows take the
//! overflow path (upstream's `run_single_expert` tier: chunked cooperative
//! trellis GEMMs over the same packed weights + weighted scatter-add into the
//! same fp32 accumulator).
//!
//! Pipeline for one token batch (T tokens, S = T*top_k slots):
//!
//! ```text
//!   [caller]        moe_sort_by_expert over the batch's GLOBAL expert ids
//!   stage_sorted:   Atlas sort outputs -> token_sorted/weight_sorted i64/f16
//!                   in LOCAL-expert order (EP-remote slots -> sentinel tail
//!                   bucket) + expert_count i64 [num_local+1] bincount
//!   ingress:        bf16 [T, H] -> f16 hidden_f16 (RAW; the kernel applies
//!                   suh+Hadamard itself while gathering)
//!   out_f32:        zeroed [T, H] accumulator (the kernel atomicAdds)
//!   tier select:    S <= 128 -> NO-SYNC shortcut (num_active = -1, no D2H,
//!                   overflow impossible); else ONE stream-sync D2H of the
//!                   local expert_offsets slice (upstream's
//!                   `expert_count.tolist()` host-sync tier)
//!   exl3_moe:       fused launch over the 0 < count <= 128 experts
//!   overflow:       per count > 128 expert: chunked f16 gather -> gate/up
//!                   exl3_gemm (f16 C) -> half SiLU·mul -> down exl3_gemm
//!                   (f32 C) -> weighted scatter-add (persistent slabs only)
//!   egress:         f32 [T, H] -> bf16 (routing probs ALREADY applied)
//! ```
//!
//! Contracts carried from the vendored kernel (`exl3_moe_kernel.cuh`):
//!  * Pointer tables are DENSE over the EP-LOCAL experts; remote slots are
//!    excluded by the sentinel bucket (`expert_count[num_local]`), which the
//!    kernel's expert loop never reaches — never null entries at reachable
//!    indices.
//!  * The fused kernel needs ONE codebook across gate/up/down and either
//!    uniform K in [`EXL3_MOE_FUSED_K_BITS`] = {2,3,4,5,6} (fixed-K
//!    instance) or mixed K with every value in [`EXL3_MOE_MIXED_K_BITS`] =
//!    {2,3,4} (k0 runtime-dispatch instance) — [`exl3_moe_fused_serves`].
//!  * temp slabs `(C, 128, H/I)` need NO zero-init (group barriers protect
//!    them); `output_state` MUST be zeroed every call.
//!  * PLAIN launch, but every block spins on group barriers in the shared
//!    locks buffer: treat like the cooperative entries — never under CUDA
//!    graph capture, never concurrently with another exl3 launch sharing the
//!    locks buffer unless stream-ordered.
//!  * Overflow is ROUTINE at serving shapes, not rare: a 4096-token batch at
//!    top_k 10 over 512 experts averages 80 rows/expert, so ordinary routing
//!    skew pushes popular experts past 128 on every full chunk. The overflow
//!    path is therefore alloc-free and sync-free (persistent `ov_*` slabs,
//!    cooperative trellis GEMMs stream-ordered behind the fused launch) and
//!    decodes at the same fp16 precision as the fused tier — no numerics seam
//!    across the 128-row boundary. The ONE host sync per batch is the
//!    expert_offsets D2H of the host-sync tier (S > 128), as upstream.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::moe_decode::Exl3MoeProj;

// Overflow (count > 128) tier — split on the 500-LoC cap. The parent-module
// `#[path]` chain resolves this against `ops/exl3_matmul/`.
#[path = "moe_prefill_overflow.rs"]
mod overflow;
pub use overflow::Exl3MoeOverflowCtx;
use overflow::run_overflow_expert;

/// The fused kernel's per-expert row cap (`TEMP_ROWS_FUSED` upstream): the
/// temp slabs hold this many rows per concurrent expert group, and experts
/// with more sorted rows are skipped (ticket-free) for the overflow path.
pub const EXL3_MOE_MAX_TOKENS_PER_EXPERT: usize = 128;

/// K values the fused `exl3_moe` module has FIXED-K instances for
/// (`exl3_moe_k{K}_n{128,256}_cb{1,2}`) — a layer whose gate/up/down share
/// one K in this set takes the fused tier. Mirrored by
/// `weight_map::EXL3_NATIVE_MOE_K_BITS` (pinned equal by test). K=8 is
/// deliberately absent: no shipped checkpoint has K=8 routed experts and each
/// K adds four full pipelined-GEMM instantiations to the module.
pub const EXL3_MOE_FUSED_K_BITS: [u32; 5] = [2, 3, 4, 5, 6];

/// K values the fused module's k0 RUNTIME-DISPATCH instances switch over —
/// the only instances a MIXED-K layer (gate/up/down at different K) can use.
/// Kept at upstream-minus-{1,5,6,7,8} on purpose: every retained case
/// instantiates the full pipelined GEMM in all four k0 variants, and widening
/// the switch to {2..6} measured 3.0x the module's compile time (53.5 s vs
/// 17.7 s) for a layout no shipped checkpoint has (all five bpw branches are
/// uniform-K per layer). See `exl3_vendor/exl3_moe_kernel.cuh`.
pub const EXL3_MOE_MIXED_K_BITS: [u32; 3] = [2, 3, 4];

/// Whether the fused tier can serve a layer with these per-projection K
/// (gate, up, down): uniform K in [`EXL3_MOE_FUSED_K_BITS`], or mixed K with
/// every value in [`EXL3_MOE_MIXED_K_BITS`]. The loader's keep-set and the
/// launch both apply this rule — a K reaching a k0 instance outside its
/// switch would SILENTLY skip that projection's GEMM.
pub fn exl3_moe_fused_serves(ks: [u32; 3]) -> bool {
    if ks[0] == ks[1] && ks[1] == ks[2] {
        EXL3_MOE_FUSED_K_BITS.contains(&ks[0])
    } else {
        ks.iter().all(|k| EXL3_MOE_MIXED_K_BITS.contains(k))
    }
}

/// Device scratch for the prefill pipeline; one set serves every MoE layer
/// (all launches are stream-ordered). See `moe::tables::Exl3MoeState` for
/// the serving-side owner; the parity example allocates its own.
#[derive(Clone, Copy, Debug)]
pub struct Exl3MoePrefillScratch {
    /// f16 `[t_cap, hidden]` RAW activation ingress (the kernel's
    /// `hidden_state`).
    pub hidden_f16: DevicePtr,
    /// f32 `[t_cap, hidden]` routed accumulator (`output_state`) — zeroed by
    /// the pipeline every call.
    pub out_f32: DevicePtr,
    /// f16 `[concurrency, 128, hidden]` gate staging slab (no zero-init).
    pub temp_state_g: DevicePtr,
    /// f16 `[concurrency, 128, hidden]` up staging slab.
    pub temp_state_u: DevicePtr,
    /// f16 `[concurrency, 128, inter]` gate intermediate slab.
    pub temp_inter_g: DevicePtr,
    /// f16 `[concurrency, 128, inter]` up intermediate slab.
    pub temp_inter_u: DevicePtr,
    /// i64 `[t_cap * top_k]` token index per LOCAL-sorted slot.
    pub token_sorted: DevicePtr,
    /// f16 `[t_cap * top_k]` routing weight per LOCAL-sorted slot.
    pub weight_sorted: DevicePtr,
    /// i64 `[e_cap + 1]` per-local-expert bincount + sentinel tail.
    pub expert_count: DevicePtr,
    /// f16 `[ov_chunk, hidden]` overflow gathered-A chunk (RAW rows of
    /// `hidden_f16`).
    pub ov_a_f16: DevicePtr,
    /// f16 `[ov_chunk, hidden]` overflow `A_had` scratch — dedicated, must
    /// NOT alias `ov_a_f16` (the gate and up GEMMs read the same A).
    pub ov_a_had_f16: DevicePtr,
    /// f16 `[ov_chunk, inter]` overflow gate GEMM out (SiLU·mul in place).
    pub ov_gate_f16: DevicePtr,
    /// f16 `[ov_chunk, inter]` overflow up GEMM out.
    pub ov_up_f16: DevicePtr,
    /// f32 `[ov_chunk, hidden]` overflow down GEMM out.
    pub ov_down_f32: DevicePtr,
    /// Token capacity of `hidden_f16`/`out_f32` (+`token_sorted`/
    /// `weight_sorted` at `t_cap * top_k` slots) — callers batch above it.
    pub t_cap: usize,
    /// Local-expert capacity of `expert_count` (slab holds `e_cap + 1`).
    pub e_cap: usize,
    /// Temp-slab count C; the launch requires `C * 8 <= sm_count`.
    pub concurrency: usize,
    /// Row chunk of the overflow GEMMs (slab rows of `ov_*`).
    pub ov_chunk: usize,
}

/// What one batch actually exercised — parity asserts on it, serving traces
/// it.
#[derive(Clone, Copy, Debug)]
pub struct Exl3MoePrefillStats {
    /// Experts the fused kernel processed (`0 < count <= 128`); -1 when the
    /// S <= 128 no-sync shortcut skipped the host readback.
    pub num_active: i64,
    /// Experts served by the reconstruct overflow path (count > 128).
    pub overflow_experts: usize,
}

/// Stage Atlas's `moe_sort_by_expert` outputs into the fused kernel's
/// LOCAL-expert-ordered forms (plain launch; kernel contract at its
/// definition in `exl3_matmul.cu`).
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_stage_sorted(
    gpu: &dyn GpuBackend,
    token_to_perm: DevicePtr,
    probs_f32: DevicePtr,
    expert_offsets: DevicePtr,
    token_sorted: DevicePtr,
    weight_sorted: DevicePtr,
    expert_count: DevicePtr,
    local_start: usize,
    num_local: usize,
    top_k: usize,
    s: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_stage_sorted")?;
    let work = s.max(num_local + 1);
    let grid = div_ceil(work as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(token_to_perm)
        .arg_ptr(probs_f32)
        .arg_ptr(expert_offsets)
        .arg_ptr(token_sorted)
        .arg_ptr(weight_sorted)
        .arg_ptr(expert_count)
        .arg_i32(local_start as i32)
        .arg_i32(num_local as i32)
        .arg_i32(top_k as i32)
        .arg_u64(s as u64)
        .launch(stream)
}

/// Launch the fused `exl3_moe` kernel (module `exl3_moe`) over the staged
/// batch. `num_active`: count of experts with `0 < count <= 128` from the
/// host-sync tier, or -1 for the S <= 128 no-sync shortcut (defaults grid).
/// The caller must NOT launch when `num_active == 0`.
///
/// PLAIN launch, but spin-barrier co-resident (grid <= sm_count by
/// construction, 1 block/SM at 90KB smem) — operationally like the
/// cooperative entries: no graph capture, one in-flight launch per locks
/// buffer.
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_fused(
    gpu: &dyn GpuBackend,
    tables: &[Exl3MoeProj; 3],
    scratch: &Exl3MoePrefillScratch,
    t: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
    num_local: usize,
    num_active: i64,
    act_limit: f32,
    locks: DevicePtr,
    sm_count: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        num_active != 0,
        "exl3_moe_fused: no active experts — skip the launch"
    );
    let ks = [tables[0].k_bits, tables[1].k_bits, tables[2].k_bits];
    ensure!(
        exl3_moe_fused_serves(ks),
        "exl3_moe_fused: gate/up/down K={ks:?} outside the fused-kernel envelope \
         (uniform K in {EXL3_MOE_FUSED_K_BITS:?}, or mixed K all in \
         {EXL3_MOE_MIXED_K_BITS:?}; mgemm serves {{2..6,8}}); the loader \
         keep-predicate must refuse this layer"
    );
    let cb = tables[0].cb;
    ensure!(
        tables[1].cb == cb && tables[2].cb == cb && (cb == 1 || cb == 2),
        "exl3_moe_fused: gate/up/down codebooks differ ({}/{}/{}) — the fused \
         kernel decodes at ONE codebook",
        tables[0].cb,
        tables[1].cb,
        tables[2].cb,
    );
    ensure!(
        hidden.is_multiple_of(128) && inter.is_multiple_of(128),
        "exl3_moe_fused: hidden {hidden} / inter {inter} must be multiples of 128"
    );
    let c = scratch.concurrency;
    ensure!(
        c >= 1 && c * 8 <= sm_count as usize,
        "exl3_moe_fused: concurrency {c} needs {}..{} SMs (have {sm_count})",
        8,
        c * 8,
    );

    let kname = if ks[0] == ks[1] && ks[1] == ks[2] {
        ks[0]
    } else {
        0
    };
    let n_tile = if hidden.is_multiple_of(256) && inter.is_multiple_of(256) {
        256
    } else {
        128
    };
    let name = format!("exl3_moe_k{kname}_n{n_tile}_cb{cb}");
    let h = gpu.kernel("exl3_moe", &name)?;
    super::raise_smem_once(gpu, h)?;

    // Upstream grid selection (exl3_moe.cu host, mirrored in the kernel-file
    // header): defaults 8 SMs/expert x min(C, 64) groups; a known-small
    // active set narrows the groups and widens each to <= 32 SMs.
    let mut num_groups = c.min(64);
    let mut group_size = 8usize;
    if num_active > 0 {
        num_groups = num_groups.min(num_active as usize);
        group_size = (sm_count as usize / num_groups).min(32);
    }

    KernelLaunch::new(gpu, h)
        .grid([group_size as u32, 1, num_groups as u32])
        .block([512, 1, 1])
        .shared_mem(super::EXL3_SMEM_MAX)
        .arg_ptr(scratch.hidden_f16)
        .arg_ptr(scratch.temp_state_g)
        .arg_ptr(scratch.temp_state_u)
        .arg_ptr(scratch.temp_inter_g)
        .arg_ptr(scratch.temp_inter_u)
        .arg_ptr(scratch.out_f32)
        .arg_ptr(tables[0].trellis_ptrs)
        .arg_ptr(tables[0].suh_ptrs)
        .arg_ptr(tables[0].svh_ptrs)
        .arg_ptr(tables[1].trellis_ptrs)
        .arg_ptr(tables[1].suh_ptrs)
        .arg_ptr(tables[1].svh_ptrs)
        .arg_ptr(tables[2].trellis_ptrs)
        .arg_ptr(tables[2].suh_ptrs)
        .arg_ptr(tables[2].svh_ptrs)
        .arg_ptr(scratch.expert_count)
        .arg_ptr(scratch.token_sorted)
        .arg_ptr(scratch.weight_sorted)
        .arg_i32(hidden as i32)
        .arg_i32(inter as i32)
        .arg_i32(num_local as i32)
        .arg_i32(top_k as i32)
        .arg_i32(EXL3_MOE_MAX_TOKENS_PER_EXPERT as i32)
        .arg_i32(c as i32)
        .arg_f32(act_limit)
        .arg_i32(0) // act_function: 0 = SiLU (qwen4_exp)
        .arg_i32(ks[0] as i32)
        .arg_i32(ks[1] as i32)
        .arg_i32(ks[2] as i32)
        .arg_ptr(locks)
        .launch(stream)?;
    let _ = t; // geometry is carried by the staged tensors
    Ok(())
}

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
    super::exl3_bf16_to_f16(gpu, input_bf16, scratch.hidden_f16, t * hidden, stream)?;

    // 2) Zero the fp32 accumulator (the kernel and the overflow epilogue
    //    both accumulate into it).
    gpu.memset_async(scratch.out_f32, 0, t * hidden * 4, stream)?;

    // 3) Tier select. S <= 128: upstream's no-sync shortcut — every expert
    //    count is <= S <= 128, so the fused kernel covers everything and no
    //    host readback is needed (added upstream because the sync was ~33%
    //    idle at MTP verify shapes). Otherwise ONE stream-sync D2H of the
    //    LOCAL slice of expert_offsets (the host-sync tier).
    let mut num_active: i64 = -1;
    let mut overflow: Vec<(usize, usize, usize)> = Vec::new(); // (e_local, span_start, count)
    if s > EXL3_MOE_MAX_TOKENS_PER_EXPERT {
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
            if count == 0 {
                continue;
            }
            if count <= EXL3_MOE_MAX_TOKENS_PER_EXPERT {
                active += 1;
            } else {
                overflow.push((e, (off[e] - lo) as usize, count));
            }
        }
        num_active = active;
    }

    // 4) Fused launch (skipped only when the host-sync tier saw no fusable
    //    expert).
    if num_active != 0 {
        exl3_moe_fused(
            gpu, tables, scratch, t, top_k, hidden, inter, num_local, num_active, act_limit, locks,
            sm_count, stream,
        )?;
    }

    // 5) Overflow experts (count > 128): chunked trellis GEMMs + weighted
    //    scatter-add, stream-ordered behind the fused kernel.
    for &(e_local, span_start, count) in &overflow {
        run_overflow_expert(
            gpu, ov, tables, scratch, e_local, span_start, count, hidden, inter, act_limit, locks,
            sm_count, stream,
        )?;
    }

    // 6) Egress: fp32 accumulator -> BF16 token-major output.
    super::exl3_f32_to_bf16(gpu, scratch.out_f32, out_bf16, t * hidden, stream)?;

    Ok(Exl3MoePrefillStats {
        num_active,
        overflow_experts: overflow.len(),
    })
}
