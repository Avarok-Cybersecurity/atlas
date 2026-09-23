// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM MLP decode forward — dense FFN and routed MoE, one token.
//!
//! Launch geometry is lifted verbatim from the two gated microtests
//! (`examples/glm5next_{ffn,moe}_microtest.rs`, Slice-10 gates 3/4/6/7), which measured this
//! exact sequence against HF 5.16.1 on real layer-0 and layer-3 weights. Nothing here
//! re-derives the equations.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::forward_prefill_gemm;
use super::weights::{Glm5NextDenseMlpWeights, Glm5NextMoeWeights, Nvfp4Proj};
use super::{Glm5NextMlpConfig, Glm5NextMlpKernels};

const W4_TILE: u32 = 64;
const ACT_BLOCK: u32 = 256;

/// `C[M, N] = A[M, K] @ B[N, K]^T`, BF16 in and out.
#[allow(clippy::too_many_arguments)]
fn gemm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gemv: KernelHandle,
    batchm: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // 🔴 Wide (prefill) shapes go to cuBLASLt. `dense_gemm_bf16` says of itself "Phase 1:
    // Correct scalar implementation ... Phase 2: Will add mma.sync" — one output element
    // per thread, no tensor cores. It measured 1.45 TFLOP/s on `mlp_dense` and 1.38 on
    // `moe_shared`; the two agreeing at different shapes is what proves it is the KERNEL
    // and not the shape. Together they were ~275 ms/step, ~20% of prefill.
    //
    // 🪤 This file kept the scalar path after the DSA and KDA blocks moved off it, so the
    // same defect survived a commit that was supposed to have killed it. The dense MLP and
    // the shared expert are BF16 and unquantized in the EXL3 K2 checkpoint (its scope is
    // `glm53_routed_experts_only` — only the routed experts are trellis), so they were
    // always eligible; nothing about the quantization stopped this.
    //
    // Same numerics boundary as everywhere else — see `glm5next_layer::cublas_wide_proj`.
    // M=1 decode and M<=16 verify keep their bit-identical tiers, so the router's M=1 call
    // below cannot reach this arm even though it shares this helper.
    if m > crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
        && crate::layers::glm5next_layer::cublas_wide_proj()
    {
        return crate::layers::ops::cublas_bf16_proj_dense(
            a, b, c, m as u32, n as u32, kk as u32, stream,
        );
    }
    // M=1 decode -> GEMV; M=2..8 (a K-token verify sweep) -> ONE weight read for all rows;
    // wider -> the tile GEMM. `ops::dense_mm_bf16` owns the policy and the grid coupling.
    crate::layers::ops::dense_mm_bf16(
        gpu,
        &crate::layers::ops::DenseMmKernels {
            gemm: k,
            gemv,
            batchm,
        },
        a,
        b,
        c,
        m,
        n,
        kk,
        stream,
    )
}

/// `C[1, N] = A[1, K] @ dequant(B)[N, K]^T` — the M=1 decode kernel.
///
/// Same NVFP4 operand triple as [`w4a16`], one output row. Used for every routed-expert
/// projection because they are all M=1 and the tile GEMM measured 9.7 GB/s there.
fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    k_sw: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    // Prefer the single-warp sibling when the target carries it. BIT-IDENTICAL — the
    // per-orig-lane partial is the same function, the shuffle tree is the same tree, and
    // the final combine is the same two-term FP32 add; only the block packing differs.
    // This site launched the base kernel directly and so had never picked up the SW win
    // that `ops::w4a16_decode_gemv` has been handing every other decode GEMV.
    if k_sw.0 != 0 {
        return crate::layers::ops::w4a16_gemv_sw_raw(
            gpu, k_sw, a, w.packed, w.scale, w.scale_2, c, n as u32, kk as u32, stream,
        );
    }
    KernelLaunch::new(gpu, k)
        .grid([crate::layers::ops::w4a16_gemv_grid_x(n as u32), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        // 🪤 by VALUE, as in `w4a16`.
        .arg_f32(w.scale_2)
        .arg_ptr(c)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// `C[M, N] = A[M, K] @ dequant(B)[N, K]^T` — NVFP4 weight, BF16 activation and output.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn w4a16(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    w: &Nvfp4Proj,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([
            (n as u32).div_ceil(W4_TILE),
            (m as u32).div_ceil(W4_TILE),
            1,
        ])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.packed)
        .arg_ptr(w.scale)
        // 🪤 by VALUE. `weight_scale_2` is a scalar argument, not a pointer.
        .arg_f32(w.scale_2)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .launch(stream)?;
    Ok(())
}

/// `out = silu(min(gate, limit)) * clamp(up, -limit, limit)` over `n` elements.
///
/// Re-exported to the grouped-prefill module as [`swiglu_rows`]: the clamp is asymmetric and
/// model-specific, so that path must run THIS activation, never `moe_silu_mul`.
pub(super) fn swiglu_rows(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    swiglu(gpu, k, gate, up, out, n, limit, stream)
}

fn swiglu(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: usize,
    limit: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([(n as u32).div_ceil(ACT_BLOCK), 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_f32(limit)
        .launch(stream)?;
    Ok(())
}

/// Scratch for one MLP site, allocated once and reused every decode step.
///
/// Sized for the widest thing this site can run: a dense layer's `intermediate_size`, a routed
/// layer's `moe_intermediate_size`, and the shared expert's width.
pub struct Glm5NextMlpWorkspace {
    /// `[rows, max_inter]` BF16 ×3 — gate, up, activated. Shared by every projection pair.
    a_gate: DevicePtr,
    a_up: DevicePtr,
    a_act: DevicePtr,
    /// `[rows, num_experts]` F32 router logits.
    logits: DevicePtr,
    /// `[rows, top_k]` I32 / F32 selection.
    ids: DevicePtr,
    wts: DevicePtr,
    /// `[rows, top_k, hidden]` BF16. 🪤 Fully written every step — remote and invalid slots are
    /// memset to zero before the loop, never left stale.
    expert_out: DevicePtr,
    /// `[rows, hidden]` BF16 shared-expert output.
    shared_out: DevicePtr,
    /// `[rows * top_k]` I32 union expert ids, `-1` = entry unused. Row-batched MoE only.
    u_eid: DevicePtr,
    /// `[rows * top_k, rows]` I32 slot per union entry per row, `-1` = row absent.
    u_slot: DevicePtr,
    /// `[rows * top_k]` I32 — sorted row → original token. Grouped prefill GEMM only.
    sorted_token_ids: DevicePtr,
    /// `[rows * top_k]` I32 — sorted row → expert id. Grouped prefill GEMM only.
    sorted_expert_ids: DevicePtr,
    /// `[num_experts + 1]` I32 prefix sum over the expert-sorted rows.
    expert_offsets: DevicePtr,
    /// `[rows, top_k]` I32 — a slot's row in the expert-sorted output. Read by the grouped
    /// GEMM's paired `glm5next_moe_combine_indexed`.
    token_to_perm: DevicePtr,
    max_inter: usize,
    /// Widest verify this scratch can serve. `1` on the serial decode path.
    max_rows: usize,
    /// `max_rows * top_k` — the routed-slot extent every grouped-path buffer is sized for.
    max_total_expanded: usize,
}

impl Glm5NextMlpWorkspace {
    pub fn new(gpu: &dyn GpuBackend, cfg: &Glm5NextMlpConfig, max_rows: usize) -> Result<Self> {
        let rows = max_rows.max(1);
        let max_inter = cfg
            .local_dense_intermediate
            .max(cfg.moe_intermediate)
            .max(cfg.local_shared_intermediate)
            .max(1);
        // The grouped MoE path activates all `top_k` slots in one launch, so the three
        // activation buffers are slot-major and `top_k` times a routed expert's width. The
        // dense path still writes only the first `inter` elements — `max_inter` stays the
        // guard for it.
        // The dense/shared arm is now `[rows, inter]`; the grouped MoE arm is still one row's
        // `top_k` slots at a time. Both share these buffers, so take the wider.
        // The row-batched MoE arm computes every (row, slot) pair in ONE launch, so its
        // activations are `[rows, top_k, moe_intermediate]` — wider than either of the above.
        let act_elems = (rows * max_inter)
            .max(rows * cfg.top_k * cfg.moe_intermediate)
            .max(1);
        Ok(Self {
            a_gate: gpu.alloc(act_elems * 2)?,
            a_up: gpu.alloc(act_elems * 2)?,
            a_act: gpu.alloc(act_elems * 2)?,
            logits: gpu.alloc(rows * cfg.num_experts * 4)?,
            ids: gpu.alloc(rows * cfg.top_k * 4)?,
            wts: gpu.alloc(rows * cfg.top_k * 4)?,
            expert_out: gpu.alloc(rows * cfg.top_k * cfg.hidden * 2)?,
            shared_out: gpu.alloc(rows * cfg.hidden * 2)?,
            u_eid: gpu.alloc(rows * cfg.top_k * 4)?,
            u_slot: gpu.alloc(rows * cfg.top_k * rows * 4)?,
            // 🪤 A59: the grouped-prefill routing tables are allocated HERE, at load, with
            // every other pool — never on the first prefill. They are tiny (a 256-row
            // sub-chunk at top_k = 8 is 8 KB each, plus 1.2 KB of offsets), so they are
            // allocated unconditionally rather than behind the env lever: a conditional
            // pool is a pool that is missing exactly when a fallback needs it.
            sorted_token_ids: gpu.alloc(rows * cfg.top_k * 4)?,
            sorted_expert_ids: gpu.alloc(rows * cfg.top_k * 4)?,
            expert_offsets: gpu.alloc((cfg.num_experts + 1) * 4)?,
            token_to_perm: gpu.alloc(rows * cfg.top_k * 4)?,
            max_inter,
            max_rows: rows,
            max_total_expanded: rows * cfg.top_k,
        })
    }

    /// Widest row group this scratch serves.
    pub(super) fn max_rows(&self) -> usize {
        self.max_rows
    }
    /// `max_rows * top_k` — the routed-slot extent the grouped buffers were sized for.
    pub(super) fn max_total_expanded(&self) -> usize {
        self.max_total_expanded
    }
    pub(super) fn ids(&self) -> DevicePtr {
        self.ids
    }
    pub(super) fn a_gate(&self) -> DevicePtr {
        self.a_gate
    }
    pub(super) fn a_up(&self) -> DevicePtr {
        self.a_up
    }
    pub(super) fn a_act(&self) -> DevicePtr {
        self.a_act
    }
    pub(super) fn expert_out(&self) -> DevicePtr {
        self.expert_out
    }
    pub(super) fn sorted_token_ids(&self) -> DevicePtr {
        self.sorted_token_ids
    }
    pub(super) fn sorted_expert_ids(&self) -> DevicePtr {
        self.sorted_expert_ids
    }
    pub(super) fn expert_offsets(&self) -> DevicePtr {
        self.expert_offsets
    }
    pub(super) fn token_to_perm(&self) -> DevicePtr {
        self.token_to_perm
    }
}

/// A BF16 SwiGLU MLP of width `inter`: `down(clamped_swiglu(gate(x), up(x)))`.
///
/// Used for both the dense layers and the shared expert — identical math, different widths.
/// With `tp_world_size > 1` the result is a **partial sum**; the caller reduces.
#[allow(clippy::too_many_arguments)]
pub fn forward_dense(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextDenseMlpWeights,
    inter: usize,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if inter == 0 || inter > ws.max_inter {
        bail!(
            "GLM dense MLP: width {inter} does not fit a workspace built for {}",
            ws.max_inter
        );
    }
    if m == 0 || m > ws.max_rows {
        bail!(
            "GLM dense MLP: {m} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    gemm(
        gpu,
        k.gemm,
        k.gemv,
        k.gemv_batchm,
        x,
        w.gate_proj,
        ws.a_gate,
        m,
        inter,
        cfg.hidden,
        stream,
    )?;
    gemm(
        gpu,
        k.gemm,
        k.gemv,
        k.gemv_batchm,
        x,
        w.up_proj,
        ws.a_up,
        m,
        inter,
        cfg.hidden,
        stream,
    )?;
    swiglu(
        gpu,
        k.swiglu,
        ws.a_gate,
        ws.a_up,
        ws.a_act,
        // Elementwise over the whole `[m, inter]` block — bit-identical to m launches of
        // `inter`, because every output element depends only on its own gate/up pair.
        m * inter,
        cfg.swiglu_limit,
        stream,
    )?;
    gemm(
        gpu,
        k.gemm,
        k.gemv,
        k.gemv_batchm,
        ws.a_act,
        w.down_proj,
        out,
        m,
        cfg.hidden,
        inter,
        stream,
    )
}

/// `C[top_k, N] = A @ dequant(expert[ids[slot]])^T` — every routed slot in ONE launch.
///
/// Bit-identical per slot to the [`w4a16_gemv`] loop it replaces; see the kernel's header.
/// Slots whose expert this rank does not own are skipped, so `c` must already hold whatever
/// those rows should contribute (zero, for the routed sum).
#[allow(clippy::too_many_arguments)]
fn w4a16_gemv_moe(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &super::weights::Glm5NextExpertPtrTable,
    c: DevicePtr,
    ids: DevicePtr,
    n: usize,
    kk: usize,
    top_k: usize,
    num_experts: usize,
    input_stride: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        // 🪤 COUPLED to the kernel's `N_PER_BLOCK_SW` = 8, and grid.y IS the slot.
        .grid([
            crate::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
            top_k as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(ids)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(num_experts as u32)
        .arg_u32(input_stride as u32)
        .launch(stream)
}

/// `C[rows, top_k, N]` — the UNION of the rows' selected experts, each expert swept ONCE.
///
/// Bit-identical per (row, slot) to [`w4a16_gemv_moe`]; see the kernel header. `u_eid` /
/// `u_slot` come from `glm5next_moe_row_union` and stay on device, so this is capturable.
///
/// 🪤 grid.y is `rows * top_k` — the UNION extent, not `top_k`. Entries the routing did not
/// fill retire immediately on `u_eid < 0`.
#[allow(clippy::too_many_arguments)]
fn w4a16_gemv_moe_batchm(
    gpu: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    t: &super::weights::Glm5NextExpertPtrTable,
    c: DevicePtr,
    u_eid: DevicePtr,
    u_slot: DevicePtr,
    n: usize,
    kk: usize,
    rows: usize,
    top_k: usize,
    num_experts: usize,
    a_row_stride: usize,
    a_slot_stride: usize,
    c_row_stride: usize,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        // 🪤 COUPLED to the kernel's `N_PER_BLOCK_SW` = 8.
        .grid([
            crate::layers::ops::w4a16_gemv_sw_grid_x(n as u32),
            (rows * top_k) as u32,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(t.packed_ptrs)
        .arg_ptr(t.scale_ptrs)
        .arg_ptr(t.scale2_vals)
        .arg_ptr(c)
        .arg_ptr(u_eid)
        .arg_ptr(u_slot)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(num_experts as u32)
        .arg_u32(a_row_stride as u32)
        .arg_u32(a_slot_stride as u32)
        .arg_u32(c_row_stride as u32)
        .launch(stream)
}

/// Kill switch for the row-batched routed path: `AVAROK_NO_GLM_MOE_ROW_BATCH=1` restores the
/// one-launch-per-row grouped dispatch. Read once — this sits on the per-layer decode path.
fn row_batch_disabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("AVAROK_NO_GLM_MOE_ROW_BATCH").as_deref() == Ok("1"))
}

/// Widest tier `w4a16_gemv_sw_moe_batchm` may be dispatched at, `AVAROK_GLM_MOE_ROW_BATCH_MAX`.
///
/// 🔬 An A/B lever, not a tuning knob: `=4` restores the pre-2026-08-31 cap exactly, so the
/// width extension can be measured against itself in ONE image instead of one image per arm —
/// the same lever that found A65's real defect. Clamped to the compiled tier family.
pub(crate) fn row_batch_max() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = std::env::var("AVAROK_GLM_MOE_ROW_BATCH_MAX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(MOE_ROW_BATCH_MAX_ROWS)
            .clamp(1, MOE_ROW_BATCH_MAX_ROWS);
        if m != MOE_ROW_BATCH_MAX_ROWS {
            tracing::warn!(
                "GLM MoE row-batch width capped at {m} (default {MOE_ROW_BATCH_MAX_ROWS})"
            );
        }
        m
    })
}

/// Widest compiled `w4a16_gemv_sw_moe_batchm_mR` tier. Mirror of the
/// `AVAROK_MOE_BATCHM_ENTRY` list in `kernels/gb10/common/w4a16_gemv.cu` and of the
/// `[KernelHandle; 7]` in `Glm5NextMlpKernels`.
///
/// 🔴 Since 2026-09-02 this is a **sub-group width, not a caller contract**. The prefill sub-chunk
/// is 16 rows wide (`glm5next_layer::PREFILL_ROWS`) because the DENSE tier widened to 16; the
/// routed experts did not follow, so [`forward_moe`] splits any wider row group into even
/// sub-groups of at most this and sweeps each one. Callers may pass any `rows` their workspace
/// holds.
pub const MOE_ROW_BATCH_MAX_ROWS: usize = 8;

/// Split `rows` into consecutive `(start, width)` sub-groups of at most `cap`, as evenly as the
/// count allows.
///
/// 🪤 Even, not greedy. A greedy split of 9 rows at cap 8 leaves a trailing group of ONE, and
/// there is no `w4a16_gemv_sw_moe_batchm_m1` tier — the array starts at m2. Balancing gives 5 + 4,
/// and at the shipping cap of `MOE_ROW_BATCH_MAX_ROWS` every group is >= 2 for every `rows >= 2`.
///
/// 🪤 A width-1 group is still REACHABLE at a small cap, where it is arithmetically forced (3 rows
/// at cap 2 has no all->=2 split). That is not a correctness hole — the caller's gate requires
/// every group to have a tier, so such a call simply runs the per-row arm — but it does mean
/// `AVAROK_GLM_MOE_ROW_BATCH_MAX=2` silently disables row batching at odd widths. Only the A/B
/// lever can reach it.
///
/// 🔴 Splitting is EXACT. A row's routed output is the sum over ITS OWN top-k slots, each slot a
/// single expert's GEMV whose accumulation order (`w4a16_gemv_partial_rows`) does not depend on
/// `R` or on which rows share the sweep; the union table only decides which experts get swept and
/// in what order the sweeps are issued, never what any row adds. So `forward_moe(rows)` returns
/// the same bits however it is grouped. What splitting costs is amortization, not accuracy: two
/// 8-row sweeps visit the union of 8 rows twice instead of the (smaller) union of 16 once.
fn moe_row_groups(rows: usize, cap: usize) -> Vec<(usize, usize)> {
    let n = rows.div_ceil(cap.max(1)).max(1);
    let mut out = Vec::with_capacity(n);
    let mut start = 0usize;
    for i in 0..n {
        let w = (rows - start).div_ceil(n - i);
        out.push((start, w));
        start += w;
    }
    out
}

/// 🪤 `glm5next_moe_row_union` is ONE block of `rows * top_k` threads. A CUDA block is capped
/// at 1024 threads, but this kernel's own scans are `O(T^2)`/`O(T^3)` over that extent and the
/// tier family was sized around 64, so 64 is the contract. Threads past a block never run:
/// exceeding it would SILENTLY drop union entries, so the dispatch refuses instead.
pub const MOE_ROW_UNION_MAX_IDS: usize = 64;

/// Say once PER ROW COUNT whether a verify shares its expert sweeps across its rows.
///
/// 🪤 A plain `Once` here is a trap: the first MoE forward of a run is the prefill/K=1 step at
/// `rows = 1`, which can never batch, so a single announcement reports "per-row" for the whole
/// process and the batched path looks like it never engaged. Latch one bit per row count.
fn announce_row_batch(batched: bool, rows: usize) {
    use std::sync::atomic::{AtomicU16, Ordering};
    static SEEN: AtomicU16 = AtomicU16::new(0);
    let bit = 1u16 << rows.min(15);
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if batched {
        tracing::info!("GLM MoE: row-batched expert union ({rows} rows, one sweep each)");
    } else {
        tracing::info!("GLM MoE: per-row expert sweeps ({rows} rows)");
    }
}

/// Say once PER ROW COUNT whether the routed experts took the grouped GEMM.
///
/// 🪤 A silent fallback here is the expensive kind: a missing `moe_sort_by_expert` or
/// `moe_w4a16_grouped_gemm_ptrtable` handle sends prefill straight back to the 8-row GEMV
/// and the only symptom is the TTFT. Latch one bit per row count, like `announce_row_batch`.
fn announce_grouped_prefill(on: bool, rows: usize) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: AtomicU32 = AtomicU32::new(0);
    if rows <= MOE_ROW_BATCH_MAX_ROWS {
        return; // the narrow paths never consider it; saying so every step is noise
    }
    let bit = 1u32 << (rows.min(31));
    if SEEN.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
        return;
    }
    if on {
        tracing::info!(
            "GLM MoE prefill: grouped tensor-core W4A16 GEMM, {rows} rows in ONE launch \
             per projection (AVAROK_GLM_MOE_PREFILL_GEMM=0 to restore the GEMV path)"
        );
    } else {
        tracing::info!(
            "GLM MoE prefill: row-batched GEMV, {rows} rows split into \
             {MOE_ROW_BATCH_MAX_ROWS}-row sweeps"
        );
    }
}

/// Kill switch for the grouped path: `AVAROK_GLM_MOE_HOST_DISPATCH=1` restores the
/// read-ids-to-host expert loop. Read once — this sits on the per-layer decode path.
fn host_dispatch_forced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("AVAROK_GLM_MOE_HOST_DISPATCH").as_deref() == Ok("1"))
}

/// Say once which expert-dispatch path this process took. A missing `w4a16_gemv_sw_moe`
/// entry point falls back SILENTLY otherwise, and the fallback is the slow one.
fn announce_dispatch(grouped: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if grouped {
            tracing::info!("GLM MoE: grouped device dispatch (no per-layer D2H)");
        } else {
            tracing::info!("GLM MoE: host dispatch (per-layer stream sync + D2H)");
        }
    });
}

/// One routed MoE site, one token. Leaves a **partial sum** in `out` whenever this rank shares
/// the experts (EP) or the shared expert (TP) with anyone else.
///
/// # 🪤 The device→host round trip
///
/// The expert loop reads the selected ids back to the host to decide which experts are local.
/// That is a synchronising `copy_d2h` on the decode critical path, once per routed layer.
/// It is deliberate for this slice: correctness first, and it is exactly what the gated
/// microtest does. The upgrade path is the pointer-table grouped GEMM
/// (`layers::moe::ptr_table_build`), which keeps the routing on device — not a change to
/// this math.
/// Routed MoE over `rows` rows.
///
/// 🔴 The routed experts amortize PARTIALLY over a verify's rows. Measured on the live routing
/// trace (42 series x 406 steps), the union of selected experts over K consecutive tokens is
/// 8.00 / 13.74 / 18.76 / 23.35 at K = 1..4, so their weight traffic grows with K however the
/// loop is written — but it grows along that curve, not along 8K.
///
/// 🔴 RETRACTED (2026-08-29): this comment used to say the routed experts "stay one row at a
/// time" and that deduplicating the union "needs a device-side sort the dispatch kernel does
/// not have yet". Both are wrong. `top_k * rows <= 64` ids resolve in ONE block by pairwise
/// scan — no sort — and `w4a16_gemv_sw_moe_batchm_mR` then sweeps each union expert once.
/// Measured on t69, K=3, six probes byte-identical either way: open512 20.25 -> 22.12 tok/s
/// (+9.2%), a 125.4 -> 114.8 ms step. At K=2 (the serving default) 18.91 -> 19.75.
///
/// 🔴 WIDENED (2026-08-31) from `rows <= 4` to `rows <= 8`. The 4 was the compiled tier family,
/// not the union's limit — `8 * 8 == 64` fits its single block exactly. This is what makes the
/// 8-row batched PREFILL sub-chunk (ANOMALIES A65) amortize its routed experts too; before it,
/// prefill batched every stage EXCEPT the experts, which by then were most of the traffic left.
///
/// The SHARED expert is a different animal again: it is the same weights for every row, so it
/// runs once over all of them regardless.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe(
    gpu: &dyn GpuBackend,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    w: &Glm5NextMoeWeights,
    x: DevicePtr,
    out: DevicePtr,
    rows: usize,
    ws: &Glm5NextMlpWorkspace,
    stream: u64,
) -> Result<()> {
    if rows == 0 || rows > ws.max_rows {
        bail!(
            "GLM MoE: {rows} rows do not fit a workspace built for {}",
            ws.max_rows
        );
    }
    if w.experts.len() != cfg.local_experts {
        bail!(
            "GLM MoE: {} bound experts but this rank owns {} of {}",
            w.experts.len(),
            cfg.local_experts,
            cfg.num_experts
        );
    }

    use crate::layers::glm5next_layer::profile;

    // 🔴 The routed experts DO amortize over a verify's rows — just not fully. The union of
    // the selected experts over K consecutive tokens is 8.00 / 13.74 / 18.76 / 23.35 at
    // K = 1..4 (live routing trace, 42 series x 406 steps), so K rows sweep that many experts
    // instead of 8K. The per-row path pays 8K; this one pays the union.
    // 🪤 The route trace reads `ids` back per row, which the batched path never does — leave
    // it on the per-row arm rather than reconstructing the trace from the union table.
    // Sub-groups the routed sweep runs at. `rows` may exceed the widest tier — the prefill
    // sub-chunk is 16 wide since the dense tier widened — so every gate below is PER GROUP.
    let groups = moe_row_groups(rows, row_batch_max());
    // ── PREFILL ONLY: the whole row group through the tensor-core grouped GEMM ──
    //
    // 🔴 `rows > MOE_ROW_BATCH_MAX_ROWS` IS the prefill test. Decode is 1 row and a
    // speculative verify is capped at `DENSE_GEMV_BATCHM_MAX_M` (= 8 = this constant), so the
    // only caller that can be wider is `Glm5NextLayer::prefill`'s sub-chunk. Both of the
    // narrow paths therefore keep the bit-identical GEMV arm, unchanged, and prefill — which
    // has never been bit-identical to decode anyway, since the dense projections leave the
    // batched GEMV above the same width — takes the GEMM.
    //
    // 🪤 The same test is why this is safe under CUDA graph capture: only decode is captured,
    // and decode can never reach here, so the host read of `expert_offsets` inside
    // `forward_moe_grouped_prefill` cannot land inside a capture.
    //
    // 🔴 The width FLOOR is load-bearing, not caution. MEASURED at 5,400 tokens, matched
    // arms on one binary — GEMV / grouped GEMM tok/s: 16 rows 63.05 / 25.14 (0.40x),
    // 64 rows 77.79 / 57.71 (0.74x), 128 rows 80.73 / 88.17 (1.09x), 256 rows 82.14 /
    // 128.10 (1.56x). The crossover is between 64 and 128. Without the floor, "default ON"
    // would have regressed the shipping serve 2.5x, because PREFILL_ROWS defaults to 16.
    let grouped_prefill = rows > MOE_ROW_BATCH_MAX_ROWS
        && rows >= forward_prefill_gemm::prefill_gemm_min_rows()
        && forward_prefill_gemm::prefill_gemm_enabled()
        && !host_dispatch_forced()
        // The route trace reads `ids` back PER ROW; the grouped path never materialises a
        // per-row id list. Leave tracing on the arm that can serve it.
        && !profile::trace_on()
        && k.moe_sort_by_expert.0 != 0
        && k.moe_grouped_gemm.0 != 0
        && k.combine_indexed.0 != 0
        && rows * cfg.top_k <= ws.max_total_expanded();
    let batched = !grouped_prefill
        && rows >= 2
        && !host_dispatch_forced()
        && !row_batch_disabled()
        && !profile::trace_on()
        && k.moe_row_union.0 != 0
        && groups.iter().all(|&(_, w)| {
            // 🪤 The union table is one block of `w * top_k` threads; past 64 ids it would
            // silently drop entries. GLM-5.3 is 8 x 8 = 64 exactly, so this is a live edge.
            w >= 2
                && w * cfg.top_k <= MOE_ROW_UNION_MAX_IDS
                && k.w4a16_gemv_sw_moe_batchm[w - 2].0 != 0
        });
    announce_row_batch(batched, rows);
    announce_grouped_prefill(grouped_prefill, rows);

    // ── router: FULL expert set, FP32 logits, replicated on every rank ──
    let t = profile::start();
    // 🔴 ONE GEMM for every row. This was `rows` M=1 GEMVs — at 256 prefill rows x 42
    // sparse layers, ~10,700 launches per chunk, each re-reading the ENTIRE router weight
    // to produce one row of logits. The comment this replaces named the blocker exactly:
    // "No FP32-out batchm twin exists". It exists now.
    //
    // The tile GEMM was no escape either: the router is the worst tile case in the stack,
    // N = 288 tiling to 18 blocks at a measured 6.8 GB/s.
    //
    // 🪤 Router logits pick the experts, so reassociation can flip a top-k tie and route a
    // token to a different expert. That is a real behaviour change, not just a numeric one,
    // which is why it is gated and measured on answer quality rather than on tok/s alone.
    // Bounded to prefill widths by the same M > 16 cut used everywhere: decode and the
    // speculative verify stay on the bit-identical GEMV and route exactly as before.
    if rows > crate::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize
        && crate::layers::glm5next_layer::cublas_wide_proj()
    {
        crate::layers::ops::cublas_bf16_proj_dense_f32_out(
            x,
            w.router,
            ws.logits,
            rows as u32,
            cfg.num_experts as u32,
            cfg.hidden as u32,
            stream,
        )?;
    } else {
        for r in 0..rows {
            gemm(
                gpu,
                k.gemm_f32,
                k.gemv_f32,
                // No FP32-out batchm twin at these widths; stays on gemv/tile.
                KernelHandle(0),
                x.offset(r * cfg.hidden * 2),
                w.router,
                ws.logits.offset(r * cfg.num_experts * 4),
                1,
                cfg.num_experts,
                cfg.hidden,
                stream,
            )?;
        }
    }
    // ONE top-k for every row. `glm5next_router_topk` already takes the row on `blockIdx.x`
    // and strides `logits`/`ids`/`wts` by it, so this is the identical per-row work in one
    // launch instead of K — and it was a `grid [1,1,1]` launch, 120 of them per K=3 step for
    // 1.75 ms (nsys 2026-08-29). Bit-identical: no row's arithmetic changes.
    KernelLaunch::new(gpu, k.router)
        .grid([rows as u32, 1, 1])
        .block([ACT_BLOCK, 1, 1])
        .arg_ptr(ws.logits)
        .arg_ptr(w.router_bias)
        .arg_ptr(ws.ids)
        .arg_ptr(ws.wts)
        .arg_u32(cfg.num_experts as u32)
        .arg_u32(cfg.top_k as u32)
        // n_group: the parser already refuses anything but 1; the kernel refuses too.
        .arg_u32(1)
        .arg_f32(cfg.routed_scale)
        .arg_u32(u32::from(cfg.renormalize))
        .arg_u32(u32::from(cfg.router_bf16_ladder))
        .launch(stream)?;
    profile::end(profile::MOE_ROUTER, t, gpu, stream);

    // 🪤 Zero FIRST. A slot this rank does not own must contribute exactly zero to the
    // all-reduced sum; leaving the previous token's expert output there is a wrong answer
    // that only appears at EP > 1 and only for tokens whose routing moved. `expert_out` is
    // `[rows, top_k, hidden]` and contiguous, so one memset covers every row.
    gpu.memset_async(ws.expert_out, 0, rows * cfg.top_k * cfg.hidden * 2, stream)?;

    for r in 0..rows {
        if batched || grouped_prefill {
            break; // the experts run once for ALL rows, after this loop
        }
        let xr = x.offset(r * cfg.hidden * 2);
        let ids_r = ws.ids.offset(r * cfg.top_k * 4);
        let expert_out_r = ws.expert_out.offset(r * cfg.top_k * cfg.hidden * 2);

        let grouped = !host_dispatch_forced() && k.w4a16_gemv_sw_moe.0 != 0;
        announce_dispatch(grouped);
        if grouped {
            // ── grouped, device-dispatched: routing never leaves the GPU ──
            //
            // 🔴 The host loop this replaces did `synchronize` + `copy_d2h(ids)` once per routed
            // layer — 42 full stream drains per decode token on GLM-5.3, and the reason the
            // decode step could not be graph-captured. It also issued one launch per LOCAL
            // expert per projection (~16/layer); this is four, whatever the routing picks.
            let t = profile::start();
            let mi = cfg.moe_intermediate;
            // gate and up: every slot reads the SAME x, so input_stride = 0.
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                xr,
                &w.ptrs.gate,
                ws.a_gate,
                ids_r,
                mi,
                cfg.hidden,
                cfg.top_k,
                cfg.num_experts,
                0,
                stream,
            )?;
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                xr,
                &w.ptrs.up,
                ws.a_up,
                ids_r,
                mi,
                cfg.hidden,
                cfg.top_k,
                cfg.num_experts,
                0,
                stream,
            )?;
            // Elementwise over all slots at once. Remote slots activate uninitialised rows; the
            // down projection skips them, so those rows are never read.
            swiglu(
                gpu,
                k.swiglu,
                ws.a_gate,
                ws.a_up,
                ws.a_act,
                cfg.top_k * mi,
                cfg.swiglu_limit,
                stream,
            )?;
            // down: slot-major activations, so input_stride = one expert's width.
            w4a16_gemv_moe(
                gpu,
                k.w4a16_gemv_sw_moe,
                ws.a_act,
                &w.ptrs.down,
                expert_out_r,
                ids_r,
                cfg.hidden,
                mi,
                cfg.top_k,
                cfg.num_experts,
                mi,
                stream,
            )?;
            profile::end(profile::MOE_EXPERTS, t, gpu, stream);

            if profile::trace_on() {
                let mut ids = vec![0u8; cfg.top_k * 4];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(ids_r, &mut ids)?;
                let decoded: Vec<i32> = (0..cfg.top_k)
                    .map(|k| {
                        i32::from_le_bytes([
                            ids[k * 4],
                            ids[k * 4 + 1],
                            ids[k * 4 + 2],
                            ids[k * 4 + 3],
                        ])
                    })
                    .collect();
                profile::stash_route(&decoded);
            }
        } else {
            // 🚩 A FULL STREAM SYNC + D2H IN THE MIDDLE OF EVERY MoE LAYER. The routing decision
            // is read back to the host so the expert GEMMs can be launched by id. Timed on its own
            // because it is the one span here that is pure latency and scales with layer count,
            // not with weight bytes.
            let t = profile::start();
            let mut ids = vec![0u8; cfg.top_k * 4];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ids_r, &mut ids)?;
            profile::end(profile::MOE_HOSTSYNC, t, gpu, stream);

            if profile::trace_on() {
                let decoded: Vec<i32> = (0..cfg.top_k)
                    .map(|k| {
                        i32::from_le_bytes([
                            ids[k * 4],
                            ids[k * 4 + 1],
                            ids[k * 4 + 2],
                            ids[k * 4 + 3],
                        ])
                    })
                    .collect();
                profile::stash_route(&decoded);
            }

            let t = profile::start();
            for slot in 0..cfg.top_k {
                let id = i32::from_le_bytes([
                    ids[slot * 4],
                    ids[slot * 4 + 1],
                    ids[slot * 4 + 2],
                    ids[slot * 4 + 3],
                ]);
                // -1 is the kernel's "slot unfilled" sentinel; it is reachable only if top_k exceeded
                // the expert count, which `validate` refuses. Treat it as zero rather than as an index.
                if id < 0 {
                    continue;
                }
                let id = id as usize;
                if id >= cfg.num_experts {
                    bail!(
                        "GLM MoE: router selected expert {id} of {}",
                        cfg.num_experts
                    );
                }
                let Some(local) = cfg.local_slot(id) else {
                    continue; // another rank owns it; its zero row is already in place.
                };
                let e = &w.experts[local];
                let dst = expert_out_r.offset(slot * cfg.hidden * 2);
                let mi = cfg.moe_intermediate;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    xr,
                    &e.gate_proj,
                    ws.a_gate,
                    mi,
                    cfg.hidden,
                    stream,
                )?;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    xr,
                    &e.up_proj,
                    ws.a_up,
                    mi,
                    cfg.hidden,
                    stream,
                )?;
                swiglu(
                    gpu,
                    k.swiglu,
                    ws.a_gate,
                    ws.a_up,
                    ws.a_act,
                    mi,
                    cfg.swiglu_limit,
                    stream,
                )?;
                w4a16_gemv(
                    gpu,
                    k.w4a16_gemv,
                    k.w4a16_gemv_sw,
                    ws.a_act,
                    &e.down_proj,
                    dst,
                    cfg.hidden,
                    mi,
                    stream,
                )?;
            }

            profile::end(profile::MOE_EXPERTS, t, gpu, stream);
        }
    }

    if grouped_prefill {
        // ONE launch per projection over the WHOLE row group — the 8-row cap does not apply.
        // Leaves the routed outputs in `ws.expert_out` in EXPERT-SORTED order; the combine
        // below reads them through `ws.token_to_perm`.
        let t = profile::start();
        forward_prefill_gemm::forward_moe_grouped_prefill(gpu, k, cfg, w, x, rows, ws, stream)?;
        profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    }

    if batched {
        let t = profile::start();
        let mi = cfg.moe_intermediate;
        // ONE sweep per sub-group. At `rows <= MOE_ROW_BATCH_MAX_ROWS` this is the single pass it
        // always was; a wider prefill sub-chunk runs it twice over disjoint row ranges, which is
        // byte-identical (see `moe_row_groups`) and keeps the routed experts at the tier width
        // that was actually measured.
        for &(r0, w_rows) in &groups {
            // The union table: one block, one thread per (row, slot) id. Stays on device.
            // 🪤 Rebuilt per sub-group over that group's slice of `ids` — the scratch is sized for
            // the widest group, and a later group overwrites the earlier one's table after its
            // sweeps have been issued on the same stream.
            KernelLaunch::new(gpu, k.moe_row_union)
                .grid([1, 1, 1])
                .block([(w_rows * cfg.top_k) as u32, 1, 1])
                .arg_ptr(ws.ids.offset(r0 * cfg.top_k * 4))
                .arg_ptr(ws.u_eid)
                .arg_ptr(ws.u_slot)
                .arg_u32(w_rows as u32)
                .arg_u32(cfg.top_k as u32)
                .launch(stream)?;

            let kb = k.w4a16_gemv_sw_moe_batchm[w_rows - 2];
            // gate and up: a row's slots all read the SAME x, so the slot stride is 0.
            w4a16_gemv_moe_batchm(
                gpu,
                kb,
                x.offset(r0 * cfg.hidden * 2),
                &w.ptrs.gate,
                ws.a_gate.offset(r0 * cfg.top_k * mi * 2),
                ws.u_eid,
                ws.u_slot,
                mi,
                cfg.hidden,
                w_rows,
                cfg.top_k,
                cfg.num_experts,
                cfg.hidden,
                0,
                cfg.top_k * mi,
                stream,
            )?;
            w4a16_gemv_moe_batchm(
                gpu,
                kb,
                x.offset(r0 * cfg.hidden * 2),
                &w.ptrs.up,
                ws.a_up.offset(r0 * cfg.top_k * mi * 2),
                ws.u_eid,
                ws.u_slot,
                mi,
                cfg.hidden,
                w_rows,
                cfg.top_k,
                cfg.num_experts,
                cfg.hidden,
                0,
                cfg.top_k * mi,
                stream,
            )?;
            // Elementwise over every (row, slot) at once. Slots this rank does not own activate
            // uninitialised rows; the down projection skips them, so those rows are never read.
            swiglu(
                gpu,
                k.swiglu,
                ws.a_gate.offset(r0 * cfg.top_k * mi * 2),
                ws.a_up.offset(r0 * cfg.top_k * mi * 2),
                ws.a_act.offset(r0 * cfg.top_k * mi * 2),
                w_rows * cfg.top_k * mi,
                cfg.swiglu_limit,
                stream,
            )?;
            // down: slot-major activations, so the slot stride is one expert's width.
            w4a16_gemv_moe_batchm(
                gpu,
                kb,
                ws.a_act.offset(r0 * cfg.top_k * mi * 2),
                &w.ptrs.down,
                ws.expert_out.offset(r0 * cfg.top_k * cfg.hidden * 2),
                ws.u_eid,
                ws.u_slot,
                cfg.hidden,
                mi,
                w_rows,
                cfg.top_k,
                cfg.num_experts,
                cfg.top_k * mi,
                mi,
                cfg.top_k * cfg.hidden,
                stream,
            )?;
        }
        profile::end(profile::MOE_EXPERTS, t, gpu, stream);
    }

    // ── shared expert: BF16, TP-sharded, NOT routed-scaled ──
    let t = profile::start();
    forward_dense(
        gpu,
        k,
        cfg,
        &w.shared,
        cfg.local_shared_intermediate,
        x,
        ws.shared_out,
        rows,
        ws,
        stream,
    )?;

    // 🔴 The combine runs BEFORE the all-reduce, so the TP-partial shared expert and the
    // EP-partial routed sum reduce together in one collective. Adding the shared output after
    // a reduce — the `layers::moe` pattern, written for a replicated shared expert — would
    // keep only this rank's half of it.
    profile::end(profile::MOE_SHARED, t, gpu, stream);
    let t = profile::start();
    // ONE combine for every row: `glm5next_moe_combine` takes the row on `blockIdx.x` and
    // strides all four buffers by it. Was K `grid [1,1,1]` launches — 1.50 ms of a K=3 step.
    //
    // 🪤 The grouped path's routed rows are EXPERT-SORTED, so it takes the `_indexed` twin —
    // same accumulation order, same single rounding, one extra indirection through
    // `token_to_perm`. Reading the sorted buffer with the plain kernel would silently combine
    // whichever tokens happened to land at `t * top_k + k`.
    if grouped_prefill {
        KernelLaunch::new(gpu, k.combine_indexed)
            .grid([rows as u32, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.expert_out)
            .arg_ptr(ws.token_to_perm)
            .arg_ptr(ws.wts)
            .arg_ptr(ws.shared_out)
            .arg_ptr(out)
            .arg_u32(cfg.hidden as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;
    } else {
        KernelLaunch::new(gpu, k.combine)
            .grid([rows as u32, 1, 1])
            .block([ACT_BLOCK, 1, 1])
            .arg_ptr(ws.expert_out)
            .arg_ptr(ws.wts)
            .arg_ptr(ws.shared_out)
            .arg_ptr(out)
            .arg_u32(cfg.hidden as u32)
            .arg_u32(cfg.top_k as u32)
            .launch(stream)?;
    }
    profile::end(profile::MOE_COMBINE, t, gpu, stream);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MOE_ROW_BATCH_MAX_ROWS, moe_row_groups};

    /// The split must cover every row exactly once, in order, never exceed the tier width,
    /// and never emit a group of ONE — there is no `w4a16_gemv_sw_moe_batchm_m1`, so a
    /// trailing single row would silently drop the whole batched arm for that sub-chunk.
    #[test]
    fn row_groups_cover_and_never_orphan_a_row() {
        for cap in 1..=MOE_ROW_BATCH_MAX_ROWS {
            for rows in 1..=64 {
                let g = moe_row_groups(rows, cap);
                assert_eq!(g[0].0, 0, "rows={rows} cap={cap}: does not start at 0");
                let mut next = 0;
                for &(start, w) in &g {
                    assert_eq!(start, next, "rows={rows} cap={cap}: gap or overlap");
                    assert!(w >= 1 && w <= cap, "rows={rows} cap={cap}: width {w}");
                    // No orphan at the width that ships. At a small cap an all->=2 split can be
                    // arithmetically impossible (3 rows at cap 2), and the caller's per-group
                    // tier gate handles that by falling back — see the fn doc.
                    if rows >= 2 && cap == MOE_ROW_BATCH_MAX_ROWS {
                        assert!(w >= 2, "rows={rows} cap={cap}: orphaned a single row");
                    }
                    next += w;
                }
                assert_eq!(next, rows, "rows={rows} cap={cap}: {next} rows covered");
            }
        }
    }

    /// The two widths this actually ships at.
    #[test]
    fn row_groups_at_the_shipping_widths() {
        assert_eq!(moe_row_groups(16, 8), vec![(0, 8), (8, 8)]);
        assert_eq!(moe_row_groups(8, 8), vec![(0, 8)]);
        assert_eq!(moe_row_groups(9, 8), vec![(0, 5), (5, 4)]);
    }
}
