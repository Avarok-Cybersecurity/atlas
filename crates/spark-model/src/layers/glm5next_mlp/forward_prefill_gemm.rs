// SPDX-License-Identifier: AGPL-3.0-only

//! GLM routed-MoE **prefill** through the shared tensor-core grouped W4A16 GEMM.
//!
//! # What this replaces, and why it is not a new kernel
//!
//! `forward_moe`'s routed arm runs `w4a16_gemv_sw_moe_batchm_mR` — a software-dequant
//! GEMV with **no `mma.sync`** — and `MOE_ROW_BATCH_MAX_ROWS` caps it at 8 rows per
//! launch *regardless of `AVAROK_GLM_PREFILL_ROWS`*. A 256-row prefill sub-chunk is
//! therefore 32 separate 8-row sweeps, and each sweep re-reads the weights of every
//! expert its 8 rows selected. Measured share of prefill: **36.9 % (2026-09-02) to
//! 49.3 % (2026-09-06) of the GPU-busy window** — the single largest bucket in both
//! profiles, and at 94-100 % of its own DRAM roofline, so the win cannot come from
//! more FLOP/s. It has to come from reading each expert's weights FEWER TIMES.
//!
//! `kernels/gb10/common/moe_w4a16_grouped_gemm.cu::moe_w4a16_grouped_gemm_ptrtable`
//! does exactly that: **one launch for all experts**, `mma.sync.aligned.m16n8k16`,
//! `expert_offsets` prefix sum, in-kernel A-gather through `sorted_token_ids`. Thirteen
//! other MoE models already run their prefill on it through
//! `layers::moe::forward_prefill_routed`. GLM never did, because `glm5next_mlp` is a
//! separate bespoke implementation.
//!
//! # 🔴 Layout compatibility — VERIFIED, not assumed
//!
//! The two kernels read the SAME bytes with the same convention. Checked term by term
//! against `w4a16_gemv.cu::w4a16_gemv_partial_rows` (the GEMV this replaces) and
//! `moe_w4a16_grouped_gemm.cu::moe_w4a16_grouped_gemm_ptrtable`:
//!
//! | term | GEMV (`w4a16_gemv.cu`) | grouped GEMM | same? |
//! |---|---|---|---|
//! | packed byte for logical `k` | `B_packed[n * (K/2) + k/2]` (`kk*8 + b`, `k = kk*16 + 2b(+1)`) | `B_expert[gn * (K/2) + gk/2]` | ✅ N-major `[N, K/2]` |
//! | nibble | even `k` → `byte & 0xF`, odd → `byte >> 4` | `(gk & 1) ? (byte >> 4) : (byte & 0xF)` | ✅ |
//! | codebook | `E2M1_LUT` | `E2M1_LUT_MOE` (same 16 values) | ✅ |
//! | block scale | `B_scale[n * (K/16) + k/16]`, E4M3 | `S_expert[gn * (K/GROUP_SIZE) + k_base/GROUP_SIZE]`, E4M3, `GROUP_SIZE 16` | ✅ |
//! | per-tensor scale | `scale2_vals[eid]`, multiplied into the block scale | `scale2_vals[expert_id]`, multiplied into the dequant | ✅ |
//! | remote expert (EP) | `packed_ptrs[eid] == 0` → return, caller's zeros stand | `B_expert == 0` → return, caller's zeros stand | ✅ |
//! | table index | GLOBAL expert id, `[num_experts]` | GLOBAL expert id, `[num_experts]` | ✅ |
//!
//! `weight_loader/glm5_next_load/nvfp4_dequant.rs` states the same three conventions
//! (ModelOpt direct-multiplier order, `GROUP_SIZE = 16` not DeepSeek-V4's 32, even flat
//! index = LOW nibble). **No permutation, no repack, no extra bytes at load** — the
//! already-uploaded `Glm5NextMoePtrTables` are passed straight through.
//!
//! # 🔴 What is NOT bit-identical — and it is NOT only re-association
//!
//! Two things move, and the second one is the bigger one:
//!
//! 1. **Association.** The GEMM accumulates a `k`-major `mma.sync` FP32 chain over
//!    `K_STEP = 16` tiles; the GEMV accumulates two interleaved FP32 `fmaf` chains per
//!    orig-lane and combines them through a warp shuffle tree.
//! 2. **Operand precision.** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` takes
//!    BF16 operands, so the grouped kernel stages its dequantised weight tile as
//!    `__float2bfloat16(E2M1_LUT[nibble] * e4m3 * scale2)` in shared memory. The GEMV keeps
//!    that product in **FP32** all the way into its `fmaf`. The grouped path therefore
//!    carries an extra ~2^-9 relative rounding PER WEIGHT ELEMENT that the GEMV does not.
//!
//! MEASURED 2026-09-22, `examples/glm5next_moe_grouped_prefill_microtest.rs` on n1 (GB10),
//! M=64, top_k=8, 16 experts, N=256, K=512, random NVFP4, against an FP32 host reference:
//!
//! | arm | max_abs | max_abs / ‖ref‖∞ | max_rel |
//! |---|---:|---:|---:|
//! | grouped GEMM vs exact FP32 ref | 1.462067 | 0.003306 | 0.128986 |
//! | GEMV (production) vs the same ref | 0.999542 | 0.002260 | 0.003889 |
//! | **grouped GEMM vs BF16-WEIGHT ref** | **0.998566** | **0.002258** | **0.003888** |
//! | grouped GEMM vs GEMV | 2.000000 | 0.004522 | 0.130725 |
//!
//! The third row is the finding: against a reference that rounds the dequantised weight to
//! BF16 first, the grouped GEMM scores `0.003888` — the GEMV's own `0.003889` to five
//! figures. **All of the residual gap is operand precision; none of it is layout.** That is
//! what makes the layout claim above evidence rather than assertion.
//!
//! It is still a numerics change, so it is gated OFF for decode and for the speculative
//! verify (`rows <= MOE_ROW_BATCH_MAX_ROWS`), which must stay bit-identical, and ON only
//! for prefill — which already left the bit-identical tier when the dense projections moved
//! to cuBLASLt above the same width.
//!
//! # Buffer discipline
//!
//! 🪤 **A59**: every buffer this path needs is allocated in `Glm5NextMlpWorkspace::new`,
//! at load, sized from `max_rows`. Nothing here allocates. The only per-call host traffic
//! is one `[num_experts + 1]` i32 read of `expert_offsets` (see [`max_m_tiles_from_offsets`]).

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::forward::Glm5NextMlpWorkspace;
use super::weights::Glm5NextMoeWeights;
use super::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// `M_TILE` of `moe_w4a16_grouped_gemm_ptrtable`. 🪤 COUPLED to `#define M_TILE 64` in
/// `kernels/gb10/common/moe_w4a16_grouped_gemm.cu`; `max_m_tiles` is counted in these.
pub(crate) const GROUPED_M_TILE: usize = 64;

/// Route the routed-expert **prefill** through the grouped GEMM.
/// `AVAROK_GLM_MOE_PREFILL_GEMM=0` restores the row-batched GEMV path exactly.
///
/// 🔬 Default ON in this branch so the A/B is one env flip on ONE image — the same lever
/// shape as `AVAROK_GLM_MOE_ROW_BATCH_MAX`. Read once: this sits on the per-layer path.
pub(crate) fn prefill_gemm_enabled() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let off = std::env::var("AVAROK_GLM_MOE_PREFILL_GEMM").as_deref() == Ok("0");
        if off {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM DISABLED \
                 (AVAROK_GLM_MOE_PREFILL_GEMM=0) — row-batched GEMV path"
            );
        }
        !off
    })
}

/// Narrowest prefill sub-chunk the grouped GEMM may take, `AVAROK_GLM_MOE_PREFILL_GEMM_MIN_ROWS`.
///
/// 🔴 MEASURED, and the reason this gate exists at all. n1/n2, one image
/// (`ws1p1-moegemm`, `bfeed3d84`), community ckpt, 5,400-token prompt, matched arms:
///
/// | sub-chunk | GEMV (production) | grouped GEMM | ratio |
/// |---|---:|---:|---:|
/// | 16 rows (the shipping default) | **63.05 tok/s** | **25.14 tok/s** | **0.40x** |
/// | 256 rows | 82.14 tok/s | **128.10 tok/s** | **1.56x** |
///
/// At 16 rows the grouped path is a 2.5x REGRESSION, and the arithmetic says why: 16 rows
/// at `top_k = 8` is 128 routed slots over 288 experts, so the union of experts a chunk
/// touches is barely smaller than the GEMV's own — there is no weight traffic to save —
/// while the path still pays `ceil(N/16) * layers` host stream drains and an `M_TILE = 64`
/// tile per active expert holding ~2.6 real rows. At 256 rows the same chunk touches
/// essentially this rank's WHOLE local expert set once, which is ~5x less weight traffic
/// per token, and the win appears.
///
/// 🪤 So a default-ON grouped path with no width floor would have silently regressed the
/// shipping serve by 2.5x, because `PREFILL_ROWS` defaults to 16. The floor is what makes
/// "default ON" safe. It is set at 64 — above the measured-bad 16, below the measured-good
/// 256 — and it is an env lever so the crossover can be swept without a rebuild.
pub(crate) fn prefill_gemm_min_rows() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        let m = std::env::var("AVAROK_GLM_MOE_PREFILL_GEMM_MIN_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(DEFAULT_GEMM_MIN_ROWS)
            .max(1);
        if m != DEFAULT_GEMM_MIN_ROWS {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM width floor overridden to {m} rows \
                 (default {DEFAULT_GEMM_MIN_ROWS}; MEASURED 0.40x at 16, 1.56x at 256)"
            );
        }
        m
    })
}

/// See [`prefill_gemm_min_rows`] for the measurement this number comes from.
pub(crate) const DEFAULT_GEMM_MIN_ROWS: usize = 64;

/// Read the REAL expert histogram to size the grid, `AVAROK_GLM_MOE_PREFILL_GEMM_EXACT_TILES=0`
/// to use the worst-case bound instead and skip the host read entirely.
///
/// 🔬 A discriminator, not a tuning knob. The exact bound costs one `copy_d2h_on_stream` per
/// routed layer per sub-chunk, and that call DRAINS THE STREAM — at a 16-row sub-chunk over a
/// 5,400-token prompt that is `338 * 42 = 14,196` full host-GPU round trips. Turning it off
/// trades those drains for a grid that is `ceil(rows*top_k/64)`x taller and almost entirely
/// early-exit. Flipping this against a fixed arm separates "the sync is the cost" from "the
/// tile padding is the cost" in ONE image instead of guessing.
pub(crate) fn prefill_gemm_exact_tiles() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| {
        let off = std::env::var("AVAROK_GLM_MOE_PREFILL_GEMM_EXACT_TILES").as_deref() == Ok("0");
        if off {
            tracing::warn!(
                "GLM routed-MoE prefill grouped GEMM: WORST-CASE grid height \
                 (AVAROK_GLM_MOE_PREFILL_GEMM_EXACT_TILES=0) — no per-layer expert_offsets D2H"
            );
        }
        !off
    })
}

/// Tiles of [`GROUPED_M_TILE`] the busiest expert needs, from the host copy of
/// `expert_offsets`.
///
/// 🔴 Why this is not the worst case. The kernel's grid is
/// `(ceil(N/64), max_m_tiles, num_experts)` and every CTA past an expert's real row count
/// early-exits. The safe bound — one expert takes every routed slot — is
/// `ceil(rows * top_k / 64)`: at `rows = 256`, `top_k = 8` that is 32, i.e. a
/// `32 x 32 x 288 = 294,912`-CTA launch of which ~97 % do nothing, three times per layer.
/// Reading the REAL offsets makes it `1 x 32 x 288`. 288 experts over 2,048 routed slots
/// average 7.1 rows each, so one tile is the normal answer.
///
/// 🪤 Cannot truncate: the result is the max over the ACTUAL per-expert counts, and it is
/// clamped to the worst case only as an upper bound. `layers::moe`'s prefill makes the same
/// trade (`moe_prefill_exact_tiles`, default-on for NVFP4, measured -120.7 ms cold TTFT).
pub(crate) fn max_m_tiles_from_offsets(offsets: &[i32], worst_case: u32) -> u32 {
    let mut prev = 0i32;
    let mut max_rows = 0i32;
    for &cur in offsets.iter().skip(1) {
        max_rows = max_rows.max(cur - prev);
        prev = cur;
    }
    (max_rows.max(0) as u32)
        .div_ceil(GROUPED_M_TILE as u32)
        .max(1)
        .min(worst_case.max(1))
}

/// `C[te, n_out] = gather(A)[te, k] @ dequant(expert weights)^T`, all experts, ONE launch.
///
/// 🪤 grid.x is COUPLED to the kernel's `N_TILE = 64`; block is 128 threads (4 warps of
/// `M_TILE/16`). Mirrors `ops::moe_w4a16_grouped_gemm_ptrtable`, which is `pub` but lives
/// behind `MoeLayer`'s own dispatch — called directly here to keep GLM off that type.
#[allow(clippy::too_many_arguments)]
fn grouped_gemm(
    gpu: &dyn GpuBackend,
    k: spark_runtime::gpu::KernelHandle,
    a: DevicePtr,
    t: &super::weights::Glm5NextExpertPtrTable,
    c: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: usize,
    n_out: usize,
    kk: usize,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k)
        .grid([(n_out as u32).div_ceil(64), max_m_tiles, num_experts as u32])
        .block([128, 1, 1])
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
pub(super) fn forward_moe_grouped_prefill(
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
    let worst_case = te.div_ceil(GROUPED_M_TILE).max(1) as u32;
    let max_m_tiles = if prefill_gemm_exact_tiles() {
        let mut off_raw = vec![0u8; (cfg.num_experts + 1) * 4];
        gpu.copy_d2h_on_stream(ws.expert_offsets(), &mut off_raw, stream)?;
        let offsets: Vec<i32> = off_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        max_m_tiles_from_offsets(&offsets, worst_case)
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
        stream,
    )?;

    // ── 4. clamped SwiGLU over every sorted row at once ──
    // 🪤 GLM's clamp is ASYMMETRIC and is NOT `moe_silu_mul`. Elementwise, so the sorted
    // layout changes nothing. Rows belonging to a remote expert hold uninitialised values
    // here; the down GEMM skips them on the same null-pointer test, so they are never read.
    super::forward::swiglu_rows(
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
        stream,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests;
