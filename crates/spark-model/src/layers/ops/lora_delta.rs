// SPDX-License-Identifier: AGPL-3.0-only

//! Runtime LoRA delta: y += scale * (x @ A^T) @ B^T, BF16 side-path.
//! Zero new CUDA kernels — reuses dense_gemv_bf16 / dense_gemm_tc /
//! dense_gemm_bf16 / bf16_scaled_add, all shipped in kernels/gb10/common/.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Resolved once at adapter load (module names per common/KERNEL.toml:
/// gemv=dense_gemv_bf16.cu, gemm_tc=dense_gemm_tc.cu, gemm=dense_gemm_bf16.cu,
/// residual_add=residual_add.cu — stem, no override).
#[derive(Clone, Copy)]
pub struct LoraKernels {
    pub gemv_k: KernelHandle,
    pub gemm_tc_k: KernelHandle, // KernelHandle(0) on miss -> gemm_k fallback
    pub gemm_k: KernelHandle,
    pub scaled_add_k: KernelHandle,
    /// M2 fused batched bgmv (per-request routing): shrink then expand+fold.
    /// Module "lora_bgmv" (stem == module — no KERNEL.toml override).
    pub bgmv_shrink_k: KernelHandle,
    pub bgmv_expand_fold_k: KernelHandle,
}

impl LoraKernels {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemm_tc_k: crate::layers::try_kernel(gpu, "gemm_tc", "dense_gemm_tc"),
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            scaled_add_k: gpu.kernel("residual_add", "bf16_scaled_add")?,
            bgmv_shrink_k: gpu.kernel("lora_bgmv", "lora_bgmv_shrink")?,
            bgmv_expand_fold_k: gpu.kernel("lora_bgmv", "lora_bgmv_expand_fold")?,
        })
    }
}

/// Frozen per-(layer,module) routing tables the bgmv reads: the `[max_loras]`
/// device pointer tables (`a_table`/`b_table`, NULL=base) + the shared
/// `[max_loras]` f32 `scale_table`, plus the projection dims. Load-time-fixed
/// device addresses (built at pool pack time), so they are stable kernel args
/// across CUDA-graph capture/replay — adapter identity flows ONLY through the
/// per-step `seq_slot` buffer. Installed by copy onto the layer next to the
/// active-slot [`LoraPair`] (which the single-seq n==1 path still uses).
#[derive(Debug, Clone, Copy)]
pub struct LoraRoute {
    pub a_table: DevicePtr,
    pub b_table: DevicePtr,
    pub scale_table: DevicePtr,
    pub k_in: u32,
    pub n_out: u32,
    pub max_rank: u32,
}

/// One adapted module. A/B are PEFT tensors VERBATIM (host F16->BF16 at load):
///   a: [rank, k_in]  row-major BF16  (PEFT lora_A [r, in_features] — already
///                                     the B-operand `[N,K]` layout dense_* expect)
///   b: [n_out, rank] row-major BF16  (PEFT lora_B [out_features, r] — likewise)
/// Both are rank-padded to the pool's max_rank (zero rows/cols beyond `rank`),
/// so kernels may uniformly run at the pool rank — bit-identical to true rank.
/// scale = lora_alpha/r, or lora_alpha/sqrt(r) under use_rslora — read per
/// adapter at load, never defaulted. Do NOT pre-fold into B (keeps tensors
/// verbatim for the M0 offline parity test); it rides the scaled_add for free.
#[derive(Debug, Clone, Copy)]
pub struct LoraPair {
    pub a: DenseWeight,
    pub b: DenseWeight,
    pub rank: u32,
    pub k_in: u32,
    pub n_out: u32,
    pub scale: f32,
    /// The pool's padded rank — the ROW STRIDE of `b` (and row count of `a`).
    /// Kernels MUST contract/produce at this dim, not `rank`: B rows are
    /// `max_rank` elements apart in the pool, so a `k = rank` expand would
    /// misread every row past the first when `rank < max_rank`. Pad rows of
    /// A and pad cols of B are zeroed at pack time, so running the shrink at
    /// `n = max_rank` and the expand at `k = max_rank` is bit-identical to
    /// the true-rank product.
    pub max_rank: u32,
}

/// Per-layer attention-side LoRA weights, installed by copy onto
/// `Qwen3AttentionLayer`.
///
/// The `q` pair folds the delta into the RAW q_proj output at offset 0, full
/// width = q_proj_dim (on a gated model the interleaved `[Q|gate]`, width
/// `2·q_heads·head_dim`), BEFORE the `deinterleave_qg` split — the PEFT
/// `lora_B` was trained against exactly that interleaved basis, so the delta
/// applies like k/v/o, just wider.
#[derive(Clone, Copy)]
pub struct LoraAttnWeights {
    /// #30: the TRUE global layer index (`0..num_hidden_layers`), stamped at
    /// install from the global `idx`. The prefill apply sites index the
    /// request slot's GLOBAL-layer-indexed pairs with THIS (not `attn_layer_idx`,
    /// an attention-only counter that diverges from the global index on hybrid
    /// GDN/attention models).
    pub layer_idx: usize,
    pub q: Option<LoraPair>,
    pub k: Option<LoraPair>,
    pub v: Option<LoraPair>,
    pub o: Option<LoraPair>,
    pub kernels: LoraKernels,
    /// M2 per-request routing tables (per module). `None` = single/global
    /// adapter with no routing (the n==1 path uses the pair above and stays
    /// byte-identical). `Some` when a multi-adapter pool is resident; the
    /// batched decode path reads these + the per-seq `seq_slot` via the bgmv.
    pub q_route: Option<LoraRoute>,
    pub k_route: Option<LoraRoute>,
    pub v_route: Option<LoraRoute>,
    pub o_route: Option<LoraRoute>,
}

/// Per-layer dense-FFN LoRA weights, installed by copy onto `DenseFfnLayer`.
#[derive(Clone, Copy)]
pub struct LoraFfnWeights {
    pub gate: Option<LoraPair>,
    pub up: Option<LoraPair>,
    pub down: Option<LoraPair>,
    pub kernels: LoraKernels,
}

/// base_out[m, n_out] += scale * (x[m, k_in] @ a^T) @ b^T.
///
/// CONTIGUITY CONTRACT: x rows contiguous with stride k_in*2 bytes, base_out
/// rows contiguous with stride n_out*2 bytes. Every v0 site satisfies this
/// (k/v/o/gate/up/down all land in dedicated contiguous buffers/regions);
/// strided cases (multi-seq per-seq qkv_buf) must loop with m=1 on offset ptrs.
///
/// GRAPH-SAFE: pure kernel launches, no alloc/sync; a/b (load-time device
/// weights), lora_xa/lora_delta (BufferArena, fixed address), and scale
/// (baked kernel arg, constant for a startup-static adapter) are all
/// pointer/value-stable across capture and replay — identical status to base
/// weights. m==1 -> GEMV; m>1 -> tensor-core GEMM (scalar fallback).
///
/// POOL LAYOUT (lora/mod.rs pack): A is [max_rank, k_in] (real rows at the
/// head, pad rows zero), B is [n_out, max_rank] row-major (pad COLS zero,
/// row stride = max_rank). Both stages therefore run at `pair.max_rank`:
/// shrink n = max_rank (xa pad cols come out zero), expand k = max_rank
/// (matches B's row stride; zero pads contribute nothing) — bit-identical
/// to a true-rank product.
#[allow(clippy::too_many_arguments)]
pub fn apply_lora_delta(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    x: DevicePtr,        // [m, pair.k_in] BF16
    base_out: DevicePtr, // [m, pair.n_out] BF16, modified in place
    m: u32,
    lora_xa: DevicePtr,    // arena scratch >= m * max_rank BF16
    lora_delta: DevicePtr, // arena scratch >= m * n_out BF16
    stream: u64,
) -> Result<()> {
    if m == 1 {
        // shrink: [1,k_in] @ A[max_rank,k_in]^T -> xa[1,max_rank]
        ops::dense_gemv(
            gpu,
            kernels.gemv_k,
            x,
            &pair.a,
            lora_xa,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        // expand: [1,max_rank] @ B[n_out,max_rank]^T -> delta[1,n_out]
        ops::dense_gemv(
            gpu,
            kernels.gemv_k,
            lora_xa,
            &pair.b,
            lora_delta,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    } else if kernels.gemm_tc_k.0 != 0 {
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            lora_xa,
            &pair.b,
            lora_delta,
            m,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    } else {
        ops::dense_gemm(
            gpu,
            kernels.gemm_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            kernels.gemm_k,
            lora_xa,
            &pair.b,
            lora_delta,
            m,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    }
    // fold: base_out += scale * delta   (kernels/gb10/common/residual_add.cu:60)
    ops::scaled_add(
        gpu,
        kernels.scaled_add_k,
        base_out,
        lora_delta,
        pair.scale,
        m * pair.n_out,
        stream,
    )
}

/// M2 per-request routed LoRA delta over a batch of `n` decode rows, each
/// naming its own adapter slot via `seq_slot[n]` (i32, `<0` = base/no delta).
/// Two launches — shrink then expand+fold — reading the module's frozen
/// `route.a_table`/`route.b_table`/`route.scale_table` (`[max_loras]` device
/// arrays, NULL/0 = base-only slot) at the load-time-fixed pool addresses.
///
/// `out[i, :] += scale_s * (x[i, :] @ A_s^T) @ B_s^T`   where `s = seq_slot[i]`.
///
/// BYTE-IDENTICAL to `n` sequential `apply_lora_delta(m=1)` calls for the same
/// `(x_i, s_i)` (the on-hardware oracle): kernel 1 is `dense_gemv_bf16` with a
/// per-row A-base gather (emits BF16 xa = the oracle's lora_xa boundary);
/// kernel 2 is the same body reading BF16 xa back, then the oracle's fold
/// (round delta→BF16, then `base += scale*bf16(delta)`), so per-slot scale is
/// applied in fp32 AFTER the BF16 delta rounding. Contraction runs at
/// `route.max_rank` (never true rank) — pad rows/cols are zero, bit-identical.
///
/// STRIDES (elements, not bytes):
/// - `x_row_stride`   : distance between `x` rows (normed = `h`; attn_out = `q_dim`).
/// - `out_row_stride` : distance between `base_out` rows. Contiguous O uses
///   `n_out`; the STRIDED K/V `qkv_buf` uses `per_seq_qkv/2` (BF16 elements) so
///   the fold lands inside the interleaved `[Q|K|V]` layout without corrupting it.
///
/// GRAPH-SAFE: only pointer/value-stable args — `x`/`base_out` are the fixed
/// forward buffers, the tables are load-time-fixed, `xa` is a fixed arena
/// scratch (`>= n*max_rank` BF16), and `seq_slot` is a fixed-address buffer
/// whose CONTENTS are re-uploaded each decode step (like positions/block_table).
/// No alloc/sync — captures inside the decode graph.
///
/// ARG ORDER is in lockstep with `lora_bgmv.cu` (cuLaunchKernel is type-blind;
/// the byte-identity oracle is the only guard — keep both in sync).
#[allow(clippy::too_many_arguments)]
pub fn apply_lora_bgmv(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    route: &LoraRoute,
    x: DevicePtr,        // [n, x_row_stride] BF16
    base_out: DevicePtr, // [n, out_row_stride] BF16, folded in place
    seq_slot: DevicePtr, // [n] i32 (<0 => base)
    n: u32,              // batch rows
    x_row_stride: u32,   // elements between x rows (>= route.k_in)
    out_row_stride: u32, // elements between base_out rows (>= route.n_out)
    lora_xa: DevicePtr,  // arena scratch >= n * max_rank BF16
    stream: u64,
) -> Result<()> {
    use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

    // Kernel 1: shrink — xa[n, max_rank] = x @ A_s^T.
    // grid = (ceil(max_rank/4), n, 1)  block = (256,1,1).
    KernelLaunch::new(gpu, kernels.bgmv_shrink_k)
        .grid([div_ceil(route.max_rank, 4), n, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(seq_slot)
        .arg_ptr(route.a_table)
        .arg_ptr(lora_xa)
        .arg_u32(n)
        .arg_u32(route.max_rank)
        .arg_u32(route.k_in)
        .arg_u32(x_row_stride)
        .launch(stream)?;

    // Kernel 2: expand + fold — base_out += scale_s * (xa @ B_s^T).
    // grid = (ceil(n_out/4), n, 1)  block = (256,1,1).
    KernelLaunch::new(gpu, kernels.bgmv_expand_fold_k)
        .grid([div_ceil(route.n_out, 4), n, 1])
        .block([256, 1, 1])
        .arg_ptr(lora_xa)
        .arg_ptr(seq_slot)
        .arg_ptr(route.b_table)
        .arg_ptr(route.scale_table)
        .arg_ptr(base_out)
        .arg_u32(n)
        .arg_u32(route.n_out)
        .arg_u32(route.max_rank)
        .arg_u32(out_row_stride)
        .launch(stream)
}
