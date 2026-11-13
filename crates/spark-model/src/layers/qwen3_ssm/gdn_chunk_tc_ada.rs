// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db),
// itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer),
// Apache License 2.0. Source: src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu:28-44,
// state_passing.cu:30-50, output.cu:9-55 (launch geometry: grids, block sizes, the output stage's
// one-wave job split). Changes: translated from C++ to Rust over Atlas's `KernelLaunch`; the job
// split is sized from the device's SM count instead of the RTX 5090's 170; fixed sm_89 geometries;
// Atlas's FLA scratch carving; a batch dimension and pointer-table state, by Atlas contributors.

//! Responsibility: the host side of the Ada tensor-core chunked GDN prefill
//! (`kernels/ada/common/gdn_chunk_tc_ada.cu`, three launches: prepare, state
//! passing, output): lazy kernel resolution, the eligibility verdict that names
//! the guard which rejected, the scratch carving and the launch sequence.
//!
//! SELECTION. The arm runs only when `[defaults] gdn_prefill_tc` resolves on
//! (`AVAROK_GDN_PREFILL_TC` overriding) AND the module `gdn_chunk_tc_ada` is in
//! the loaded image. Only `kernels/ada` compiles the module, so the row alone
//! never selects it: `kernels/hopper` declares the row true for its own spine,
//! and there the module is absent and this file does nothing, logs nothing and
//! looks nothing up. On Ada the row is false until a GPU receipt flips it, so
//! the default serve is byte-identical to the scalar FLA spine.
//!
//! NUMERICS. Not bit-identical to the FLA route: every per-chunk product runs
//! on mma.sync (BF16 for K K^T, Q K^T and Q S; TF32 with FP32 accumulation for
//! the triangular solve products and the state update). The state stays FP32
//! in MMA accumulators; W, U, v_new and S_c are rounded to BF16 as the FLA route
//! rounds them. The bound it is held to is `gdn_chunk_tc_ada::oracle::{OUTPUT,
//! STATE}` against an FP64 recurrence (`examples/gdn_chunk_tc_ada_microtest.rs`).

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;

#[cfg(test)]
pub(crate) mod oracle;

#[cfg(test)]
mod emulate;

#[cfg(test)]
mod maps_tests;

#[cfg(test)]
#[path = "gdn_chunk_tc_ada_tests.rs"]
mod tests;

/// The module `kernels/ada/common/gdn_chunk_tc_ada.cu` compiles to.
pub(crate) const MODULE: &str = "gdn_chunk_tc_ada";
pub(crate) const ENTRY_PREPARE: &str = "gdn_chunk_tc_ada_prepare";
pub(crate) const ENTRY_STATE: &str = "gdn_chunk_tc_ada_state";
pub(crate) const ENTRY_OUTPUT: &str = "gdn_chunk_tc_ada_output";

/// Tokens per chunk and the head width the kernels are compiled for.
pub(crate) const CHUNK: u32 = 64;
pub(crate) const DIM: u32 = 128;
/// Stage 1: 4 warps per (chunk, value head).
pub(crate) const PREPARE_THREADS: u32 = 128;
/// Stage 2: 8 warps per (value head, 16-column value strip).
pub(crate) const STATE_THREADS: u32 = 256;
pub(crate) const STATE_STRIP: u32 = 16;
pub(crate) const STATE_STRIPS: u32 = DIM / STATE_STRIP;
/// Stage 3: 4 warps, 4 resident CTAs per SM.
pub(crate) const OUTPUT_THREADS: u32 = 128;
pub(crate) const OUTPUT_CTAS_PER_SM: u32 = 4;

/// Dynamic shared memory per stage, mirrored from the kernel's `static_assert`s
/// (each entry also exports `<entry>_dyn_smem_bytes`, so the registry raises a
/// launch that asks for less):
/// prepare: T_inv 64x64 f32 + max(Schur scratch 4x16x20, WU panel 64x16) f32
/// + 64x16 bf16 output stage + g, beta, beta*exp(g).
pub(crate) const PREPARE_SMEM: u32 = (64 * 64 + 4 * 16 * 20 + 64 * 16 / 2 + 3 * 64) * 4;
/// state: W 64x128 bf16 + U/v_decay 64x16 f32 + K 64x128 bf16 + 4 state
/// snapshots 16x32 f32 + g.
pub(crate) const STATE_SMEM: u32 = (64 * 128 / 2 + 64 * 16 + 64 * 128 / 2 + 4 * 32 * 16 + 64) * 4;
/// output: Q 64x128 bf16 + two 2048-element bf16 stages.
pub(crate) const OUTPUT_SMEM: u32 = (64 * 128 + 2 * 2048) * 2;

/// Fallback machine width when the driver cannot be asked: the reference part
/// (`kernels/ada/HARDWARE.toml` `sm_count`). Only the output grid reads it.
const REFERENCE_SM_COUNT: u32 = 34;

/// The three resolved entry points plus the device width the output grid is
/// sized from (queried once, at resolution).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Kernels {
    pub prepare: KernelHandle,
    pub state: KernelHandle,
    pub output: KernelHandle,
    pub sm_count: u32,
}

/// Lazily resolved handles, one per layer (a model swap builds new layers, so
/// a handle never outlives the image it came from). Default is empty; the
/// field lives on `Qwen3SsmLayer` as `gdn_tc_ada`.
#[derive(Default)]
pub(crate) struct Handles {
    cell: OnceLock<Option<Kernels>>,
}

impl Handles {
    /// The kernels, or `None` when the module is not in this image (every
    /// target but Ada: no lookup is issued) or an entry failed to resolve.
    pub(crate) fn resolve(&self, gpu: &dyn GpuBackend) -> Option<Kernels> {
        *self.cell.get_or_init(|| resolve_kernels(gpu))
    }
}

fn resolve_kernels(gpu: &dyn GpuBackend) -> Option<Kernels> {
    if !gpu.has_module(MODULE) {
        return None;
    }
    let probe = |entry: &str| crate::layers::try_target_kernel(gpu, MODULE, entry);
    let (prepare, state, output) = (
        probe(ENTRY_PREPARE),
        probe(ENTRY_STATE),
        probe(ENTRY_OUTPUT),
    );
    if prepare.0 == 0 || state.0 == 0 || output.0 == 0 {
        tracing::warn!(
            "{MODULE} is in the image but an entry did not resolve (prepare={} state={} \
             output={}); the GDN prefill keeps the FLA spine",
            prepare.0 != 0,
            state.0 != 0,
            output.0 != 0
        );
        return None;
    }
    let sm_count = gpu.sm_count().unwrap_or(REFERENCE_SM_COUNT).max(1);
    Some(Kernels {
        prepare,
        state,
        output,
        sm_count,
    })
}

/// The geometry of one call, in elements (strides) and heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TcAdaShape {
    pub batch: u32,
    pub seq_len: u32,
    pub nk: u32,
    pub nv: u32,
    pub kd: u32,
    pub vd: u32,
    pub qk_stride: u32,
    pub v_stride: u32,
    pub gb_stride: u32,
}

impl TcAdaShape {
    /// Atlas's packed prefill layout: q/k/v inside one `conv_dim`-wide row per
    /// token, gate and beta interleaved in one `2 * nv` FP32 row.
    pub(crate) fn packed(
        batch: u32,
        seq_len: u32,
        nk: usize,
        nv: usize,
        kd: usize,
        vd: usize,
        conv_dim: usize,
    ) -> Self {
        Self {
            batch,
            seq_len,
            nk: nk as u32,
            nv: nv as u32,
            kd: kd as u32,
            vd: vd as u32,
            qk_stride: conv_dim as u32,
            v_stride: conv_dim as u32,
            gb_stride: (nv * 2) as u32,
        }
    }

    pub(crate) fn num_chunks(&self) -> u32 {
        self.seq_len.div_ceil(CHUNK)
    }
}

/// One call's operands. `h_state` is `[batch][nv][128][128]` FP32, or with
/// `h_state_is_table` a device array of `batch` per-request state pointers;
/// it is read and written in place. `out` is `[batch * seq_len][nv][128]` BF16.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TcAdaArgs {
    pub h_state: DevicePtr,
    pub h_state_is_table: bool,
    pub q: DevicePtr,
    pub k: DevicePtr,
    pub v: DevicePtr,
    pub gate: DevicePtr,
    pub beta: DevicePtr,
    pub out: DevicePtr,
    pub shape: TcAdaShape,
    /// A mid-chunk SSM tail capture is active for this pass: the arm declines
    /// and the FLA ladder handles the pass exactly as it does today.
    pub midchunk_capture: bool,
}

/// Why the arm did not run although it was requested and the module is
/// present. NAME THE GUARD THAT REJECTED: a requested path that silently falls
/// back measures as "no effect" (the PR #296 lesson in `ssm_gdn_a3.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Reject {
    Empty,
    ExactReplay,
    MidchunkCapture,
    HeadDims,
    HeadRatio,
    Strides,
    Alignment,
    NoScratch,
    ScratchTooSmall,
    GridLimit,
}

impl Reject {
    pub(crate) fn why(self) -> &'static str {
        match self {
            Self::Empty => "zero tokens",
            Self::ExactReplay => {
                "warm exact replay keeps the token-sequential recurrence (chunk grid anchored at the snapshot)"
            }
            Self::MidchunkCapture => "a mid-chunk SSM tail capture is active for this pass",
            Self::HeadDims => "head dims differ from the compiled tile (k_dim = v_dim = 128)",
            Self::HeadRatio => "value heads are not a multiple of key heads",
            Self::Strides => "a token stride is not a multiple of 8 (16-byte cp.async rows)",
            Self::Alignment => "q/k/v are not 16-byte aligned",
            Self::NoScratch => "no GDN FLA scratch is allocated for this model",
            Self::ScratchTooSmall => "the GDN FLA scratch is smaller than this call needs",
            Self::GridLimit => "batch or chunk count exceeds a grid dimension",
        }
    }

    /// The once-latch key (`ModelStats::once`), one per reason.
    pub(crate) fn key(self) -> &'static str {
        match self {
            Self::Empty => "log:gdn_tc_ada_reject_empty",
            Self::ExactReplay => "log:gdn_tc_ada_reject_exact_replay",
            Self::MidchunkCapture => "log:gdn_tc_ada_reject_midchunk",
            Self::HeadDims => "log:gdn_tc_ada_reject_head_dims",
            Self::HeadRatio => "log:gdn_tc_ada_reject_head_ratio",
            Self::Strides => "log:gdn_tc_ada_reject_strides",
            Self::Alignment => "log:gdn_tc_ada_reject_alignment",
            Self::NoScratch => "log:gdn_tc_ada_reject_no_scratch",
            Self::ScratchTooSmall => "log:gdn_tc_ada_reject_scratch_small",
            Self::GridLimit => "log:gdn_tc_ada_reject_grid",
        }
    }

    /// Routing by design (logged at info) rather than a misconfiguration (warn).
    pub(crate) fn expected(self) -> bool {
        matches!(
            self,
            Self::Empty | Self::ExactReplay | Self::MidchunkCapture
        )
    }
}

/// The verdict, pure so the grammar is testable without a GPU.
pub(crate) fn reject(a: &TcAdaArgs, scratch_bytes: usize, exact_replay: bool) -> Option<Reject> {
    let s = &a.shape;
    let aligned = |p: DevicePtr| p.0.is_multiple_of(16);
    if s.seq_len == 0 || s.batch == 0 {
        Some(Reject::Empty)
    } else if exact_replay {
        Some(Reject::ExactReplay)
    } else if a.midchunk_capture {
        Some(Reject::MidchunkCapture)
    } else if s.kd != DIM || s.vd != DIM {
        Some(Reject::HeadDims)
    } else if s.nk == 0 || s.nv < s.nk || !s.nv.is_multiple_of(s.nk) {
        Some(Reject::HeadRatio)
    } else if !s.qk_stride.is_multiple_of(8) || !s.v_stride.is_multiple_of(8) {
        Some(Reject::Strides)
    } else if !aligned(a.q) || !aligned(a.k) || !aligned(a.v) {
        Some(Reject::Alignment)
    } else if scratch_bytes == 0 {
        Some(Reject::NoScratch)
    } else if ScratchLayout::bytes(s.batch, s.num_chunks(), s.nv) > scratch_bytes {
        Some(Reject::ScratchTooSmall)
    } else if s.batch > 65_535 || s.nv > 65_535 {
        Some(Reject::GridLimit)
    } else {
        None
    }
}

/// The FLA scratch carved in the FLA order (W, U, S, uc, gc), so the two routes
/// use the same bytes for the same call: every region holds
/// `batch * num_chunks * nv` blocks, blocked by `(b * num_chunks + c) * nv + h_v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScratchLayout {
    pub w: DevicePtr,
    pub u: DevicePtr,
    pub h_chunk: DevicePtr,
    pub v_new: DevicePtr,
    pub gc: DevicePtr,
}

impl ScratchLayout {
    fn blocks(batch: u32, num_chunks: u32, nv: u32) -> usize {
        batch as usize * num_chunks as usize * nv as usize
    }

    pub(crate) fn carve(base: DevicePtr, batch: u32, num_chunks: u32, nv: u32) -> Self {
        let n = Self::blocks(batch, num_chunks, nv);
        let (c, d) = (CHUNK as usize, DIM as usize);
        let w = base;
        let u = w.offset(n * c * d * 2);
        let h_chunk = u.offset(n * c * d * 2);
        let v_new = h_chunk.offset(n * d * d * 2);
        let gc = v_new.offset(n * c * d * 2);
        Self {
            w,
            u,
            h_chunk,
            v_new,
            gc,
        }
    }

    /// Bytes `carve` spans: the FLA sizing formula (`buffers/sizes.rs`).
    pub(crate) fn bytes(batch: u32, num_chunks: u32, nv: u32) -> usize {
        let n = Self::blocks(batch, num_chunks, nv);
        let (c, d) = (CHUNK as usize, DIM as usize);
        n * c * d * 2 + n * c * d * 2 + n * d * d * 2 + n * c * d * 2 + n * c * 4
    }
}

/// Stage 3's grid x: at most one resident wave of `OUTPUT_CTAS_PER_SM` CTAs per
/// SM, chunks spread evenly over it (upstream `output.cu`, retuned from 170
/// SMs to the device's count); a small call keeps one chunk per CTA.
pub(crate) fn output_grid_chunks(num_chunks: u32, nv: u32, batch: u32, sm_count: u32) -> u32 {
    let jobs = num_chunks as u64 * nv as u64 * batch as u64;
    let target = sm_count.max(1) as u64 * OUTPUT_CTAS_PER_SM as u64;
    let per_block = jobs.div_ceil(target).max(1);
    (num_chunks as u64).div_ceil(per_block).max(1) as u32
}

/// The three launches, in stream order. The caller has checked [`reject`].
pub(crate) fn launch(
    gpu: &dyn GpuBackend,
    kern: &Kernels,
    a: &TcAdaArgs,
    scratch: DevicePtr,
    stream: u64,
) -> Result<()> {
    let s = &a.shape;
    let nc = s.num_chunks();
    let l = ScratchLayout::carve(scratch, s.batch, nc, s.nv);
    KernelLaunch::new(gpu, kern.prepare)
        .grid([nc, s.nv, s.batch])
        .block([PREPARE_THREADS, 1, 1])
        .shared_mem(PREPARE_SMEM)
        .arg_ptr(a.k)
        .arg_ptr(a.v)
        .arg_ptr(a.gate)
        .arg_ptr(a.beta)
        .arg_ptr(l.w)
        .arg_ptr(l.u)
        .arg_ptr(l.gc)
        .arg_u32(s.seq_len)
        .arg_u32(nc)
        .arg_u32(s.nk)
        .arg_u32(s.nv)
        .arg_u32(s.qk_stride)
        .arg_u32(s.v_stride)
        .arg_u32(s.gb_stride)
        .launch(stream)?;
    KernelLaunch::new(gpu, kern.state)
        .grid([s.nv * STATE_STRIPS, s.batch, 1])
        .block([STATE_THREADS, 1, 1])
        .shared_mem(STATE_SMEM)
        .arg_ptr(a.h_state)
        .arg_ptr(l.w)
        .arg_ptr(l.u)
        .arg_ptr(a.k)
        .arg_ptr(l.gc)
        .arg_ptr(l.v_new)
        .arg_ptr(l.h_chunk)
        .arg_u32(s.seq_len)
        .arg_u32(nc)
        .arg_u32(s.nk)
        .arg_u32(s.nv)
        .arg_u32(s.qk_stride)
        .arg_u32(a.h_state_is_table as u32)
        .launch(stream)?;
    KernelLaunch::new(gpu, kern.output)
        .grid([
            output_grid_chunks(nc, s.nv, s.batch, kern.sm_count),
            s.nv,
            s.batch,
        ])
        .block([OUTPUT_THREADS, 1, 1])
        .shared_mem(OUTPUT_SMEM)
        .arg_ptr(a.q)
        .arg_ptr(a.k)
        .arg_ptr(l.v_new)
        .arg_ptr(l.gc)
        .arg_ptr(l.h_chunk)
        .arg_ptr(a.out)
        .arg_u32(s.seq_len)
        .arg_u32(nc)
        .arg_u32(s.nk)
        .arg_u32(s.nv)
        .arg_u32(s.qk_stride)
        .launch(stream)
}

impl Qwen3SsmLayer {
    /// Run the Ada tensor-core chunked GDN prefill for this call if it is
    /// selected and eligible. `Ok(false)` means "not taken": the caller's
    /// existing ladder runs, unchanged. `Ok(true)` means the output and the
    /// state are written.
    pub(super) fn prefill_gdn_tc_ada(
        &self,
        a: &TcAdaArgs,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !crate::layers::ops::target_defaults::resolved()
            .gdn_prefill_tc
            .value
        {
            return Ok(false);
        }
        // Absent module (every target but Ada): no lookup, no log.
        let Some(kern) = self.gdn_tc_ada.resolve(ctx.gpu) else {
            return Ok(false);
        };
        let scratch = ctx.buffers.gdn_fla_scratch();
        let scratch_bytes = if scratch.is_null() {
            0
        } else {
            ctx.buffers.sizes().gdn_fla_scratch
        };
        if let Some(why) = reject(a, scratch_bytes, ctx.gdn_exact_replay) {
            if ctx.stats.once(why.key()) {
                if why.expected() {
                    tracing::info!("GDN prefill: {MODULE} not taken: {}", why.why());
                } else {
                    tracing::warn!(
                        "AVAROK_GDN_PREFILL_TC / [defaults] gdn_prefill_tc is on but the Ada \
                         tensor-core chunked prefill is NOT running: {}",
                        why.why()
                    );
                }
            }
            return Ok(false);
        }
        if ctx.stats.once("log:gdn_tc_ada_active") {
            tracing::info!(
                "GDN prefill: Ada tensor-core chunked path ACTIVE ({ENTRY_PREPARE} -> \
                 {ENTRY_STATE} -> {ENTRY_OUTPUT}, output grid sized for {} SMs)",
                kern.sm_count
            );
        }
        launch(ctx.gpu, &kern, a, scratch, stream)?;
        Ok(true)
    }
}
