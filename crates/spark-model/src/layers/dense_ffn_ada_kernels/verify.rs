// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the dense FFN's FP8 speculative-verify forward over M = 2..8
//! rows with one weight pass per projection (`kernels/ada/common/fp8_gemv_ada_batchm.cu`),
//! behind the `ffn_fp8_verify_gemv` row (off by default on every target).
//!
//! # What it replaces, and the claim
//!
//! An FP8 layer's `forward_k2` / `forward_k3` / `forward_km` redirect to
//! `forward_prefill` (`native_small_batch_uses_prefill`): `w8a16_gemv_batch4`
//! at M <= 4, and the padded tile GEMMs or the W8A8 arm at M 5..8. The batch-M
//! kernels read each weight byte once for all M rows, like batch4, but with the
//! sm_89 register dequant and at every M up to 8.
//!
//! [`forward_km_fp8`] reproduces, row for row and BIT FOR BIT, what the M=1
//! decode forward (`DenseFfnLayer::forward`) computes for each row: it picks
//! the same down arm `forward` would pick from the same layer handles and
//! levers ([`super::super::fp8_down::fp8_down_arm`]), and launches the batch-M
//! twin of each kernel in that arm. The twins are byte-identical to M launches
//! of their M=1 kernels (the `.cu` header gives the argument; the GPU oracle is
//! `examples/fp8_ada_batchm_microtest.rs`; the host model of the loop order is
//! pinned in `verify_tests.rs`). So with the row on, a verify row's FFN output
//! equals the decode output for the same token, at every M.
//!
//! At M <= 4 that is also what the prefill ladder computes today
//! (`w8a16_gemv_batch4` is byte-identical to `w8a16_gemv` per row, and
//! `fp8_gemv_ada` to `w8a16_gemv`). At M 5..8 the prefill ladder runs a
//! different arithmetic (MMA tiles reassociate; W8A8 quantizes the activation),
//! so the row moves those verify logits TOWARD the decode values. The receipt
//! for the row is therefore byte parity on the microtest, greedy SHA equality
//! at `--num-drafts 3`, and verify time.
//!
//! # Handles
//!
//! `DenseFfnLayer` has no field for these kernels, so they are looked up at
//! layer construction through the backend's op cache
//! ([`super::Fp8FfnKernels::resolve`] calls [`prime`]) and read back here, a
//! cache hit, per call. Nothing is looked up with the row off.
//!
//! # API (master plan 3.7.3)
//!
//! `forward_km_fp8(layer, x, m, out, ctx, stream) -> Result<bool>`: `x` is
//! `[m, hidden]` BF16, `out` is `[m, hidden]` BF16 (callers pass
//! `ctx.buffers.moe_output()`, as `forward_km` writes it), and the FFN's
//! `expert_gate_out` / `expert_up_out` scratch is used for the intermediate.
//! `Ok(false)` = not eligible and nothing was launched; the caller keeps its
//! current path.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layer::ForwardContext;
use crate::layers::dense_ffn::fp8_down::{Fp8DownArm, fp8_down_arm};
use crate::layers::dense_ffn::{DenseFfnLayer, FfnActivation};
use crate::layers::ops;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// Fewest verify rows the batch-M kernels serve (M=1 is the decode GEMV).
pub const MIN_ROWS: usize = 2;
/// Most verify rows the batch-M kernels serve.
pub const MAX_ROWS: usize = 8;
/// Module (file stem) of the batch-M kernels.
pub const BATCHM_MODULE: &str = "fp8_gemv_ada_batchm";

const PLAIN: [&str; 7] = [
    "fp8_gemv_ada_batchm_2",
    "fp8_gemv_ada_batchm_3",
    "fp8_gemv_ada_batchm_4",
    "fp8_gemv_ada_batchm_5",
    "fp8_gemv_ada_batchm_6",
    "fp8_gemv_ada_batchm_7",
    "fp8_gemv_ada_batchm_8",
];
const DUAL: [&str; 7] = [
    "fp8_gemv_ada_batchm_dual_2",
    "fp8_gemv_ada_batchm_dual_3",
    "fp8_gemv_ada_batchm_dual_4",
    "fp8_gemv_ada_batchm_dual_5",
    "fp8_gemv_ada_batchm_dual_6",
    "fp8_gemv_ada_batchm_dual_7",
    "fp8_gemv_ada_batchm_dual_8",
];
const SILU: [&str; 7] = [
    "fp8_gemv_ada_batchm_silu_input_2",
    "fp8_gemv_ada_batchm_silu_input_3",
    "fp8_gemv_ada_batchm_silu_input_4",
    "fp8_gemv_ada_batchm_silu_input_5",
    "fp8_gemv_ada_batchm_silu_input_6",
    "fp8_gemv_ada_batchm_silu_input_7",
    "fp8_gemv_ada_batchm_silu_input_8",
];
const PLAIN_L2EF: [&str; 7] = [
    "fp8_gemv_ada_batchm_l2ef_2",
    "fp8_gemv_ada_batchm_l2ef_3",
    "fp8_gemv_ada_batchm_l2ef_4",
    "fp8_gemv_ada_batchm_l2ef_5",
    "fp8_gemv_ada_batchm_l2ef_6",
    "fp8_gemv_ada_batchm_l2ef_7",
    "fp8_gemv_ada_batchm_l2ef_8",
];
const DUAL_L2EF: [&str; 7] = [
    "fp8_gemv_ada_batchm_l2ef_dual_2",
    "fp8_gemv_ada_batchm_l2ef_dual_3",
    "fp8_gemv_ada_batchm_l2ef_dual_4",
    "fp8_gemv_ada_batchm_l2ef_dual_5",
    "fp8_gemv_ada_batchm_l2ef_dual_6",
    "fp8_gemv_ada_batchm_l2ef_dual_7",
    "fp8_gemv_ada_batchm_l2ef_dual_8",
];
const SILU_L2EF: [&str; 7] = [
    "fp8_gemv_ada_batchm_l2ef_silu_input_2",
    "fp8_gemv_ada_batchm_l2ef_silu_input_3",
    "fp8_gemv_ada_batchm_l2ef_silu_input_4",
    "fp8_gemv_ada_batchm_l2ef_silu_input_5",
    "fp8_gemv_ada_batchm_l2ef_silu_input_6",
    "fp8_gemv_ada_batchm_l2ef_silu_input_7",
    "fp8_gemv_ada_batchm_l2ef_silu_input_8",
];

/// The three batch-M kernels for one row count.
#[derive(Clone, Copy, Debug)]
pub struct RowKernels {
    /// `[m, k] x [n, k]^T`, the twin of `fp8_gemv_ada`.
    pub plain: KernelHandle,
    /// gate + up of the same rows, the twin of `fp8_gemv_ada_dual`.
    pub dual: KernelHandle,
    /// down with the SwiGLU computed inline, the twin of `fp8_gemv_ada_silu_input`.
    pub silu_input: KernelHandle,
}

impl RowKernels {
    /// `[plain, dual, silu_input]` entry names for `m` rows, the evict-first
    /// twins when `l2ef`; `None` outside `MIN_ROWS..=MAX_ROWS`.
    pub fn names(m: usize, l2ef: bool) -> Option<[&'static str; 3]> {
        if !(MIN_ROWS..=MAX_ROWS).contains(&m) {
            return None;
        }
        let i = m - MIN_ROWS;
        Some(if l2ef {
            [PLAIN_L2EF[i], DUAL_L2EF[i], SILU_L2EF[i]]
        } else {
            [PLAIN[i], DUAL[i], SILU[i]]
        })
    }

    /// The handles for `m` rows on this backend, through its op cache: a map
    /// read once [`prime`] ran. `None` when `m` is out of range or the target
    /// did not compile the module (no lookup is issued then).
    pub fn cached(gpu: &dyn GpuBackend, m: usize, l2ef: bool) -> Option<Self> {
        let [plain, dual, silu] = Self::names(m, l2ef)?;
        if !gpu.has_module(BATCHM_MODULE) {
            return None;
        }
        let get = |entry| super::cached_target_kernel(gpu, (BATCHM_MODULE, entry));
        Some(Self {
            plain: get(plain),
            dual: get(dual),
            silu_input: get(silu),
        })
    }
}

/// Construction-time lookup of every row count's handles (the names the
/// forward will read), so the forward never issues a first lookup.
pub(crate) fn prime(gpu: &dyn GpuBackend, l2ef: bool) {
    for m in MIN_ROWS..=MAX_ROWS {
        let _ = RowKernels::cached(gpu, m, l2ef);
    }
}

/// The launches one verify forward makes, mirroring `forward`'s FP8 arms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyRoute {
    /// dual, `silu_mul` (the layer's `act_mul`), plain down. `forward`'s default.
    SplitSilu,
    /// dual, then down with the SwiGLU inline (`AVAROK_NO_DECODE_SPLIT_SILU`).
    FusedSilu,
    /// gate, up, the activation multiply, down: one plain GEMV each.
    PerProjection,
}

/// Which route a verify forward over `m` rows takes, or `None` (decline).
///
/// The arm is `forward`'s own choice for this layer (same handles, same
/// lever), so each row's arithmetic is the decode arithmetic; the batch-M
/// twin of every kernel in that arm must be present, else the call declines
/// rather than mix kernels.
pub(crate) fn plan(
    layer: &DenseFfnLayer,
    k: &RowKernels,
    m: usize,
    hidden: usize,
    inter: usize,
    split_silu_lever: bool,
) -> Option<VerifyRoute> {
    let fp8 = layer.fp8_weights.as_ref()?;
    let shapes_ok = (MIN_ROWS..=MAX_ROWS).contains(&m)
        && hidden.is_multiple_of(16)
        && inter.is_multiple_of(16)
        && weight_ok(&fp8.gate_proj, inter, hidden)
        && weight_ok(&fp8.up_proj, inter, hidden)
        && weight_ok(&fp8.down_proj, hidden, inter);
    // A packed-Q2 layer is served by its own arm, and an adapter is never
    // installed on an FP8 layer (`set_lora_weights` refuses), but a decline is
    // the safe answer if either ever holds.
    if !shapes_ok || layer.q2_weights.is_some() || layer.lora.is_some() {
        return None;
    }
    let arm = fp8_down_arm(
        layer.activation == FfnActivation::SiLU,
        layer.w8a16_gemv_dual_k.0 != 0,
        layer.w8a16_gemv_silu_input_k.0 != 0,
        layer.act_mul.0 != 0,
        layer.w8a16_gemv_k.0 != 0,
        split_silu_lever,
    );
    let (route, needed) = match arm {
        Fp8DownArm::SplitSilu => (VerifyRoute::SplitSilu, [k.dual, k.plain]),
        Fp8DownArm::FusedSilu => (VerifyRoute::FusedSilu, [k.dual, k.silu_input]),
        // `forward`'s per-projection arm launches `w8a16_gemv_k` three times
        // and `act_mul` once; with either handle absent it has nothing to
        // mirror.
        Fp8DownArm::PerProjection if layer.w8a16_gemv_k.0 != 0 && layer.act_mul.0 != 0 => {
            (VerifyRoute::PerProjection, [k.plain, k.plain])
        }
        Fp8DownArm::PerProjection => return None,
    };
    needed.iter().all(|h| h.0 != 0).then_some(route)
}

/// A block-scaled `[n, k]` weight at the shape the FFN reads.
fn weight_ok(w: &Fp8Weight, n: usize, k: usize) -> bool {
    w.scale_format == WeightQuantFormat::Fp8BlockScaled && w.n as usize == n && w.k as usize == k
}

/// The 3.7.3 entry point: the verify FFN over `m` rows into `out`, or
/// `Ok(false)` (row off, kernels absent, or the layer not eligible).
pub fn forward_km_fp8(
    layer: &DenseFfnLayer,
    x: DevicePtr,
    m: usize,
    out: DevicePtr,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<bool> {
    if !ops::port_rows::ffn_fp8_verify_gemv::row() {
        return Ok(false);
    }
    let l2ef = ops::port_rows::l2_evict_first_weights::row();
    let Some(k) = RowKernels::cached(ctx.gpu, m, l2ef) else {
        return Ok(false);
    };
    forward_km_fp8_with(layer, &k, x, m, out, ctx, stream)
}

/// [`forward_km_fp8`] over explicit handles (the row already decided).
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_km_fp8_with(
    layer: &DenseFfnLayer,
    k: &RowKernels,
    x: DevicePtr,
    m: usize,
    out: DevicePtr,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<bool> {
    let hidden = ctx.config.hidden_size;
    let inter = ctx.config.intermediate_size;
    let Some(route) = plan(layer, k, m, hidden, inter, ctx.levers.decode_split_silu) else {
        return Ok(false);
    };
    let Some(fp8w) = layer.fp8_weights.as_ref() else {
        return Ok(false);
    };
    if ctx.stats.once("log:ffn_fp8_verify_gemv") {
        tracing::info!(
            "[avarok] dense FFN verify: FP8 batch-M GEMV ({BATCHM_MODULE}, M 2..8, {route:?}); \
             each row byte-identical to the M=1 decode FFN. AVAROK_FFN_FP8_VERIFY_GEMV=0 restores \
             the prefill ladder."
        );
    }
    let (h, i, rows) = (hidden as u32, inter as u32, m as u32);
    let gpu = ctx.gpu;
    let gate_out = ctx.buffers.expert_gate_out();
    let up_out = ctx.buffers.expert_up_out();
    let (g, u, d) = (&fp8w.gate_proj, &fp8w.up_proj, &fp8w.down_proj);
    match route {
        VerifyRoute::SplitSilu => {
            gemv_batchm_dual(gpu, k.dual, x, g, gate_out, u, up_out, i, h, stream)?;
            ops::silu_mul(
                gpu,
                layer.act_mul,
                gate_out,
                up_out,
                gate_out,
                rows * i,
                stream,
            )?;
            gemv_batchm(gpu, k.plain, gate_out, d, out, h, i, stream)?;
        }
        VerifyRoute::FusedSilu => {
            gemv_batchm_dual(gpu, k.dual, x, g, gate_out, u, up_out, i, h, stream)?;
            gemv_batchm_silu_input(gpu, k.silu_input, gate_out, up_out, d, out, h, i, stream)?;
        }
        VerifyRoute::PerProjection => {
            gemv_batchm(gpu, k.plain, x, g, gate_out, i, h, stream)?;
            gemv_batchm(gpu, k.plain, x, u, up_out, i, h, stream)?;
            ops::silu_mul(
                gpu,
                layer.act_mul,
                gate_out,
                up_out,
                gate_out,
                rows * i,
                stream,
            )?;
            gemv_batchm(gpu, k.plain, gate_out, d, out, h, i, stream)?;
        }
    }
    Ok(true)
}

/// `out[m, n] = a[m, k] x E4M3(w[n, k])^T * scales`, M fixed by the handle.
///
/// Kernel: `fp8_gemv_ada_batchm_<M>(A, B, block_scale, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w: &Fp8Weight,
    out: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w.weight)
        .arg_ptr(w.row_scale)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// Gate and up of the same `[m, k]` rows in one launch.
///
/// Kernel: `fp8_gemv_ada_batchm_dual_<M>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K)`
/// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gemv_batchm_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    w1: &Fp8Weight,
    out1: DevicePtr,
    w2: &Fp8Weight,
    out2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(w1.weight)
        .arg_ptr(w1.row_scale)
        .arg_ptr(out1)
        .arg_ptr(w2.weight)
        .arg_ptr(w2.row_scale)
        .arg_ptr(out2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// `down(silu(gate) * up)` over `[m, k]` gate and up rows.
///
/// Kernel: `fp8_gemv_ada_batchm_silu_input_<M>(gate, up, B, block_scale, C, N, K)`
/// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gemv_batchm_silu_input(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    w: &Fp8Weight,
    out: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(w.weight)
        .arg_ptr(w.row_scale)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
#[path = "verify_tests.rs"]
mod tests;
