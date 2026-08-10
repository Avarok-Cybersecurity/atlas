// SPDX-License-Identifier: AGPL-3.0-only

//! Token-overlay kernel launchers (Feature 2). Thin `KernelLaunch` wrappers over
//! `kernels/gb10/common/token_overlay.cu`:
//! - [`embed_rowdiff`] — build-time: which adapter base rows differ from served.
//! - [`embed_overlay_routed`] — forward: replace overridden vocab rows post-gather.
//! - [`lmhead_overlay_routed`] — forward: recompute overridden logit columns.
//!
//! Argument order is in LOCKSTEP with the `.cu` signatures (cuLaunchKernel is
//! type-blind). All device tables are load-time-fixed addresses; the only
//! per-step arg is `seq_slot` (NULL ⇒ uniform `active`) — graph-capture safe.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use crate::layers::try_kernel;

/// The four token-overlay kernels, resolved once at model construction via
/// [`try_kernel`] (null-on-miss ⇒ the feature is silently unused rather than a
/// hard init failure on a kernel image that predates the overlay).
#[derive(Clone, Copy)]
pub struct OverlayKernels {
    pub rowdiff: KernelHandle,
    pub embed_overlay: KernelHandle,
    pub lmhead_overlay_bf16: KernelHandle,
    pub lmhead_overlay_f32: KernelHandle,
}

impl Default for OverlayKernels {
    /// All-null handles: the overlay feature is silently unused (every hook's
    /// `kernel.0 == 0` guard fires). `KernelHandle` has no `Default` derive, so
    /// this is spelled out.
    fn default() -> Self {
        Self {
            rowdiff: KernelHandle(0),
            embed_overlay: KernelHandle(0),
            lmhead_overlay_bf16: KernelHandle(0),
            lmhead_overlay_f32: KernelHandle(0),
        }
    }
}

impl OverlayKernels {
    pub fn new(gpu: &dyn GpuBackend) -> Self {
        Self {
            rowdiff: try_kernel(gpu, "token_overlay", "embed_rowdiff_bf16"),
            embed_overlay: try_kernel(gpu, "token_overlay", "embed_overlay_routed_bf16"),
            lmhead_overlay_bf16: try_kernel(gpu, "token_overlay", "lmhead_overlay_routed_bf16"),
            lmhead_overlay_f32: try_kernel(gpu, "token_overlay", "lmhead_overlay_routed_f32"),
        }
    }
}

/// `flags[r] = (max_i |base[r,i] - served[r,i]| > thresh)`. Grid one thread/row.
#[allow(clippy::too_many_arguments)]
pub fn embed_rowdiff(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    base: DevicePtr,   // [rows, h] bf16
    served: DevicePtr, // [rows, h] bf16
    flags: DevicePtr,  // [rows] u8 out
    rows: u32,
    h: u32,
    thresh: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(base)
        .arg_ptr(served)
        .arg_ptr(flags)
        .arg_u32(rows)
        .arg_u32(h)
        .arg_f32(thresh)
        .launch(stream)
}

/// In-place row-replace of overridden vocab rows on the residual stream after
/// the embed gather (BEFORE `scale_embeddings`). Per row `r`: `s = seq_slot[r]`
/// (or `active` when NULL); `s<0` skip; `ids[r]>=vocab` skip (no overlay entry
/// for a token beyond the overlay's served-vocab snapshot — CWE-125 guard);
/// `slot=slot_map_tab[s][ids[r]]`; `slot<0` or `slot>=n_tab[s]` skip; copy
/// `rows_tab[s][slot]` over `out[r]`.
#[allow(clippy::too_many_arguments)]
pub fn embed_overlay_routed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ids: DevicePtr,      // [n] u32 token id per row
    seq_slot: DevicePtr, // [n] i32 or NULL(0)
    active: i32,
    slot_map_tab: DevicePtr, // u64[L]
    rows_tab: DevicePtr,     // u64[L]
    n_tab: DevicePtr,        // u32[L] embed n_override per slot
    out: DevicePtr,          // [n, h] bf16 in place
    num_tokens: u32,
    h: u32,
    vocab: u32, // slot_map length (served vocab at overlay build)
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ids)
        .arg_ptr(seq_slot)
        .arg_i32(active)
        .arg_ptr(slot_map_tab)
        .arg_ptr(rows_tab)
        .arg_ptr(n_tab)
        .arg_ptr(out)
        .arg_u32(h)
        .arg_u32(vocab)
        .launch(stream)
}

/// In-place recompute of overridden logit columns (BEFORE softcap). One warp per
/// `(row, j)`; `j` indexes the overridden-id slot of that row's adapter. Picks
/// the bf16 or f32 logits kernel per `is_fp32`.
#[allow(clippy::too_many_arguments)]
pub fn lmhead_overlay_routed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle, // bf16 or f32 variant, selected by caller
    hidden: DevicePtr,    // [m, h] bf16
    seq_slot: DevicePtr,  // [m] i32 or NULL(0)
    active: i32,
    rows_tab: DevicePtr, // u64[L]
    ids_tab: DevicePtr,  // u64[L]
    n_tab: DevicePtr,    // u32[L]
    logits: DevicePtr,   // [m, vocab] bf16 or f32 in place
    m: u32,
    max_n_override: u32,
    h: u32,
    vocab: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, max_n_override, 1])
        .block([32, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(seq_slot)
        .arg_i32(active)
        .arg_ptr(rows_tab)
        .arg_ptr(ids_tab)
        .arg_ptr(n_tab)
        .arg_ptr(logits)
        .arg_u32(h)
        .arg_u32(vocab)
        .launch(stream)
}
