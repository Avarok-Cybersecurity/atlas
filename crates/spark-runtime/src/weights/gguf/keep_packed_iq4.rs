// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the GGUF loader's keep-packed 4-bit route (port lane L19, row `gguf_native_iq4`): the
//! switch, the per-tensor decision, the packed upload, the pre-flight size, and the GPU expansion of the IQ4
//! and Q5_K tensors the route does not keep.
//!
//! WHAT IT DOES. With the row on, a dense Qwen3.5-family GGUF (`qwen35`) keeps its 4-bit and 6-bit
//! projections as the file's raw super-blocks in VRAM instead of expanding them to BF16:
//!   * `IQ4_XS` (ggml id 23) -> [`WeightDtype::Iq4Xs`], `Q4_K` (12) -> [`WeightDtype::Q4K`], `Q6_K` (14)
//!     -> [`WeightDtype::Q6K`];
//!   * for the transform-free projections (the dense FFN and the full-attention q/k/v/o,
//!     [`super::names::is_keep_packed_proj`]) and the two big GDN input projections, whose value-head ROW
//!     reorder moves whole rows and is applied to the packed bytes ([`super::value_transform::reorder_packed_rows`],
//!     one row = K/256 whole super-blocks, so no block is split);
//!   * and the output head when the model side has said it can serve a packed head ([`set_keep_head`]).
//!
//! The consumers are spark-model's `port::gguf_q4` `W4Linear` (decode GEMV at 1..=8 rows and MMQ prefill on
//! the raw blocks, `kernels/ada/common/`) installed by the loader routes of the W4 lanes. Everything else
//! (norms, the GDN `out_proj` column reorder, the MTP head, every other type) takes the loader's existing
//! path; an `IQ4_XS`, `IQ4_NL` or `Q5_K` tensor on that path is expanded to BF16 on the GPU by the Ada
//! module `dequant_iq4xs_q5k_bf16_ada` when it is present (bit-exact with the CPU decoder, which runs
//! otherwise).
//!
//! With the row on, the IQ4 types also get their block layout in the container (`container.rs`, two guarded
//! arms) and a qwen35 file's MTP head (`blk.<L>.nextn.*`) is mapped to Atlas's `mtp.*` names
//! ([`super::names::translate_with_mtp`]).
//!
//! OFF (the default on every target): [`enabled`] is false, [`try_load`] and [`est_bytes`] return `None`,
//! [`qwen35_mtp_layer`] returns `None`, the container refuses the IQ4 types as before, and the loader is
//! byte-identical to the pre-sprint one. spark-runtime never reads a row: the row is resolved in spark-model
//! and published here by spark-server's `port_apply` before any model loads ([`set_enabled`]).

use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};

use super::{container, names, value_transform};
use crate::gpu::{DevicePtr, GpuBackend};
use crate::kernel_args::KernelLaunch;
use crate::weights::{WeightDtype, WeightTensor};

static ENABLED: AtomicBool = AtomicBool::new(false);
static KEEP_HEAD: AtomicBool = AtomicBool::new(false);

/// Publish the resolved `gguf_native_iq4` row. Called once at startup, before any model loads.
pub fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Release);
}

/// The published row (false until [`set_enabled`]).
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Acquire)
}

/// Keep the output head (`lm_head.weight`) packed too. Only a model side that serves a packed head through
/// `W4Linear` sets this (the packed head is lane L24's); until then the head is expanded as before, because
/// the pre-sprint packed-head path (`model/lm_head_q6k.rs`) resolves a module the Ada targets do not build.
pub fn set_keep_head(on: bool) {
    KEEP_HEAD.store(on, Ordering::Release);
}

/// See [`set_keep_head`].
pub fn keep_head() -> bool {
    KEEP_HEAD.load(Ordering::Acquire)
}

/// ggml type ids this route reads.
pub const ID_Q4_K: u32 = 12;
pub const ID_Q5_K: u32 = 13;
pub const ID_Q6_K: u32 = 14;
pub const ID_IQ4_NL: u32 = 20;
pub const ID_IQ4_XS: u32 = 23;

/// The Ada GPU dequant module (`kernels/ada/common/dequant_iq4xs_q5k_bf16_ada.cu`).
pub const DEQUANT_MODULE: &str = "dequant_iq4xs_q5k_bf16_ada";

/// The keep-packed dtype and the super-block bytes of a ggml id, for the three packed types.
pub fn packed_type(id: u32) -> Option<(WeightDtype, usize)> {
    match id {
        ID_IQ4_XS => Some((WeightDtype::Iq4Xs, 136)),
        ID_Q4_K => Some((WeightDtype::Q4K, 144)),
        ID_Q6_K => Some((WeightDtype::Q6K, 210)),
        _ => None,
    }
}

/// What the loader does with one tensor under the row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Upload the raw super-blocks unchanged, tagged with this dtype.
    Packed(WeightDtype),
    /// Apply the GDN value-head row reorder to the packed bytes (`after_qk` as in
    /// [`value_transform::packed_reorder_rows`]), then upload, tagged with this dtype.
    PackedRowReorder { dtype: WeightDtype, after_qk: bool },
    /// Expand to BF16 on the GPU with the Ada dequant module (IQ4_XS, IQ4_NL, Q5_K without a value
    /// transform).
    GpuDequant,
    /// The loader's existing path.
    Existing,
}

/// The per-tensor decision, a pure function of the ggml id, the mapped HF name, the architecture and the
/// head switch (the row itself is checked by the callers).
pub fn plan(id: u32, hf: &str, arch: &str, keep_head: bool) -> Plan {
    // The MTP head stays BF16: its loader reads plain BF16 weights.
    let is_mtp = hf.starts_with("mtp.");
    if let Some((dtype, _)) = packed_type(id)
        && names::is_dense_qwen35(arch)
        && !is_mtp
    {
        if names::is_keep_packed_proj(hf) && !value_transform::needs(hf) {
            return Plan::Packed(dtype);
        }
        if let Some(after_qk) = value_transform::packed_reorder_rows(hf) {
            return Plan::PackedRowReorder { dtype, after_qk };
        }
        // Only a Q6_K head is served packed (the factory installs one for `WeightDtype::Q6K` only); an
        // IQ4_XS or Q4_K head falls through to the GPU or existing expansion.
        if keep_head && hf == "lm_head.weight" && dtype == WeightDtype::Q6K {
            return Plan::Packed(dtype);
        }
    }
    if matches!(id, ID_IQ4_XS | ID_IQ4_NL | ID_Q5_K) && !value_transform::needs(hf) {
        return Plan::GpuDequant;
    }
    Plan::Existing
}

/// `(n, k)` of a 2-D HF shape whose rows are whole 256-weight super-blocks, else `None`.
fn rows_of_superblocks(hf_shape: &[usize]) -> Option<(usize, usize)> {
    match hf_shape {
        [n, k] if *n > 0 && *k > 0 && k.is_multiple_of(256) => Some((*n, *k)),
        _ => None,
    }
}

/// Device bytes a tensor the route keeps packed occupies (the pre-flight estimate), or `None` when the
/// tensor is expanded (the caller counts BF16) or the row is off.
pub(crate) fn est_bytes(id: u32, hf: &str, arch: &str, hf_shape: &[usize]) -> Option<usize> {
    if !enabled() {
        return None;
    }
    est_bytes_with(id, hf, arch, hf_shape, keep_head())
}

/// [`est_bytes`] with the switches passed in (pure, for tests).
pub fn est_bytes_with(
    id: u32,
    hf: &str,
    arch: &str,
    hf_shape: &[usize],
    keep_head: bool,
) -> Option<usize> {
    match plan(id, hf, arch, keep_head) {
        Plan::Packed(_) | Plan::PackedRowReorder { .. } => {
            let (n, k) = rows_of_superblocks(hf_shape)?;
            let (_, bb) = packed_type(id)?;
            Some(n * (k / 256) * bb)
        }
        Plan::GpuDequant | Plan::Existing => None,
    }
}

/// The loader hook (one call per mapped `Direct` tensor, before the existing dequant path). `Ok(None)`
/// means "not handled here": the caller's existing path runs, unchanged. Always `Ok(None)` with the row off.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_load(
    gpu: &dyn GpuBackend,
    id: u32,
    hf: &str,
    arch: &str,
    raw: &[u8],
    hf_shape: &[usize],
    gdn: Option<value_transform::GdnDims>,
    force_cpu: bool,
) -> Result<Option<WeightTensor>> {
    if !enabled() {
        return Ok(None);
    }
    try_load_with(
        gpu,
        id,
        hf,
        arch,
        raw,
        hf_shape,
        gdn,
        force_cpu,
        keep_head(),
    )
}

/// [`try_load`] with the head switch passed in (no process state, for tests).
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_load_with(
    gpu: &dyn GpuBackend,
    id: u32,
    hf: &str,
    arch: &str,
    raw: &[u8],
    hf_shape: &[usize],
    gdn: Option<value_transform::GdnDims>,
    force_cpu: bool,
    keep_head: bool,
) -> Result<Option<WeightTensor>> {
    let packed = |dtype: WeightDtype, bytes: &[u8]| -> Result<Option<WeightTensor>> {
        let ptr = gpu.alloc(bytes.len())?;
        gpu.copy_h2d(bytes, ptr)?;
        Ok(Some(WeightTensor {
            ptr,
            shape: hf_shape.to_vec(),
            dtype,
        }))
    };
    match plan(id, hf, arch, keep_head) {
        Plan::Packed(dtype) => {
            let Some(want) = est_bytes_with(id, hf, arch, hf_shape, keep_head) else {
                // Not whole super-block rows: the existing path expands it.
                return Ok(None);
            };
            ensure!(
                raw.len() == want,
                "keep-packed {hf}: {} bytes on disk, {want} expected for shape {hf_shape:?}",
                raw.len()
            );
            packed(dtype, raw)
        }
        Plan::PackedRowReorder { dtype, after_qk } => {
            let (Some(want), Some(dims)) = (est_bytes_with(id, hf, arch, hf_shape, keep_head), gdn)
            else {
                return Ok(None);
            };
            ensure!(
                raw.len() == want,
                "keep-packed {hf}: {} bytes on disk, {want} expected for shape {hf_shape:?}",
                raw.len()
            );
            let (_, bb) = packed_type(id).context("packed type")?;
            let permuted =
                value_transform::reorder_packed_rows(raw, hf_shape, &dims, after_qk, 256, bb)
                    .with_context(|| format!("packed GDN row reorder {hf}"))?;
            packed(dtype, &permuted)
        }
        Plan::GpuDequant if !force_cpu && gpu.has_module(DEQUANT_MODULE) => {
            let numel: usize = hf_shape.iter().product();
            let ptr = dequant_to_bf16_gpu(gpu, id, raw, numel)?;
            Ok(Some(WeightTensor {
                ptr,
                shape: hf_shape.to_vec(),
                dtype: WeightDtype::BF16,
            }))
        }
        Plan::GpuDequant | Plan::Existing => Ok(None),
    }
}

/// `(kernel, weights a block, bytes a block)` of the Ada dequant entry for a ggml id.
pub fn dequant_entry(id: u32) -> Option<(&'static str, usize, usize)> {
    match id {
        ID_IQ4_XS => Some(("dequant_iq4_xs_to_bf16_ada", 256, 136)),
        ID_IQ4_NL => Some(("dequant_iq4_nl_to_bf16_ada", 32, 18)),
        ID_Q5_K => Some(("dequant_q5_k_to_bf16_ada", 256, 176)),
        _ => None,
    }
}

/// Upload `raw` and expand it to a fresh BF16 device buffer of `numel` values on the Ada module, on the
/// default stream, synchronized (a load-time kernel: the raw upload is freed before returning).
pub fn dequant_to_bf16_gpu(
    gpu: &dyn GpuBackend,
    id: u32,
    raw: &[u8],
    numel: usize,
) -> Result<DevicePtr> {
    let (func, qk, bb) =
        dequant_entry(id).with_context(|| format!("no Ada dequant for ggml {id}"))?;
    ensure!(
        numel.is_multiple_of(qk) && raw.len() == numel / qk * bb,
        "Ada dequant ggml {id}: {} bytes for {numel} values",
        raw.len()
    );
    let n_blocks = numel / qk;
    let kernel = gpu.kernel(DEQUANT_MODULE, func)?;
    let q_ptr = gpu.alloc(raw.len())?;
    gpu.copy_h2d(raw, q_ptr)?;
    let out = gpu.alloc(numel * 2)?;
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, kernel)
        .grid([u32::try_from(n_blocks).context("block count")?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_ptr)
        .arg_ptr(out)
        .arg_u32(n_blocks as u32)
        .arg_u32(bb as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    gpu.free(q_ptr)?;
    Ok(out)
}

/// The MTP head block of a dense qwen35 file under the row (`None` with the row off, for another
/// architecture, or when the file has no `blk.<L>.nextn.*` tensor).
pub(crate) fn qwen35_mtp_layer(gguf: &container::GgufFile, arch: &str) -> Option<usize> {
    if !enabled() || !names::is_dense_qwen35(arch) {
        return None;
    }
    names::mtp_layer(gguf.tensors.iter().map(|t| t.name.as_str()))
}

/// The boot line the loader prints when the row is on.
pub fn boot_line() -> String {
    format!(
        "gguf_native_iq4=on: keeping IQ4_XS / Q4_K / Q6_K projections of qwen35 GGUFs packed \
         (output head {}), IQ4 / Q5_K expanded on the GPU when `{DEQUANT_MODULE}` is built, \
         MTP head names mapped",
        if keep_head() { "packed" } else { "expanded" }
    )
}

#[cfg(test)]
#[path = "keep_packed_iq4_tests.rs"]
mod tests;
