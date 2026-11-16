// SPDX-License-Identifier: AGPL-3.0-only
// Host side translated for Atlas from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/marlin.cu, Copyright contributors to the vLLM project, Apache-2.0.
// Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0, adapted from https://github.com/IST-DASLab/marlin.
// Changes: the module layout and the public constructors are Atlas's; see the per-file headers.

//! Responsibility: the Marlin W4A16 route's public surface (lane L18b): the constructors that turn a
//! BF16 weight (round-to-nearest int4, opt-in), a GPTQ checkpoint or an AWQ checkpoint into a
//! [`W4Box`] backed by the Marlin int4 kernels of `kernels/ada/common/marlin_ada.cu` (lane L18a), and
//! the `marlin_w4a16` row.
//!
//! # How a weight flows
//!
//! ```text
//! BF16 [n, k] --quantize::rtn_int4_sym--+
//! GPTQ tensors --checkpoint::gptq------+--> reference::LogicalInt4 --repack::to_host--> MarlinHost
//! CT packed    --compressed::ct--------+        (q [k][n], scales,          (Marlin tiles,
//! AWQ tensors  --checkpoint::awq-------+        zeros)                       permuted scales)
//! MarlinHost --weight::MarlinWeight::upload--> device --w4linear::MarlinLinear--> W4Box
//! ```
//!
//! [`reference::LogicalInt4`] is the one logical form: every source is parsed into it, every oracle
//! (repack round trip, CPU dequant, CPU GEMM) is computed from it, and `dequant_bf16` reconstructs it
//! from the device bytes.
//!
//! # Row invariance
//!
//! `decode` (rows 1..=64) runs ONE thread configuration per weight, fixed at load, and never the
//! `m_block_size_8` tile: the K partition of a Marlin launch depends only on the thread configuration,
//! the grid and the shape, not on the row count (`marlin_template.h:328-358`, `parallel` is 1 for every
//! decode launch), so row `r`'s bits do not depend on how many rows ride with it. vLLM's own selector
//! (`determine_exec_config`) changes the configuration with M and is used for `prefill` only. See
//! `config.rs`.
//!
//! # Gating
//!
//! Nothing here runs unless a loader route calls a constructor, and every route keys on the
//! `marlin_w4a16` row ([`row_enabled`], off on every target; the BF16 routes call
//! [`route::quantize_projection`], which returns `None` with the row off) or on a GPTQ/AWQ
//! checkpoint the operator chose. The kernels exist only in the Ada build: on any other target [`available`] is false and a
//! constructor returns an error naming the module, without issuing a kernel lookup.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::port::w4::{W4Box, W4Opts};

pub mod checkpoint;
pub mod compressed;
pub mod config;
pub mod kernels;
pub mod quantize;
pub mod reference;
pub mod repack;
pub mod route;
pub mod w4linear;
pub mod weight;

pub use checkpoint::{AwqWeights, GptqWeights, GptqZeroFormat, ScaleDtype};
pub use reference::LogicalInt4;
pub use repack::MarlinHost;
pub use w4linear::{MarlinLinear, release};

/// Rows one decode launch serves (`thread_m_blocks` 4, one 64-row tile).
pub const MAX_DECODE_ROWS: usize = 64;
/// The bit width every Marlin weight built here has.
pub const BITS: u8 = 4;
/// The group size of the opt-in RTN mode and of the checkpoints the pruned kernel set serves.
pub const DEFAULT_GROUP: usize = 128;

/// The Marlin weight type (`vllm::ScalarType` of the B operand).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MarlinType {
    /// `kU4B8`: unsigned 4-bit codes with a fixed bias of 8 (GPTQ symmetric, RTN), `w = (q - 8) * s`.
    U4B8,
    /// `kU4`: unsigned 4-bit codes with per-group zero points (AWQ, GPTQ asymmetric),
    /// `w = (q - z) * s`.
    U4,
}

impl MarlinType {
    /// The label in the kernel entry names (`marlin_ada_bf16_<label>_...`).
    pub fn label(self) -> &'static str {
        match self {
            Self::U4B8 => "u4b8",
            Self::U4 => "u4",
        }
    }

    /// True when the weight carries zero points.
    pub fn has_zp(self) -> bool {
        matches!(self, Self::U4)
    }
}

/// True when this build carries the Marlin kernels (the Ada target). Issues no kernel lookup.
pub fn available(gpu: &dyn GpuBackend) -> bool {
    gpu.has_module(kernels::MODULE)
}

/// The resolved `marlin_w4a16` row (`AVAROK_MARLIN_W4A16`, else the target's declaration; false on
/// every target in this sprint).
pub fn row_enabled() -> bool {
    crate::layers::ops::port_rows::marlin_w4a16::row()
}

/// Check the load-time options for the RTN mode: 4 bits and a group size the kernel set serves.
pub fn check_rtn_opts(n: usize, k: usize, opts: &W4Opts) -> Result<()> {
    ensure!(
        opts.bits == BITS,
        "marlin: the RTN mode is int4 only, --w4-bits {} was asked",
        opts.bits
    );
    let group = usize::from(opts.group);
    ensure!(
        kernels::group_supported(group, k),
        "marlin: group {group} at K={k} has no kernel instance (the Ada build carries g{})",
        DEFAULT_GROUP
    );
    ensure!(
        opts.clip_search <= quantize::MAX_CLIP_SEARCH,
        "marlin: --w4-clip-search {} is above the largest candidate count {}",
        opts.clip_search,
        quantize::MAX_CLIP_SEARCH
    );
    config::supports_shape(n, k, group).map_err(|e| anyhow::anyhow!("marlin: {e}"))
}

/// Opt-in round-to-nearest int4 from a BF16 weight already on the device.
///
/// `src` is the `[n, k]` BF16 row-major weight (the `DenseWeight` layout); it is read back to the
/// host, quantized there (symmetric `kU4B8`, group `opts.group`, `opts.clip_search` shrink
/// candidates per group), repacked to the Marlin layout and uploaded. `src` is not freed: the caller
/// frees its BF16 source after this returns. `opts.int8_prefill` has no Marlin meaning and is
/// ignored. Load time only (host round trip, synchronizes `stream`).
pub fn quantize_rtn_from_bf16(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    n: usize,
    k: usize,
    opts: &W4Opts,
    stream: u64,
) -> Result<W4Box> {
    check_rtn_opts(n, k, opts)?;
    ensure!(
        available(gpu),
        "marlin: module {} is not compiled for this target (the Marlin route needs the Ada build)",
        kernels::MODULE
    );
    let mut bytes = vec![0u8; n * k * 2];
    gpu.synchronize(stream)?;
    gpu.copy_d2h_on_stream(src, &mut bytes, stream)?;
    let w: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let rtn = quantize::RtnOpts {
        group: usize::from(opts.group),
        clip_search: opts.clip_search,
    };
    let logical = quantize::rtn_int4_sym(&w, n, k, &rtn)?;
    from_logical(gpu, &logical, stream)
}

/// A GPTQ int4 checkpoint tensor set (`qweight`, `scales`, optional `qzeros` and `g_idx`).
/// Act-order checkpoints (a non-trivial `g_idx`) are refused: the kernel has no activation reorder.
pub fn from_gptq(gpu: &dyn GpuBackend, w: &GptqWeights<'_>, stream: u64) -> Result<W4Box> {
    let logical = checkpoint::gptq_to_logical(w)?;
    from_logical(gpu, &logical, stream)
}

/// An AWQ int4 checkpoint tensor set (`qweight`, `qzeros`, `scales`, all AWQ-interleaved).
pub fn from_awq(gpu: &dyn GpuBackend, w: &AwqWeights<'_>, stream: u64) -> Result<W4Box> {
    let logical = checkpoint::awq_to_logical(w)?;
    from_logical(gpu, &logical, stream)
}

/// A compressed-tensors `pack-quantized` int4 tensor set (`weight_packed`, `weight_scale`, optional
/// contiguous `weight_g_idx`); see [`compressed`].
pub fn from_compressed(
    gpu: &dyn GpuBackend,
    w: &compressed::CtPackedWeights<'_>,
    stream: u64,
) -> Result<W4Box> {
    let logical = compressed::ct_to_logical(w)?;
    from_logical(gpu, &logical, stream)
}

/// Any logical int4 weight: repack on the host, upload, wrap.
pub fn from_logical(gpu: &dyn GpuBackend, logical: &LogicalInt4, stream: u64) -> Result<W4Box> {
    let linear = build(gpu, logical, stream)?;
    Ok(MarlinLinear::boxed(linear))
}

/// [`from_logical`] without the type erasure, for the microtest and the self-check, which need the
/// concrete [`MarlinLinear`] (its `run_rows` takes an explicit decode policy).
pub fn build(gpu: &dyn GpuBackend, logical: &LogicalInt4, stream: u64) -> Result<MarlinLinear> {
    logical.validate()?;
    ensure!(
        available(gpu),
        "marlin: module {} is not compiled for this target (the Marlin route needs the Ada build)",
        kernels::MODULE
    );
    let host = repack::to_host(logical)?;
    let w = weight::MarlinWeight::upload(gpu, &host, stream)?;
    Ok(MarlinLinear::new(w))
}

#[cfg(test)]
mod quantize_tests;
#[cfg(test)]
mod repack_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod w4linear_tests;
