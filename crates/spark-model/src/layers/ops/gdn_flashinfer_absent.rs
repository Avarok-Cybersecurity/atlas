// SPDX-License-Identifier: AGPL-3.0-only

//! `not(unix)` counterpart of [`super::gdn_flashinfer`].
//!
//! The real module reaches the FlashInfer GDN kernel through
//! `dlopen("libatlasgdn.so")` — POSIX dynamic loading, and a `.so` that is only
//! built for Linux. Windows has neither, so the path is unavailable there.
//!
//! That is not a degradation introduced here: the feature is opt-in behind
//! `ATLAS_GDN_FLASHINFER=1`, and its own docs note that "the binary builds and
//! runs without the library". Both call sites are already guarded by
//! [`available`] and fall back to the scalar FLA scan, so reporting `false`
//! here reproduces exactly what a unix host does when the library is absent or
//! the flag is unset.
//!
//! Declared under the same module path as the real one (`ops::gdn_flashinfer`)
//! so the two call sites need no `cfg`.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// Always false: there is no `libatlasgdn.so` to dlopen on this platform, so
/// callers take the FLA fallback.
pub fn available() -> bool {
    false
}

/// Unreachable through the guarded call sites, which check [`available`] first.
/// Bails rather than silently returning `Ok(())`: a caller that reached this
/// without checking would otherwise get an unwritten output buffer treated as a
/// completed scan.
#[allow(clippy::too_many_arguments)]
pub fn flashinfer_gdn_prefill(
    _gpu: &dyn GpuBackend,
    _qkv: DevicePtr,
    _gate_beta: DevicePtr,
    _output: DevicePtr,
    _h_state: DevicePtr,
    _scale: f32,
    _total: u32,
    _nk: u32,
    _nv: u32,
    _kd: u32,
    _vd: u32,
    _conv_dim: u32,
    _gb_stride: u32,
    _num_seqs: u32,
    _stream: u64,
) -> Result<()> {
    bail!("FlashInfer GDN prefill requires dlopen(libatlasgdn.so); unavailable on this platform")
}
