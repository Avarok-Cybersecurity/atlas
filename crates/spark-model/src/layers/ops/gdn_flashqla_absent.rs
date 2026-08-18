// SPDX-License-Identifier: AGPL-3.0-only

//! `not(unix)` counterpart of [`super::gdn_flashqla`].
//!
//! The real module reaches the FlashQLA GDN kernel through
//! `dlopen("libatlasqla.so")` — POSIX dynamic loading, and a `.so` that is only
//! built for Linux. Windows has neither, so the path is unavailable there.
//!
//! Declared under the same module path as the real one (`ops::gdn_flashqla`)
//! so the call sites need no `cfg`.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

pub fn initialize(_gpu: &dyn GpuBackend) -> bool {
    false
}

/// Always false: there is no `libatlasqla.so` to dlopen on this platform.
pub fn available() -> bool {
    false
}

pub fn use_log_gate(_exact_replay: bool, _kd: usize, _vd: usize) -> bool {
    false
}

/// Unreachable through the guarded call sites, which check [`available`] first.
#[allow(clippy::too_many_arguments)]
pub fn flashqla_gdn_prefill(
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
    bail!("FlashQLA GDN prefill requires dlopen(libatlasqla.so); unavailable on this platform")
}
