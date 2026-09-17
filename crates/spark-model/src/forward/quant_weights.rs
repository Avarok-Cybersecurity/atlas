// SPDX-License-Identifier: AGPL-3.0-only
//! Per-quant weight abstraction.
//!
//! Plug-in trait the vendor-agnostic forward modules (e.g.
//! [`super::qwen3_5`]) call instead of reaching into a concrete weight
//! type like `MlxInt8Weight`. Each backend's weight loader implements
//! `QuantWeights` and overrides whichever fused variants it ships
//! kernels for; the rest fall back to default impls that compose the
//! unfused primitives or fail loudly.
//!
//! Convention: `gemv` is mandatory (every backend ships one). Fused
//! variants (`gemv_silu_gate`, `gemv_silu_gate_resid`,
//! `gemv_gate_up_with`) are advisory — backends override only the
//! ones they have fused kernels for. Defaults either fall back to a
//! correct-but-slower composition (`gemv_gate_up_with`) or error
//! loudly so the caller can choose between hard-requiring the fused
//! kernel and degrading to a manual silu+mul+gemv pipeline.

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::mlx_int8::{self, MlxInt8Weight};

/// A quantised weight tensor that can drive matvec / matmul ops on a
/// `GpuBackend`. Implementations live with each backend's weight
/// loader (Metal: `MlxInt8Weight`; future CUDA: `Nvfp4Weight`,
/// `Fp8DenseWeight`, …).
pub trait QuantWeights: Send + Sync {
    /// Output dimension `N` of the underlying `[N, K]` weight.
    fn out_features(&self) -> u32;

    /// Input dimension `K` of the underlying `[N, K]` weight.
    fn in_features(&self) -> u32;

    /// Decode-path matvec: `y = self @ x`.
    ///
    /// `x` is a BF16 buffer of length `in_features()`; `y` must hold at
    /// least `out_features()` BF16 slots.
    fn gemv(&self, gpu: &dyn GpuBackend, x: DevicePtr, y: DevicePtr, stream: u64) -> Result<()>;

    /// Dual-output GEMV with shared input: `gate_y = self @ x`,
    /// `up_y = other @ x`. The default impl is two serial `gemv`
    /// calls — correct on any backend, just slower than the fused
    /// kernel some backends ship (e.g. Metal's
    /// `mlx_int8_gemv_gate_up`). Backends with a fused dual-output
    /// path override this to halve x-side memory bandwidth and
    /// remove a launch.
    ///
    /// `where Self: Sized` keeps this method off the dyn-trait surface
    /// (it's only callable through generic-parameter dispatch, which
    /// is what the forward modules use anyway).
    fn gemv_gate_up_with(
        &self,
        other: &Self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate_y: DevicePtr,
        up_y: DevicePtr,
        stream: u64,
    ) -> Result<()>
    where
        Self: Sized,
    {
        debug_assert_eq!(self.out_features(), other.out_features());
        debug_assert_eq!(self.in_features(), other.in_features());
        self.gemv(gpu, x, gate_y, stream)?;
        other.gemv(gpu, x, up_y, stream)?;
        Ok(())
    }

    /// Fused FFN tail: `y = self @ (silu(gate) ⊙ up)`.
    ///
    /// Default impl errors — backends that ship the fused kernel
    /// override; backends that don't can either error out (forcing
    /// the caller to do the unfused dance) or override with their
    /// own composition.
    fn gemv_silu_gate(
        &self,
        _gpu: &dyn GpuBackend,
        _gate: DevicePtr,
        _up: DevicePtr,
        _y: DevicePtr,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "QuantWeights::gemv_silu_gate is not implemented for this weight type — \
             override with a fused kernel or compose silu+mul+gemv on the caller side"
        )
    }

    /// Same as [`Self::gemv_silu_gate`] but additionally folds the
    /// layer-output residual addition into the same kernel:
    ///   `y[n] = x_resid[n] + sum_k self[n, k] * (silu(gate[k]) ⊙ up[k])`.
    ///
    /// Default impl errors. Backends that ship a `_resid` variant of
    /// the fused kernel override.
    fn gemv_silu_gate_resid(
        &self,
        _gpu: &dyn GpuBackend,
        _gate: DevicePtr,
        _up: DevicePtr,
        _x_resid: DevicePtr,
        _y: DevicePtr,
        _stream: u64,
    ) -> Result<()> {
        bail!(
            "QuantWeights::gemv_silu_gate_resid is not implemented for this weight type — \
             override with a fused kernel or compose silu+mul+gemv+add on the caller side"
        )
    }
}

// ── Backend impls ────────────────────────────────────────────────────
//
// Lives next to the trait so the orphan rule applies (the trait is
// foreign to spark-runtime, but native here). Each impl is a thin
// forwarding shim onto the concrete weight type's inherent methods,
// keeping the optimised fused-kernel paths intact.

impl QuantWeights for MlxInt8Weight {
    fn out_features(&self) -> u32 {
        self.out_features
    }
    fn in_features(&self) -> u32 {
        self.in_features
    }
    fn gemv(&self, gpu: &dyn GpuBackend, x: DevicePtr, y: DevicePtr, stream: u64) -> Result<()> {
        MlxInt8Weight::gemv(self, gpu, x, y, stream)
    }
    fn gemv_gate_up_with(
        &self,
        other: &Self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        gate_y: DevicePtr,
        up_y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        // Atlas Metal ships a fused dual-output kernel
        // (`mlx_int8_gemv_gate_up`); use it instead of two serial
        // gemvs to halve x-side bandwidth and remove a launch.
        mlx_int8::gemv_gate_up(gpu, self, other, x, gate_y, up_y, stream)
    }
    fn gemv_silu_gate(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        MlxInt8Weight::gemv_silu_gate(self, gpu, gate, up, y, stream)
    }
    fn gemv_silu_gate_resid(
        &self,
        gpu: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        x_resid: DevicePtr,
        y: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        MlxInt8Weight::gemv_silu_gate_resid(self, gpu, gate, up, x_resid, y, stream)
    }
}
