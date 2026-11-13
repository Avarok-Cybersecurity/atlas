// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: building an `Lfm3ConvLayer`: geometry checks, kernel
//! resolution (the conv entries plus the shared norm, residual and GEMV
//! kernels), the padding-row window, the optional FP8 decode projections, and
//! the accessors the LFM3 loader (lane L37) uses.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::launch::ConvKernels;
use super::{Lfm3ConvCfg, Lfm3ConvFp8, Lfm3ConvLayer, Lfm3ConvWeights};
use crate::layers::FfnComponent;
use crate::weight_map::{DenseWeight, Fp8Weight, WeightQuantFormat};

/// Every kernel the layer launches. A `0` handle marks an optional tier that
/// is absent; required handles are resolved with `gpu.kernel` and fail the
/// load when missing.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerKernels {
    pub conv: ConvKernels,
    /// `rms_norm_vanilla` (weight * x), in place on the sublayer outputs.
    pub rms_norm: KernelHandle,
    /// `norm::rms_norm_residual_vanilla`, the operator norm.
    pub rms_norm_residual: KernelHandle,
    /// `norm::residual_add_rms_norm_vanilla`, residual + ffn norm.
    pub residual_add_rms_norm: KernelHandle,
    /// `residual_add::bf16_residual_add`.
    pub residual_add: KernelHandle,
    /// `gemv::dense_gemv_bf16` (M = 1).
    pub gemv: KernelHandle,
    /// `dense_gemv_bf16_batchm` (M = 2..=16, each row bit-identical to M = 1).
    /// Optional: absent, those widths go to cuBLASLt.
    pub gemv_batchm: KernelHandle,
    /// Block-scaled FP8 GEMV (M = 1): the Ada drop-in `fp8_gemv_ada` where the
    /// target compiles it, else `w8a16_gemv` (same arguments). Optional.
    pub fp8_gemv: KernelHandle,
    /// `w8a16_gemv_batch4::w8a16_gemv_batch16` (M = 2..=16). Optional.
    pub fp8_gemv_batch16: KernelHandle,
}

impl LayerKernels {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        let fp8_gemv = {
            let h = crate::layers::try_target_kernel(gpu, "fp8_gemv_ada", "fp8_gemv_ada");
            if h.0 != 0 {
                h
            } else {
                crate::layers::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv")
            }
        };
        Ok(Self {
            conv: ConvKernels::resolve(gpu)?,
            rms_norm: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            rms_norm_residual: gpu.kernel("norm", "rms_norm_residual_vanilla")?,
            residual_add_rms_norm: gpu.kernel("norm", "residual_add_rms_norm_vanilla")?,
            residual_add: gpu.kernel("residual_add", "bf16_residual_add")?,
            gemv: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemv_batchm: crate::layers::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            fp8_gemv,
            fp8_gemv_batch16: crate::layers::try_kernel(
                gpu,
                "w8a16_gemv_batch4",
                "w8a16_gemv_batch16",
            ),
        })
    }
}

fn require(w: &DenseWeight, what: &str, layer: usize) -> Result<()> {
    ensure!(
        !w.weight.is_null(),
        "LFM3 conv layer {layer}: weight `{what}` is NULL (not loaded)"
    );
    Ok(())
}

impl Lfm3ConvLayer {
    /// Build the layer and resolve its kernel handles (contract 3.7.7).
    ///
    /// Fails, naming the cause, when the geometry is not the implemented one,
    /// a weight is missing, the FFN is not dense, or the target's kernel image
    /// does not carry `lfm3_shortconv_ada` (only `kernels/ada` compiles it).
    /// Allocates one padding window (`cfg.state_bytes()`, zeroed).
    pub fn new(gpu: &dyn GpuBackend, cfg: &Lfm3ConvCfg, w: Lfm3ConvWeights) -> Result<Self> {
        cfg.validate()?;
        let li = cfg.layer_idx;
        for (dw, name) in [
            (&w.operator_norm, "operator_norm"),
            (&w.in_proj, "conv.in_proj"),
            (&w.conv, "conv.conv"),
            (&w.out_proj, "conv.out_proj"),
            (&w.post_operator_norm, "post_operator_norm"),
            (&w.ffn_norm, "ffn_norm"),
            (&w.post_ffn_norm, "post_ffn_norm"),
        ] {
            require(dw, name, li)?;
        }
        ensure!(
            matches!(w.ffn, FfnComponent::Dense(_)),
            "LFM3 conv layer {li}: the feed-forward must be a dense SwiGLU (FfnComponent::Dense)"
        );
        let k = LayerKernels::resolve(gpu)
            .with_context(|| format!("LFM3 conv layer {li}: kernel resolution"))?;
        let pad_state = gpu.alloc(cfg.state_bytes())?;
        gpu.memset(pad_state, 0, cfg.state_bytes())?;
        Ok(Self {
            cfg: *cfg,
            operator_norm: w.operator_norm,
            in_proj: w.in_proj,
            conv: w.conv,
            out_proj: w.out_proj,
            post_operator_norm: w.post_operator_norm,
            ffn_norm: w.ffn_norm,
            ffn: w.ffn,
            post_ffn_norm: w.post_ffn_norm,
            fp8: None,
            k,
            pad_state,
        })
    }

    /// Install FP8 decode copies of the conv projections (lane L37's opt-in
    /// `lfm3_fp8_all` route, or the vendor FP8 checkpoint). The BF16 weights
    /// stay installed: prefill chunks wider than 16 rows read them through
    /// cuBLASLt, so the caller must not free them.
    ///
    /// Both weights must be `WeightQuantFormat::Fp8BlockScaled` with FP32
    /// `[ceil(N/128), ceil(K/128)]` scales (what `fp8_gemv_ada` and
    /// `w8a16_gemv` index) and the conv projection shapes; the M = 1 FP8 GEMV
    /// must have resolved.
    pub fn set_fp8_proj(&mut self, in_proj: Fp8Weight, out_proj: Fp8Weight) -> Result<()> {
        let (li, h) = (self.cfg.layer_idx, self.cfg.hidden as u32);
        for (wt, name, n) in [(&in_proj, "in_proj", 3 * h), (&out_proj, "out_proj", h)] {
            ensure!(
                wt.scale_format == WeightQuantFormat::Fp8BlockScaled,
                "LFM3 conv layer {li}: FP8 {name} is tagged {:?}; the FP8 GEMVs read \
                 Fp8BlockScaled FP32 block scales",
                wt.scale_format
            );
            ensure!(
                wt.n == n && wt.k == h,
                "LFM3 conv layer {li}: FP8 {name} is [{}, {}], expected [{n}, {h}]",
                wt.n,
                wt.k
            );
            ensure!(
                !wt.weight.is_null() && !wt.row_scale.is_null(),
                "LFM3 conv layer {li}: FP8 {name} has a NULL weight or scale"
            );
        }
        ensure!(
            self.k.fp8_gemv.0 != 0,
            "LFM3 conv layer {li}: FP8 projections need fp8_gemv_ada or w8a16_gemv in the image"
        );
        self.fp8 = Some(Lfm3ConvFp8 { in_proj, out_proj });
        Ok(())
    }

    /// Geometry this layer was built with.
    pub fn cfg(&self) -> &Lfm3ConvCfg {
        &self.cfg
    }

    /// The BF16 conv projections `(in_proj, out_proj)`, for the FP8 route's
    /// load-time quantization.
    pub fn bf16_proj(&self) -> (DenseWeight, DenseWeight) {
        (self.in_proj, self.out_proj)
    }

    /// Whether FP8 decode projections are installed.
    pub fn has_fp8_proj(&self) -> bool {
        self.fp8.is_some()
    }

    /// The feed-forward, for routes that install quantized FFN weights after
    /// construction (the dense FFN's own setters).
    pub fn ffn_mut(&mut self) -> &mut FfnComponent {
        &mut self.ffn
    }

    /// The window a batch row reads and writes: the sequence's own, or the
    /// layer's padding window for the pool-dummy `SsmLayerState` the model
    /// hands padding rows. Any other state type is a wiring error.
    pub(crate) fn row_state(&self, state: &dyn crate::layer::LayerState) -> Result<DevicePtr> {
        row_state_of(state, self.pad_state, self.cfg.layer_idx)
    }
}

/// [`Lfm3ConvLayer::row_state`] without the layer, for unit tests.
pub(crate) fn row_state_of(
    state: &dyn crate::layer::LayerState,
    pad: DevicePtr,
    layer_idx: usize,
) -> Result<DevicePtr> {
    let any = state.as_any();
    if let Some(s) = any.downcast_ref::<super::Lfm3ConvState>() {
        ensure!(
            !s.conv.is_null(),
            "LFM3 conv layer {layer_idx}: the sequence's window was already released"
        );
        return Ok(s.conv);
    }
    if any.is::<crate::layer::SsmLayerState>() {
        return Ok(pad);
    }
    anyhow::bail!(
        "LFM3 conv layer {layer_idx}: expected Lfm3ConvState (or the pool-dummy SsmLayerState of a \
         padding row), got another layer state type"
    )
}
