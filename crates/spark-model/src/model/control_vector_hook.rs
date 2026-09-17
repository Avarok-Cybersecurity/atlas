// SPDX-License-Identifier: AGPL-3.0-only

//! Installing a control vector, and applying it at the end of every layer.
//!
//! See `docs/design/qwen4exp-control-vectors.md`. Two entry points:
//!
//! * [`TransformerModel::load_control_vector`] — boot-time, once.
//! * [`TransformerModel::cvec_after_layer`] — called from EVERY model-level
//!   layer loop, immediately after `layer.prefill()` / `layer.decode()`
//!   returns.
//!
//! # Why one function, called from eleven places
//!
//! The intervention has to land on every forward path the model can take —
//! four prefill loops, three decode loops, two verify loops and the two
//! profiling/draft loops. A path that misses it serves partially-steered
//! output, and nothing catches that: the text is well-formed, every
//! correctness gate passes, and only a behavioural difference between (say)
//! a chunked and an unchunked prompt would ever hint at it.
//!
//! So the per-site call is deliberately trivial and takes the whole
//! [`ForwardContext`], which means the two things a site could get wrong —
//! the highway base and the row offset — are computed HERE, once. The K-row
//! verify and the fused decode+prefill step put their rows at a non-zero
//! `hc_row_offset`; a site that passed a bare `hc_streams()` would silently
//! steer the wrong rows, which is how the batched-verify highway-row defect
//! held MTP acceptance at 0.19 while every gate passed.

use anyhow::{Context, Result};
use spark_runtime::gpu::DevicePtr;

use crate::control_vector::{ControlVector, ControlVectorSpec};
use crate::layer::ForwardContext;
use crate::model::TransformerModel;

impl TransformerModel {
    /// Load and install a control vector. Boot-time only; any failure aborts.
    ///
    /// Refuses a model with no mHC highway. That refusal is also the MVP's
    /// scope gate: the intervention site is `hc_streams`, which only exists
    /// when `hc_mult > 0`, so a non-mHC architecture has nowhere to put this
    /// and would otherwise accept the flag and do nothing.
    pub fn load_control_vector(&mut self, spec: &ControlVectorSpec) -> Result<()> {
        anyhow::ensure!(
            self.config.hc_mult > 0,
            "control vectors need an mHC highway to act on, and model_type \
             {:?} has hc_mult = 0. The intervention site is the per-layer \
             highway; without one there is nothing to steer.",
            self.config.model_type
        );
        let cv = ControlVector::load(
            self.gpu.as_ref(),
            spec,
            self.config.hidden_size,
            self.config.num_hidden_layers,
        )
        .with_context(|| {
            format!(
                "installing control vector {} on {}",
                spec.path.display(),
                self.config.model_type
            )
        })?;
        self.control_vector = Some(cv);
        Ok(())
    }

    /// Whether a control vector is installed. Used by the paths that must
    /// refuse to fold or skip work when one is active.
    #[inline]
    pub fn has_control_vector(&self) -> bool {
        self.control_vector.is_some()
    }

    /// Apply the control vector to layer `layer_idx`'s output.
    ///
    /// Call site: immediately after `layer.prefill()`/`layer.decode()` returns
    /// in a model-level layer loop, where the highway holds that layer's
    /// output and the next layer has not yet read it.
    ///
    /// A no-op — no launch at all — when no vector is installed or when this
    /// layer falls outside the active range. Graph-capture legal: one
    /// stream-ordered launch against a boot-time allocation, no synchronize
    /// and no environment read.
    /// `path` names the calling forward path (`"decode"`, `"verify_batched"`,
    /// …). It is REQUIRED, and it is what makes the probe a coverage proof
    /// rather than a spot check: with `AVAROK_CVEC_PROBE=1` the log lists
    /// which paths actually applied the vector, so a path nobody wired simply
    /// never appears. That is the only way to catch the failure this feature
    /// is most exposed to — partially-steered output that every correctness
    /// gate passes.
    #[inline]
    pub(crate) fn cvec_after_layer(
        &self,
        ctx: &ForwardContext<'_>,
        path: &'static str,
        layer_idx: usize,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(cv) = self.control_vector.as_ref() else {
            return Ok(());
        };
        if !cv.applies_to(layer_idx) || num_tokens == 0 {
            return Ok(());
        }
        // The one place the highway base is computed. `hc_row_offset` is the
        // row this pass owns — non-zero for mixed decode+prefill steps and the
        // K-row verify — and the highway is row-major `[row, hc_mult, H]` FP32.
        let stride = ctx.config.hc_mult * ctx.config.hidden_size * 4;
        let highway = DevicePtr(ctx.buffers.hc_streams().0 + (ctx.hc_row_offset * stride) as u64);

        let pre = if cv.probe_enabled() {
            cv.cos_probe(
                ctx.gpu,
                highway,
                layer_idx,
                num_tokens,
                ctx.config.hc_mult,
                stream,
            )?
        } else {
            0.0
        };

        cv.apply(
            ctx.gpu,
            highway,
            layer_idx,
            num_tokens,
            ctx.config.hc_mult,
            stream,
        )
        .with_context(|| format!("control vector at layer {layer_idx} ({path})"))?;

        if cv.probe_enabled() {
            let post = cv.cos_probe(
                ctx.gpu,
                highway,
                layer_idx,
                num_tokens,
                ctx.config.hc_mult,
                stream,
            )?;
            // `post` must be ~0 and `pre` clearly non-zero. A `post` that
            // equals `pre` means the launch did not touch these rows — the
            // row-offset bug, not a missing call.
            tracing::info!(
                "CVEC_PROBE path={path} layer={layer_idx} rows={num_tokens} \
                 row_offset={} pre={pre:.6} post={post:.6}",
                ctx.hc_row_offset
            );
        }
        Ok(())
    }

    /// Mean `|cos(h, v)|` over layer `layer_idx`'s highway — the validation
    /// probe. Synchronizes and copies D2H, so it is debug-only and must never
    /// run inside CUDA-graph capture.
    ///
    /// Returns `None` when no vector is installed or this layer is outside the
    /// range, so a caller can tell "not measured" from "measured zero".
    pub fn cvec_cos_probe(
        &self,
        ctx: &ForwardContext<'_>,
        layer_idx: usize,
        num_tokens: usize,
        stream: u64,
    ) -> Result<Option<f32>> {
        let Some(cv) = self.control_vector.as_ref() else {
            return Ok(None);
        };
        if !cv.applies_to(layer_idx) || num_tokens == 0 {
            return Ok(None);
        }
        let stride = ctx.config.hc_mult * ctx.config.hidden_size * 4;
        let highway = DevicePtr(ctx.buffers.hc_streams().0 + (ctx.hc_row_offset * stride) as u64);
        cv.cos_probe(
            ctx.gpu,
            highway,
            layer_idx,
            num_tokens,
            ctx.config.hc_mult,
            stream,
        )
        .map(Some)
    }
}
