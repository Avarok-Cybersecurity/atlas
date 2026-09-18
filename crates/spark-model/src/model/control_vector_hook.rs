// SPDX-License-Identifier: AGPL-3.0-only

//! Installing control vectors, and applying the one a request selected at the
//! end of every layer.
//!
//! See `docs/design/qwen4exp-control-vectors.md`. Entry points:
//!
//! * [`TransformerModel::load_control_vector`] — boot-time, once per vector.
//! * [`TransformerModel::cvec_after_layer`] — called from EVERY model-level
//!   layer loop, immediately after `layer.prefill()` / `layer.decode()`
//!   returns.
//!
//! # Why the selection is a parameter
//!
//! `cvec_id` is passed in rather than read off the model, because it is a
//! property of the REQUEST, not of the engine. Every call site takes it from
//! the sequence it is processing, so a path cannot accidentally apply the
//! previous request's vector — the failure an engine-held "currently active"
//! field would have, silently and with plausible output.
//!
//! A batch is guaranteed uniform in `cvec_id` by the admission cohort filter
//! (`spark-server/src/scheduler/admission.rs`), which is the same mechanism
//! that keeps a LoRA batch single-adapter. That guarantee is what lets a
//! multi-sequence path read the id from its first sequence.
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
    /// Load and register a control vector under `name`. Boot-time only; any
    /// failure aborts.
    ///
    /// Refuses a model with no mHC highway — the intervention site is
    /// `hc_streams`, which only exists when `hc_mult > 0`, so a non-mHC
    /// architecture has nowhere to put this and would otherwise accept the
    /// flag and do nothing.
    pub fn load_control_vector(&mut self, name: &str, spec: &ControlVectorSpec) -> Result<u64> {
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
        self.control_vectors.insert(name.to_string(), cv)
    }

    /// Whether any control vector is registered.
    #[inline]
    pub fn has_control_vectors(&self) -> bool {
        !self.control_vectors.is_empty()
    }

    /// Resolve a control vector by NAME to its id, for a request's selection.
    /// `None` when the name is not registered — the caller turns that into a
    /// 400 rather than silently serving unsteered.
    pub fn control_vector_id(&self, name: &str) -> Option<u64> {
        self.control_vectors.by_name(name).map(|e| e.id)
    }

    /// The registered names, for error messages and `GET /v1/models`.
    pub fn control_vector_names(&self) -> Vec<String> {
        self.control_vectors.names().map(str::to_string).collect()
    }

    /// Apply the control vector `cvec_id` selects to layer `layer_idx`'s
    /// output. `cvec_id == 0` means the request opted out and this is a no-op.
    ///
    /// Call site: immediately after `layer.prefill()`/`layer.decode()` returns
    /// in a model-level layer loop, where the highway holds that layer's
    /// output and the next layer has not yet read it.
    ///
    /// Graph-capture legal: one stream-ordered launch against a boot-time
    /// allocation, no synchronize and no environment read. NOTE that a
    /// captured graph bakes in the vector pointer, so a serve with any vector
    /// registered runs decode eagerly — see `decode_graphs_allowed`.
    #[inline]
    pub(crate) fn cvec_after_layer(
        &self,
        ctx: &ForwardContext<'_>,
        path: &'static str,
        cvec_id: u64,
        layer_idx: usize,
        num_tokens: usize,
        stream: u64,
    ) -> Result<()> {
        if cvec_id == 0 || num_tokens == 0 {
            return Ok(());
        }
        let Some(cv) = self.control_vectors.resolve(cvec_id) else {
            // An id that resolves to nothing means the request was admitted
            // against a registry that no longer holds its vector. Steering
            // silently less than asked is the failure this feature must not
            // have, so it is an error, not a skip.
            anyhow::bail!(
                "control vector id {cvec_id:#x} is not registered (path={path}, \
                 layer={layer_idx}); registered: {:?}",
                self.control_vector_names()
            );
        };
        if !cv.applies_to(layer_idx) {
            return Ok(());
        }
        let (highway, pre) = self.cvec_site(ctx, cv, layer_idx, num_tokens, stream)?;
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
                "CVEC_PROBE path={path} cvec={cvec_id:#x} layer={layer_idx} \
                 rows={num_tokens} row_offset={} pre={pre:.6} post={post:.6}",
                ctx.hc_row_offset
            );
        }
        Ok(())
    }

    /// The one place the highway base is computed, plus the optional `pre`
    /// probe reading taken before anything writes it.
    ///
    /// `hc_row_offset` is the row this pass owns — non-zero for mixed
    /// decode+prefill steps and the K-row verify — and the highway is
    /// row-major `[row, hc_mult, H]` FP32.
    fn cvec_site(
        &self,
        ctx: &ForwardContext<'_>,
        cv: &ControlVector,
        layer_idx: usize,
        num_tokens: usize,
        stream: u64,
    ) -> Result<(DevicePtr, f32)> {
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
        Ok((highway, pre))
    }

    /// Whether decode CUDA graphs may be captured.
    ///
    /// A captured graph bakes in the control vector's device pointer and
    /// scale, so replaying it for a request that selected a DIFFERENT vector
    /// (or none) would steer with the wrong one — silently, since the output
    /// is well-formed either way. LoRA has the same hazard and answers it the
    /// same way: rotating the active adapter destroys captured graphs and
    /// requires eager decode.
    ///
    /// Measured on GB10, graphs are speed-NEUTRAL for qwen4_exp (16.4 replay
    /// vs 16.5 eager), so this costs essentially nothing on the hardware this
    /// model is served on.
    #[inline]
    pub fn decode_graphs_allowed(&self) -> bool {
        self.control_vectors.is_empty()
    }
}
