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
    /// Agree with the other ranks about what is registered, at BOOT.
    ///
    /// The per-request check in `impl_a2` already refuses an id it cannot
    /// resolve, so a divergence never steers half the model. But it refuses
    /// *during a request*, and the head has no path to turn that into a
    /// response — the caller waits until its own timeout. Failing closed as a
    /// thirty-minute hang is still a bad failure.
    ///
    /// This removes the possibility instead of improving the symptom. If the
    /// registries agree here, no id the head can later send is unresolvable on
    /// a worker, so the per-request check becomes unreachable — which is what
    /// it should have been all along. It stays in place regardless: cheap, and
    /// "unreachable" is a claim about today's call graph, not a guarantee.
    ///
    /// Placed before the worker enters its command loop, where both ranks run
    /// the same code and a collective is a natural barrier.
    pub fn ep_check_control_vector_registry(&self) -> Result<()> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(()); // single process: nobody to disagree with
        };
        let world = comm.world_size();
        if world <= 1 {
            return Ok(());
        }
        let local = self.control_vectors.fingerprint();

        // All-gather, not broadcast-and-compare. A broadcast tells the workers
        // what the head has, so only a worker can detect a divergence — the
        // head proceeds, logs agreement it never verified, and serves on with a
        // dead worker. Requests then hang exactly as before, one layer further
        // out. Every rank has to be able to refuse, so every rank needs every
        // fingerprint.
        //
        // `all_gather` is the right primitive because it moves `Uint8`: raw
        // bytes, reduced by nothing. `all_reduce` is hard-wired to bf16 (it is
        // the activation reducer) and would put these through a float.
        let send = self.gpu.alloc(8)?;
        let recv = self.gpu.alloc(8 * world)?;
        let gathered = (|| -> Result<Vec<u64>> {
            self.gpu.copy_h2d(&local.to_le_bytes(), send)?;
            comm.all_gather(send.0, recv.0, 8)?;
            self.gpu.synchronize(self.gpu.default_stream())?;
            let mut buf = vec![0u8; 8 * world];
            self.gpu.copy_d2h(recv, &mut buf)?;
            Ok(buf
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect())
        })();
        // Free both before propagating: a boot that is about to fail should not
        // also leak, and these are the only two allocations on this path.
        let free_err = self.gpu.free(send).and_then(|()| self.gpu.free(recv));
        let gathered = gathered?;
        free_err?;

        if gathered.iter().any(|&f| f != local) {
            let mine = self
                .control_vectors
                .entries()
                .map(|e| {
                    format!(
                        "  '{}' id={:#018x} {}",
                        e.name,
                        e.id,
                        e.vector.config_identity()
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mine = if mine.is_empty() {
                "  (none)".to_string()
            } else {
                mine
            };
            let table = gathered
                .iter()
                .enumerate()
                .map(|(r, f)| {
                    format!(
                        "  rank {r}: {f:#018x}{}",
                        if r == comm.rank() { " (this rank)" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            anyhow::bail!(
                "control-vector registries differ across ranks:\n{table}\nThe \
                 fingerprint covers every registered vector's NAME and its \
                 CONFIGURATION — file sha256, mode, scale, layer range — so this \
                 catches a different file, scale, mode or range under the same name, \
                 not just a missing one.\nThis rank has:\n{mine}\nCompare against the \
                 other ranks' `control vector '<name>' identity:` lines. Refusing to \
                 start: the alternative is a request that hangs when it first selects \
                 a vector."
            );
        }
        if local != 0 {
            tracing::info!(
                "control-vector registry agrees across {world} ranks \
                 (fingerprint {local:#018x})"
            );
        }
        Ok(())
    }

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

    /// Arm per-layer activation capture for DERIVING a vector.
    ///
    /// Same mHC requirement as installing one: the thing being measured is the
    /// highway, so a model without one has nothing to capture.
    pub fn arm_control_vector_capture(&mut self) -> Result<()> {
        anyhow::ensure!(
            self.config.hc_mult > 0,
            "control-vector capture needs an mHC highway to measure, and \
             model_type {:?} has hc_mult = 0",
            self.config.model_type
        );
        self.cvec_capture = Some(crate::control_vector_capture::ControlVectorCapture::new(
            self.gpu.as_ref(),
            self.config.num_hidden_layers,
            self.config.hidden_size,
        )?);
        Ok(())
    }

    /// Zero the capture accumulator — run between the positive and negative
    /// corpus passes.
    pub fn reset_control_vector_capture(&self) -> Result<()> {
        let cap = self
            .cvec_capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("control-vector capture is not armed"))?;
        cap.reset(self.gpu.as_ref())
    }

    /// Write the capture accumulator to `path`; returns the token count it
    /// represents (the divisor for the mean).
    pub fn dump_control_vector_capture(&self, path: &std::path::Path) -> Result<u64> {
        let cap = self
            .cvec_capture
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("control-vector capture is not armed"))?;
        cap.dump(self.gpu.as_ref(), path)
    }

    /// Tokens folded into the capture so far, or `None` when not armed.
    pub fn control_vector_capture_tokens(&self) -> Option<u64> {
        self.cvec_capture.as_ref().map(|c| c.tokens())
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
        if num_tokens == 0 {
            return Ok(());
        }
        // Capture runs FIRST and unconditionally, because derivation must see
        // UNSTEERED activations — measuring the model after steering it would
        // fold the vector back into the direction derived from it. It also runs
        // before the `cvec_id == 0` return, since a derivation pass by
        // definition selects no vector.
        if let Some(cap) = self.cvec_capture.as_ref() {
            let stride = ctx.config.hc_mult * ctx.config.hidden_size * 4;
            let base = DevicePtr(ctx.buffers.hc_streams().0 + (ctx.hc_row_offset * stride) as u64);
            cap.accumulate(
                ctx.gpu,
                base,
                layer_idx,
                num_tokens,
                ctx.config.hc_mult,
                stream,
            )?;
        }
        if cvec_id == 0 {
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
