// SPDX-License-Identifier: AGPL-3.0-only

//! Drain the MoE expert-telemetry staging into a request's accumulator.
//!
//! The device side (`layers::moe::telemetry`) stages every MoE layer's
//! routing for the pass; this reads it back once, after the pass, and folds
//! it into the `SequenceState` of the request that asked for it.
//!
//! Called only from the PREFILL entry points. Decode passes are not
//! attributed in v1: batched decode invokes the FFN once per sequence with
//! the offset baked into the input pointer and no row index reaching the
//! layer, so a staged row cannot be tied back to a sequence. Prefill has one
//! sequence and rows `0..n`, which is exact. The response reports this scope
//! so a consumer cannot mistake prompt routing for whole-request routing.

use crate::model::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Fold this prefill pass's staged routing into `seq`.
    ///
    /// `rows` is the number of token positions the pass ran. Never returns
    /// an error to the caller: telemetry is observational, and a request
    /// that asked to report experts should still get its completion if the
    /// readback fails. A failure is logged and leaves the accumulator short,
    /// which the conservation check downstream will show.
    pub(crate) fn drain_expert_telemetry(&self, seq: &mut SequenceState, rows: usize, stream: u64) {
        let Some(staging) = self.expert_telemetry.as_ref() else {
            return;
        };
        if seq.expert_activation.is_none() {
            // The request did not ask to report experts. The staging was
            // still written (the copies are recorded per layer, not per
            // request) — it is simply not read.
            return;
        }
        let gpu = self.gpu.as_ref();
        // The prefill dispatch already synchronized to read logits; this
        // guard is for the paths that might not, since a D2H that races the
        // routing kernels would read the PREVIOUS pass's ids and present
        // them as this prompt's.
        if let Err(e) = gpu.synchronize(stream) {
            tracing::warn!("expert telemetry: sync failed, routing not recorded: {e}");
            return;
        }

        let staged = staging.rows_for(rows);
        let (ids, weights) = match staging.drain(gpu, staged) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("expert telemetry: readback failed: {e}");
                return;
            }
        };

        let top_k = staging.top_k();
        let acc = seq
            .expert_activation
            .as_mut()
            .expect("checked above that the request reports experts");
        if rows > staged {
            // A prompt wider than the staging buffer. Recorded rather than
            // dropped: the difference between "these are the experts this
            // prompt used" and "these are the experts its first N tokens
            // used" is exactly what a category mapping would get wrong.
            acc.note_unattributed_rows((rows - staged) as u64);
        }
        let per_layer = staged * top_k;
        if per_layer == 0 {
            return;
        }
        for layer in 0..self.layers.len() {
            let base = layer * per_layer;
            if base + per_layer > ids.len() {
                break;
            }
            for row in 0..staged {
                let lo = base + row * top_k;
                acc.fold_row(layer, &ids[lo..lo + top_k], &weights[lo..lo + top_k]);
            }
        }
    }
}
