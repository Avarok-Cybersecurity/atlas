// SPDX-License-Identifier: AGPL-3.0-only

//! Drain the MoE expert-telemetry staging into a request's accumulator.
//!
//! The device side (`layers::moe::telemetry`) stages every MoE layer's
//! routing for the pass; this reads it back once, after the pass, and folds
//! it into the `SequenceState` of the request that asked for it.
//!
//! Prefill and decode are drained separately because they attribute
//! differently. A prefill pass is one sequence over rows `0..n`; a decode
//! step is one row per sequence, and which sequence owns row `i` is
//! `seqs[i]` — resolved on the host at drain time, which is why no
//! device-side sequence identity is needed. The row index reaches the layer
//! as an explicit argument through `FfnComponent::forward`.
//!
//! MTP verify rows are staged but NOT folded: a rejected draft token's
//! routing belongs to a token that was rolled back, and the acceptance count
//! that would say which rows survived is computed per verify variant. Those
//! rows are counted as unattributed rather than dropped, so a consumer of an
//! MTP-on serve can see that decode coverage is partial instead of inferring
//! it from a suspiciously small number.

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

    /// Fold one decode step's staged routing into the sequences that produced
    /// it, row by row.
    ///
    /// Row `i` belongs to `seqs[i]` by construction: the dispatchers place
    /// sequence `i`'s token at row `i` of the batch, and the layer stages
    /// under the row index its caller passed. The mapping is resolved here,
    /// on the host, per step — so it cannot go stale the way a device-side
    /// map uploaded before a graph replay could.
    ///
    /// Never returns an error: telemetry is observational, and a request that
    /// asked to report experts should still get its completion if the
    /// readback fails.
    pub(crate) fn drain_decode_expert_telemetry(
        &self,
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) {
        let Some(staging) = self.expert_telemetry.as_ref() else {
            return;
        };
        if !seqs.iter().any(|s| s.expert_activation.is_some()) {
            // Nobody in this batch asked. The staging was still written — the
            // copies are recorded per layer, not per request — it is simply
            // not read.
            return;
        }
        let gpu = self.gpu.as_ref();
        if let Err(e) = gpu.synchronize(stream) {
            tracing::warn!("expert telemetry: decode sync failed, step not recorded: {e}");
            return;
        }
        let rows = staging.rows_for(seqs.len());
        let (ids, weights) = match staging.drain(gpu, rows) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!("expert telemetry: decode readback failed: {e}");
                return;
            }
        };
        let top_k = staging.top_k();
        let per_layer = rows * top_k;
        if per_layer == 0 {
            return;
        }
        for (row, seq) in seqs.iter_mut().enumerate().take(rows) {
            let Some(acc) = seq.expert_activation.as_mut() else {
                continue;
            };
            for layer in 0..self.layers.len() {
                let base = layer * per_layer;
                if base + per_layer > ids.len() {
                    break;
                }
                let lo = base + row * top_k;
                acc.fold_decode_row(layer, &ids[lo..lo + top_k], &weights[lo..lo + top_k]);
            }
        }
        // A batch wider than the staging buffer: the tail sequences ran but
        // were not staged, and saying so is the difference between "these
        // sequences used no experts" and "we did not look".
        for seq in seqs.iter_mut().skip(rows) {
            if let Some(acc) = seq.expert_activation.as_mut() {
                acc.note_decode_unattributed_rows(1);
            }
        }
    }

    /// Record that `rows` decode/verify positions ran without being folded.
    ///
    /// The MTP verify path calls this: its rows are staged but excluded from
    /// folding in v1, and an excluded row that went uncounted would read as a
    /// request that simply routed less.
    pub(crate) fn note_decode_unattributed(&self, seq: &mut SequenceState, rows: u64) {
        if self.expert_telemetry.is_none() {
            return;
        }
        if let Some(acc) = seq.expert_activation.as_mut() {
            acc.note_decode_unattributed_rows(rows);
        }
    }
}
