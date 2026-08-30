// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request MoE expert-activation accumulator: the host-side half.
//!
//! One of these hangs off a `SequenceState` while a request is in flight.
//! The model folds each pass's staged routing into it (see
//! [`super::telemetry`]); the scheduler reads it when the sequence finishes
//! and puts the summary on the response.
//!
//! Dense storage — `[layer][expert]` counts and mass — because the fold runs
//! per token per layer and a hash lookup there would cost more than the
//! whole tap. A 61-layer, 384-expert model spends 187 KB per in-flight
//! request that asked for telemetry; requests that did not ask allocate
//! nothing.

/// What one request's tokens did to the router, per MoE layer.
#[derive(Clone, Debug)]
pub struct ExpertActivationAcc {
    num_layers: usize,
    num_experts: usize,
    top_k: u32,
    /// `[layer * num_experts + expert]` — how many routed token-slots chose
    /// this expert.
    counts: Vec<u32>,
    /// `[layer * num_experts + expert]` — summed post-renormalization
    /// routing weight. This is the quantity a category's expert set is
    /// budgeted on: an expert picked often but weakly contributes less to
    /// the layer's output than one picked rarely at high weight.
    mass: Vec<f32>,
    /// Routed token-slots folded so far, i.e. rows × layers. Lets a reader
    /// check `Σcounts == tokens_routed × top_k` and catch a tap that
    /// silently recorded nothing.
    tokens_routed: u64,
    /// Rows a pass dropped because they exceeded the staging width. Non-zero
    /// means the numbers describe a prefix of the request, and the summary
    /// says so rather than presenting itself as complete.
    dropped_rows: u64,
}

impl ExpertActivationAcc {
    pub fn new(num_layers: usize, num_experts: usize, top_k: u32) -> Self {
        Self {
            num_layers,
            num_experts,
            top_k,
            counts: vec![0; num_layers * num_experts],
            mass: vec![0.0; num_layers * num_experts],
            tokens_routed: 0,
            dropped_rows: 0,
        }
    }

    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn tokens_routed(&self) -> u64 {
        self.tokens_routed
    }

    pub fn dropped_rows(&self) -> u64 {
        self.dropped_rows
    }

    /// Record that `rows` rows of a pass could not be staged.
    pub fn note_dropped_rows(&mut self, rows: u64) {
        self.dropped_rows += rows;
    }

    /// Fold one row (one token position) of one layer.
    ///
    /// `ids` / `weights` are that row's `top_k` slots. A slot whose weight is
    /// zero is skipped: that is how a folded zero/identity expert and an
    /// unwritten staging tail both present, and neither is routing this
    /// request performed. An id at or beyond `num_experts` is likewise
    /// skipped — models with zero-computation experts (LongCat) score a
    /// wider set than they have expert weights for.
    pub fn fold_row(&mut self, layer_idx: usize, ids: &[u32], weights: &[f32]) {
        if layer_idx >= self.num_layers {
            return;
        }
        let base = layer_idx * self.num_experts;
        let mut folded_any = false;
        for (&id, &w) in ids.iter().zip(weights.iter()) {
            if w == 0.0 || !w.is_finite() {
                continue;
            }
            let e = id as usize;
            if e >= self.num_experts {
                continue;
            }
            self.counts[base + e] += 1;
            self.mass[base + e] += w;
            folded_any = true;
        }
        if folded_any {
            self.tokens_routed += 1;
        }
    }

    /// Per-layer sparse view: `(layer, [(expert, count, mass)])`, layers
    /// ascending, experts ascending within a layer, zero-count experts
    /// dropped. This is the shape the response carries — dense arrays of
    /// mostly zeros would be two orders of magnitude larger on the wire.
    pub fn to_layers(&self) -> Vec<(usize, Vec<(u32, u32, f32)>)> {
        let mut out = Vec::new();
        for layer in 0..self.num_layers {
            let base = layer * self.num_experts;
            let mut experts = Vec::new();
            for e in 0..self.num_experts {
                let c = self.counts[base + e];
                if c > 0 {
                    experts.push((e as u32, c, self.mass[base + e]));
                }
            }
            if !experts.is_empty() {
                out.push((layer, experts));
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "activation_acc_tests.rs"]
mod tests;
