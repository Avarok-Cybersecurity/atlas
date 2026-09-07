// SPDX-License-Identifier: AGPL-3.0-only

//! Which MoE experts a request's tokens routed to.
//!
//! Produced when a request sets `report_expert_metadata` on a serve started
//! with `--expert-telemetry`, and carried on `usage.expert_activation`. The
//! `expert-categories` benchmark aggregates it over a category's prompts to
//! decide which experts that category needs; `--expert-category` later loads
//! only those.

/// One MoE layer's activations, as parallel arrays.
///
/// Parallel rather than a list of triples because the arrays are the bulk of
/// the payload: a 61-layer model with ~200 distinct experts per layer is
/// ~12k entries, and `[[17,240,12.6],…]` spends a bracket pair per expert.
/// All three arrays have the same length and are ordered by ascending
/// expert id.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertLayerActivation {
    /// Absolute model layer index — the numbering a MODEL.toml
    /// `[expert_categories] layers."<L>"` key uses.
    pub layer: usize,
    pub experts: Vec<u32>,
    /// Routed token-slots that chose each expert.
    pub counts: Vec<u32>,
    /// Summed post-renormalization routing weight per expert. This is what
    /// an expert set is budgeted on (`atlas_core::moe_policy::budget_experts`)
    /// — an expert chosen often but weakly contributes less than one chosen
    /// rarely at high weight.
    pub mass: Vec<f32>,
}

impl ExpertLayerActivation {
    /// Sum another turn's counts and mass into this layer, keeping expert
    /// ids ascending so the arrays stay in the order consumers expect.
    fn merge(&mut self, other: &Self) {
        for (i, &e) in other.experts.iter().enumerate() {
            match self.experts.binary_search(&e) {
                Ok(at) => {
                    self.counts[at] += other.counts[i];
                    self.mass[at] += other.mass[i];
                }
                Err(at) => {
                    self.experts.insert(at, e);
                    self.counts.insert(at, other.counts[i]);
                    self.mass.insert(at, other.mass[i]);
                }
            }
        }
    }
}

/// The whole per-request report.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertActivationReport {
    /// Which part of the request these numbers describe:
    /// `"prefill+decode"` when the serve attributes both, `"prefill"` when it
    /// can only attribute the prompt. Stated on the wire because the two are
    /// not interchangeable — a consumer that assumed whole-request coverage
    /// from a prefill-only report would under-count every generated token.
    pub scope: &'static str,
    pub top_k: u32,
    pub num_experts: u32,
    /// Token positions folded into the counts. `Σcounts == tokens_routed *
    /// top_k` when every routed slot carried weight — the check that
    /// separates "this prompt used few experts" from "the tap recorded
    /// nothing".
    pub tokens_routed: u64,
    /// Token positions that ran but are NOT in the counts (a prompt wider
    /// than the staging buffer, or the non-final chunks of a two-phase
    /// prefill). Non-zero means the report covers a prefix.
    pub unattributed_rows: u64,
    /// Of `tokens_routed`, how many came from DECODE rather than the prompt.
    pub decode_tokens_routed: u64,
    /// Decode positions that ran without being folded: MTP verify rows, which
    /// v1 stages but does not attribute because a rejected draft's routing
    /// belongs to a rolled-back token. Non-zero on a speculating serve, and
    /// the number a consumer needs to tell partial decode coverage from a
    /// request that genuinely routed little.
    pub decode_unattributed_rows: u64,
    /// Only layers that routed appear; dense layers of a hybrid model are
    /// absent rather than present-and-empty.
    pub layers: Vec<ExpertLayerActivation>,
}

impl ExpertActivationReport {
    /// Fold another turn's report into this one.
    ///
    /// A tool-calling request prefills several times — once per turn — and
    /// each prefill produces its own report. Summing them gives the experts
    /// the REQUEST needed, which is what a category mapping is about; taking
    /// the last turn's alone would silently describe only the final prompt.
    pub fn merge(&mut self, other: &Self) {
        self.tokens_routed += other.tokens_routed;
        self.unattributed_rows += other.unattributed_rows;
        self.decode_tokens_routed += other.decode_tokens_routed;
        self.decode_unattributed_rows += other.decode_unattributed_rows;
        // A merge keeps the WEAKER scope: if either turn could not attribute
        // decode, the merged report cannot claim it did.
        if other.scope == "prefill" {
            self.scope = "prefill";
        }
        for layer in &other.layers {
            match self.layers.iter_mut().find(|l| l.layer == layer.layer) {
                Some(existing) => existing.merge(layer),
                None => {
                    let at = self.layers.partition_point(|l| l.layer < layer.layer);
                    self.layers.insert(at, layer.clone());
                }
            }
        }
    }

    /// Build from the model-side accumulator.
    pub fn from_acc(acc: &spark_model::layers::ExpertActivationAcc) -> Self {
        let layers = acc
            .to_layers()
            .into_iter()
            .map(|(layer, experts)| {
                let mut ids = Vec::with_capacity(experts.len());
                let mut counts = Vec::with_capacity(experts.len());
                let mut mass = Vec::with_capacity(experts.len());
                for (e, c, m) in experts {
                    ids.push(e);
                    counts.push(c);
                    mass.push(m);
                }
                ExpertLayerActivation {
                    layer,
                    experts: ids,
                    counts,
                    mass,
                }
            })
            .collect();
        Self {
            // The accumulator only sees decode rows on a build that drains
            // them, so the presence of decode attribution IS the scope.
            scope: if acc.decode_tokens_routed() > 0 || acc.decode_unattributed_rows() > 0 {
                "prefill+decode"
            } else {
                "prefill"
            },
            top_k: acc.top_k(),
            num_experts: acc.num_experts() as u32,
            tokens_routed: acc.tokens_routed(),
            unattributed_rows: acc.unattributed_rows(),
            decode_tokens_routed: acc.decode_tokens_routed(),
            decode_unattributed_rows: acc.decode_unattributed_rows(),
            layers,
        }
    }
}
