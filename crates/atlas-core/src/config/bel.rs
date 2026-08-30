// SPDX-License-Identifier: AGPL-3.0-only

//! Boot-time expert loading: which experts this serve will hold resident.
//!
//! Built once at boot from `--expert-category <name>` and the model's
//! MODEL.toml `[expert_categories]` table, then read by two places that must
//! agree exactly:
//!
//!  * the weight loaders, which skip the tensors of experts not in the plan;
//!  * `MoeLayer`, which masks those experts out of the router so the top-k
//!    cannot select one whose weights were never loaded.
//!
//! One plan, two readers — if they could disagree, the router would select an
//! expert backed by a null pointer and the serve would die mid-request.

/// The resident-expert plan for one serve.
#[derive(Debug, Clone, PartialEq)]
pub struct BelPlan {
    /// The category name as given on the command line, for logs and for the
    /// response payload — a request should be able to see which subset it
    /// was served by.
    pub category: String,
    /// The routing-mass coverage the table was measured at, carried so a log
    /// line can say how selective this plan is without re-deriving it.
    pub coverage: f32,
    /// `allowed[layer][expert]`. `None` for a layer the table does not
    /// mention, which means "load this layer in full" — a MoE layer missing
    /// from the table is rejected at boot, so the only `None`s here are
    /// non-MoE layers.
    allowed: Vec<Option<Box<[bool]>>>,
    num_experts: usize,
}

impl BelPlan {
    /// Build from a category's `(layer, expert ids)` table.
    ///
    /// `num_layers` and `num_experts` come from the loaded model's config,
    /// not the table, so a table measured on a different checkpoint is
    /// rejected here rather than silently indexing a different expert space.
    pub fn new(
        category: impl Into<String>,
        coverage: f32,
        num_layers: usize,
        num_experts: usize,
        layers: impl IntoIterator<Item = (usize, Vec<u16>)>,
    ) -> Result<Self, String> {
        let category = category.into();
        let mut allowed: Vec<Option<Box<[bool]>>> = vec![None; num_layers];
        for (layer, ids) in layers {
            if layer >= num_layers {
                return Err(format!(
                    "expert category '{category}' names layer {layer}, but the model has \
                     {num_layers} layers — the table was measured on a different checkpoint"
                ));
            }
            let mut row = vec![false; num_experts].into_boxed_slice();
            for id in ids {
                let e = id as usize;
                if e >= num_experts {
                    return Err(format!(
                        "expert category '{category}' layer {layer} names expert {e}, but the \
                         model has {num_experts} experts — the table was measured on a \
                         different checkpoint"
                    ));
                }
                row[e] = true;
            }
            allowed[layer] = Some(row);
        }
        Ok(Self {
            category,
            coverage,
            allowed,
            num_experts,
        })
    }

    /// Whether this layer's experts are restricted at all.
    pub fn restricts_layer(&self, layer: usize) -> bool {
        self.allowed.get(layer).is_some_and(Option::is_some)
    }

    /// Whether `expert` is resident in `layer`. A layer the plan does not
    /// mention is unrestricted, so everything in it is resident.
    pub fn is_loaded(&self, layer: usize, expert: usize) -> bool {
        match self.allowed.get(layer) {
            Some(Some(row)) => row.get(expert).copied().unwrap_or(false),
            _ => true,
        }
    }

    /// This layer's additive router mask: `0.0` for a resident expert,
    /// `-inf` for one that was never loaded. `None` for an unrestricted
    /// layer, which needs no mask at all.
    ///
    /// Additive and uploaded once, so a category listing every expert is a
    /// numerical no-op — the negative control a BEL run is judged against.
    pub fn router_mask(&self, layer: usize) -> Option<Vec<f32>> {
        let row = self.allowed.get(layer)?.as_ref()?;
        Some(
            (0..self.num_experts)
                .map(|e| {
                    if row.get(e).copied().unwrap_or(false) {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect(),
        )
    }

    /// Resident experts in `layer`, or `None` if unrestricted.
    pub fn layer_count(&self, layer: usize) -> Option<usize> {
        let row = self.allowed.get(layer)?.as_ref()?;
        Some(row.iter().filter(|b| **b).count())
    }

    /// `(resident, total)` summed over restricted layers — the memory story.
    pub fn totals(&self) -> (usize, usize) {
        let mut resident = 0;
        let mut total = 0;
        for layer in 0..self.allowed.len() {
            if let Some(n) = self.layer_count(layer) {
                resident += n;
                total += self.num_experts;
            }
        }
        (resident, total)
    }

    /// Layers this plan restricts, ascending.
    pub fn restricted_layers(&self) -> Vec<usize> {
        (0..self.allowed.len())
            .filter(|&l| self.restricts_layer(l))
            .collect()
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }
}

#[cfg(test)]
#[path = "bel_tests.rs"]
mod tests;
