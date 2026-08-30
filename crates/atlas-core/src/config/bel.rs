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

/// One category folded into a plan, with the coverage its table was measured
/// at. A plan built from several categories keeps them all: a union has no
/// single coverage, and reporting one of them would misdescribe the rest.
#[derive(Debug, Clone, PartialEq)]
pub struct CategorySource {
    pub name: String,
    pub coverage: f32,
    /// `(layer, expert ids)` for this category alone.
    pub layers: Vec<(usize, Vec<u16>)>,
}

/// The resident-expert plan for one serve.
#[derive(Debug, Clone, PartialEq)]
pub struct BelPlan {
    /// The categories this plan was built from, in the order given on the
    /// command line. More than one means the UNION of their expert sets: a
    /// serve told to handle both Python and SQL must hold what either needs,
    /// since a request does not announce its category.
    pub sources: Vec<CategorySource>,
    /// `allowed[layer][expert]`. `None` for a layer no source mentions, which
    /// means "load this layer in full" — a MoE layer missing from the table is
    /// rejected at boot, so the only `None`s here are non-MoE layers.
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
        Self::from_sources(
            vec![CategorySource {
                name: category.into(),
                coverage,
                layers: layers.into_iter().collect(),
            }],
            num_layers,
            num_experts,
        )
    }

    /// Build the UNION of several categories' expert sets.
    ///
    /// Union, not intersection: the serve cannot know which category a request
    /// belongs to, so it must hold everything any of the named categories
    /// routes to. A layer is restricted only if at least one source restricts
    /// it, and an expert is resident if any source keeps it.
    ///
    /// `num_layers` and `num_experts` come from the loaded model's config, not
    /// from the tables, so a table measured on a different checkpoint is
    /// rejected here rather than silently indexing a different expert space.
    pub fn from_sources(
        sources: Vec<CategorySource>,
        num_layers: usize,
        num_experts: usize,
    ) -> Result<Self, String> {
        if sources.is_empty() {
            return Err("an expert-loading plan needs at least one category".to_string());
        }
        let mut allowed: Vec<Option<Box<[bool]>>> = vec![None; num_layers];
        for src in &sources {
            let category = &src.name;
            for (layer, ids) in &src.layers {
                let layer = *layer;
                if layer >= num_layers {
                    return Err(format!(
                        "expert category '{category}' names layer {layer}, but the model has \
                         {num_layers} layers — the table was measured on a different checkpoint"
                    ));
                }
                let row = allowed[layer]
                    .get_or_insert_with(|| vec![false; num_experts].into_boxed_slice());
                for id in ids {
                    let e = *id as usize;
                    if e >= num_experts {
                        return Err(format!(
                            "expert category '{category}' layer {layer} names expert {e}, but \
                             the model has {num_experts} experts — the table was measured on a \
                             different checkpoint"
                        ));
                    }
                    row[e] = true;
                }
            }
        }
        Ok(Self {
            sources,
            allowed,
            num_experts,
        })
    }

    /// Human label for logs and for the response payload: the category, or
    /// the categories joined by `+` when this is a union.
    pub fn label(&self) -> String {
        self.sources
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join("+")
    }

    /// The coverage every source was measured at, or `None` when they differ —
    /// a union of tables generated at different thresholds has no single
    /// coverage, and reporting one of them would misdescribe the others.
    pub fn uniform_coverage(&self) -> Option<f32> {
        let first = self.sources.first()?.coverage;
        self.sources
            .iter()
            .all(|s| s.coverage == first)
            .then_some(first)
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
