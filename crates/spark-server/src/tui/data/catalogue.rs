// SPDX-License-Identifier: AGPL-3.0-only

//! The Library's row model: recipes joined against locally cached weights.
//!
//! Two independent lists become one, keyed on the HuggingFace id. The join is
//! outer in both directions on purpose — the three states are genuinely
//! different things a user needs to tell apart:
//!
//! * **recipe + weights** — runnable right now.
//! * **recipe, no weights** — runnable after a download.
//! * **weights, no recipe** — servable, but you are on your own for flags.
//!
//! Sorted in exactly that order, so "what can I run right now" is the top of
//! the list rather than something to scroll for.
//!
//! **One row per MODEL, not per recipe.** Several recipes for one checkpoint
//! differ in quantization, context and speculation — a real choice, but not one
//! that belongs in a list where the model id then repeats three times with only
//! a trailing stem to tell them apart. The choice moves to a card view behind
//! the row, which has room to show each recipe's measured rationale.

use crate::recipe::Recipe;
use crate::tui::data::library::{LibraryEntry, human_size};

/// One row: every recipe for a model, its local checkpoint, or both.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The HuggingFace id — the join key, and what `spark serve` is given.
    pub model: String,
    /// Every recipe naming this model, in id order. Empty for a local-only row.
    pub recipes: Vec<Recipe>,
    pub local: Option<LibraryEntry>,
}

impl Entry {
    pub fn has_recipe(&self) -> bool {
        !self.recipes.is_empty()
    }

    /// The recipe to describe the row by, when only one can be shown.
    ///
    /// The first Atlas recipe, falling back to the first of any kind: a vLLM
    /// recipe still carries the description and params worth rendering, it just
    /// cannot be launched from here.
    pub fn primary(&self) -> Option<&Recipe> {
        self.recipes
            .iter()
            .find(|r| r.is_atlas())
            .or_else(|| self.recipes.first())
    }

    pub fn has_weights(&self) -> bool {
        self.local.as_ref().is_some_and(|l| l.has_weights)
    }

    /// Whether a compiled kernel target exists for this checkpoint.
    ///
    /// **Deliberately independent of `has_recipe`.** A recipe with no compiled
    /// target still serves, on generic kernels — conflating the two badges
    /// would tell the user their model is unsupported when it is merely
    /// unoptimized.
    pub fn optimized(&self) -> bool {
        self.local.as_ref().is_some_and(|l| l.optimized)
    }

    /// Ready to serve with a validated config and no download.
    pub fn runnable_now(&self) -> bool {
        self.has_weights() && self.recipes.iter().any(Recipe::is_atlas)
    }

    /// Sort key: runnable, then recipe-without-weights, then local-only.
    fn rank(&self) -> u8 {
        match (self.runnable_now(), self.has_recipe(), self.has_weights()) {
            (true, _, _) => 0,
            (_, true, _) => 1,
            _ => 2,
        }
    }

    /// The size on disk, or a dash when the weights are not here.
    pub fn size_text(&self) -> String {
        match &self.local {
            Some(l) if l.has_weights => human_size(l.size_bytes),
            Some(_) => "partial".into(),
            None => "—".into(),
        }
    }

    /// The one-line subtitle under the model id.
    pub fn subtitle(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(r) = self.primary() {
            if !r.model_params.is_empty() {
                parts.push(r.model_params.clone());
            }
            if !r.quantization.is_empty() {
                parts.push(r.quantization.clone());
            }
            if !r.category.is_empty() {
                parts.push(r.category.clone());
            }
        }
        if let Some(l) = &self.local {
            if !l.model_type.is_empty() {
                parts.push(l.model_type.clone());
            }
            if l.layers > 0 {
                parts.push(format!("{}L", l.layers));
            }
        }
        parts.dedup();
        parts.join(" · ")
    }

    /// Does this row match a filter? Matches the id, the recipe id and the
    /// architecture, because all three are things people type.
    pub fn matches(&self, needle: &str) -> bool {
        if needle.is_empty() {
            return true;
        }
        let needle = needle.to_lowercase();
        let hay = [
            self.model.to_lowercase(),
            // Every recipe id, so a search for a specific stem still finds the
            // model row that now hides it.
            self.recipes
                .iter()
                .map(|r| r.id.to_lowercase())
                .collect::<Vec<_>>()
                .join(" "),
            self.local
                .as_ref()
                .map(|l| l.model_type.to_lowercase())
                .unwrap_or_default(),
        ];
        hay.iter().any(|h| h.contains(&needle))
    }
}

/// Join recipes and local checkpoints into one sorted list, **one row per
/// model**.
///
/// A model with several recipes yields ONE row carrying all of them. The
/// recipes differ in quantization, context and speculation — a real choice, and
/// one the card view behind the row presents properly. Repeating the model id
/// once per recipe put that choice in the wrong place.
pub fn join(recipes: &[Recipe], local: &[LibraryEntry]) -> Vec<Entry> {
    let mut rows: Vec<Entry> = Vec::new();

    // BTreeMap so the grouped recipes land in a stable id order regardless of
    // the order the fetch returned them in.
    let mut by_model: std::collections::BTreeMap<&str, Vec<Recipe>> =
        std::collections::BTreeMap::new();
    for recipe in recipes {
        by_model
            .entry(recipe.model.as_str())
            .or_default()
            .push(recipe.clone());
    }
    for (model, mut group) in by_model {
        group.sort_by(|a, b| a.id.cmp(&b.id));
        rows.push(Entry {
            model: model.to_string(),
            recipes: group,
            local: local.iter().find(|l| l.id == model).cloned(),
        });
    }
    // Local checkpoints no recipe covers still belong in the list: they are
    // servable, and omitting them would make the Library disagree with the
    // cache the user can see on disk.
    for entry in local {
        if recipes.iter().any(|r| r.model == entry.id) {
            continue;
        }
        rows.push(Entry {
            model: entry.id.clone(),
            recipes: Vec::new(),
            local: Some(entry.clone()),
        });
    }

    rows.sort_by(|a, b| a.rank().cmp(&b.rank()).then_with(|| a.model.cmp(&b.model)));
    rows
}

#[cfg(test)]
#[path = "catalogue_tests.rs"]
mod tests;
