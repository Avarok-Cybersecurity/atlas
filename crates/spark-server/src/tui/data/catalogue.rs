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

use crate::recipe::Recipe;
use crate::tui::data::library::{LibraryEntry, human_size};

/// One row: a recipe, a local checkpoint, or both.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The HuggingFace id — the join key, and what `spark serve` is given.
    pub model: String,
    pub recipe: Option<Recipe>,
    pub local: Option<LibraryEntry>,
}

impl Entry {
    pub fn has_recipe(&self) -> bool {
        self.recipe.is_some()
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
        self.has_weights() && self.recipe.as_ref().is_some_and(Recipe::is_atlas)
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
        if let Some(r) = &self.recipe {
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
            self.recipe
                .as_ref()
                .map(|r| r.id.to_lowercase())
                .unwrap_or_default(),
            self.local
                .as_ref()
                .map(|l| l.model_type.to_lowercase())
                .unwrap_or_default(),
        ];
        hay.iter().any(|h| h.contains(&needle))
    }
}

/// Join recipes and local checkpoints into one sorted list.
///
/// A model with several recipes yields several rows — the recipes differ in
/// quantization and context, which is exactly the choice being offered, so
/// collapsing them would hide it.
pub fn join(recipes: &[Recipe], local: &[LibraryEntry]) -> Vec<Entry> {
    let mut rows: Vec<Entry> = Vec::new();

    for recipe in recipes {
        let matched = local.iter().find(|l| l.id == recipe.model).cloned();
        rows.push(Entry {
            model: recipe.model.clone(),
            recipe: Some(recipe.clone()),
            local: matched,
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
            recipe: None,
            local: Some(entry.clone()),
        });
    }

    rows.sort_by(|a, b| {
        a.rank()
            .cmp(&b.rank())
            .then_with(|| a.model.cmp(&b.model))
            .then_with(|| {
                let ar = a.recipe.as_ref().map(|r| r.id.as_str()).unwrap_or("");
                let br = b.recipe.as_ref().map(|r| r.id.as_str()).unwrap_or("");
                ar.cmp(br)
            })
    });
    rows
}

#[cfg(test)]
#[path = "catalogue_tests.rs"]
mod tests;
