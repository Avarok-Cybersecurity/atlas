// SPDX-License-Identifier: AGPL-3.0-only

//! Starting points: what Enter offers on a model no recipe covers.
//!
//! A local-only checkpoint used to be a dead end — the list said "no recipe"
//! and Enter refused, leaving the user to reconstruct a serve command from
//! scratch outside the dashboard. This module synthesizes candidate configs
//! to start FROM: the published recipes re-aimed at this model, plus one
//! blank card that is nothing but the server's own defaults.
//!
//! None of it is a measurement. A published recipe's description is a
//! gate-measured rationale for one checkpoint; copying its settings onto a
//! different checkpoint keeps the settings and loses the evidence. Every
//! synthesized card therefore carries `Recipe::starting_point`, its
//! description says what was copied from where, and the donor's `updated`
//! date is dropped — a date on the card would read as "verified then".

use crate::recipe::Recipe;
use crate::tui::data::catalogue::Entry;

use super::lib_state::LibState;

/// Lowercased with separators removed, so `qwen3.6`, `qwen3_6_moe` and
/// `Qwen3.6-27B` can find each other. Family names and HF ids disagree about
/// separators far more than about letters.
fn norm(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

/// Does this donor look like the same family as the target model?
///
/// The recipe id's leading path segment is its family (`qwen3.6/...`), and it
/// is matched against both the HF id and the checkpoint's `model_type` — the
/// architecture string is the stronger signal when the HF id is renamed
/// beyond recognition, which quantized re-uploads routinely are.
fn same_family(donor: &Recipe, model: &str, model_type: &str) -> bool {
    let family = norm(donor.id.split('/').next().unwrap_or_default());
    if family.is_empty() {
        return false;
    }
    norm(model).contains(&family) || norm(model_type).contains(&family)
}

/// One donor recipe, re-aimed at `model` and marked as the guess it is.
fn template_from(donor: &Recipe, model: &str) -> Recipe {
    let mut t = donor.clone();
    t.model = model.to_string();
    t.starting_point = Some(donor.id.clone());
    t.description = format!(
        "Starting point, not a measurement: settings copied from {}, which was \
         measured on {} — none of it has been verified on {}. Review each value \
         before launching.",
        donor.id, donor.model, model
    );
    // The donor's date describes the donor's file. On this card it would read
    // as "this combination was verified then", which nothing ever did. The
    // same goes for the rest of the donor's checkpoint metadata: maintainer,
    // parameter count and quantization describe the donor's model, and left
    // in place they would caption THIS card with another checkpoint's facts.
    // The `defaults:` settings are the only thing genuinely being offered.
    t.updated.clear();
    t.maintainer.clear();
    t.category.clear();
    t.model_params.clear();
    t.quantization.clear();
    t.kv_dtype.clear();
    t
}

/// The no-donor card: serve with every flag at the server's own default.
fn blank(model: &str) -> Recipe {
    Recipe {
        id: "starting-point/atlas-defaults".into(),
        version: "0".into(),
        model: model.to_string(),
        runtime: Some("atlas".into()),
        container: String::new(),
        min_nodes: 1,
        description: "Starting point, not a measurement: no settings pinned, so every \
                      flag is the server's own default. There is nothing to edit on \
                      this card — launch as-is, or pick a donor card to start from \
                      its settings."
            .into(),
        maintainer: String::new(),
        category: String::new(),
        model_params: String::new(),
        quantization: String::new(),
        kv_dtype: String::new(),
        updated: String::new(),
        defaults: std::collections::BTreeMap::new(),
        starting_point: Some("no donor — the server's own defaults".into()),
    }
}

/// The recipes whose parameters may be offered for `model`, best match first.
///
/// Donors are Atlas single-node recipes only: a vLLM recipe cannot be
/// launched from here at all, and a multi-node donor carries `ep_size`/
/// `min_nodes` this dashboard's single-node launcher would refuse — offering
/// either would be offering settings whose launch is a dead end. Family
/// matches sort first. One function for both consumers — the starting-point
/// cards and the Config form's borrow picker — because two copies of this
/// filter is how one of them ends up offering a vLLM donor.
pub(super) fn ranked_donors<'a>(
    recipes: &'a [Recipe],
    model: &str,
    model_type: &str,
) -> Vec<&'a Recipe> {
    let mut donors: Vec<&Recipe> = recipes
        .iter()
        .filter(|r| r.is_atlas() && r.min_nodes <= 1)
        .collect();
    donors.sort_by_key(|r| (!same_family(r, model, model_type), r.id.clone()));
    donors
}

/// Every starting point for `entry`, best guess first. The blank card is
/// always last, so the list is never empty and the safest option is the
/// fallback rather than the default.
pub(super) fn starting_points(recipes: &[Recipe], entry: &Entry) -> Vec<Recipe> {
    let model_type = entry
        .local
        .as_ref()
        .map(|l| l.model_type.as_str())
        .unwrap_or_default();
    let mut out: Vec<Recipe> = ranked_donors(recipes, &entry.model, model_type)
        .into_iter()
        .map(|d| template_from(d, &entry.model))
        .collect();
    out.push(blank(&entry.model));
    out
}

impl LibState {
    /// Build and hold the starting points for the selected no-recipe model.
    ///
    /// Rebuilt on every entry rather than cached: the set depends on the
    /// index, which a background refresh replaces wholesale, and a stale set
    /// would offer cards copied from recipes that no longer exist.
    pub(super) fn open_starting_points(&mut self) {
        let Some(entry) = self.current() else {
            return;
        };
        let cards = starting_points(&self.index.recipes, entry);
        let model = entry.model.clone();
        self.starting = Some((model, cards));
    }
}

#[cfg(test)]
#[path = "lib_start_tests.rs"]
mod tests;
