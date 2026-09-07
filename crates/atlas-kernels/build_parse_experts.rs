// SPDX-License-Identifier: AGPL-3.0-only

//! Parse `[expert_categories]` from a model's MODEL.toml.
//!
//! Written by the `expert-categories` benchmark (atlas-plugin), consumed at
//! boot by `spark serve --expert-category <name>` to load only that
//! category's experts (BEL). The table is structural model metadata, so it
//! is baked into `TargetPtxSet` at build time like `[dflash]` — runtime
//! boxes do not ship the kernels/ tree.
//!
//! Self-contained on purpose (own raw type, no `super::` references):
//! `tests/expert_categories_parse.rs` compiles this SAME file, because a
//! build script's `#[cfg(test)]` modules are never run by `cargo test`
//! (see `tests/kernel_shadow_detector.rs` for the precedent and the hole
//! that rule once shipped).
//!
//! Malformed content FAILS THE BUILD — a typo'd expert id that silently
//! defaulted would surface as the wrong experts resident on a serve.
//! Only a genuinely absent `[expert_categories]` section yields an empty
//! result.

/// One `[expert_categories.<name>]` table, normalized: layers sorted
/// ascending by index, expert ids sorted ascending, categories returned
/// sorted by name (deterministic codegen).
#[derive(Clone, Debug, PartialEq)]
pub struct ExpertCategoryRaw {
    pub name: String,
    pub coverage: f64,
    pub layers: Vec<(usize, Vec<u16>)>,
}

/// Render parsed categories as the body of a `&[ExpertCategory]` static
/// (see `build_codegen.rs`). Ordering is already normalized by the parser;
/// this only formats.
///
/// Emitted into a `static`, never inline in `all_ptx_sets()`: const
/// promotion does not reach inside a `vec![]` body, so an inline literal
/// would be a temporary that cannot borrow for `'static`. For the same
/// reason the id arrays carry no `[..]` — slice indexing is not const, and
/// `&[u16; N]` coerces to `&[u16]` against the field type on its own.
pub fn render_expert_categories(cats: &[ExpertCategoryRaw]) -> String {
    cats.iter()
        .map(|c| {
            let layers = c
                .layers
                .iter()
                .map(|(l, ids)| {
                    let ids = ids
                        .iter()
                        .map(|i| format!("{i}u16"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({l}usize, &[{ids}])")
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "ExpertCategory {{ name: \"{}\", coverage: {:?}f32, layers: &[{layers}] }}",
                c.name, c.coverage as f32,
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Read `<model_dir>/MODEL.toml` and parse its `[expert_categories]`.
/// Missing file or absent section → empty (the model simply has no
/// categorization yet). Anything else malformed → panic (build failure).
pub fn parse_expert_categories(model_dir: &std::path::Path) -> Vec<ExpertCategoryRaw> {
    let path = model_dir.join("MODEL.toml");
    if !path.exists() {
        return Vec::new();
    }
    let src = path.display().to_string();
    let content =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{src}: read failed: {e}"));
    let toml: toml::Value =
        toml::from_str(&content).unwrap_or_else(|e| panic!("Bad TOML in {src}: {e}"));
    parse_expert_categories_value(&toml, &src)
}

/// Parse from an already-loaded TOML document. `src` names the source in
/// panic messages.
pub fn parse_expert_categories_value(toml: &toml::Value, src: &str) -> Vec<ExpertCategoryRaw> {
    let Some(section) = toml.get("expert_categories") else {
        return Vec::new();
    };
    let table = section
        .as_table()
        .unwrap_or_else(|| panic!("{src}: [expert_categories] must be a table of categories"));

    let mut out: Vec<ExpertCategoryRaw> = Vec::with_capacity(table.len());
    for (name, body) in table {
        out.push(parse_one_category(name, body, src));
    }
    // BTreeMap iteration is already name-ordered, but do not rely on the
    // toml crate's map choice — deterministic codegen is a contract here.
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn parse_one_category(name: &str, body: &toml::Value, src: &str) -> ExpertCategoryRaw {
    let ctx = format!("{src}: [expert_categories.{name}]");
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        panic!(
            "{ctx}: category names must be non-empty [A-Za-z0-9_-]+ — they are matched \
             verbatim against `--expert-category`"
        );
    }
    let body = body
        .as_table()
        .unwrap_or_else(|| panic!("{ctx}: must be a table with `coverage` and `layers`"));

    // `prompts` / `tokens_routed` are provenance the generating benchmark
    // records beside the mapping; parsed nowhere, but legal.
    for key in body.keys() {
        if !matches!(
            key.as_str(),
            "coverage" | "layers" | "prompts" | "tokens_routed"
        ) {
            panic!(
                "{ctx}: unknown key `{key}` (allowed: coverage, layers, prompts, tokens_routed)"
            );
        }
    }

    let coverage = match body.get("coverage") {
        Some(v) => v
            .as_float()
            .or_else(|| v.as_integer().map(|i| i as f64))
            .unwrap_or_else(|| panic!("{ctx}: `coverage` must be a number")),
        None => panic!(
            "{ctx}: `coverage` is required (the routing-mass fraction this table was generated at)"
        ),
    };
    if !(coverage > 0.0 && coverage <= 1.0) {
        panic!("{ctx}: `coverage` must be in (0.0, 1.0], got {coverage}");
    }

    let layers_val = body.get("layers").unwrap_or_else(|| {
        panic!("{ctx}: `layers` table is required (one int array per MoE layer)")
    });
    let layers_table = layers_val
        .as_table()
        .unwrap_or_else(|| panic!("{ctx}: `layers` must be a table: layers.\"<index>\" = [ids]"));
    if layers_table.is_empty() {
        panic!("{ctx}: `layers` is empty — a category with no experts cannot be loaded");
    }

    let mut layers: Vec<(usize, Vec<u16>)> = Vec::with_capacity(layers_table.len());
    for (layer_key, ids_val) in layers_table {
        let layer: usize = layer_key.parse().unwrap_or_else(|_| {
            panic!("{ctx}: layer key {layer_key:?} is not a non-negative integer")
        });
        let arr = ids_val
            .as_array()
            .unwrap_or_else(|| panic!("{ctx}: layers.\"{layer}\" must be an array of expert ids"));
        if arr.is_empty() {
            panic!("{ctx}: layers.\"{layer}\" is empty — every listed MoE layer needs ≥1 expert");
        }
        let mut ids: Vec<u16> = Vec::with_capacity(arr.len());
        for v in arr {
            let id = v
                .as_integer()
                .unwrap_or_else(|| panic!("{ctx}: layers.\"{layer}\" contains a non-integer"));
            if !(0..=u16::MAX as i64).contains(&id) {
                panic!(
                    "{ctx}: layers.\"{layer}\" expert id {id} out of range [0, {}]",
                    u16::MAX
                );
            }
            ids.push(id as u16);
        }
        ids.sort_unstable();
        if ids.windows(2).any(|w| w[0] == w[1]) {
            panic!("{ctx}: layers.\"{layer}\" contains duplicate expert ids");
        }
        layers.push((layer, ids));
    }
    layers.sort_unstable_by_key(|(l, _)| *l);
    // Duplicate layer keys cannot survive TOML table parsing, but two keys
    // like "3" and "03" can alias the same index — reject that too.
    if layers.windows(2).any(|w| w[0].0 == w[1].0) {
        panic!("{ctx}: two layer keys resolve to the same layer index");
    }

    ExpertCategoryRaw {
        name: name.to_string(),
        coverage,
        layers,
    }
}
