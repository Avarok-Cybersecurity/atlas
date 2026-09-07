// SPDX-License-Identifier: AGPL-3.0-only

//! `[expert_categories]` MODEL.toml parse pin.
//!
//! Compiles the same `build_parse_experts.rs` the build script uses — a
//! build script's own `#[cfg(test)]` modules are never run by `cargo test`
//! (the hole `tests/kernel_shadow_detector.rs` was written to close).
//!
//! What must hold: a malformed table FAILS THE BUILD. An expert id that
//! silently defaulted or got truncated would put the wrong experts on the
//! GPU under `--expert-category`, and nothing downstream could tell.

#[path = "../build_parse_experts.rs"]
mod build_parse_experts;

use build_parse_experts::{
    parse_expert_categories, parse_expert_categories_value, render_expert_categories,
};

fn parse(toml_src: &str) -> Vec<build_parse_experts::ExpertCategoryRaw> {
    let doc: toml::Value = toml::from_str(toml_src).expect("fixture must be valid TOML");
    parse_expert_categories_value(&doc, "fixture")
}

// ---------------------------------------------------------------- Path A

#[test]
fn parses_categories_normalized_and_sorted() {
    let cats = parse(
        r#"
[model]
name = "irrelevant"

[expert_categories.sql]
coverage = 0.9
prompts = 32
tokens_routed = 2803
layers."10" = [7, 3]
layers."2" = [9]

[expert_categories.code-python]
coverage = 0.85
layers."2" = [1, 4]
"#,
    );

    assert_eq!(cats.len(), 2);
    // Categories sorted by name, not TOML order.
    assert_eq!(cats[0].name, "code-python");
    assert_eq!(cats[1].name, "sql");
    assert!((cats[0].coverage - 0.85).abs() < 1e-9);

    let sql = &cats[1];
    // Layers sorted ascending by index, ids sorted ascending within a layer.
    assert_eq!(sql.layers, vec![(2usize, vec![9u16]), (10, vec![3, 7])]);
}

#[test]
fn absent_section_yields_no_categories() {
    assert!(parse("[model]\nname = \"x\"\n").is_empty());
}

#[test]
fn missing_model_toml_yields_no_categories() {
    let dir = std::env::temp_dir().join("atlas-ec-parse-missing");
    std::fs::create_dir_all(&dir).unwrap();
    let _ = std::fs::remove_file(dir.join("MODEL.toml"));
    assert!(parse_expert_categories(&dir).is_empty());
}

// ---------------------------------------------------------------- Path B
// Edge cases where a silent default or a truncation would be invisible
// downstream: BEL would load a different expert set than EC measured.

#[test]
fn expert_id_above_u16_is_rejected_not_truncated() {
    // 65536 truncates to 0 in u16 — a real expert id, silently wrong.
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.\"0\" = [65536]\n")
    })
    .unwrap_err();
    let msg = panic_msg(&err);
    assert!(msg.contains("out of range"), "got: {msg}");
}

#[test]
fn negative_expert_id_is_rejected() {
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.\"0\" = [-1]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("out of range"));
}

#[test]
fn duplicate_expert_ids_in_a_layer_are_rejected() {
    // A duplicate would inflate a "how many experts does this category need"
    // count without changing what is loaded.
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.\"0\" = [4, 4]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("duplicate"));
}

#[test]
fn aliasing_layer_keys_are_rejected() {
    // "3" and "03" both parse to layer 3; one would silently win.
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.\"3\" = [1]\nlayers.\"03\" = [2]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("same layer index"));
}

#[test]
fn coverage_boundaries_one_is_accepted_zero_is_not() {
    let cats = parse("[expert_categories.a]\ncoverage = 1.0\nlayers.\"0\" = [1]\n");
    assert_eq!(cats[0].coverage, 1.0);

    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.0\nlayers.\"0\" = [1]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("(0.0, 1.0]"));
}

#[test]
fn empty_layer_array_is_rejected() {
    // An empty array means "load no experts in this layer" — every token
    // routed there would hit an unloaded expert.
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.\"0\" = []\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("needs ≥1 expert"));
}

#[test]
fn empty_layers_table_is_rejected() {
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\n[expert_categories.a.layers]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("empty"));
}

// ---------------------------------------------------------------- Path C

#[test]
fn missing_coverage_is_rejected() {
    // PCND: coverage is the provenance of the mapping; defaulting it would
    // make two tables generated at different thresholds indistinguishable.
    let err = std::panic::catch_unwind(|| parse("[expert_categories.a]\nlayers.\"0\" = [1]\n"))
        .unwrap_err();
    assert!(panic_msg(&err).contains("`coverage` is required"));
}

#[test]
fn missing_layers_is_rejected() {
    let err =
        std::panic::catch_unwind(|| parse("[expert_categories.a]\ncoverage = 0.9\n")).unwrap_err();
    assert!(panic_msg(&err).contains("`layers` table is required"));
}

#[test]
fn unknown_key_is_rejected() {
    // A typo'd key ("layer" for "layers") must not read as "no layers".
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.\"0\" = [1]\nexpertz = 3\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("unknown key"));
}

#[test]
fn non_integer_layer_key_is_rejected() {
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.a]\ncoverage = 0.9\nlayers.first = [1]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("not a non-negative integer"));
}

#[test]
fn category_name_with_a_space_is_rejected() {
    // Names are matched verbatim against --expert-category; a space would be
    // unquotable on the command line in practice.
    let err = std::panic::catch_unwind(|| {
        parse("[expert_categories.\"code python\"]\ncoverage = 0.9\nlayers.\"0\" = [1]\n")
    })
    .unwrap_err();
    assert!(panic_msg(&err).contains("category names must be"));
}

#[test]
fn shipped_model_tomls_all_parse() {
    // Every MODEL.toml in the tree must survive this parser: it runs for all
    // of them on every build, and a panic here is a broken build, not a
    // broken test run.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels");
    let mut seen = 0usize;
    for hw in std::fs::read_dir(&root).expect("kernels/ must exist") {
        let hw = hw.unwrap().path();
        if !hw.is_dir() {
            continue;
        }
        for model in std::fs::read_dir(&hw).unwrap() {
            let model = model.unwrap().path();
            if model.join("MODEL.toml").exists() {
                parse_expert_categories(&model);
                seen += 1;
            }
        }
    }
    assert!(seen > 10, "expected many MODEL.toml files, found {seen}");
}

// ------------------------------------------------------------- codegen

#[test]
fn renders_a_static_body_that_compiles_as_written() {
    let cats = parse(
        "[expert_categories.sql]\ncoverage = 0.9\nlayers.\"2\" = [9]\nlayers.\"10\" = [7, 3]\n",
    );
    let rendered = render_expert_categories(&cats);
    assert_eq!(
        rendered,
        "ExpertCategory { name: \"sql\", coverage: 0.9f32, \
         layers: &[(2usize, &[9u16]), (10usize, &[3u16, 7u16])] }"
    );
    // Two properties this exact string encodes, both of which were real
    // compile failures before they were fixed, and neither of which any
    // other test would notice:
    //   - no `[..]`: slice indexing is not a const operation, so an id
    //     array written as `&[..][..]` cannot appear in a `static`;
    //   - the value goes in a named static, not inline in `all_ptx_sets()`,
    //     because const promotion does not reach inside a `vec![]` body.
    assert!(!rendered.contains("[..]"), "id arrays must not be sliced");
}

#[test]
fn renders_empty_for_an_uncategorized_model() {
    // Every target emits the static; a model with no categories emits an
    // empty one rather than being skipped.
    assert_eq!(render_expert_categories(&[]), "");
}

fn panic_msg(err: &Box<dyn std::any::Any + Send>) -> String {
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}
