// SPDX-License-Identifier: AGPL-3.0-only

//! The corpus itself, and the draw that selects from it.
//!
//! File ORDER decides which prompts a run of fewer than 32 measures, so it
//! is part of the measurement's identity, not a formatting detail. These
//! tests pin the order and the shape; the hash pin makes an edit to the
//! corpus visible as a failing test rather than as a quietly different
//! category table.

use super::*;

const CATEGORIES: [&str; 20] = [
    "code-python",
    "code-rust",
    "code-javascript",
    "code-c-systems",
    "shell-devops",
    "sql",
    "regex",
    "json-config",
    "math",
    "science-physics",
    "science-biology",
    "medicine-clinical",
    "finance-business",
    "legal-formal",
    "history-humanities",
    "philosophy-ethics",
    "translation",
    "creative-writing",
    "general-chat",
    "tool-calling",
];
const PER_CATEGORY: usize = 100;

// ---------------------------------------------------------------- Path A

#[test]
fn corpus_has_every_category_at_full_size() {
    let rows = load().expect("the compiled-in corpus must parse");
    assert_eq!(rows.len(), CATEGORIES.len() * PER_CATEGORY);
    assert_eq!(categories(&rows), CATEGORIES.to_vec());
    for c in CATEGORIES {
        let n = rows.iter().filter(|r| r.category == c).count();
        assert_eq!(n, PER_CATEGORY, "category {c} has {n} rows");
    }
}

#[test]
fn rows_of_a_category_are_contiguous() {
    // The draw takes the first N of each category by scanning in order; a
    // category split across the file would still work, but the file would no
    // longer read as what it is.
    let rows = load().unwrap();
    for c in CATEGORIES {
        let idx: Vec<usize> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.category == c)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            idx.last().unwrap() - idx.first().unwrap(),
            idx.len() - 1,
            "category {c} is not contiguous"
        );
    }
}

#[test]
fn ids_are_unique() {
    let rows = load().unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for r in &rows {
        assert!(seen.insert(r.id.clone()), "duplicate row id {}", r.id);
    }
}

// ---------------------------------------------------------------- Path B

#[test]
fn draw_takes_the_first_n_of_each_category_in_file_order() {
    // The BFCL lesson: the draw is positional, so anything that reorders the
    // file silently changes which prompts are scored. This pins that a draw
    // of 2 is the first two rows of each category as the file has them.
    let rows = load().unwrap();
    let drawn = draw(&rows, 2, &[]).unwrap();
    assert_eq!(drawn.len(), CATEGORIES.len() * 2);

    for c in CATEGORIES {
        let expected: Vec<&str> = rows
            .iter()
            .filter(|r| r.category == c)
            .take(2)
            .map(|r| r.id.as_str())
            .collect();
        let got: Vec<&str> = drawn
            .iter()
            .filter(|r| r.category == c)
            .map(|r| r.id.as_str())
            .collect();
        assert_eq!(got, expected, "draw for {c} is not the file's first two");
    }
}

#[test]
fn draw_larger_than_the_corpus_takes_everything_available() {
    let rows = load().unwrap();
    assert_eq!(draw(&rows, 999, &[]).unwrap().len(), rows.len());
}

#[test]
fn a_category_filter_selects_only_those_rows() {
    let rows = load().unwrap();
    let drawn = draw(&rows, 4, &["math".to_string(), "sql".to_string()]).unwrap();
    assert_eq!(drawn.len(), 8);
    assert!(
        drawn
            .iter()
            .all(|r| r.category == "math" || r.category == "sql")
    );
}

#[test]
fn prompts_are_short_and_self_contained() {
    // Long prompts dilute the category signal with generic filler, and the
    // measurement is prompt-side, so the bound is part of the instrument.
    let rows = load().unwrap();
    for r in &rows {
        let words = r.prompt.split_whitespace().count();
        assert!(
            (8..=45).contains(&words),
            "{} has {words} words, outside 8..=45: {:?}",
            r.id,
            r.prompt
        );
    }
}

#[test]
fn no_two_prompts_are_identical() {
    // A duplicated prompt would count twice toward one category's mass,
    // biasing whichever experts it happens to route to.
    let rows = load().unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for r in &rows {
        assert!(
            seen.insert(r.prompt.as_str()),
            "duplicate prompt text at {}",
            r.id
        );
    }
}

// ---------------------------------------------------------------- Path C

#[test]
fn an_unknown_category_is_refused_by_name() {
    let rows = load().unwrap();
    let err = draw(&rows, 1, &["klingon".to_string()])
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown category 'klingon'"), "got: {err}");
    assert!(
        err.contains("code-python"),
        "the error must list what IS available: {err}"
    );
}

#[test]
fn selection_parsing_treats_all_and_empty_as_everything() {
    assert!(parse_selection("all").is_empty());
    assert!(parse_selection("  ").is_empty());
    assert_eq!(
        parse_selection("math, sql ,code-rust"),
        vec!["math", "sql", "code-rust"]
    );
}

#[test]
fn the_manifest_agrees_with_the_rows() {
    // A report quotes the manifest. If it could drift from the rows, the
    // report would describe a corpus that was not the one measured.
    let (m, rows) = load_with_manifest().expect("corpus parses");
    assert_eq!(m.categories, CATEGORIES.to_vec());
    assert_eq!(m.prompts_per_category, PER_CATEGORY);
    assert_eq!(rows.len(), m.categories.len() * m.prompts_per_category);
    assert_eq!(m.name, "atlas-expert-categorization-corpus");
}

#[test]
fn corpus_content_is_pinned() {
    // Changing the corpus changes the measurement. This hash makes that show
    // up as a failing test, at which point the descriptor's `updated` date
    // must move too — that date is what decides whether two runs are
    // comparable.
    let hash = content_hash(&load().unwrap());
    assert_eq!(
        hash, CORPUS_SHA256,
        "the corpus changed — update CORPUS_SHA256 and bump DESCRIPTOR.updated"
    );
}

/// SHA-256 over `category\x01id\x01prompt` rows joined by `\x02`. Content,
/// not bytes, so reformatting the JSONL does not trip it but editing a
/// prompt does.
const CORPUS_SHA256: &str = "c898b248ab284636f406ed2589f03d746cae88dcc15262e8ae521ecf276b9f63";
