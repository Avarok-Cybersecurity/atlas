// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// Against the real file
// ---------------------------------------------------------------------------

/// ★ Every pattern in the committed CODEOWNERS must be one this understands.
///
/// An unsupported pattern silently mentions nobody. That is not dangerous — the
/// mentions are advisory — but it is invisible, and invisible is how a reviewer
/// stops being notified without anyone noticing.
#[test]
fn every_pattern_in_the_real_file_is_supported() {
    let rules = load(&repo_root());
    assert!(!rules.is_empty(), "CODEOWNERS did not parse");
    for rule in &rules {
        assert!(
            is_supported(&rule.pattern),
            "{:?} uses a construct this module does not implement — implement it \
             or its owners go unmentioned",
            rule.pattern
        );
    }
}

/// The real rules must actually resolve for the paths this repository changes
/// most. A parser that produces rules nothing matches is no better than none.
#[test]
fn real_paths_resolve_to_real_owners() {
    let rules = load(&repo_root());
    for path in [
        "kernels/gb10/common/paged_decode_attn_fp8.cu",
        "crates/spark-model/src/layers/ops/gdn.rs",
        "crates/atlas-kernels/build.rs",
        ".github/workflows/ci.yml",
        "docs/adr/0012-closure-hash-cascade.md",
        "Cargo.toml",
    ] {
        assert_eq!(
            owners_of(&rules, path),
            ["@tbraun96", "@rsafier", "@SeedSource"],
            "{path} must retain the committed owner set"
        );
    }
}

// ---------------------------------------------------------------------------
// Matching rules
// ---------------------------------------------------------------------------

#[test]
fn a_directory_pattern_covers_its_whole_subtree() {
    let rules = parse("crates/spark-model/ @a\n");
    assert_eq!(owners_of(&rules, "crates/spark-model/src/lib.rs"), ["@a"]);
    assert_eq!(owners_of(&rules, "crates/spark-model/Cargo.toml"), ["@a"]);
    assert!(
        owners_of(&rules, "crates/spark-server/src/lib.rs").is_empty(),
        "a sibling crate is not covered"
    );
}

/// A pattern with no `/` matches that basename at any depth — this is why
/// `Cargo.toml` covers every crate's manifest, not just the workspace one.
#[test]
fn a_bare_filename_matches_at_any_depth() {
    let rules = parse("Cargo.toml @a\n");
    assert_eq!(owners_of(&rules, "Cargo.toml"), ["@a"]);
    assert_eq!(owners_of(&rules, "crates/spark-server/Cargo.toml"), ["@a"]);
    assert!(owners_of(&rules, "crates/spark-server/Cargo.lock").is_empty());
}

/// A pattern containing a `/` is anchored to the root, so a same-named
/// directory deeper in the tree does not match.
#[test]
fn a_pattern_with_a_slash_is_anchored_to_the_root() {
    let rules = parse("kernels/gb10/ @a\n");
    assert_eq!(owners_of(&rules, "kernels/gb10/common/x.cu"), ["@a"]);
    assert!(
        owners_of(&rules, "vendor/kernels/gb10/x.cu").is_empty(),
        "an anchored pattern must not match deeper"
    );
}

#[test]
fn a_star_matches_within_one_segment_only() {
    let rules = parse("/docs/*.md @a\n");
    assert_eq!(owners_of(&rules, "docs/readme.md"), ["@a"]);
    assert!(
        owners_of(&rules, "docs/adr/0001.md").is_empty(),
        "`*` must not cross a separator"
    );
}

#[test]
fn multiple_stars_within_one_segment_are_matched() {
    let rules = parse("/docs/a*b*.md @a\n");
    assert_eq!(owners_of(&rules, "docs/atlas-bench-check.md"), ["@a"]);
    assert!(owners_of(&rules, "docs/atlas-bench-check.txt").is_empty());
    assert!(owners_of(&rules, "docs/a/b/c.md").is_empty());
}

/// ★ Last match wins, as GitHub does. A catch-all first and a specific rule
/// after it is the normal shape of the file, and getting the order backwards
/// would give everything to the catch-all.
#[test]
fn the_last_matching_rule_wins() {
    let rules = parse("* @everyone\ncrates/spark-model/ @model-owner\n");
    assert_eq!(owners_of(&rules, "README.md"), ["@everyone"]);
    assert_eq!(
        owners_of(&rules, "crates/spark-model/src/lib.rs"),
        ["@model-owner"]
    );
}

/// A pattern with no owners REMOVES ownership rather than being ignored — it is
/// how CODEOWNERS un-assigns a subtree.
#[test]
fn a_pattern_with_no_owners_clears_ownership() {
    let rules = parse("* @everyone\nvendor/\n");
    assert!(
        owners_of(&rules, "vendor/thing.rs").is_empty(),
        "a bare pattern must clear, not fall through to the catch-all"
    );
}

#[test]
fn comments_and_blank_lines_are_ignored() {
    let rules = parse("# a comment\n\n  \n* @a # trailing comment\n");
    assert_eq!(rules.len(), 1);
    assert_eq!(rules[0].pattern, "*");
    assert_eq!(rules[0].owners, ["@a"]);
}

#[test]
fn owners_across_paths_are_deduplicated_and_sorted() {
    let rules = parse("* @z\ncrates/ @a @z\n");
    let paths = vec![
        "crates/x/src/lib.rs".to_string(),
        "crates/y/src/lib.rs".to_string(),
        "README.md".to_string(),
    ];
    assert_eq!(owners_for_paths(&rules, &paths), ["@a", "@z"]);
}

#[test]
fn a_missing_codeowners_file_yields_no_rules_rather_than_failing() {
    let empty = std::env::temp_dir().join(format!("atlas-noowners-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    assert!(load(&empty).is_empty());
}

/// Unsupported gitignore constructs must report unsupported rather than
/// quietly matching nothing while looking like they work.
#[test]
fn unsupported_globs_are_reported_not_silently_wrong() {
    for pattern in [
        "docs/**/notes.md",
        "docs/note?.md",
        "docs/note[12].md",
        r"docs/note\*.md",
    ] {
        assert!(!is_supported(pattern), "{pattern}");
        let rules = parse(&format!("{pattern} @a\n"));
        assert!(owners_of(&rules, "docs/note1.md").is_empty(), "{pattern}");
    }
}
