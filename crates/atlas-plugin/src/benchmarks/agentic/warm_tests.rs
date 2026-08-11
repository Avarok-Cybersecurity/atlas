// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_default_is_the_harness_ssot_path() {
    // warm_cargo_cache.sh:30, score_run.py:_warm_target_dir and run_tier.sh:100
    // all default to ${HOME}/.cargo/atlas-warm-target. Pointing anywhere else is
    // the bug: on a box where the harness has already warmed 34 GB of rlibs, a
    // different path is cold and every agent `cargo test` cold-builds axum.
    let d = dir_from(None, Some("/home/x".into()), "atlas-warm-target").unwrap();
    assert_eq!(d, std::path::Path::new("/home/x/.cargo/atlas-warm-target"));
    let t = dir_from(None, Some("/home/x".into()), "atlas-warm-template").unwrap();
    assert_eq!(
        t,
        std::path::Path::new("/home/x/.cargo/atlas-warm-template")
    );
}

#[test]
fn an_explicit_override_wins_and_an_empty_one_does_not() {
    let d = dir_from(Some("/mnt/warm".into()), Some("/home/x".into()), "leaf").unwrap();
    assert_eq!(d, std::path::Path::new("/mnt/warm"));
    // `${VAR:-default}` treats an empty VAR as unset; so must we, or an
    // accidentally-empty export silently relocates the cache to a cold path.
    let d = dir_from(Some("".into()), Some("/home/x".into()), "leaf").unwrap();
    assert_eq!(d, std::path::Path::new("/home/x/.cargo/leaf"));
}

#[test]
fn no_home_is_an_error_not_a_relative_path() {
    assert!(dir_from(None, None, "leaf").is_err());
    assert!(dir_from(None, Some("".into()), "leaf").is_err());
}

#[test]
fn the_template_covers_the_feature_superset_and_every_touched_crate() {
    // warm_cargo_cache.sh: cargo keys a cached rlib by (crate, version,
    // feature-set, profile), so a missing feature loses the warm hit entirely.
    for dep in [
        "axum",
        "tokio",
        "serde",
        "serde_json",
        "tower",
        "tower-http",
        "hyper",
        "reqwest",
        "anyhow",
        "thiserror",
        "tracing",
        "tracing-subscriber",
    ] {
        assert!(
            TEMPLATE_MANIFEST.contains(&format!("\n{dep} =")),
            "{dep} missing from the warm template"
        );
    }
    assert!(TEMPLATE_MANIFEST.contains("[dev-dependencies]"));
    assert!(TEMPLATE_MANIFEST.contains(r#"features = ["json", "macros", "ws", "multipart"]"#));
    // The template must reference the crates it declares or their rlibs are
    // never compiled into the warm dir.
    for used in ["axum::", "tokio::", "serde_json::", "tower::"] {
        assert!(
            TEMPLATE_MAIN.contains(used),
            "{used} not touched by main.rs"
        );
    }
    assert!(TEMPLATE_MAIN.contains("ATLAS_HARNESS_PORT"));
}

#[test]
fn writing_the_template_is_idempotent_so_cargo_does_not_rebuild_it() {
    let dir = std::env::temp_dir().join(format!("atlas-warm-tpl-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    write_template(&dir).unwrap();
    let manifest = dir.join("Cargo.toml");
    let main = dir.join("src/main.rs");
    assert_eq!(
        std::fs::read_to_string(&manifest).unwrap(),
        TEMPLATE_MANIFEST
    );
    assert_eq!(std::fs::read_to_string(&main).unwrap(), TEMPLATE_MAIN);
    let before = std::fs::metadata(&manifest).unwrap().modified().unwrap();
    write_template(&dir).unwrap();
    assert_eq!(
        std::fs::metadata(&manifest).unwrap().modified().unwrap(),
        before,
        "an unchanged rewrite bumps mtime and forces a rebuild every run"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
