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

    let tier = include_str!("../../../../../bench/fp8_dgx2_drift/harness/run_tier.sh");
    assert!(tier.contains(
        "ATLAS_WARM_TARGET_DIR=\"${ATLAS_WARM_TARGET_DIR:-${HOME}/.cargo/atlas-warm-target}\""
    ));
    let warm = include_str!("../../../../../bench/fp8_dgx2_drift/harness/warm_cargo_cache.sh");
    assert!(warm.contains(
        "WARM_TARGET_DIR=\"${ATLAS_WARM_TARGET_DIR:-${HOME}/.cargo/atlas-warm-target}\""
    ));
    assert!(warm.contains(
        "TEMPLATE_DIR=\"${ATLAS_WARM_TEMPLATE_DIR:-${HOME}/.cargo/atlas-warm-template}\""
    ));
    let scorer = include_str!("../../../../../bench/fp8_dgx2_drift/harness/score_run.py");
    assert!(scorer.contains("\".cargo\", \"atlas-warm-target\""));
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
    let shell = include_str!("../../../../../bench/fp8_dgx2_drift/harness/warm_cargo_cache.sh");
    let shell_manifest = shell
        .split_once("<<'TOML'\n")
        .and_then(|(_, rest)| rest.split_once("\nTOML\n"))
        .map(|(manifest, _)| manifest)
        .expect("warm_cargo_cache.sh must carry its Cargo.toml heredoc");
    let rust_manifest: toml::Value = toml::from_str(TEMPLATE_MANIFEST).unwrap();
    let shell_manifest: toml::Value = toml::from_str(shell_manifest).unwrap();
    assert_eq!(rust_manifest, shell_manifest);

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
    let dependencies = rust_manifest["dependencies"].as_table().unwrap();
    for (dep, expected) in [
        ("axum", &["json", "macros", "ws", "multipart"][..]),
        ("tokio", &["full"][..]),
        ("serde", &["derive"][..]),
        ("tower", &["full"][..]),
        ("tower-http", &["full"][..]),
        ("hyper", &["full"][..]),
        ("reqwest", &["json"][..]),
        ("tracing-subscriber", &["env-filter"][..]),
    ] {
        let actual: Vec<&str> = dependencies[dep]["features"]
            .as_array()
            .unwrap_or_else(|| panic!("{dep} must declare features"))
            .iter()
            .map(|feature| feature.as_str().unwrap())
            .collect();
        assert_eq!(actual, expected, "{dep} feature superset drifted");
    }

    let shell_main = shell
        .split_once("<<'RUST'\n")
        .and_then(|(_, rest)| rest.split_once("\nRUST\n"))
        .map(|(main, _)| main)
        .expect("warm_cargo_cache.sh must carry its main.rs heredoc");
    let shell_main = shell_main
        .strip_prefix(
            "// Touches each dependency so its rlib is compiled into the warm target dir.\n",
        )
        .expect("shell template purpose comment");
    assert_eq!(TEMPLATE_MAIN.trim(), shell_main.trim());
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
    let main_before = std::fs::metadata(&main).unwrap().modified().unwrap();
    std::thread::sleep(Duration::from_millis(20));
    write_template(&dir).unwrap();
    assert_eq!(
        std::fs::metadata(&manifest).unwrap().modified().unwrap(),
        before,
        "an unchanged rewrite bumps mtime and forces a rebuild every run"
    );
    assert_eq!(
        std::fs::metadata(&main).unwrap().modified().unwrap(),
        main_before,
        "an unchanged main.rs rewrite also forces a rebuild"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
