// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn sandbox(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("atlas-agentic-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("src")).unwrap();
    d
}

fn full_project(name: &str) -> std::path::PathBuf {
    let d = sandbox(name);
    std::fs::write(
        d.join("Cargo.toml"),
        "[package]\nname = \"p\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    std::fs::write(
        d.join("src/main.rs"),
        "fn main() {}\n#[cfg(test)]\nmod t {}\n",
    )
    .unwrap();
    d
}

const FULL_RUN: &[&str] = &[
    "cargo init",
    "cargo test",
    "setsid cargo run --release > /tmp/s.log 2>&1 &",
    "timeout 15 curl -s http://127.0.0.1:3001/ping",
    "timeout 5 fuser -k 3001/tcp",
];

#[test]
fn a_complete_run_meets_every_step() {
    let d = full_project("full");
    let cmds: Vec<String> = FULL_RUN.iter().map(|s| s.to_string()).collect();
    let f = followed_directions(&cmds, &d);
    assert!(f.overall(), "{:?}", f.steps);
    assert_eq!(f.met(), 6);
}

#[test]
fn the_lazy_early_stop_is_caught_even_though_the_code_is_correct() {
    // The exact case this metric exists for: a correct project written and
    // then abandoned. `webserver_ok` would still be true; process fidelity
    // must not be.
    let d = full_project("lazy");
    let f = followed_directions(&["cargo init".to_string()], &d);
    assert!(!f.overall());
    let by_name: std::collections::BTreeMap<_, _> = f.steps.iter().copied().collect();
    assert!(by_name["wrote_project"] && by_name["wrote_tests"]);
    assert!(!by_name["ran_tests"] && !by_name["curled"] && !by_name["tore_down"]);
}

#[test]
fn tests_count_from_either_a_tests_dir_or_an_attribute() {
    let d = sandbox("testsdir");
    std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
    std::fs::write(d.join("src/main.rs"), "fn main() {}").unwrap();
    assert!(
        !followed_directions(&[], &d)
            .steps
            .iter()
            .any(|(n, ok)| *n == "wrote_tests" && *ok)
    );
    std::fs::create_dir_all(d.join("tests")).unwrap();
    assert!(
        followed_directions(&[], &d)
            .steps
            .iter()
            .any(|(n, ok)| *n == "wrote_tests" && *ok)
    );
}

#[test]
fn cargo_detection_needs_a_real_subcommand_boundary() {
    assert!(contains_cargo("cargo test", &["test"]));
    assert!(contains_cargo("cd x && cargo   test --release", &["test"]));
    assert!(contains_cargo("cargo nextest run", &["nextest"]));
    // "cargo testify" is not "cargo test".
    assert!(!contains_cargo("cargo testify", &["test"]));
    // A path that merely contains the word must not count.
    assert!(!contains_cargo("ls /home/cargo-tests", &["test"]));
}

#[test]
fn the_walk_skips_target_so_build_output_is_not_evidence() {
    let d = sandbox("walk");
    std::fs::create_dir_all(d.join("target/release/build")).unwrap();
    std::fs::write(d.join("target/release/build/x.rs"), "#[test] fn t() {}").unwrap();
    std::fs::write(d.join("src/main.rs"), "fn main() {}").unwrap();
    assert!(!has_tests(&d), "a #[test] under target/ must not count");
}

#[test]
fn free_port_returns_a_usable_ephemeral_port() {
    // What `free_port` actually promises: a port the OS handed out as free at
    // the moment of the call. It does NOT promise the port is still free
    // afterwards — nothing can, since any process may take it — so this asserts
    // the range and does not re-bind. An earlier version of this test did
    // re-bind and flaked under a parallel `cargo test`.
    for _ in 0..4 {
        let p = free_port().unwrap();
        assert!(p > 1024, "expected an ephemeral port, got {p}");
    }
}

#[tokio::test]
async fn a_missing_cargo_toml_fails_fast_without_building() {
    let d = sandbox("nocargo");
    let r = webserver_test(&d, None, Duration::from_secs(1), Duration::from_secs(1)).await;
    assert!(!r.webserver_ok && !r.build_ok);
    assert!(r.error.contains("Cargo.toml"), "{}", r.error);
}
