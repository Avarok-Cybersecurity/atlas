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

fn step(d: &std::path::Path, name: &str) -> bool {
    followed_directions(&[], d)
        .steps
        .iter()
        .any(|(n, ok)| *n == name && *ok)
}

#[test]
fn tests_count_from_either_a_tests_dir_or_an_attribute() {
    let d = sandbox("testsdir");
    std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
    std::fs::write(d.join("src/main.rs"), "fn main() {}").unwrap();
    assert!(!step(&d, "wrote_tests"));
    // followed_directions.py:90 requires the dir to contain Rust. `cargo init`
    // can leave an empty `tests/` the agent never filled in.
    std::fs::create_dir_all(d.join("tests")).unwrap();
    assert!(!step(&d, "wrote_tests"), "an empty tests/ is not evidence");
    std::fs::write(d.join("tests/it.rs"), "#[test] fn t() {}").unwrap();
    assert!(step(&d, "wrote_tests"));
    std::fs::remove_dir_all(d.join("tests")).unwrap();
    assert!(!step(&d, "wrote_tests"));
    std::fs::write(d.join("src/main.rs"), "fn main() {}\n#[test] fn t() {}").unwrap();
    assert!(step(&d, "wrote_tests"));
}

#[test]
fn an_async_axum_test_attribute_counts() {
    // `#[tokio::test]` is how an Axum handler test is written; the built-in
    // rubric looked only for #[test]/#[cfg(test)] and scored these runs as
    // "never added tests" where followed_directions.py:97 credits them.
    let d = sandbox("tokiotest");
    std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
    std::fs::write(
        d.join("src/main.rs"),
        "fn main() {}\n#[tokio::test]\nasync fn t() {}\n",
    )
    .unwrap();
    assert!(step(&d, "wrote_tests"));
}

#[test]
fn a_project_without_a_main_rs_is_not_a_written_project() {
    let d = sandbox("nomain");
    std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
    std::fs::write(d.join("build.rs"), "fn main() {}").unwrap();
    assert!(!step(&d, "wrote_project"));
    std::fs::write(d.join("src/main.rs"), "fn main() {}").unwrap();
    assert!(step(&d, "wrote_project"));
}

#[cfg(unix)]
#[test]
fn the_evidence_walk_neither_follows_a_symlink_nor_counts_one() {
    // The scorer walks a tree written by the thing it is scoring, after that
    // thing has finished and with no timeout over it. `ln -s . a` three times
    // used to make this walk 3^40 paths — the kernel's ELOOP ceiling bounds the
    // depth and nothing bounds the breadth — so the model could hang the whole
    // benchmark from a shell command. A link is also not evidence: the two
    // symlinked files below are somebody else's `main.rs` and tests.
    let d = sandbox("symlinks");
    std::fs::write(d.join("Cargo.toml"), "[package]").unwrap();
    let elsewhere = d.parent().unwrap().join("atlas-agentic-symlinks-donor");
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(elsewhere.join("main.rs"), "fn main() {}").unwrap();
    std::fs::write(elsewhere.join("it.rs"), "#[test] fn t() {}").unwrap();
    std::os::unix::fs::symlink(elsewhere.join("main.rs"), d.join("src/main.rs")).unwrap();
    std::fs::create_dir_all(d.join("tests")).unwrap();
    std::os::unix::fs::symlink(elsewhere.join("it.rs"), d.join("tests/it.rs")).unwrap();
    std::os::unix::fs::symlink(&elsewhere, d.join("borrowed")).unwrap();
    // A finite borrowed directory makes a follow-links mutant fail promptly;
    // the old three-link self-cycle could consume the whole CI job first.
    assert!(!step(&d, "wrote_project"), "a symlinked main.rs is not one");
    assert!(!step(&d, "wrote_tests"), "a symlinked test file is not one");
    std::fs::write(d.join("tests/real.rs"), "fn helper() {}").unwrap();
    assert!(step(&d, "wrote_tests"), "a real test file still counts");
}

#[test]
fn the_detectors_are_word_anchored_like_the_harness_regexes() {
    let d = full_project("anchored");
    let m = |cmd: &str, name: &str| {
        followed_directions(&[cmd.to_string()], &d)
            .steps
            .iter()
            .any(|(n, ok)| *n == name && *ok)
    };
    // `\bp?kill\b` — a log line mentioning a killed process is not a teardown.
    assert!(!m("echo the build was killed", "tore_down"));
    assert!(!m("cargo install cargo-skill", "tore_down"));
    assert!(m("kill 4242", "tore_down"));
    assert!(m("pkill -f server", "tore_down"));
    assert!(m("timeout 5 fuser -k 3001/tcp", "tore_down"));
    // `\bcurl\b` must fire without a trailing space, and `\bnc\s+-z\b`.
    assert!(m("timeout 15 curl\n", "curled"));
    assert!(m("nc -z 127.0.0.1 3001", "curled"));
    assert!(!m("echo curlybraces", "curled"));
    // `\btarget/(?:debug|release)/\S`
    assert!(m("./target/release/app &", "ran_server"));
    assert!(!m("ls mytarget/release/", "ran_server"));
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

#[tokio::test]
async fn the_ephemeral_port_stays_reserved_while_the_project_builds() {
    let d = sandbox("port-reservation");
    std::fs::write(
        d.join("Cargo.toml"),
        "[package]\nname = \"port-reservation\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    std::fs::write(d.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(
        d.join("build.rs"),
        r#"fn main() {
    let port: u16 = std::env::var("ATLAS_HARNESS_PORT").unwrap().parse().unwrap();
    assert!(
        std::net::TcpListener::bind(("127.0.0.1", port)).is_err(),
        "the scorer released its selected port before the build"
    );
}
"#,
    )
    .unwrap();

    let result = webserver_test(
        &d,
        None,
        Duration::from_secs(30),
        Duration::from_millis(100),
    )
    .await;
    assert!(result.build_ok, "{}", result.error);
    let _ = std::fs::remove_dir_all(d);
}

#[tokio::test]
async fn a_missing_cargo_toml_fails_fast_without_building() {
    let d = sandbox("nocargo");
    let r = webserver_test(&d, None, Duration::from_secs(1), Duration::from_secs(1)).await;
    assert!(!r.webserver_ok && !r.build_ok);
    assert!(r.error.contains("Cargo.toml"), "{}", r.error);
}

#[tokio::test]
async fn a_missing_src_dir_fails_fast_without_building() {
    // score_run.py:306 refuses this pair before spending a build on it.
    let d = std::env::temp_dir().join(format!("atlas-agentic-nosrc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("Cargo.toml"), "[package]\nname=\"p\"\n").unwrap();
    let r = webserver_test(&d, None, Duration::from_secs(1), Duration::from_secs(1)).await;
    assert!(!r.webserver_ok && !r.build_ok);
    assert!(r.error.contains("src/"), "{}", r.error);
    let _ = std::fs::remove_dir_all(&d);
}

#[tokio::test]
async fn a_pong_counts_even_when_the_server_never_closes_the_socket() {
    // The regression this exists for: the old line-based reader only flushed a
    // newline-less body at EOF, so a server holding the connection open was
    // scored "/ping did not answer 'pong'" while it was answering all along.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let held = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        let mut buf = [0u8; 1024];
        let _ = sock.read(&mut buf).await;
        sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\npong")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_secs(10)).await;
    });
    let body = ping(port)
        .await
        .expect("a connected socket must yield a body");
    assert!(body.to_lowercase().contains("pong"), "{body:?}");
    held.abort();
}

#[test]
fn a_bind_failure_is_named_rather_than_reported_as_silence() {
    // score_run.py:418 singles this out: a port collision is a harness fault
    // and must never be indistinguishable from a model that wrote nothing.
    let p = std::env::temp_dir().join(format!("atlas-ws-err-{}.log", std::process::id()));
    std::fs::write(
        &p,
        "thread 'main' panicked: Address already in use (os error 98)",
    )
    .unwrap();
    let s = server_stderr(&p);
    assert!(s.contains("port in use"), "{s}");
    assert!(s.contains("os error 98"), "{s}");
    std::fs::write(&p, "   \n").unwrap();
    assert_eq!(server_stderr(&p), "", "empty stderr must add no noise");
    let _ = std::fs::remove_file(&p);
}
