// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

/// The same `cargo test`, from a tier that had to compile the crate.
// `\x20` only because a `\`-continuation eats the leading spaces cargo prints.
const COLD_TEST: &str = "\
\x20  Compiling pingpong v0.1.0 (/home/claude/.atlas/runs/agentic-webserver/sandbox/run-00)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 12.43s
     Running unittests src/main.rs (/home/claude/.cargo/atlas-warm-target/debug/deps/pingpong-9a8b7c6d5e4f3a2b)

running 1 test
test tests::ping_returns_pong ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s
";

/// …and from the next tier, which found the shared warm target dir already
/// holding it. Same work, same outcome, four differences the model must not see.
const WARM_TEST: &str = "\
\x20   Finished `test` profile [unoptimized + debuginfo] target(s) in 0.08s
     Running unittests src/main.rs (/home/claude/.cargo/atlas-warm-target/debug/deps/pingpong-9a8b7c6d5e4f3a2b)

running 1 test
test tests::ping_returns_pong ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
";

const CURL_A: &str = "\
% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100     4  100     4    0     0   1333      0 --:--:-- --:--:-- --:--:--  1333
pong
[stderr]
3001/tcp:            1417733
";

const CURL_B: &str = "\
% Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100     4  100     4    0     0    987      0 --:--:-- --:--:-- --:--:--   991
pong
[stderr]
3001/tcp:            2233441
";

// ── convergence ────────────────────────────────────────────────────

#[test]
fn two_runs_of_the_same_work_read_identically() {
    assert_ne!(COLD_TEST, WARM_TEST, "the fixtures must actually differ");
    assert_eq!(normalize(COLD_TEST), normalize(WARM_TEST));
    assert_ne!(CURL_A, CURL_B);
    assert_eq!(normalize(CURL_A), normalize(CURL_B));
}

#[test]
fn normalisation_is_idempotent() {
    for raw in [
        COLD_TEST,
        WARM_TEST,
        CURL_A,
        "2026-08-06T21:09:12.345Z INFO listening on 0.0.0.0:3001",
        "-rw-r--r-- 1 claude claude 234 Aug  6 21:09 Cargo.toml",
        "real\t0m1.234s\nuser\t0m0.512s\nsys\t0m0.061s",
        "kill: (12345) - No such process",
        "",
    ] {
        let once = normalize(raw);
        assert_eq!(normalize(&once), once, "not idempotent: {raw:?}");
    }
}

// ── what must NOT be touched ───────────────────────────────────────

#[test]
fn compiler_diagnostics_pass_through_untouched() {
    // The failure text is the whole signal: it is how the model finds its bug
    // and how a human reads the run afterwards.
    let raw = "error[E0433]: failed to resolve: use of undeclared crate `axkm`\n \
        --> src/main.rs:1:5\n  |\n1 | use axkm::Router;\n  |     ^^^^ use of undeclared crate\n\
        warning: unused import: `std::env`\n\
        error: could not compile `pingpong` (bin \"pingpong\") due to 1 previous error";
    assert_eq!(normalize(raw), raw);
}

#[test]
fn the_port_survives_everywhere_it_appears() {
    // ATLAS_HARNESS_PORT is part of the prompt's contract; a model that cannot
    // read the port back out of tool output cannot tear its server down.
    for raw in [
        "3001/tcp:            1417733",
        "curl: (7) Failed to connect to localhost port 3001 after 0 ms: Connection refused",
        "listening on 0.0.0.0:3001",
        "kill $(lsof -t -i:3001) failed to kill 3001/tcp",
    ] {
        assert!(normalize(raw).contains("3001"), "port lost: {raw}");
    }
}

#[test]
fn the_port_survives_the_lines_that_also_talk_about_killing() {
    // The prompt's teardown instruction is "kill whatever is listening on its
    // port", so the model writes that sentence into its script and its
    // comments — and `cat` shows it the result. Naming a pid word licensed
    // rewriting every four-digit number on the line, so these three came back
    // with the PORT replaced by `<pid>`: the model would then read its own file
    // back wrong, and an `edit` built from what it saw would no longer match.
    for raw in [
        "# kill whatever is listening on port 3001",
        "let port = 3001; // process the request",
        "echo \"Killing the process on port 3001\"",
        "Killed process 12345 on port 3001",
    ] {
        assert!(normalize(raw).contains("3001"), "port lost: {raw}");
    }
    // …and the pid on that same line is still a pid.
    assert_eq!(
        normalize("Killed process 12345 on port 3001"),
        "Killed process <pid> on port 3001"
    );
}

#[test]
fn a_pid_word_is_a_word_and_not_a_substring() {
    // `contains("process")` fired on "Processing" and `contains("kill")` on
    // "skill", and a match licenses rewriting every four-digit number on the
    // line — so the model's own program output was rewritten under it.
    assert_eq!(
        normalize("Processing 1000 records"),
        "Processing 1000 records"
    );
    assert_eq!(
        normalize("installed cargo-skill 1234 files"),
        "installed cargo-skill 1234 files"
    );
    // The forms the genuine cases actually take all still fire.
    assert_eq!(normalize("kill 12345"), "kill <pid>");
    assert_eq!(normalize("Killed process 12345"), "Killed process <pid>");
    assert_eq!(normalize("killing 12345 now"), "killing <pid> now");
    assert_eq!(normalize("pid=12345"), "pid=<pid>");
    assert_eq!(normalize("PIDs: 12345 12346"), "PIDs: <pid> <pid>");
}

#[test]
fn exit_statuses_and_test_counts_survive() {
    let raw = "test result: FAILED. 1 passed; 2 failed; 0 ignored\n[exit exit status: 101]";
    let out = normalize(raw);
    assert!(out.contains("1 passed; 2 failed"), "{out}");
    assert!(out.contains("101"), "an exit status is not a pid: {out}");
}

#[test]
fn a_source_listing_is_returned_verbatim() {
    // `cat src/main.rs` goes through here. If a byte of it changed, an `edit`
    // built from what the model saw would stop matching the file on disk.
    let raw = "use axum::{routing::get, Router};\n\
        async fn ping() -> &'static str { \"pong\" }\n    \
        let port: u16 = std::env::var(\"ATLAS_HARNESS_PORT\")\n        \
        .unwrap_or_else(|_| \"3001\".to_string())\n        .parse().unwrap();\n\
        // 2000 requests in 3 batches\n";
    assert_eq!(normalize(raw), raw);
}

// ── what must be ───────────────────────────────────────────────────

#[test]
fn durations_are_erased_wherever_they_are_reported() {
    let cases = [
        (
            "    Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.23s",
            "    Finished `dev` profile [unoptimized + debuginfo] target(s) in <elapsed>",
        ),
        (
            "    Finished `release` profile [optimized] target(s) in 1m 03s",
            "    Finished `release` profile [optimized] target(s) in <elapsed>",
        ),
        ("real\t0m1.234s", "real\t<elapsed>"),
        ("  sys     0m0.061s", "  sys\t<elapsed>"),
        (
            "curl: (7) Failed to connect to localhost port 3001 after 0 ms: refused",
            "curl: (7) Failed to connect to localhost port 3001 after <elapsed>: refused",
        ),
    ];
    for (raw, want) in cases {
        assert_eq!(normalize(raw), want);
    }
}

#[test]
fn clocks_and_dates_are_erased() {
    assert_eq!(
        normalize("2026-08-06T21:09:12.345Z INFO ready"),
        "<timestamp> INFO ready"
    );
    assert_eq!(
        normalize("2026-08-06 21:09:12 INFO ready"),
        "<timestamp> INFO ready"
    );
    assert_eq!(
        normalize("-rw-r--r-- 1 claude claude 234 Aug  6 21:09 Cargo.toml"),
        "-rw-r--r-- 1 claude claude 234 <date> Cargo.toml"
    );
    assert_eq!(
        normalize("drwxr-xr-x 3 claude claude 4096 Aug  6  2025 src"),
        "drwxr-xr-x 3 claude claude 4096 <date> src"
    );
    // A bare date carries no clock and does not move during a run.
    assert_eq!(normalize("version 2026-08-06"), "version 2026-08-06");
}

#[test]
fn timestamps_and_dates_inside_identifiers_are_not_erased() {
    for raw in [
        "build-2026-08-06T21:09:12abc",
        "build_2026-08-06T21:09:12",
        "build-2026-08-06T21:09:12_suffix",
        "artifact-Aug  6  2025suffix",
    ] {
        assert_eq!(normalize(raw), raw);
    }
}

#[test]
fn pids_are_erased_where_they_are_pids() {
    assert_eq!(
        normalize("3001/tcp:            1417733"),
        "3001/tcp:            <pid>"
    );
    assert_eq!(
        normalize("kill: (12345) - No such process"),
        "kill: (<pid>) - No such process"
    );
    assert_eq!(normalize("1417733"), "<pid>");
    // No anchor, no rewrite: a free number in ordinary output is data.
    assert_eq!(
        normalize("total 4096 bytes written"),
        "total 4096 bytes written"
    );
}

#[test]
fn cold_and_warm_progress_lines_are_dropped() {
    let out = normalize(COLD_TEST);
    assert!(!out.contains("Compiling"), "{out}");
    assert!(out.contains("Finished"), "the result line must stay: {out}");
    assert!(out.contains("test result: ok. 1 passed"), "{out}");
    assert_eq!(
        normalize(
            "    Updating crates.io index\n   Locking 40 packages\n     Downloaded axum v0.8.4\nok"
        ),
        "ok"
    );
}

#[test]
fn libtest_thread_ordering_is_removed() {
    // Two runs of the same green suite, the pool having finished them in
    // different orders. Nothing else about the block may move.
    let a = "running 2 tests\ntest tests::ping ... ok\ntest tests::health ... ok\n\ntest result: ok. 2 passed";
    let b = "running 2 tests\ntest tests::health ... ok\ntest tests::ping ... ok\n\ntest result: ok. 2 passed";
    assert_eq!(normalize(a), normalize(b));
    assert!(normalize(a).starts_with("running 2 tests\ntest tests::health"));
    assert!(normalize(a).ends_with("test result: ok. 2 passed"));
}

#[test]
fn a_failing_test_keeps_its_report() {
    let raw = "test tests::ping ... FAILED\n\nfailures:\n\n---- tests::ping stdout ----\n\
        thread 'tests::ping' panicked at src/main.rs:42:9:\nassertion failed: body == \"pong\"";
    assert_eq!(normalize(raw), raw);
}

#[test]
fn ps_output_keeps_the_answer_and_drops_the_bookkeeping() {
    // `ps aux | grep webserver` is a command this model actually issues, and
    // every numeric column in the reply moves between runs.
    let a = "claude   1471629  0.5  0.1 1234567 89012 ?        Sl   22:20   0:03 target/debug/webserver";
    let b = "claude   2233441 12.0  0.2 2345678 91234 ?        Ssl  23:41   0:11 target/debug/webserver";
    assert_eq!(
        normalize(a),
        "claude <pid> <cpu> <mem> <vsz> <rss> ? Sl <start> <time> target/debug/webserver"
    );
    // STAT is real state, so the two differ there and nowhere else.
    assert_eq!(
        normalize(a).replace(" Sl ", " ST "),
        normalize(b).replace(" Ssl ", " ST ")
    );
    assert_eq!(normalize(&normalize(a)), normalize(a));
    // A header, and anything that is not ps output, is left alone.
    let header = "USER       PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND";
    assert_eq!(normalize(header), header);
}

#[test]
fn line_structure_is_preserved() {
    // Nothing but noise lines may disappear: the model counts on `read`-like
    // shapes (`1: use axum`) lining up with what it wrote.
    let raw = "a\n\nb\n";
    assert_eq!(normalize(raw), raw);
}
