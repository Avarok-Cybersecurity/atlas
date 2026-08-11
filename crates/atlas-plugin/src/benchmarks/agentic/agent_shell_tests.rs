// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::benchmarks::agentic::agent::tests::{cfg, sandbox};

// ── the memory bound ───────────────────────────────────────────────

#[test]
fn a_runaway_command_cannot_grow_the_capture_without_bound() {
    // `yes` writes as fast as the drain empties the pipe, and the drain used to
    // append every byte to a `Vec` that stopped growing only when the command
    // did. 64 MiB here stands in for the tens of gigabytes a 180 s default
    // timeout allows; what matters is that the bound does not depend on it.
    let mut capture = Capture::default();
    for _ in 0..64 {
        capture.push(&[b'x'; 1 << 20]);
    }
    assert!(
        capture.held() <= 2 * CAPTURE_END,
        "held {} bytes",
        capture.held()
    );
    assert_eq!(capture.dropped, (64 << 20) - 2 * CAPTURE_END);
}

#[test]
fn the_capture_keeps_both_ends_and_says_what_it_dropped() {
    let mut capture = Capture::default();
    capture.push(b"error[E0433]: first\n");
    capture.push(&vec![b'-'; 4 * CAPTURE_END]);
    capture.push(b"\nerror: could not compile `pingpong`\n");
    let text = capture.text();
    assert!(text.starts_with("error[E0433]: first\n"), "head lost");
    assert!(
        text.ends_with("error: could not compile `pingpong`\n"),
        "tail lost"
    );
    assert!(
        text.contains("bytes dropped from the middle"),
        "{}",
        &text[..64]
    );
}

#[test]
fn a_character_split_across_the_seam_is_not_mangled() {
    // Head and tail decoded separately would each see half of the two-byte `é`
    // and emit a replacement character.
    let mut capture = Capture::default();
    capture.push(&"é".repeat(CAPTURE_END).into_bytes());
    assert!(!capture.text().contains('\u{fffd}'));
}

// ── truncation ─────────────────────────────────────────────────────

#[test]
fn truncation_is_idempotent_and_stays_inside_the_output_cap() {
    // The agent loop truncates every tool result, including the ones that came
    // back from here already truncated. Overshooting the cap by the length of
    // the elision note made that second pass cut the string again, nesting one
    // note inside another and dropping the text on both sides of the first.
    for text in [
        "a".repeat(100_000),
        "é".repeat(100_000),
        "line\n".repeat(20_000),
    ] {
        let once = truncate(&text);
        assert!(once.len() <= MAX_TOOL_OUTPUT, "{} bytes", once.len());
        assert_eq!(truncate(&once), once, "truncation is not idempotent");
        assert_eq!(once.matches("elided from the middle").count(), 1);
    }
}

// ── process containment ────────────────────────────────────────────

#[cfg(unix)]
#[tokio::test]
async fn a_timed_out_command_takes_the_children_it_forked_with_it() {
    // Killing `sh` alone left the fork running. It is not idle: `cargo build`
    // is the command that actually times out here, and it kept compiling —
    // against the same warm target dir, on the CPU the next iteration is timed
    // on, until the end of the whole iteration when `reap` finally runs.
    let sb = sandbox("orphan");
    let mut c = cfg(sb.clone());
    c.command_timeout = Duration::from_millis(300);
    let out = run_shell(
        &c,
        "(sleep 2; touch survivor) & sleep 60",
        c.command_timeout,
    )
    .await
    .unwrap();
    assert!(out.contains("timed out"), "{out}");
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(
        !sb.join("survivor").exists(),
        "a forked child outlived the timeout that killed its shell"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn a_setsid_server_is_left_for_the_reaper() {
    // The counterpart: the prompt tells the model to detach its server, so the
    // group kill must NOT reach it. Tearing it down here would end the run's
    // server the moment any unrelated command timed out.
    let sb = sandbox("detached");
    let mut c = cfg(sb.clone());
    c.command_timeout = Duration::from_millis(300);
    let out = run_shell(
        &c,
        "setsid sh -c '(sleep 2; touch detached) > /dev/null 2>&1' & sleep 60",
        c.command_timeout,
    )
    .await
    .unwrap();
    assert!(out.contains("timed out"), "{out}");
    tokio::time::sleep(Duration::from_secs(4)).await;
    assert!(
        sb.join("detached").exists(),
        "a detached server must survive an unrelated command's timeout"
    );
    super::super::reap(&sb).await;
}
