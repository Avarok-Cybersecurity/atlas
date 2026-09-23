// SPDX-License-Identifier: AGPL-3.0-only

//! Every prefill-command sender must hand the workers their vision rows.
//!
//! The `0xFFFFFFF0` preamble is written out by hand at five places in the
//! scheduler, and the worker's handler reads a fixed sequence of words for
//! each one. A sixth sender that copies the four existing lines and stops
//! there does not degrade gracefully: on a text-only model nothing happens, on
//! a vision-capable model the worker blocks in a collective the head never
//! enters, and in between — if the ordering ever lets it through — rank 0
//! splices the encoder's rows while the workers keep raw `<|image|>`
//! embeddings and the per-layer all-reduce mixes the two. That last state is
//! the one worth a test: the model stays FLUENT and answers about a picture it
//! was never shown, so every check that looks at rank 0 alone passes
//! (rsafier, PR #1066 — Qwen3.8 read **HARBOR** as *Superman* with a 0.998
//! tower cosine and an exact splice, on one rank of two).
//!
//! Source-scanning rather than runtime, because the failure is structural: the
//! two halves of a wire protocol written in different files with nothing
//! tying them together. A runtime test would need two ranks and would only
//! cover the paths it happened to exercise.

use std::path::{Path, PathBuf};

/// Lines allowed between the command send and the vision sync — enough for
/// the three argument words, the token bulk broadcast, and the comments that
/// explain them, but not enough to land in an unrelated block.
const MAX_GAP: usize = 40;

const SEND: &str = "ep_broadcast_cmd_for_seq";
const PREFILL_CMD: &str = "0xFFFFFFF0";
const SYNC: &str = "ep_sync_vision_embeds";

fn scheduler_sources() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/scheduler");
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("scheduler sources must be readable") {
            let path = entry.expect("readable dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    assert!(!out.is_empty(), "scanned no scheduler sources");
    out
}

/// A line that actually SENDS the prefill command, not one that mentions it.
fn is_prefill_send(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or(line);
    code.contains(SEND) && code.contains(PREFILL_CMD)
}

fn is_sync(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or(line);
    code.contains(SYNC)
}

#[test]
fn every_prefill_broadcast_is_followed_by_the_vision_row_sync() {
    let mut senders = 0usize;
    let mut unpaired = Vec::new();
    for path in scheduler_sources() {
        let text = std::fs::read_to_string(&path).expect("source readable");
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !is_prefill_send(line) {
                continue;
            }
            senders += 1;
            let end = (i + 1 + MAX_GAP).min(lines.len());
            if !lines[i + 1..end].iter().any(|l| is_sync(l)) {
                unpaired.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        senders >= 5,
        "expected at least the five known prefill-command senders, found {senders} — if a send \
         site moved, this scan is no longer looking where the protocol is written"
    );
    assert!(
        unpaired.is_empty(),
        "these prefill-command send sites do not hand the workers their vision rows within \
         {MAX_GAP} lines:\n  {}\n\nAdd `model.ep_sync_vision_embeds(&<the prompt tokens>)?;` \
         after the token broadcast. The worker reads that word for EVERY 0xFFFFFFF0 on a \
         vision-capable model; omitting it blocks the worker, and in the orderings where it \
         does not, leaves the ranks all-reducing different image embeddings while the model \
         stays fluent.",
        unpaired.join("\n  ")
    );
}

/// The worker half: exactly one matching receive, in the `0xFFFFFFF0` arm.
///
/// Guards the other direction — a sync added to the head with no counterpart
/// desynchronises the wire just as thoroughly.
#[test]
fn the_worker_reads_the_vision_rows_once_per_prefill_command() {
    let worker = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../spark-model/src/model/impl_a2.rs")
        .canonicalize()
        .expect("the worker command loop must exist");
    let text = std::fs::read_to_string(&worker).expect("source readable");
    let calls = text.lines().filter(|l| is_sync(l)).count();
    assert_eq!(
        calls, 1,
        "the worker's command loop must call {SYNC} exactly once — found {calls}. Two calls \
         read two words off the wire for one the head sent; none leaves the workers spliceless."
    );
    let arm = text
        .find("0xFFFFFFF0 => {")
        .expect("the prefill arm must exist");
    let sync_at = text.find(SYNC).expect("the sync call must exist");
    assert!(
        sync_at > arm,
        "the vision sync must live INSIDE the 0xFFFFFFF0 arm; reading it anywhere else pairs \
         it with a different command"
    );
}
