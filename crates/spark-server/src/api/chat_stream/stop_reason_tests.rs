// SPDX-License-Identifier: AGPL-3.0-only

//! THE invariant behind the `stop_reason` extension field:
//!
//! **The terminal delta must carry the guard name, and carrying it must
//! not tempt anyone into moving `finish_reason`.**
//!
//! #927 / #1000 / #1002. Round-13 cell V (`--prefill-varlen-batch`)
//! produced 6 of 16 responses truncated at 49 tokens by the content-loop
//! / fuzzy / SimHash degeneration watchdogs, and every one of them
//! reported `finish_reason: "length"` on the OpenAI-compatible wire. The
//! client could not tell a quality cut from a budget stop — it is the
//! same bytes either way, and the two want opposite recoveries (raise
//! `max_tokens` and continue, vs. reroll the turn).
//!
//! The fix `guard_stop_wire_reason` named was an extension FIELD, not a
//! new `finish_reason` value, and both halves of that sentence are
//! load-bearing:
//!   * the field must actually be populated — `state.guard_stop` is
//!     already computed at the terminal-delta site (it is the same value
//!     the `--dump` body reports), so the only way this breaks is a
//!     refactor quietly dropping it on the way out;
//!   * `finish_reason` must NOT move. Relabelling guard cuts to `"stop"`
//!     measurably cost 2/10 then 6/10 episodes of the agentic gate, and
//!     minting a fifth enum value hard-fails strictly typed clients
//!     (Rust `async-openai` fails deserialization outright; pydantic-ai
//!     raised on OpenRouter's non-standard `"error"`).
//!
//! Behavioural coverage of the serialized shape lives with the encoder
//! (`openai::encode_stream::tests` for the chunk, `openai::tests::
//! chat_wire` for the blocking body). What those cannot see is whether
//! the PRODUCER still hands the encoder a real value, because reaching
//! `handle_done` behaviourally needs a whole `StreamCtx` and a live
//! scheduler channel. That property is structural, so this test is too —
//! same reasoning, and same in-tree precedent (`cancel_guard_tests`), as
//! the invariant one directory over.

/// The Done arm, read as source. Kept as a `const` so a rename of the
/// file is a compile error here rather than a silently vacuous test.
const HANDLE_DONE: &str = include_str!("handle_done.rs");

/// Slice the source from `needle` for `window` bytes, clamped to a char
/// boundary. Used to keep each assertion scoped to one statement group
/// instead of matching anywhere in a 460-line file.
fn window_after(src: &str, needle: &str, window: usize) -> String {
    let start = src.find(needle).unwrap_or_else(|| {
        panic!("anchor vanished from handle_done.rs: {needle:?} — update this test with the code")
    });
    let mut end = (start + window).min(src.len());
    while !src.is_char_boundary(end) {
        end -= 1;
    }
    src[start..end].to_string()
}

/// The source of one free function, from its signature to the closing
/// brace in column 0. Exact rather than windowed, so an assertion over
/// "everything in this function" cannot quietly start reading the next
/// item's doc comment when the file grows.
fn fn_body(src: &str, signature: &str) -> String {
    let start = src.find(signature).unwrap_or_else(|| {
        panic!(
            "anchor vanished from handle_done.rs: {signature:?} — update this test with the code"
        )
    });
    let rest = &src[start..];
    let end = rest
        .find("\n}\n")
        .unwrap_or_else(|| panic!("no column-0 closing brace after {signature:?}"));
    rest[..end].to_string()
}

/// POSITIVE. The terminal `StreamDelta::Finish` names the guard, and it
/// names it by reading `state.guard_stop` — the MERGED view of both
/// guard families. The scheduler's `ActiveSeq::guard_stop` arrives on
/// `StreamEvent::Done` and is folded in with `.or()` in
/// `chat_stream::mod`; the stream-side watchdogs in `handle_token` write
/// the same field. Recomputing the name from anything else here would
/// silently cover only one family — which is exactly the shape of the
/// bug that made two shipped fixes inert (see the `.or()` note at the
/// merge site).
#[test]
fn terminal_delta_carries_the_merged_guard_name() {
    let finish = window_after(HANDLE_DONE, "deltas.push(StreamDelta::Finish {", 2_048);
    assert!(
        finish.contains("stop_reason: state.guard_stop,"),
        "the terminal delta must carry `stop_reason: state.guard_stop` — without it the \
         guard name never reaches the wire and a degeneration cut is indistinguishable \
         from a budget stop (#927 / #1000 / #1002, round-13 cell V: 6/16 responses cut \
         at 49 tokens, all reporting bare `finish_reason: \"length\"`).\n\nFinish block:\n{finish}"
    );
}

/// SSOT. The delta and the `--dump` body must report the SAME guard, by
/// reading the same expression. They are ~60 lines apart and describe
/// one event; if they can drift, a dump captured while chasing a
/// degeneration will disagree with the wire the client actually saw, and
/// the receipt stops being evidence.
#[test]
fn dump_body_and_wire_field_read_the_same_source() {
    assert!(
        HANDLE_DONE.contains("\"guard_stop\": state.guard_stop,"),
        "the --dump body must keep reading `state.guard_stop`"
    );
    assert!(
        HANDLE_DONE.contains("stop_reason: state.guard_stop,"),
        "the wire field must read the same expression the dump does"
    );
}

/// NEGATIVE, and the point of the whole design. Adding a place to say
/// WHICH guard fired is exactly when someone reaches for a nicer
/// `finish_reason` — "we can report `stop` now, the detail is in
/// `stop_reason`". That trade was already measured and lost: the
/// agentic gate fell to 2/10, then 6/10. The guard rung must still
/// resolve to `"length"`.
#[test]
fn guard_rung_still_resolves_to_length() {
    let rung = window_after(
        HANDLE_DONE,
        "} else if stream_guard_stop.is_some() {",
        1_024,
    );
    let body = rung
        .split_once("} else {")
        .map(|(before, _)| before.to_string())
        .unwrap_or(rung);
    assert!(
        body.contains("\"length\""),
        "the stream-guard rung must still resolve to \"length\" — a degeneration cut is a \
         truncation, and every client keys truncation recovery on it (openai-python raises \
         LengthFinishReasonError, Instructor raises IncompleteOutputException, aider's \
         continuation fires). `stop_reason` is a companion to that value, never a licence \
         to change it.\n\nRung:\n{body}"
    );
    assert!(
        !body.contains("\"stop\""),
        "guard cuts must not report \"stop\": measured at 2/10 then 6/10 agentic-gate \
         episodes.\n\nRung:\n{body}"
    );
}

/// NEGATIVE. No fifth `finish_reason` value was minted alongside the new
/// field. `"timeout"` is the one deliberate, shipped exception
/// (`ir::FINISH_REASON_TIMEOUT`) and is referenced by its constant, not
/// spelled as a literal, so any NEW non-standard string shows up here as
/// a bare quoted word in the resolver.
///
/// ★ Scoped to the function BODY, not a byte window. The first version
/// took 1200 bytes from the signature and ran off the end of the `fn`
/// into `wire_finish_reason_tests`, where the phrase "length is a lie"
/// appears in a doc comment — a failure with nothing wrong in the code
/// it was guarding. A structural test that over-reads its own subject
/// is worse than no test: it trains people to ignore it.
#[test]
fn no_new_nonstandard_finish_reason_value() {
    let resolver = fn_body(HANDLE_DONE, "fn resolve_wire_finish_reason");
    const ALLOWED: &[&str] = &["\"length\"", "\"tool_calls\"", "\"stop\""];
    let mut rest = resolver.as_str();
    while let Some(open) = rest.find('"') {
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else { break };
        let lit = &rest[open..open + close + 2];
        assert!(
            ALLOWED.contains(&lit),
            "unexpected finish_reason literal {lit} in resolve_wire_finish_reason — typed \
             clients hard-fail on unknown enum VALUES, which is why the guard detail rides \
             the `stop_reason` FIELD instead (#1002). Carry \"timeout\" via \
             `ir::FINISH_REASON_TIMEOUT`, as the deadline rung already does."
        );
        rest = &rest[open + close + 2..];
    }
}
