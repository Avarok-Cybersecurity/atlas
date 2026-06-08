# Agent C — Tokenizer / Streaming Detokenizer / Sanitizer Findings

## TL;DR

The "withcurl" / "andprove" / "toverify" / "RustAxum" / "pongendpoint" glue artifacts all appear in the **reasoning_content** path, not the content path. The Atlas content detokenizer is the SOTA HF `DecodeStream` (O(1) per token, BPE-correct). The reasoning detokenizer uses full re-decode + byte-slice, which is mathematically equivalent to streaming for ByteLevel BPE. Neither path drops or glues characters in the cases observed.

**Primary verdict:** these glue artifacts are **model-side (H6 confirmed)** — FP8 KV / MoE routing drift. The model is emitting `curl` after `Ġwith` instead of `Ġcurl` (missing leading-space byte). Direct evidence in `dump.jsonl`: "axut" 57×, "axum" too — the model is also misspelling "axum" as "axut" (pure model error, no Atlas code path could produce that letter substitution).

**One latent Atlas bug found** (not the cause of artifacts but worth fixing): the reasoning emit pipeline drains `state.emitted = stable_end` before sanitizer hold-back runs; at `</think>` fire, `reasoning_tag_scan_buf` is **never flushed**, losing up to ~18 trailing bytes of the final thinking block.

## Per-hypothesis verdicts (with file:line evidence)

**H1 — BPE back-merge in incremental decode → UNLIKELY.** Qwen3.6 uses `ByteLevel` decoder (verified at `/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/.../tokenizer.json` — `decoder.type = "ByteLevel"`, `trim_offsets=false`). HF ByteLevel decoder is flat byte-char→byte map + concat — no back-merge possible. `full[state.emitted..stable_end]` at `handle_token.rs:148-150` is always a linear suffix.

**H2 — Sanitizer tag_max hold-back eats normal text → UNLIKELY for this stream.** Qwen3.6 → `qwen3_coder` parser (`crates/spark-server/tool_defaults.toml:13`). qwen3_coder's `leak_markers` is non-empty (`crates/spark-server/src/tool_parser/qwen3_coder.rs:137-221`), so sanitizer fast-path at `sanitizer.rs:46-48` is NOT taken. `tag_max ≈ 19` (longest = `</function_results>`), `hold ≈ 18`. The "no match" branch at `sanitizer.rs:189-205` holds the tail but **never drops** — bytes re-emerge when buffer exceeds hold. Normal prose is only delayed, not lost mid-stream.

**H3 — Reasoning-chunk emission off-by-one / UTF-8 slip → UNLIKELY.** `state.emitted` is in bytes (consistent with `stable_end = full.trim_end_matches('\u{FFFD}').len()` also bytes). FFFD trim only strips trailing 3-byte FFFD codepoints; both endpoints always char-aligned (`handle_token.rs:146-150`).

**H4 — FFFD trim eating valid bytes → UNLIKELY.** `trim_end_matches` strips only trailing FFFD; next tick completes the multi-byte and emits the now-stable suffix. Standard pattern, no Unicode in artifacts.

**H5 — opencode markdown renderer eats whitespace → REJECTED.** The same artifacts appear in `bench/hotfix3-debug/dump.jsonl` directly inside the `reasoning_content` JSON string field. Atlas is emitting these bytes-for-bytes to the wire. opencode rendering is not the cause.

**H6 — Model precision (FP8 KV drift) → wrong token emitted → CONFIRMED as primary cause.** Direct evidence in `dump.jsonl`:
- "axut" appears **57 times**, "axum" also present — model is inconsistently misspelling the user's literal "Axum"
- "RustAxum" 36×, "Rust Axum" 19× — same model emitting both
- Every glue artifact follows pattern `Ġword<then>word_no_Ġ` (e.g., `Ġwith` + `curl` instead of `Ġwith` + `Ġcurl`)

## Specific dump.jsonl fragments + token-boundary cause

| Fragment | Likely BPE tokens | Cause |
|---|---|---|
| `ping/pongendpoint` | `Ġping`,`/`,`pong`,`endpoint` (no Ġ on endpoint) | Model dropped Ġ-prefix |
| `tests andprove` | `Ġtests`,`Ġand`,`prove` | Same |
| `curl toverify` | `Ġcurl`,`Ġto`,`verify` | Same |
| `server withcurl` | `Ġserver`,`Ġwith`,`curl` | Same |
| `withoutworrying` | `Ġwithout`,`worrying` | Same |
| `axut` (57×) | model emits wrong token of `Ax|um` split | Pure FP8 drift |
| `test-rust-axut-vit` | 5-token corruption chain of `test-rust-axum-v19` | Same |

There is NO Atlas code path that splices `Ġword`+`word` and drops a leading space; the streaming decoder renders what the model emits.

## Recommended fix shape

**For these artifacts:** no Atlas fix needed in tokenizer/sanitizer paths. The right place is Phase 2c late-layer FP8 KV / MoE routing work (see Agents A/B sibling investigations). The note in `MEMORY.md`'s `project_qwen36_phase2b_softmax_expf.md` already identifies late-attn-layers FP8 KV drift as the known weak point.

**Real latent Atlas bug found (one-line fix):** in `handle_token.rs:89-127` (the `tok == think_end_token_id` branch), after the residual emit, also flush `state.reasoning_tag_scan_buf`:

```rust
if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
    let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
    if !tail.trim().is_empty() {
        sse_events.push(reasoning_chunk(tail));
    }
}
```

This prevents the last ~18 bytes of every thinking block from being silently dropped on `</think>`. Not the cause of the current artifacts, but a real bug.

## Key file/line references

- `/workspace/atlas-mtp/crates/spark-server/src/api/chat_stream/handle_token.rs:89-269` — reasoning path (full re-decode + slice + sanitizer)
- `/workspace/atlas-mtp/crates/spark-server/src/api/chat_stream/handle_token.rs:272-289` — content path (HF DecodeStream, correct)
- `/workspace/atlas-mtp/crates/spark-server/src/api/chat_stream/handle_token.rs:559-599` — `apply_stop_string_holdback` (correct, true no-op when buffer_len=0)
- `/workspace/atlas-mtp/crates/spark-server/src/api/sanitizer.rs:36-209` — `sanitize_content_chunk` (hold-back correct, but reasoning caller doesn't flush)
- `/workspace/atlas-mtp/crates/spark-server/src/tokenizer.rs:82-101` — HF DecodeStream wrapper
- `/workspace/atlas-mtp/crates/spark-server/src/tool_parser/qwen3_coder.rs:137-221` — Qwen3.6 leak markers (tag_max ≈ 19)
- `/workspace/atlas-mtp/crates/spark-server/tool_defaults.toml:13` — Qwen3.6 → `qwen3_coder`
