# Team Synthesis — Penultimate Root Cause of opencode Degradation

Four agents, independent investigation, converged on the same upstream cause.

## The single penultimate root cause

**MTP K=2 verify-time argmax bypasses the entire pre-sample logit pipeline.**

`crates/spark-model/src/model/trait_impl/verify_b.rs:309-315` runs `ops::argmax_bf16(...)` over raw BF16 logits with NO masking. `crates/spark-server/src/scheduler/verify_k2_step.rs` emits the result via `emit_token(...)`. The full LogitsProcessor pipeline (`mid_word`, `post_close`, `tool_during_think`, `forced_think_end`, `pin_tool_call`, `forced_token`, **`grammar_bitmask`**, F2 confidence) lives only in `decode_logits_seq::process_seq_logits` and is invoked only from the non-MTP path `decode_logits_step.rs:78-97`.

With `--speculative --mtp-quantization bf16` on (the production default for this image), **every emitted token escapes all 8 of those processors**. The visible symptoms cascade:

- Grammar bitmask never applied to verify → target argmax returns tokens that violate the tool-call grammar (Agent B)
- `gs.accept_token(illegal_tok)` silently returns `false` → xgrammar NPDA desyncs from emitted stream → downstream bitmasks corrupted
- Model emits invented tool names (`description`), empty `filePath`, garbled JSON args (Agent A)
- mid_word mask never applied → `</think>` cuts mid-word ("Let mefix", "I'lldo")
- post_close mask never applied → stray `<think>` re-entry mid-response
- pin_to_tool_call never applied → after `</think>`, model wanders into prose instead of emitting `<tool_call>`
- All of these compound at `prompt_tokens ≈ 13k`, the empirical turning point Agent A identified

## Secondary issue (false-positive cascade)

My **`hotfix-3` content-loop watchdog inside `emit_token`** then amputates the partially-emitted JSON. Repeated rejected MTP drafts (`prev_draft=1152` 4× in a row, `prev_draft=90` 1867× in a row across the session) look like a period-N content loop to the watchdog. It fires inside the tool body where xgrammar's structural termination would have resolved naturally — the response ends `finish_reason=length` with a half-formed `{"filePath": "test-rust-ax"` style truncation. Server log lines 2699, 2787, 2973, 3005, 3130, 3162 each pair a watchdog fire with a malformed tool call in the dump.

## Convergence table

| Agent | Angle | Finding | Verdict |
|-------|-------|---------|---------|
| A | Tool-call patterns | Watchdog amputating valid JSON at prompt_tok>13k | downstream symptom |
| B | xgrammar audit | MTP verify uses unmasked argmax; `accept_token` failure silent | **CONFIRMED root** |
| C | Tokenizer/decode | Decoder is correct; "axut"/"withcurl" are model-side FP8 drift | model-side, separate |
| D | MTP rejection | True accept rate 41.5%; **target verify skips entire pipeline** | **CONFIRMED root (same as B)** |

## What is NOT the cause

- The grammar itself is fine (Agent B: `minLength:1` on required strings, tool names restricted to declared set)
- The streaming detokenizer is fine (Agent C: ByteLevel BPE handled correctly via HF DecodeStream)
- The sanitizer hold-back logic is fine (Agent C: never drops, only delays)
- The stop-string hold-back is a no-op when `stop_strings=[]` (Agent C)
- MTP rejection rate is healthy (Agent D: 41.5%, not the 0.32% the misleading log suggested)
- The model's "withcurl"/"axut" garbage IS real model-side FP8 drift, but it's a SEPARATE concern from the tool-call malformations — fixing the pipeline issue won't fix the model's spelling

## Fix plan, prioritized

### P1 — Wire LogitsProcessor pipeline into the MTP verify path (THE fix)

Phase C-2 scaffold (committed `f46d9f4`) is exactly the right architecture for this. The 8 processors are extracted; the `run_pipeline` driver exists. Two integration points are still required:

**Integration point 1: Non-MTP path** — replace the inline block in `decode_logits_seq::process_seq_logits` lines ~62-331 with `run_pipeline(&mut f32_logits, a, &ctx)`. This is the originally-planned C-2 wiring.

**Integration point 2: MTP verify paths** — for each of `verify_k2_step.rs`, `verify_k3_step.rs`, `verify_dflash_step.rs`, `mtp_step.rs`, `spec_step.rs`:
- D2H-copy verify-position-0 logits (and position-1 for K>=2)
- Run `run_pipeline` over them
- Compute argmax on the **processed** logits
- Compare to draft → accept/reject

Cost ≈ 0.8 ms/step per Agent D's estimate. Worth it.

### P2 — Restore `!a.inside_tool_body` gate on content-loop watchdog

In both `decode_logits_content.rs::handle_content_token` AND `emit_step.rs` (the hotfix-3 mirror I added). Inside a tool body, xgrammar guarantees structural termination — the watchdog is layered overhead and produces false positives on legitimate JSON. Keep the `parameter>\n` real-loop catch via a higher MIN_REPEATS threshold gated on grammar state being terminal.

### P3 — Make `gs.accept_token` failure non-silent

`emit_step.rs:107-111` — when `accept_token(tok)` returns `false` outside thinking, currently silent. Add `tracing::warn!` + set `a.finished=true`. Today's silent desync corrupts all downstream masks for the rest of the response. After P1 lands, P3 becomes a defense-in-depth check that should ~never fire.

### P4 — K2 ACCEPT-rate periodic summary log

Replace `is_multiple_of(50)` accept gate at `verify_k2_step.rs:119` with a periodic `info!("K2 last 100 steps: accepts=X rejects=Y rate=Z%")`. Prevents future misdiagnosis like the 0.32% scare today.

### P5 — Flush `reasoning_tag_scan_buf` on `</think>`

`handle_token.rs:89-127` — the `tok == think_end_token_id` branch never flushes the reasoning sanitizer tail. Up to ~18 trailing bytes of every thinking block lost. 5-line patch (Agent C provided diff).

### P6 — Investigate `param_as_function_salvage` synthesizing tool names

`server.log:3106, 3184` — when a truncated envelope contains `"description": "..."`, the salvage path treats `description` as the function name, surfacing a phantom tool to opencode. After P1+P2 land, truncations should disappear, but the salvage path's name-promotion logic is still worth auditing.

### Fast mitigation (deploy now while P1 is built)

Serve Qwen3.6 with `--num-drafts 0` (or `--speculative` removed). Disables MTP entirely; every token flows through `process_seq_logits` which runs the full pipeline. Throughput drops to ~30 tok/s baseline (from ~60 with MTP working), but agentic flows become correct.

## Not in scope of this fix

- Phase 2c FP8 KV drift at long-context (causes the "axut"/"withcurl" model-side errors). Needs deeper precision work — see `project_qwen36_phase2b_softmax_expf.md`.

## Next session checkpoints

1. Ship fast-mitigation image (`--num-drafts 0`) for immediate user testing
2. Land P1 (LogitsProcessor pipeline wired into MTP verify) — multi-day, careful
3. P2 + P3 + P4 + P5 can ride along with P1 image
