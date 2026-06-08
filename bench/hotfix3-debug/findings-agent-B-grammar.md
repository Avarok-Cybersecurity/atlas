# Agent B — Grammar / xgrammar Investigation

## TL;DR

The grammar is compiled correctly (json_schema with `minLength:1` on required strings, tool names restricted via fixed `<function=NAME>` literals) and IS applied on the main decode path. **Root cause is H3+H7: MTP speculative decoding bypasses the grammar bitmask on the verify side.** K=2 verify returns raw argmax over the full vocab for BOTH positions and `emit_token` calls `gs.accept_token` after the fact — when the verified token violates grammar, `accept_token` silently returns `false`, the token is still emitted, and the xgrammar NPDA desyncs from the output stream, corrupting all subsequent masks. With 2497 K=2 verifies in the hotfix3 run and ~50% bonus-token emission rate, hundreds of tokens were sampled unmasked, allowing the model to emit invented tool names (`description`), empty fields, etc. Drafts ARE masked (`mtp_head.rs:285-300`); verifies are NOT.

## Per-hypothesis verdict

- **H1 (empty strings allowed): UNLIKELY** — `enforce_min_length_on_required_strings` at `crates/spark-server/src/grammar/schema.rs:60-95` injects `minLength:1` on every required string property.
- **H2 (unbounded tool names): UNLIKELY** — `compile_qwen3_coder_tool_grammar` at `crates/spark-server/src/grammar/compile_tools.rs:211` builds literal `<function={tool_name}>` per-tool begin strings; xgrammar TagDispatch restricts continuations to these alternatives.
- **H3 (MTP samples without bitmask): CONFIRMED** — `verify_k2_step.rs:36-47`, `:80-82`, `:142` emit `v0`/`v1` raw argmax. `verify_b.rs:309-315` is GPU `argmax_bf16` with no mask path.
- **H4 (parser-side bug): UNLIKELY** — `parse_qwen3_coder_call` in `parse_single_b.rs:6` parses both JSON and XML faithfully; validator rejects `description` in `validation.rs:322-334`, matching server log line at 19:45:20.
- **H5 (schema sanitization drops constraints): UNLIKELY** — `sanitize_schema_for_grammar` at `schema.rs:109-300` preserves required/properties recursively.
- **H6 (FSM state inherited across turns): NOT INVESTIGATED** — `compile_grammar_state` runs per request (`emit_step.rs:277`), so a fresh state is built per turn; possible secondary issue but didn't trace.
- **H7 (draft sampling bypasses bitmask): UNLIKELY for drafts / CONFIRMED for verify** — drafts respect mask via `mtp_head.rs:285-300`. Verify side has no mask at all.

## Strongest hypothesis: H3 — verify uses unmasked argmax

Code references:
1. `/workspace/atlas-mtp/crates/spark-server/src/scheduler/verify_k2_step.rs:46-47` — `let [v0_argmax, v1_argmax] = result;` then `emit_token(a, v1, …)` at line 82 emits the unmasked verified token.
2. `/workspace/atlas-mtp/crates/spark-model/src/model/trait_impl/verify_b.rs:309-315` — `ops::argmax_bf16(…)` over raw BF16 logits, no bitmask invocation.
3. `/workspace/atlas-mtp/crates/spark-server/src/scheduler/emit_step.rs:107-111` — `gs.accept_token(tok)` is called for every emitted token (verify or normal); failure silently returns false (see `state.rs:135-140`), allowing matcher desync.
4. `/workspace/atlas-mtp/crates/spark-server/src/scheduler/decode_logits_seq.rs:319-331` — the ONLY bitmask-application site in the codebase. Only reached on non-MTP decode path.
5. `/workspace/atlas-mtp/bench/hotfix3-debug/server.log` — 2497 K2 verify events, 5+ validation failures (`Unknown tool 'description'`, `non-empty 'filePath'`).

Also: `verify_dflash_step.rs:41`, `verify_k3_step.rs:31`, `verify_k4_step.rs` all share the same unmasked-argmax pattern.

## Recommended fix shape

1. **Apply grammar bitmask to verify-time argmax.** In `verify_b.rs/verify_c.rs/verify_d.rs/verify_dflash`, upload the per-position grammar bitmask (or D2H-copy K logit vectors and CPU-mask with `apply_bitmask_to_logits`, ~1 ms for K=2) before computing argmax. The bonus tokens then come from the same constrained distribution drafts use.
2. **Make `gs.accept_token` failure non-silent.** In `emit_step.rs:110`, when `accept_token` returns false outside thinking, warn + either drop the token and force a re-decode or terminate cleanly. Today's silent desync corrupts all downstream masks for the rest of the response.
3. **Per-position mask refresh for K>=2 verify.** Use the `truncate_drafts_at_grammar_boundary` advance+rollback pattern (`spec_step.rs:382`) to compute a fresh bitmask for each of the K verify positions — single-snapshot masks are correct only for position 0.
