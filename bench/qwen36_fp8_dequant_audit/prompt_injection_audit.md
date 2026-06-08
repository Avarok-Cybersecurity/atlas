# Atlas Prompt-Injection Audit

**Scope**: `/workspace/atlas-mtp/crates/spark-server/src/` — does Atlas
mutate the prompt before the model sees it?

**Verdict**: **YES — Atlas mutates the user-supplied message array in
several places before the chat template is rendered.** All mutations
happen in `chat_completions_inner` *before* `tokenizer.apply_chat_template_openai`
is called (template.rs:103/120), so the LM definitely sees them.
None of these are opt-in per-request; they are always-on policy
heuristics (some gated by counters / heuristics on prior turns).

## Always-on injections (request never opts out)

### 1. Tool-parser behavioral system prompt — `api/chat/mod.rs:106-118`
Whenever `tools_active` is true, the parser's `system_prompt()` text is
**prepended to (or pushed as) `messages[0]`** before templating.
Hermes example (`tool_parser/hermes.rs:33-41`): adds full "You are a
function calling AI model... `<tools>{json}</tools>`... return JSON in
`<tool_call>` tags" scaffolding. The client never sent this — it's
always added when tools are present. Pattern:
`first.content.text = format!("{}\n\n{}", tool_prompt, first.content.text)`.

### 2. F7 stall reminder — `chat_phases.rs:104-117` → `failures/stall.rs:101-117`
If `build_f7_stall_reminder` detects ≥N identical tool-arg paths across
turns, `append_f7_reminder_to_last_user` mutates the **last user/tool
message's body in place** with a `<system-reminder>... You have already
issued the following tool calls multiple times...</system-reminder>`
block. Fires automatically; no flag.

### 3. F23 progress-tracker reminder — `chat_phases.rs:120-128` → `failures/stall.rs:130-155`
On low forward-progress score, `prepend_reminder_to_system` wraps text
in `<atlas_runtime_notice>...</atlas_runtime_notice>` and **prepends to
`messages[0]`** (or inserts a synthetic system msg if none).

### 4. F31 synthesized tool result — `chat_phases.rs:131-137`
Inserts a fabricated `[atlas-stall-guard]` tool_result message at
"refuse" threshold. The model sees a tool response that the client
never sent.

### 5. F32 duplicate failed tool_result — `chat_phases.rs:140-142`
Duplicates the most-recent failed tool_result at the conversation tail
to re-surface it for the model.

### 6. F39 circuit-breaker banner — `chat_phases.rs:144-154`
On repeated permanent-failure tool retries, prepends a banner via
`prepend_reminder_to_system`.

### 7. F49 duplicate-write banner + F50 original-error append —
`chat_phases.rs:157-169`. Banner prepended to system, original error
appended at conversation tail.

### 8. F35/F52 failure_recovery clause — `chat_phases.rs:172-188`
When the most recent message is a tool error, prepends a hardcoded
`<failure_recovery>...</failure_recovery>` block to the system message
that lists steps (a)/(b)/(c) the model should take.

### 9. F29 environment_facts injection — `chat_phases.rs:191-199` →
`failures/stall.rs:227-...`. After ≥2 `command not found` observations
of a binary, prepends `<environment_facts>...` block to system message.

### 10. Loop-detector "IMPORTANT" hint — `chat/loop_detect.rs:131-146`
On loop/spinning verdict, appends an `<IMPORTANT>` block to the last
message: "Your recent turns have produced output very similar...".

### 11. Task-pin verbatim-goal reminder — `chat/loop_detect.rs:149-182` →
`task_pin.rs:62-83`. **THIS IS THE EXACT PATTERN THE USER WORRIED
ABOUT.** When `should_inject` fires (`loop_active || ≥3 tool errors`),
appends `<system-reminder>You have had N consecutive failed or repeated
tool calls... The user's ORIGINAL request was: «{verbatim quote}»...
</system-reminder>` to the last two tool/user messages (or last
message). Triggered automatically by heuristic; no per-request opt-in.

### 12. Observation-mask body rewrite — `chat/mod.rs:162-182` →
`observation_mask.rs:110-143`. Walks history, REPLACES stale
tool/user error bodies with `[stale tool failure N/M: <excerpt>… full
body elided…]` envelopes. Most recent 2 errors are preserved.
**This rewrites past messages the user/client sent.**

### 13. Responses-API instructions stacking — `openai/responses_lowering.rs:48-57`
When resuming via `previous_response_id`, drops prior synthetic-system
messages and inserts new `instructions` as synthetic-system at pos 0.

### 14. `<atlas_runtime_notice>` wrapper — `failures/stall.rs:134-148`
All `prepend_reminder_to_system` content is wrapped in this XML tag,
which is fabricated by Atlas; the client never emits it.

## Template-level (gated, not always-on)

### 15. Spontaneous `<think>` mechanism — `scheduler/phase_promote_prefills.rs:40-159`
**Does NOT inject tokens.** If thinking is disabled but the first
sampled token is `think_start`, the scheduler relabels the run as
"inside thinking" with a budget. The token came from the model.

### 16. `<think></think>\n\n` raw-completions prefix — `api/completions.rs:74-78`
For `/v1/completions` only, prepends literal `"<think></think>\n\n"`
to the user's raw prompt **when the prompt does not already contain
`</think>`** and the tokenizer supports thinking. This is a user-visible
prompt transformation on the `/v1/completions` endpoint.

### 17. Jinja templates — `tokenizer/chat_impl.rs:112-219`
Standard `apply_chat_template`. Templates (`jinja-templates/*.jinja`)
can add their own `<think>` / tool-format scaffolding, e.g.
`nemotron_h.jinja:212` adds `<|im_start|>assistant\n<think></think>\n<tool_call>\n`
suffix when `tools and not disable_tool_steering`. Gated by MODEL.toml
`[behavior].disable_tool_steering`.

## Confirmed NOT injection

- `decode_logits_seq.rs:238-261` ("Change 3b"): masks all non-tool-call-start
  logits to `-inf`. Model still samples; no token injected.
- `api/sanitizer.rs:51-...`: post-decode output stripping only; not pre-decode.
- F70 anchor bias (`decode_logits_seq.rs:263-279`): reverted, dead code.

## Summary table

| Item | Where | Mutates prompt? | Per-request opt-in? |
|------|-------|----------------|---------------------|
| Tool-parser system prompt | chat/mod.rs:106 | yes, prepend to sys | no — auto when tools |
| F7 stall reminder | failures/stall.rs:101 | yes, last user/tool | no — auto heuristic |
| F23 progress reminder | failures/stall.rs:130 | yes, prepend sys | no — auto heuristic |
| F31 synth tool_result | failures/circuit.rs:110 | yes, fabricated msg | no — auto heuristic |
| F32 dup tool_result | chat_phases.rs:140 | yes, dup tail | no — auto |
| F39 circuit banner | chat_phases.rs:153 | yes, prepend sys | no — auto |
| F49+F50 dup-write | chat_phases.rs:165-168 | yes, prepend+append | no — auto |
| F35 failure_recovery | chat_phases.rs:184 | yes, prepend sys | no — auto |
| F29 env_facts | failures/stall.rs:261 | yes, prepend sys | no — auto |
| Loop hint | chat/loop_detect.rs:143 | yes, append last msg | no — auto |
| **task_pin goal reminder** | task_pin.rs:62 | **yes, "user's ORIGINAL request was…"** | **no — auto** |
| Observation mask | observation_mask.rs:110 | yes, rewrites past bodies | no — auto |
| `/v1/completions` `<think>` prefix | completions.rs:74 | yes, literal prepend | conditional on tokenizer + prompt |
| Responses instructions | responses_lowering.rs:48 | yes, sys insert | per Responses spec |

**Bottom line**: Atlas runs a stateful agentic-failure-handling layer
that mutates the prompt array in ≥13 distinct ways before tokenization.
None of these is exposed to the client as an opt-out; they are all
keyed on heuristics over the message history. The user's specific
concern ("user said X originally" between turns) corresponds 1:1 to
the `task_pin` reminder at `task_pin.rs:62-83`, triggered by
`loop_detect.rs:149`.
