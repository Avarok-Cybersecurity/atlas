# Production Tool-Call Wandering Control — Research

**Target failure**: Qwen3.6-35B-A3B-FP8 on Atlas, ~35% of agentic coding runs
emit 5-7 `bash` calls without ever calling `write` — wanders without finishing.

**Atlas already has** (`crates/spark-server/src/api/stream_guards.rs:26`,
`tool_handlers.rs:142`, `:200`, `:262`):

- `bump_f12_tool_call_count(count, max=12, stop)` — hard per-response cap
- `check_loop_watchdog(text, …)` — repeated-line detector on content stream
- `ATLAS_MAX_TOOL_CALLS_PER_RESPONSE` env override
- `tool_calls_emitted_count` carried in `StreamState`

These are **deterministic caps** on output stream. They do not address:
behavioural nudges, planning hints, or *per-tool-name* budgets (the actual
wandering pattern is "many `bash`, zero `write`", not "too many tools total").

---

## 1. vLLM

- **`tool_choice='required'` infinite loop** on Qwen3-32B and Kimi-K2:
  vllm-project/vllm#21026, MoonshotAI/Kimi-K2#41. Grammar forces a tool call
  every turn → model picks the same tool forever. **No native budget.**
- **RFC#29632 — Force EOS when grammar terminates**
  (`vllm/v1/structured_output/__init__.py::_fill_bitmasks`): once
  `grammar.is_terminated()` returns true, mask everything except EOS instead
  of opening the bitmask. Prevents *trailing* wandering after a tool-call JSON
  closes, not multi-call wandering.
- **`parallel_tool_calls=false`** (chat-completion request flag): limits per
  *response* to ≤1 tool call, forcing a round trip. Behavioural cap, not
  iteration cap.
- **`max_tokens` / `max_completion_tokens`** is the only hard server-side
  budget. No `max_tool_calls`, no `finish_reason="tool_limit"`.

vLLM's V1 sampler (`vllm/v1/sample/`) and engine (`vllm/v1/engine/`) have
**no tool-call accounting** — counting is a client/orchestrator concern.

---

## 2. SGLang

- `python/sglang/srt/function_call/function_call_parser.py`:
  `FunctionCallParser.parse_non_stream()` / `parse_stream_chunk()` extract
  calls but never count or limit them.
- `ToolIterationLimiter(max_iterations=5)` appears in SGLang's *agent example
  templates* (Strands docs), not in `srt/`. Iteration counting lives in the
  caller, not the server.
- `srt/managers/*` (tp_worker, scheduler) track `max_total_num_tokens` and
  `max_running_requests` — no tool-call hooks.

**Net**: SGLang has zero server-side tool budget. Same model as vLLM.

---

## 3. TensorRT-LLM / NVIDIA NIM

- NIM exposes `tool_choice`, `tools`, `parallel_tool_calls` (OpenAI-shape)
  — pure passthrough.
- `tensorrt_llm/_torch/speculative/` is the spec-decode tree; nothing
  agent-specific.
- No public "max_tool_calls" hook. NIM's agentic-AI examples
  (forums.developer.nvidia.com/t/358730) explicitly recommend an external
  agent runtime (LangGraph) for iteration control.

---

## 4. llama.cpp server

- No `--max-tool-calls` flag in `tools/server/`.
- ggml-org/llama.cpp#20164: Qwen3.5-35B / Qwen3-Coder-Next looping on tool
  calls with optional params under long context. Maintainer fix advice:
  *drop optional params from the tool schema* — i.e. push the constraint
  into the grammar, not the agent loop.
- `gary149/llama-agent` (Rust agent wrapping llama.cpp) advertises
  **doom-loop detection**: blocks repeated *identical* tool calls. No source
  visible at the URL probed (404 on `main.rs`), but the README claim is
  hash-of-(name+args) match against a recent-call window.

---

## 5. HuggingFace smolagents — **best documented pattern**

`src/smolagents/agents.py`:

| Line | Symbol | Behaviour |
|------|--------|-----------|
| 998-1025 | main loop | `while not returned_final_answer and self.step_number <= max_steps:` |
| 929-931, 1038-1050 | `_handle_max_steps_reached` | calls `provide_final_answer(task)` → forces a *synthesis* LLM call from history, appends `AgentMaxStepsError` to memory |
| 876-887, 1122-1134 | `_generate_planning_step` | periodic re-planning, **populates `{remaining_steps}` template variable** = `max_steps - step` |
| 913-917, 933-938 | `_validate_final_answer` | runs `final_answer_checks` callbacks; on false, logs + continues |

`src/smolagents/prompts/code_agent.yaml`, `update_plan_post_messages`
contains literally:

> **"Beware that you have {remaining_steps} steps remaining."**

This is the canonical **behavioural hint**: a user-position message injected
at each planning interval that tells the model its remaining budget. Combined
with the deterministic `max_steps` cap, the model both *knows* and is
*forced* to converge.

---

## 6. CrewAI — force-final-answer pattern

`base_agent_executor_mixin.py::_should_force_answer` (PR #1812 fix, issue
#1656):

```python
return self.iterations >= self.max_iter   # post-fix
```

When tripped, `handle_max_iterations_exceeded()` prints "Maximum iterations
reached. Requesting final answer." and **appends a user message to the LLM
context instructing it to finalise**, then makes one more `llm.call()`. This
is the same shape as smolagents `_handle_max_steps_reached` but with an
explicit synthetic-user-turn injection.

---

## 7. AutoGen

`ConversableAgent`: `max_consecutive_auto_reply` (int) + `is_termination_msg`
(callable). When the count hits the cap OR the predicate matches the latest
message, the loop exits. **No injection**, just cap. Used as a
client-orchestration primitive, not a server feature.

---

## 8. Claude Code / Claude Agent SDK

- `max_turns` exposed on `ClaudeAgentOptions` (counts tool-use turns only).
  Enforcement is in the **CLI**, not the Python/Rust SDKs — SDK just relays
  `error_max_turns` from the CLI (`claude-agent-sdk-python/_internal/query.py`
  comment).
- **System-reminders pattern** (michaellivs.com analysis, anthropics/
  claude-code#52018, #56867): Claude Code ships ~37 reactive
  `<system-reminder>` strings injected at lifecycle events (tool-result,
  turn-end, compaction). Examples:
  - "You've used `write` 3 times, prefer `edit` for surgical changes."
  - "The task tools haven't been used recently… This is just a gentle reminder."
  - "You are running low on iterations. Wrap up your current task."

  Reminders are placed as **`last_user` / `new_user` / `assistant_prefix`**,
  not in the system prompt — user-position attention is empirically much
  higher. This is the **soft-nudge** counterpart to the hard cap.
- `max_budget_usd` caps by $ spend; same termination shape as `max_turns`.

---

## 9. OpenAI Assistants API

`max_prompt_tokens` + `max_completion_tokens` per run only. Hitting either
terminates the run with `status=incomplete`, `incomplete_details.reason`. No
tool-call count budget; users implement that by polling `run_steps` and
calling `runs.cancel`.

---

## 10. LangGraph

`interrupt()` + `Command(resume=value)` lets a node **pause graph execution**,
return control to the caller, then resume with an injected value. This is
the substrate for "synthetic user-turn injection": you pre-emptively
interrupt before turn N+1, mutate state (e.g. append a "you've explored
enough, write the file now" user message via `graph.update_state(values,
as_node=…)`), then resume. Not built into model serving — it's an
orchestrator-level reset mechanism.

---

# Five patterns Atlas can borrow

Mapped to existing `tool_handlers.rs` / `stream_guards.rs` / `chat_stream`.

### Pattern A — **Per-tool-name budget** (deterministic cap)
*Why*: the Atlas failure is "many `bash`, no `write`", which the existing
total cap of 12 does **not** catch.

Extend `StreamState.tool_calls_emitted_count: usize` to
`HashMap<String, usize>` (keyed on tool name), and add
`ATLAS_TOOL_BUDGET="bash=5,grep=10"`. In `handle_complete_tool_call`,
`handle_tool_call_start`, `handle_tool_call_end` look up the per-name cap
before bumping. Reuse the same `stop_string_triggered` exit path that the
total cap already uses.

**LoC**: ~60 lines in `stream_guards.rs` (parser + `bump_per_tool_count`),
~30 lines call-site touches in `tool_handlers.rs`, ~20 lines `StreamCtx`
plumbing. **~110 LoC total.**

### Pattern B — **Repeat-call doom-loop detector** (deterministic + behavioural)
*Why*: catches `gary149/llama-agent`'s pattern — same `name+args` hash
appearing twice in a row is almost always wandering.

Add `recent_tool_hashes: VecDeque<u64>` to `StreamState` (cap N=4). In
`handle_complete_tool_call`, compute `xxhash64(name + canonical_json(args))`;
if it matches any of the last N-1 entries, `stop_string_triggered = true`
with a `finish_reason="repeat_loop"` (new variant). Mirror in
`handle_tool_call_end` (the streaming-delta path).

**LoC**: ~70 LoC in `stream_guards.rs` + state.rs, ~25 LoC call sites.
**~95 LoC total.**

### Pattern C — **Synthetic user-turn injection mid-stream** (behavioural hint)
*Why*: smolagents `{remaining_steps}` + Claude Code system-reminders both
work by **moving the hint to user position**, where the model attends more.

After tool-call N (N tunable, e.g. 3), have `handle_complete_tool_call`
short-circuit before issuing chunk N+1: close the assistant turn with
`finish_reason="tool_calls"`, let the harness round-trip the tool result,
then **`chat_stream_dispatch.rs`** prepends a synthetic user message:

> "You have made N tool calls (budget remaining: M). Please write the file
> now and finish, or explain why you cannot."

This requires a hook in the dispatcher rather than the streamer. The
streamer flags `state.inject_finish_reminder = true`; the dispatcher reads
it on the next request and pushes a user message into the `messages` array
before forwarding to the model.

**LoC**: ~40 LoC streamer flag + ~80 LoC dispatcher injection + ~30 LoC
config (`ATLAS_FINISH_REMINDER_AFTER`, the template). **~150 LoC total.**

### Pattern D — **Tool-required → tool-forbidden flip** (reset mechanism)
*Why*: vLLM RFC#29632 forces EOS when grammar terminates. Atlas can flip
the grammar mid-conversation: after K tool calls, on the next turn,
**inject `tool_choice="none"`** (or remove the tool-call grammar from the
xgrammar mask) so the model is *forbidden* from emitting another tool call
and must produce free-form text. This is the closest analogue to LangGraph's
interrupt-and-restart.

In Atlas this lives at `chat_stream_dispatch.rs:~80-110` where the
`tools`/`tool_choice` payload is built. Read `state.tool_calls_emitted_count`
from the prior response, if `> threshold` rewrite `tool_choice="none"`
before calling the engine.

**LoC**: ~50 LoC in `chat_stream_dispatch.rs` + ~20 LoC threshold env.
**~70 LoC total.**

### Pattern E — **`finish_reason="tool_limit"` propagation**
*Why*: callers (opencode, Claude Code) currently see Atlas's cap-triggered
stop as a normal `stop` or `length`, can't tell the model wandered. CrewAI
and AutoGen both expose distinct termination reasons; OpenAI Assistants
uses `incomplete_details.reason`.

Add a new `FinishReason::ToolLimit` variant (or a sub-field
`atlas_termination: "tool_budget_per_name" | "tool_repeat_loop" |
"tool_total_cap"`) emitted in the final SSE chunk. Caller-side hooks
(`tool_handlers.rs::handle_tool_call_end` and `handle_done.rs`) already
consult `stop_string_triggered` — extend to read a new
`StreamState.termination_reason: Option<TerminationReason>` and serialise
it in the final chunk.

**LoC**: ~30 LoC enum + serialisation, ~20 LoC call-site sets. **~50 LoC total.**

---

# Categorisation summary

| Pattern | Type | Source of inspiration |
|---|---|---|
| A — per-tool-name budget | **deterministic cap** | smolagents `max_steps` (per-axis), Claude `max_turns` |
| B — repeat-call hash | **deterministic cap** | gary149/llama-agent doom-loop |
| C — synthetic user reminder | **behavioural hint** | smolagents `{remaining_steps}`, Claude `<system-reminder>`, CrewAI `_force_answer` |
| D — `tool_choice="none"` flip | **reset mechanism** | vLLM RFC#29632 EOS-on-terminated, LangGraph interrupt |
| E — `finish_reason="tool_limit"` | **observability** (enables external resets) | CrewAI `_should_force_answer`, OpenAI `incomplete_details` |

# Recommended ordering for Atlas

1. **E first** (50 LoC, lowest risk) — gives observability into when caps
   trip, lets you measure the 35% failure rate properly before mutating
   behaviour.
2. **A** (110 LoC) — addresses the exact failure mode ("many bash, no
   write"); single-axis cap is the smallest change that should
   demonstrably move the metric.
3. **B** (95 LoC) — orthogonal to A; catches the *other* failure mode
   (identical bash twice).
4. **C** (150 LoC) — once A+B are stable, the hint shifts the wandering
   *upstream* and reduces how often the deterministic caps need to fire.
5. **D** last — most invasive (mutates request payload, not just stream),
   highest behavioural blast radius. Only if A+B+C aren't enough.

Total: ~475 LoC across 5 patterns, gated independently by env vars.

---

# Sources

- vLLM tool-call infinite loops: vllm-project/vllm#21026, MoonshotAI/Kimi-K2#41
- vLLM force-EOS RFC: vllm-project/vllm#29632
  (`vllm/v1/structured_output/__init__.py::_fill_bitmasks`)
- SGLang parser: `sgl-project/sglang/python/sglang/srt/function_call/function_call_parser.py`
- llama.cpp tool-loop bug: ggml-org/llama.cpp#20164
- gary149/llama-agent README (doom-loop detection)
- smolagents loop: `huggingface/smolagents/src/smolagents/agents.py` lines
  876-887, 913-917, 929-931, 998-1025, 1038-1050, 1122-1134
- smolagents prompts: `huggingface/smolagents/src/smolagents/prompts/code_agent.yaml`,
  `update_plan_post_messages` ("Beware that you have {remaining_steps} steps remaining.")
- CrewAI fix: crewAIInc/crewAI#1656, PR#1812 (`base_agent_executor_mixin.py`)
- AutoGen: `microsoft/autogen` `ConversableAgent.max_consecutive_auto_reply`,
  `is_termination_msg`
- Claude Agent SDK: `anthropics/claude-agent-sdk-python/_internal/query.py`
  (`error_max_turns` relayed from CLI)
- Claude Code system-reminders: anthropics/claude-code#52018, #56867;
  michaellivs.com/blog/system-reminders-steering-agents
- OpenAI Assistants `max_prompt_tokens` / `max_completion_tokens` / `run_steps`
- LangGraph `interrupt()` / `Command(resume=…)` / `graph.update_state(…, as_node=…)`
- Atlas existing infra:
  `crates/spark-server/src/api/stream_guards.rs:26` (`bump_f12_tool_call_count`);
  `crates/spark-server/src/api/chat_stream/tool_handlers.rs:142,200,262`
  (call sites); `chat_stream_dispatch.rs` (request payload assembly).
