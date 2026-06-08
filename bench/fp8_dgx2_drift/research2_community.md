# Community Research: Qwen3-family Agentic Failures & opencode Tool Protocol
**Window**: Jan 2026 – May 2026
**Atlas target**: Qwen3.6-35B-A3B-FP8, thinking_in_tools=true, EBNF grammar (XGrammar), qwen3_coder tool parser
**Source mix**: GitHub issues (vLLM, llama.cpp, opencode, Roo-Code, Continue, sglang, Ollama, QwenLM, XGrammar), HF model-card discussions, vLLM recipes, NVIDIA DGX Spark forum, community chat-template repos. Reddit search returned no direct hits; HN had no on-topic threads in window. Qwen Discord is gated/not web-indexed — recovered indirectly via cross-posted HF discussions.

---

## 1. Failure modes that match ours

### 1.1 Repeated tool calls with one parameter always missing (long context)
**llama.cpp #20164** — "Tool calling may repeatedly fail under long context when a tool has multiple optional parameters (Qwen3.5-35B, Qwen3-Coder-Next)." Failure emerges around ~30k tokens (~20% of window). Model invokes the same tool over and over and omits a *different* optional parameter each retry (e.g., `read(path=…)` without `offset`, then without `limit`). Workaround that worked locally: convert all optional params to required. Reporter referenced "AutoParser branch" but didn't share patch.

**Roo-Code #10150 / #7406** — Same family: `list_files without value for required parameter 'path'`, `apply_diff without 'path'`. Affects qwen3-coder via Open WebUI/Ollama. Settings that helped: `num_ctx ≥ 65536`, `repeat_penalty=1.1`, OpenAI-compatible endpoint over native Ollama.

### 1.2 Empty tool_calls / silent acknowledgement without emission
**HF Qwen3.6-27B discussion #13** — Reports `<|im_sep_user|>` token injection causing prompt-echo loops; "Alright now I need to do X" with no tool emission. Affects both 27B-dense and 35B-A3B variants. Suspected chat-template issue. Pointer to `abysslover/qwen36_tool_calling_failure` GitHub repo and `froggeric/Qwen-Fixed-Chat-Templates` HF repo.

### 1.3 Tool calls eaten by the reasoning parser
**vLLM #39056** (Apr 2026, smoking gun for `thinking_in_tools=true`) — `qwen3_reasoning_parser` extracts everything before `</think>` into the `reasoning` field; the downstream `qwen3_coder` tool parser only inspects `content`. When the model emits XML tool-call markup *inside* the thinking block (exactly Atlas's `thinking_in_tools=true` regime), tool calls vanish from the non-streaming response. **Fix PR #39055 proposed: promote embedded `<tool_call>` blocks out of reasoning into content before tool parsing.** Affects exactly Qwen3.5-35B-A3B-FP8 (almost certainly also 3.6-35B-A3B-FP8).

### 1.4 Object-typed args where strings expected
**opencode #6918** — qwen3-coder emits the `edit` tool with `oldString`/`newString` as JSON objects (sometimes nested dicts) instead of strings. Open. Matches our "semantically-wrong tool args after 5+ turns" report — model hallucinates a richer schema than the tool actually exposes.

### 1.5 Argument truncation mid-string
**unsloth/Qwen3-Coder-Next-GGUF #2 + opencode threads** — `write` tool arguments get chopped mid-way under opencode. Fix: raise opencode's output-token cap AND set `preserve_thinking=true` so the KV-cache prefix isn't repeatedly invalidated.

---

## 2. Workarounds the community has shipped

### 2.1 Replace the official Qwen3.6 chat template
`froggeric/Qwen-Fixed-Chat-Templates` consolidates 19+ fixes for both 3.5 and 3.6. Key ones relevant to Atlas:

- **v15** — `consecutive_failures` counter with two-tier error escalation prevents "retry stall / reasoning spiral" (model repeating identical failing tool calls). Matches our "repeated tool-call attempts at same failed action" symptom.
- **v18** — Strict structural guards (`"error":`, `Traceback`, `Exception:`) replace substring matching; stops false-positive retry triggers when a successful JSON body merely *contains* the word "error".
- **v19** — `preserve_thinking=true` default. Stops "empty `<think></think>` poisoning" and keeps 100% prefix-cache hit. Earlier templates stripped past thoughts, breaking KV cache reuse and producing degraded long-context behavior.
- Native XML tool format `<function=name><parameter=k>v</parameter></function>` restored for vLLM `qwen3_coder` parser compatibility (3.6 official template emits a flavor that breaks parsing at pos 1122 in some frameworks).
- `<|think_on|>` / `<|think_off|>` inline toggles instead of `enable_thinking` kwarg (kwarg path is broken in vLLM up to 0.9 — vLLM #35574).

### 2.2 Use `qwen3_xml` parser instead of `qwen3_coder` for 3.5
NVIDIA DGX Spark forum (Apr 2026) — switching from `--tool-call-parser qwen3_coder` to `--tool-call-parser qwen3_xml` plus the `qwen3.5-enhanced.jinja` template took one user from 100% failure to ~10% failure on 12-hour agentic runs. Same poster notes 3.6 "resolves these out of the box" — but the HF discussions above contradict that for the A3B-FP8 variant specifically.

### 2.3 Disable speculative decoding (MTP) for agentic loads
DGX Spark thread reports `--speculative-config method=mtp num_speculative_tokens=2` *degrades* tool-call success rates on 3.6-35B-A3B-FP8 (regressions from ~100% to 97% on ToolCall-15, with TC14 outright failing). Atlas's MTP=2 default may be aggravating drift; worth gating MTP off when `thinking_in_tools=true`.

### 2.4 Make optional params required, or split tools
Two independent reports (llama.cpp #20164, Roo-Code #7406): the *single* most reliable mitigation for the "wrong file path / repeated calls" loop is to remove optional parameters from the tool schema. Schema-level workaround, no model change needed.

### 2.5 Anthropic SDK endpoint over OpenAI-compatible
DGX Spark thread: one user reported better stability calling Atlas via the Anthropic Messages format than OpenAI ChatCompletions. Plausible explanation: the Anthropic flow doesn't round-trip `tool_calls` through a JSON `function.arguments` string; it preserves structured content blocks, avoiding the object-vs-string coercion bug from §1.4.

---

## 3. Sampler recipes specifically tuned for Qwen3.6 agentic

Consolidated from Qwen tech report, vLLM recipes page, and Glukhov's reference guide:

| Param | Thinking mode | Non-thinking | Agentic-tuned (3.6-35B-A3B-FP8) |
|---|---|---|---|
| temperature | 0.6 | 0.7 | **0.6** (deterministic tool args) |
| top_p | 0.95 | 0.8 | **0.95** |
| top_k | 20 | 20 | **20** |
| min_p | 0.0 | 0.0 | 0.0 |
| presence_penalty | 0.0 | 0.0–2.0 | **1.5** for prose, **0.0 for coding** (Qwen tech report) |
| repetition_penalty | 1.0 | 1.0 | **1.1** (Roo-Code empirical, llama.cpp confirmed) — stops same-tool-loop |

**Key non-obvious points:**
- 3.6-MoE specifically benefits from `presence_penalty>0` because expert-routing-per-token creates "repetition loops" at the token level even when the high-level plan is fine. Glukhov: "MoE requires different penalties."
- Stop tokens / EOS must be **exempt from repetition_penalty** (HF convention; Atlas already has this per project_qwen36_fp8_post_think_eos.md — good).
- For coding agents specifically, drop `presence_penalty` to 0 — non-zero values were reported to corrupt indentation and bracket-matching in `write` calls.

---

## 4. Issues that turned out to be chat-template / thinking-mode misconfig

Almost all of them. The pattern across vLLM #35574, vLLM #39056, vLLM #19051, Ollama #14493, llama.cpp #20164, QwenLM/Qwen3 #1831, ms-swift #5836, sglang #16653, opencode #6918:

1. **Wrong pipeline routing**: Qwen3.5 trained on XML tool format, but Ollama / older vLLM routed it to the JSON Hermes-style pipeline → 6 concrete format mismatches → 100% silent tool-call drop.
2. **Reasoning parser eats tool calls in thinking blocks** (vLLM #39056, sglang #16653) — exactly Atlas's regime.
3. **`enable_thinking=false` kwarg ignored** (vLLM #35574) until vLLM 0.9 + qwen3 reasoning parser. Many users thought their model "still thinks" but real bug was the toggle being a no-op.
4. **`arguments | items` Jinja filter crashes** on minijinja (C++ runtimes including some vLLM builds, all llama.cpp builds). QwenLM #1831 fix #6.
5. **Unclosed `<think>` block from previous turn** corrupts every subsequent turn (Ollama #14493 bug #2, abysslover repo).
6. **Tool-choice "required" returns 400** when combined with reasoning (vLLM #19051).

---

## 5. opencode tool-protocol issues — current status (as of May 2026)

### Confirmed bugs:
- **#13146** (Feb 2026, OPEN): `description` field on bash tool is required in the Zod schema but invisible in the schema exposed to the model. No server-side fix shipped. Two proposed options (make optional vs. expose in schema) — neither merged.
- **#14519** (Feb 2026, CLOSED): "expected string, received undefined" for `command`/`description`. Marked closed but no resolution version stated in the issue body.
- **#15675** (Mar 2026, OPEN): `write` tool client hangs because the server never sends `tool_call_update` with `status=completed`. File is written to disk, but UI/SDK waits forever. **Server-side bug, opencode-internal.**
- **#20902, #21000, #25664**: bash tool hangs on (a) backgrounded child processes inheriting stdout fds, (b) fast-exiting processes that finish before the watcher attaches, (c) `pkill -f` matching opencode itself.
- **#14473**: in headless/server mode, default `ask` permission for external dirs hangs the tool call forever — relevant if Atlas tests run opencode non-interactive.

### Server-side fixes that *did* ship since April 2026 (from opencode changelog):
- **v1.14.42** — HTTP API returns structured 400 bodies on schema validation errors (was opaque before)
- **v1.14.46** — MCP tool-discovery resilient to broken `outputSchema`; boolean HTTP query type alignment
- **v1.15.1** — Custom tool metadata + argument descriptions preserved from Zod schemas; invalid module exports skipped instead of crashing tool loading
- **v1.15.6** — Schema failures surface as friendly errors; plugin load errors don't cascade

**None of these fix the `description`-field schema-vs-required mismatch.** The bash-tool hang on empty `tool_calls` / empty `parameters` lists is *not* explicitly addressed by any shipped fix in window.

### Practical implication for Atlas:
Atlas must continue producing `description` even though opencode's exposed schema doesn't ask for it. Best path: **inject a synthetic `description` field server-side in Atlas's response post-processor** when the model emits a bash call without one. Or, surface this requirement in the grammar (EBNF) as a required field.

---

## Top-5 ranked actionable fixes Atlas could adopt

1. **Promote `<tool_call>` blocks out of the reasoning region before the tool parser runs** (mirrors vLLM PR #39055). Direct fix for thinking_in_tools=true tool-loss. Highest ROI — single-file change in our reasoning_parser path, exactly the symptom we're chasing in MTP fp8 drift. Saves us from blaming the model when the parser is eating the call.

2. **Adopt the `froggeric/Qwen-Fixed-Chat-Templates` v19 template (or fork it)** — gets us `preserve_thinking=true` (KV-cache hit retention), `consecutive_failures` retry-stall counter, structural error-detection guards, and native XML tool format the qwen3_coder parser actually expects. Likely fixes both "wrong file path" and "repeated same tool-call" symptoms simultaneously.

3. **Add `repetition_penalty=1.1` to Atlas's Qwen3.6 default sampler profile, exempting EOS and tool-call start/end tokens.** Two independent communities (Roo-Code, llama.cpp) converged on this value as the single most effective fix for tool-loop drift on Qwen3-coder family. Atlas already exempts stop-tokens from rep-penalty per memory — this is incremental.

4. **Gate `thinking_in_tools=true` off when MTP=2 is active, or vice versa.** DGX Spark forum data shows MTP=2 regresses Qwen3.6-35B-A3B-FP8 tool-call success. Atlas's MTP scheduler likely interacts badly with the `<think>`/`</think>` boundary placement that the reasoning parser depends on. Cheaper than re-engineering MTP: just refuse the combination at config-load time.

5. **Inject synthetic `description` field for bash tool calls server-side; gate `tool_calls=[]` empty-list emission behind an EBNF rule that forbids it.** Direct compensation for opencode #13146 (unfixed upstream) and the empty-tool_calls hang. EBNF rule already in Atlas (XGrammar) — add `tool_call ::= … description-required …` for `bash`. Two-line grammar change.

---

## Sources

- [llama.cpp #20164 — long-context optional-params tool-call failure](https://github.com/ggml-org/llama.cpp/issues/20164)
- [vLLM #39056 — qwen3_reasoning_parser eats tool calls in thinking blocks](https://github.com/vllm-project/vllm/issues/39056)
- [vLLM #35574 — enable_thinking=false ignored](https://github.com/vllm-project/vllm/issues/35574)
- [QwenLM/Qwen3 #1831 — 21 chat-template fixes](https://github.com/QwenLM/Qwen3/issues/1831)
- [HF froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)
- [HF abysslover/qwen36_tool_calling_failure repo + Qwen3.6-27B discussion #13](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/13)
- [HF Qwen3.6-35B-A3B discussion #30 — Claude Code tool failure](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/30)
- [Ollama #14493 — Qwen3.5-27B tool-calling broken, rep-penalty silently ignored](https://github.com/ollama/ollama/issues/14493)
- [Roo-Code #10150 / #7406 — qwen3-coder missing required 'path'](https://github.com/RooCodeInc/Roo-Code/issues/7406)
- [opencode #13146 — bash description-field schema mismatch (OPEN)](https://github.com/anomalyco/opencode/issues/13146)
- [opencode #14519 — invalid arguments command/description (CLOSED, no version)](https://github.com/anomalyco/opencode/issues/14519)
- [opencode #15675 — write tool client hang (OPEN)](https://github.com/anomalyco/opencode/issues/15675)
- [opencode #6918 — qwen3-coder edit tool object-vs-string args](https://github.com/anomalyco/opencode/issues/6918)
- [opencode changelog (Apr–May 2026 fixes)](https://opencode.ai/changelog)
- [vLLM recipes — Qwen3.5/3.6 official guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [NVIDIA DGX Spark forum — Qwen3.5 tool calling finally fixed](https://forums.developer.nvidia.com/t/qwen3-5-tool-calling-finally-fixed-possibly/366451)
- [NVIDIA DGX Spark forum — Qwen3.6-35B-A3B-FP8 landed](https://forums.developer.nvidia.com/t/qwen-qwen3-6-35b-a3b-and-fp8-has-landed/366822)
- [Glukhov — Agentic LLM inference parameters reference (Qwen3.6, Gemma 4)](https://www.glukhov.org/llm-performance/benchmarks/agentic-inference-parameters-reference/)
- [Cursor forum — false-positive loop detection on qwen3-coder-plus](https://forum.cursor.com/t/false-positive-loop-detection-when-using-custom-model-qwen3-coder-plus-with-repetitive-reasoning-text-before-different-tool-calls/145252)
- [Continue #5419 — Qwen3 agent-mode tool-loading 404](https://github.com/continuedev/continue/issues/5419)
- [sglang #16653 — Qwen3-Next-80B-Thinking reasoning + tool parser errors](https://github.com/sgl-project/sglang/issues/16653)
- [unsloth Qwen3-Coder-Next-GGUF #2 — Jinja template errors w/ opencode/LM Studio](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/discussions/2)
