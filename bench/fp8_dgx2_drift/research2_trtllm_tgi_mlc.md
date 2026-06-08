# TRT-LLM / TGI / MLC-LLM — Multi-Turn Agentic & Tool-Call Handling (Nov 2025 → May 2026)

**Audience:** Atlas (Qwen3.6-35B-A3B-FP8 on GB10, opencode multi-turn drift)
**Date:** 2026-05-26
**Companion to:** `research_trtllm_pipeline.md`, `research_tgi_pipeline.md`, `research_sglang_pipeline.md`
**Scope:** *only* post-Oct-2025 changes the prior files do not cover.

---

## 0. TL;DR

Three things have happened since the prior dives that matter for Atlas's FP8 multi-turn drift:

1. **XGrammar 0.1.34 (Apr 29 2026)** shipped a built-in `qwen_3_5` / `qwen_3_coder` structural-tag builder generating the exact `<tool_call>\n<function=NAME>\n…\n</function>\n</tool_call>` envelope (with `</think>` reasoning prefix) as a single `TriggeredTagsFormat`. **XGrammar 0.2.0 (May 1)** cut structural-tag compile time at 1000 tools from 21.7s → 2.1s — strict-mode is no longer cost-prohibitive.
2. **MLC-LLM commit d75d64e (Apr 20 2026)** added `strip_reasoning_in_history` to the Conversation protocol and a dedicated `qwen3` template. Qwen3's HF chat template strips `<think>...</think>` from historical assistant turns; MLC's generic ChatML didn't, which caused small Qwen3 models to emit `<|im_end|>` prematurely inside the next turn's `<think>`. **This is Atlas's drift fingerprint.**
3. **TRT-LLM PR #12061 (Mar 11 2026)** fixed two silent agentic bugs: (a) `reasoning_content` was not being forwarded back into the chat template; (b) the Qwen3-Coder tool parser was silently dropping every tool call whose `function_name` was not already in `self._tool_indices` — i.e. any near-miss hallucinated tool name vanished and the loop free-ran. PR #12199 (Mar 20) then made interleaved thinking first-class.

TGI is dead: PR #3344 "Maintenance mode" (Dec 11 2025) and a single docs commit since. The architecture in `research_tgi_pipeline.md` is final.

---

## 1. TensorRT-LLM (Nov 2025 → v1.3.0rc16, May 26 2026)

### 1.1 Tool-Parser Activity

Highlights from constant churn: **PR #12061** (Mar 11) agentic flow fixes — see §1.2; **PR #12104** (Mar 20) `auto` option for tool + reasoning parsers; **PR #12199** (Mar 20) interleaved thinking for trtllm-serve; PR #12173 (Mar 13) `anyOf` schemas in Qwen3-Coder; PR #13684 (May 21) Nano-v3 treats `<tool_call>` as implicit end-of-reasoning.

### 1.2 PR #12061 — Two Bugs Atlas Likely Shares

**Bug A — `chat_utils.py:_parse_assistant_message_content`:**
```python
+ reasoning_content = message.get("reasoning") or message.get("reasoning_content")
+ if reasoning_content is not None:
+     result["reasoning_content"] = reasoning_content
```
TRT-LLM was silently dropping the prior turn's reasoning when re-parsing chat history. For Qwen3.6's interleaved-thinking template this means the model sees a re-rendered prompt with `<think></think>` empties where its prior chain-of-thought lived, then re-derives it each turn at higher entropy — a direct path-drift generator on FP8.

**Bug B — `qwen3_coder_parser.py`:**
```python
- if function_name in self._tool_indices:
+ if function_name:
…
- except Exception:
-     logger.warning(f"invalid tool call for {fname} dropped")
+ res.append(ToolCallItem(tool_index=-1, name=fname, …))
```
The parser was **silently dropping any tool call whose name didn't exactly match the request `tools[]`**. An FP8-induced near-miss hallucinated tool name became a no-op: no `tool_call` event, the JSON gets re-emitted as prose, agent loop free-runs.

### 1.3 Guided-Decoding Hardening

- **1.0**: structural tag in C++ runtime, xgrammar 0.1.21
- **1.1**: guided + spec-decoding interop; guided under disaggregated serving
- **1.2.1** (Apr 20): xgrammar + flashinfer bump
- **rc12** (Apr 17): VLM guided-decoding `vocab_size_padded` crash fix
- **rc15** (May 21): GIL fix for guided-decoding host func; PP warmup barrier
- **rc16** (May 26): `GUIDE_TYPE_STRUCTURAL_TAG` request-management fixes

The `CapturableGuidedDecoder` host-func pipeline is stable; Atlas's gap is still no structural-tag *strict-mode* wiring on top of its existing matcher.

### 1.4 Hybrid SSM (Mamba / Qwen3-Next) — The Negative Result

Mamba-related activity: PR #12185 (rc10) prefix caching for Mamba hybrid; PR #13453 (rc10) Mamba-2 rollback replay for spec-decode; PR #13151 (rc14) Mamba cache fix under MTP + CUDA-graph padding; PR #12896 (rc15) MTP block-reuse; **PR #14471 (rc16, May 26)** *"Disable mamba replay by default"*.

NVIDIA shipped Mamba-2 rollback in March, then **turned it off by default in late May**. Either correctness regressed under load or perf cost outweighed spec-decode gain. Atlas's NGram v21 reached the same conclusion (73% rejection, 1.5× slower) independently. **Disable SSM rollback by default in Atlas spec-decode too** — confirmed industry-wide negative result, not just SM121.

`Qwen3HybridConfig.layer_types` derivation moved to `from_hf` (#13832 / #14410): Qwen3 HF configs differ in how they declare attention-vs-SSM interleaving. Atlas's loader should sanity-check `layer_types`.

---

## 2. HuggingFace TGI — Frozen

PR #3344 "Maintenance mode" (Dec 11 2025); last commit docs-only Mar 21 2026. No `qwen3`, tool-calling, or grammar-engine swap work landed. `research_tgi_pipeline.md` architecture is final. HF's serving work moved to `transformers serve`.

---

## 3. MLC-LLM + XGrammar (Nov 2025 → May 2026)

### 3.1 XGrammar Version Map

| Ver | Date | Key changes |
|---|---|---|
| 0.1.26 | 2025-10-20 | Batched `fill_next_token_mask`, batched accept |
| 0.1.28 | 2025-12-09 | bf16/fp16 bitmask CPU; async H2D in triton |
| 0.1.29 | 2025-12-19 | **`TraverseDraftTree`** for spec-decode |
| 0.1.32 | 2026-03-04 | MinimaxXmlFormat / DeepSeekXmlFormat; structural-tag refactor |
| 0.1.33 | 2026-03-27 | **`Fork()`, `BatchRollback`**; token-level grammar (`Token`/`ExcludeToken`/`TokenTagDispatch`); cross-grammar cache; GLM-4.7 |
| **0.1.34** | **2026-04-29** | **`qwen_3_5` (= Qwen3.5/3.6/Coder) structural tag; `tool_choice` modes; DeepSeek-V4; Gemma-4** |
| **0.2.0** | **2026-05-01** | **Compile @1000 tools: 21.7s → 2.1s (-90%)**; unified `reasoning` parameter |
| 0.2.1 | 2026-05-17 | Qwen XML builder unification; FSM state-merge fix; `accept_token` HF-processor fix |

### 3.2 The Critical Capability: `qwen_3_5` Builtin Structural Tag

PR #603 (Apr 28) added the builder, registered for both `qwen_3_5` and `qwen_3_coder`. Final form:

```python
TOOL_CALL_BEGIN_PREFIX = "<tool_call>\n<function="
TOOL_CALL_END = "\n</function>\n</tool_call>"
TOOL_CALL_TRIGGER = "<tool_call>\n<function="
THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
```

Per tool: `TagFormat(begin=PREFIX+name+">\n", content=JSONSchemaFormat(json_schema=params, style="qwen_xml"), end=END)`. Wrapped in `TriggeredTagsFormat(triggers=[TRIGGER], tags=tags, excludes=THINK_EXCLUDE_TOKENS)`. Reasoning prefix: `SequenceFormat([TagFormat(begin="", content=AnyTextFormat(), end="</think>"), ConstStringFormat("\n\n")])`.

`style="qwen_xml"` is a **Qwen-specific JSON-schema-to-XML-content converter** (0.1.32, refined 0.2.1) — generates `<parameter=NAME>VALUE</parameter>` from a JSON schema, hard-constrained.

**This is the single biggest external piece Atlas does not yet have.** Atlas's `crates/xgrammar/src/structural_tag/` has the FSM primitives (`TriggeredTagsFormat`, `TagFormat`, `JSONSchemaFormat`) but no Qwen-style builder, no `qwen_xml` JSON-schema lowering, no wire-up from OpenAI chat-completions `tools`.

### 3.3 `tool_choice` Semantics (0.1.34 + 0.2.0)

Well-defined modes any engine should mirror:
- `"auto"` → `TriggeredTagsFormat` (text *or* tool call)
- `"required"` → `TagsWithSeparatorFormat(..., at_least_one=True)` (no plain text)
- `"forced"` → single `TagFormat` (exactly one tool, exactly one call)
- `"none"` → text-only; tool-tag logits masked
- `{"type": "allowed_tools", …}` → filter + apply nested mode

The `reasoning` parameter is now model-dependent: for `qwen_3_5` `reasoning=False` forces `<think></think>`; for `qwen_3` it strips reasoning from history; for `llama`/`qwen_3_coder`/etc it omits the reasoning section. **Behaviour is in the builder** — engines don't decide.

### 3.4 MLC-LLM Engine Wiring — Critical Commit

`d75d64e` (Apr 20 2026) Conversation protocol changes:

> Qwen3's HF chat template strips `<think>...</think>` blocks from historical assistant messages (turns before the last user message) before rendering the prompt. mlc-llm's `qwen2` template did not, so prior thinking traces get echoed back into context verbatim. On small Qwen3 variants this pushes the model to emit `<|im_end|>` prematurely inside its next-turn `<think>` block, truncating the response before `</think>` is ever produced.

Substitute "small Qwen3 0.6B" with "Qwen3.6-35B-A3B at FP8 quantisation noise" — the failure mode rhymes. Historical `<think>` traces become near-OOD continuation prompts for turn N+1's `<think>`. The fix is two lines: add `strip_reasoning_in_history: bool` to Conversation; register a `qwen3` template setting it true.

Also: commit `d46f65f` (Apr 5) added `qwen3_5` and `qwen3_5_nothink` conversation templates.

### 3.5 XGrammar 0.2.0 Compile-Time Cliff Removed

PR #616 fixes: rule-name suffix caching, lookahead-analyzer single-pass, drop `MinimizeDFA` in `Choices()`, CSR-based `MergeEquivalentSuccessors`. If Atlas's `crates/xgrammar/` is forked pre-0.2 SHA, **those wins are not yet in**.

### 3.6 New Primitives Atlas Should Track

- **`BatchRollback`** (0.1.33): single call to roll back N matchers M tokens.
- **`Fork()`** on `GrammarMatcher` (0.1.33): clone matcher state for spec-decode tree exploration.
- **`TraverseDraftTree`** (0.1.29): bitmask generation across a draft tree in one call (use in NGram v21).
- **Token-level grammar** (`Token`/`ExcludeToken`/`TokenTagDispatch`, 0.1.33): match by token-id rather than byte stream. `<tool_call>` and `</think>` are single tokens for Qwen3.6 — token-edge grammar **eliminates the F72 byte-anchor problem by construction**.

---

## 4. Cross-Engine Summary (Qwen3 Multi-Turn Tool Calling, May 2026)

| Aspect | TRT-LLM rc16 | TGI (frozen) | MLC + XGrammar 0.2.1 | Atlas |
|---|---|---|---|---|
| Tool-call grammar | XGrammar struct-tag strict | outlines JSON regex | XGrammar `qwen_3_5` builtin | Matcher present, no Qwen3 builder |
| Reasoning fwd to history | Yes (#12061) | Verbatim echo | **Stripped from history** | Verbatim echo |
| Tool-name silent drop | Fixed (#12061) | N/A | Grammar enforces | **Likely present** |
| Spec-decode rollback | Per-slot triple | N/A | `BatchRollback`+`Fork()` | Hand-rolled v21 only |
| Hybrid-SSM rollback | Shipped then **disabled** | N/A | N/A | v21 1.5× slower |
| Struct-tag compile @1000 | 2.1s (xg 0.2) | N/A | 2.1s | 21.7s if pre-0.2 |
| Token-level grammar edges | Yes | N/A | Yes (0.1.33) | Not yet |
| Interleaved thinking | Yes (#12199) | N/A | Yes | Partial |

---

## 5. Top-5 Atlas Gaps (Ranked)

### #1 — Strip `<think>...</think>` from historical assistant turns (Qwen3.5/3.6/Coder/Coder-Next)

**Evidence:** MLC commit `d75d64e` adds exactly this; Qwen3's HF chat template does it. Atlas's `tokenizer/jinja_helpers.rs` has `strip_thinking` for *harmony* only — Qwen3 historical traces are almost certainly being echoed verbatim, becoming near-OOD continuation prompts for the next turn's `<think>` block. Direct multi-turn-drift generator on FP8.

**Effort:** 1 day. For `qwen3_5_*` and `qwen3_coder_*`, strip `<think>...</think>` (+ trailing whitespace) from all assistant messages *before* the final user message at chat-template-apply time. **Preserve the most recent assistant turn** intact (tool-call prefill scenarios depend on it).

### #2 — Adopt XGrammar's `qwen_3_5` built-in structural-tag builder; default Qwen3.6-FP8 to strict mode

**Evidence:** XGrammar 0.1.34 ships the builder; 0.2.0 cut compile cost 90%. Both arguments against strict mode (no builder, too slow) are gone. Atlas free-generates Qwen3 tool envelopes and post-parses with regex (F72 byte-anchor), which is exactly the surface where FP8 quantisation noise causes drift.

**Effort:** 3-5 days. Port the builder into `crates/xgrammar/src/structural_tag/`; wire through `crates/spark-server/src/api/chat_stream/` so when `tools` is non-empty and the model is qwen3.5/3.6/coder/coder-next, a structural-tag grammar is compiled and attached to the matcher; post-hoc tool parser bypassed in strict mode. Gate via `ATLAS_QWEN3_STRICT_TOOLS=1`.

### #3 — Audit `tool_parser` crate for silent tool-name drop

**Evidence:** TRT-LLM PR #12061 removed `if function_name in self._tool_indices` and `try/except → "invalid tool call dropped"`. If Atlas has the same hard filter, FP8 near-miss tool names become silent no-events — agent loop free-runs.

**Effort:** 4 hours. `rg "in self._tool_indices\|invalid tool call\|tool_indices" crates/`. Either drop the check (TRT-LLM's path) or emit a `ToolCallDelta` with `tool_index=-1` and a warning event the harness can act on.

### #4 — Forward `reasoning_content` from prior turns when rendering subsequent turns (interleaved thinking)

**Evidence:** TRT-LLM PR #12061 fixed `_parse_assistant_message_content` to forward `reasoning` / `reasoning_content`. PR #12199 made interleaved thinking first-class. Without it, the next turn's prompt has `<think></think>` empties where prior reasoning lived; the model re-derives the chain from scratch at higher entropy than the cached version.

**Effort:** 2-3 days. Two changes: (a) inject `reasoning_content` from prior assistant turns into the prompt at re-render time, *if* the model is interleaved-thinking-capable. (b) Coordinate with #1: rule is "strip from all but the most recent assistant turn." MLC's `d75d64e` enforces exactly this boundary.

### #5 — Disable Mamba/SSM rollback in speculative decoding by default

**Evidence:** TRT-LLM PR #14471 (May 26 2026) disables `mamba replay` default after shipping it in March (#13453). Industry-wide signal that the technique is correctness-fragile and/or perf-negative. Atlas's `project_qwen36_drift_gdn_clean.md` already shows NGram v21 with hand-rolled SSM rollback is 1.5× slower at 73% draft-rejection — independent corroboration.

**Effort:** 0.5 days. Flip the default of Atlas's SSM-rollback flag during speculative verification. Document the negative result. Applies only to spec-decode rollback — token-level grammar `matcher.rollback` is unaffected.

---

## 6. Sources

- TRT-LLM releases: https://github.com/NVIDIA/TensorRT-LLM/releases
- TRT-LLM PR #12061 (agentic fixes): https://github.com/NVIDIA/TensorRT-LLM/pull/12061
- TRT-LLM PR #12199 (interleaved thinking): https://github.com/NVIDIA/TensorRT-LLM/pull/12199
- TRT-LLM PR #14471 (disable mamba replay): https://github.com/NVIDIA/TensorRT-LLM/pull/14471
- TRT-LLM PR #13453 (mamba-2 rollback replay): https://github.com/NVIDIA/TensorRT-LLM/pull/13453
- XGrammar releases: https://github.com/mlc-ai/xgrammar/releases
- XGrammar PR #603 (Qwen3.6 tool calling): https://github.com/mlc-ai/xgrammar/pull/603
- XGrammar PR #616 (compile perf): https://github.com/mlc-ai/xgrammar/pull/616
- XGrammar PR #547 (token-level grammar): https://github.com/mlc-ai/xgrammar/pull/547
- XGrammar PR #610 (align with chat templates): https://github.com/mlc-ai/xgrammar/pull/610
- XGrammar PR #609 (unify reasoning): https://github.com/mlc-ai/xgrammar/pull/609
- MLC-LLM `d75d64e` (strip reasoning Qwen3): https://github.com/mlc-ai/mlc-llm/commit/d75d64e
- MLC-LLM `d46f65f` (Qwen3.5 templates): https://github.com/mlc-ai/mlc-llm/commit/d46f65f
- TGI maintenance-mode PR #3344: https://github.com/huggingface/text-generation-inference/pull/3344
- XGrammar structural-tag tutorial: https://github.com/mlc-ai/xgrammar/blob/main/docs/structural_tag/tool_calling_and_reasoning.md
