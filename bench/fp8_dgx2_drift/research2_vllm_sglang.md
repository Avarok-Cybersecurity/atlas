# Research 2: vLLM and SGLang — multi-turn agentic flows, tool-call faithfulness, thinking-mode, KV/SSM cache (post-Oct 2025)

Scope: vLLM `main` and SGLang `main` as of 2026-05. Atlas comparator: full re-prefill per request with prefix caching, MTP K=2 spec decode, FP8 KV, qwen3_coder tool parser with EBNF + XGrammar.

---

## 1. Request-to-request token handling (incremental vs full re-prefill)

### vLLM v1

- vLLM v1 does **not** do incremental token deltas across HTTP requests. Each `POST /v1/chat/completions` is rendered fresh by the chat-completions serving layer (`vllm/entrypoints/openai/chat_completion/serving.py`, refactored from the old monolithic `serving_chat.py` into per-feature subpackages), tokenized via the Renderer, then submitted as a brand-new `EngineCoreRequest` with a unique id. The only persistent state between requests is the **automatic prefix cache (APC)** keyed by block-hash; block-hash chains reuse cached KV when leading tokens are byte-exact, otherwise unmatched suffix is fully re-prefilled.
- V1's only "diff" mechanism is internal scheduler<->worker per-step; no shared state across requests besides the KV hash table.
- **Hybrid (SSM/GDN) caveat from #42948**: on hybrid groups the first KV block can lose its hash entry on every block reassignment because the chunked hybrid coordinator stores single-entry cache keys without reference counting — reissuing the same prompt collapses to 0% APC hit. Qwen3-Next / Qwen3.5 / Qwen3.6 are in this class.

### SGLang

- SGLang exposes a per-request RadixCache, and for hybrid SSM models it ships `MambaRadixCache` (Hybrid Models Meet SGLang blog) with two variants:
  - **V1 (default)**: no extra buffer, lower memory, but **incompatible with spec-decode** (in-place SSM updates can't roll back).
  - **V2 (`--mamba-scheduler-strategy extra_buffer --page-size 64`)**: requires FLA kernel backend, allocates an isolated Mamba cache slot per draft token, sandboxing each candidate.
- For multi-turn agentic workloads, SGLang has an explicit feature request (#20144) and an `X-Overwrite-Cache-ID` header proposal: evict prior Mamba snapshots of the same agent so only one Mamba state is retained per rollout — large SSM states (thousands of KV equivalents each) would otherwise cause "bad cache rate". Worth noting as a real production pattern.
- Like vLLM, SGLang re-renders + re-tokenizes per request — no diff-API for token streams across requests. Reuse is entirely via the radix tree.

### Implication for Atlas

Atlas's "full re-prefill per request with prefix caching" matches both. The win Atlas misses sits one level deeper: **how cache keys are derived** for hybrid layers (Mamba state per chunk-boundary) and **what gets stored** between turns (separate SSM snapshots vs single live state).

---

## 2. Sampler / SamplingParams for Qwen3-Coder

Officially recommended Qwen3-Coder family parameters (480B-A35B, 30B-A3B, Coder-Next, all variants):

| param | value |
|-------|-------|
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| repetition_penalty | 1.05 |
| presence_penalty | 0 |

vLLM Qwen3-Next recipe + SGLang cookbook both pull these from the model's `generation_config.json` automatically — SGLang explicitly says "no manual configuration is needed; SGLang applies the recommended sampling parameters from the model's generation_config.json".

**What Atlas should check**:
1. Does Atlas honor `generation_config.json` on load? If users set temperature=0 or some default, the recommended distribution shifts dramatically and tool-call XML stability degrades.
2. `repetition_penalty=1.05` MUST be applied — multiple downstream reports (Ollama #14493, vLLM #38994) blame "silent discard of repetition_penalty" for tool-call instability + late-turn loops. Atlas already exempts stop tokens from rep_penalty (good — HF convention), but the multi-turn cases need the penalty applied to long-running prose tokens specifically.
3. `top_k=20` is *much* tighter than typical inference defaults. Wide-top-k destabilizes the `<function=`/`<parameter=` XML tail.

---

## 3. `<think>` block handling across turns

This is the single most important multi-turn finding, and both engines have shipped fixes / known issues post-Oct 2025.

### vLLM
- `Qwen3ReasoningParser` (`vllm/reasoning/qwen3_reasoning_parser.py`, also in Rust as `rust/src/reasoning-parser/src/qwen3.rs` via `DelimitedReasoningParser`) operates on strings with both token-id and string-level partition. Strips opening `<think>` if generated, partitions on `</think>`. Stateless across requests.
- Bug #39056 (vLLM 0.19): Qwen3.5-FP8 + `qwen3` reasoning + `qwen3_coder` tool parser non-streaming — reasoning parser eats `<tool_call>` if emitted inside `<think>`, tool parser then sees empty content → tool_calls=[]. **Fix**: extract embedded tool-call XML from reasoning and promote to content. Atlas should audit — if model is forced to think and tool XML emits *during* thinking, qwen3_coder would also miss it.
- vLLM recommends switching from `qwen3_coder` to **`qwen3_xml`** (PR #25028, `vllm/tool_parsers/qwen3xml_tool_parser.py`). qwen3_xml uses `xml.parsers.expat` SAX state machine with `_preprocess_xml_chunk` normalization (`<function=name>` → `<function name="name">`), `_find_next_complete_element` buffering, and `_auto_close_open_parameter_if_needed`. NVIDIA DGX Spark/GB10 forum reports `qwen3_xml + enhanced jinja` sustains 6-hour agentic sessions where `qwen3_coder` fails after 2 hours.

### SGLang
- SGLang ships `Qwen3CoderDetector` (`python/sglang/srt/function_call/qwen3_coder_detector.py`) — regex-based, no grammar/EBNF, no thinking-tag awareness. The bug #8331 ("too eager") was patched in #9023 by adding a guard: only parse tool calls when tools are actually provided in the request. Atlas should have an analogous guard.
- **Critical multi-turn bug documented in Qwen3.6 #131**: the upstream chat template emits *empty* `<think></think>` blocks for historical assistant turns even when there was no reasoning content. This serializes the same logical history differently each turn → **prefix-cache invalidation** + token-budget creep. Fix is one line in the Jinja template: gate on `reasoning_content` truthiness. If Atlas uses the unpatched upstream Qwen3-Coder / Qwen3.5 / Qwen3.6 chat template, it will leak empty `<think></think>` and lose prefix-cache hits between turns even when content is byte-identical.
- **SGLang's known multi-turn quality regression**: SGLang preprocessing drops `reasoning_content` from API-supplied messages → multi-step tool use with Qwen3-Thinking degrades because the model expects to see prior thinking. Workaround documented by Qwen: pass `content` as-is without extracting thinking, let the chat template re-wrap it. Atlas needs to decide which side of this contract it implements.

---

## 4. Tool-call parser behavior

| Engine | Parser | Mechanism | Streaming | Recovery |
|---|---|---|---|---|
| vLLM | qwen3_coder | regex + state machine (`is_tool_call_started`, `in_function`, `in_param`, `json_started`) | Dual-mode: full regex when non-stream; stateful cursor when stream | Falls back to `</tool_call>` or next `<parameter=` if `</parameter>` missing |
| vLLM | qwen3_xml (PR #25028, newer) | `xml.parsers.expat` SAX event handler | Chunk-buffered XML element completion | `_preprocess_xml_chunk` normalizes non-XML `=`-syntax; `_auto_close_open_parameter_if_needed` heals |
| SGLang | qwen3_coder_detector | regex, cursor-based `parsed_pos` | Streaming via `parse_streaming_increment` | `break` on incomplete tag waits for more tokens; multiple parameter terminator candidates evaluated, nearest wins |
| HF upstream qwen3_coder (Qwen team) | regex + state-machine, parameter type coercion from tool schema | Same dual-mode | `_reset_streaming_state()` per new message |

**None of the three use XGrammar/EBNF for the tool-call XML itself.** They all use regex + state machines and recover heuristically from malformed XML. vLLM's structured outputs *can* use xgrammar/guidance for JSON schemas, but the Qwen3 tool-call XML is not grammar-enforced upstream. Atlas's "EBNF grammar enforcement + XGrammar" path is **more aggressive than either vLLM or SGLang here** — that's a feature if your grammar matches the model's training distribution exactly, but a footgun if any cell of the EBNF disagrees with what the model emits (e.g., whitespace, parameter ordering, or `<function=name>` vs `<function name="name">`).

---

## 5. Speculative-decode rollback for SSM/hybrid

### vLLM
- Multiple post-Oct 2025 issues, none fully fixed: #39273 (ngram on GDN — SSM advances by N proposed tokens but rollback isn't N-aware), #41190 (qwen3_next_mtp/DFlash TP=2 cudaErrorIllegalAddress at `num_accepted_tokens_event.synchronize()`), #40880 (MTP × CUDA-graph on Qwen3-Next → degenerate output via `vllm/v1/spec_decode/eagle.py method="mtp"`), #40831 (TurboQuant KV × any spec-decode → token loops).
- Common workarounds: drop spec-decode, switch to `kv-cache-dtype=fp8_e5m2`, or disable CUDA graphs.
- Root cause: `postprocess_mamba()` uses block-aligned checkpoints that don't reflect accepted-token count. vLLM is moving to per-draft Mamba slot allocation (SGLang-style), "in progress".

### SGLang
- SGLang already shipped **cache-isolation** for spec-decode on hybrid (Hybrid Models Meet SGLang PyTorch blog): per-draft slot, promote-on-acceptance. Combined with `extra_buffer` strategy + page_size 64 + FLA kernel backend, this works for EAGLE-Tree with top-K>1.
- Performance on H200 + Qwen3-Next-80B-A3B-Instruct-FP8: 257 → 307 → 325 tok/s as `num_speculative_tokens` goes 2 → 3 → 4 with topk=4/draft=8.

### Implication for Atlas

Atlas does FP8 KV + MTP K=2 + spec-decode on a hybrid (Qwen3-Coder-Next / Qwen3.5 / Qwen3.6). The MTP path runs through Atlas's own hybrid kernel — but the SSM-state rollback issue is the **same bug class** that vLLM has open and SGLang fixed via slot isolation. If Atlas's MTP rollback only rewinds KV-cache and leaves SSM/conv state unrolled-back (the v20 → v21 lesson from the local TRT-LLM NGram work, already memorized in `project_x_engagement` / `gdn_regtile_results`), there will be drift on rejected proposals — exactly the pattern that shows up as "tool args truncate after turn N" or "thinking leaks into content".

---

## Top-5 ranked things Atlas might be missing or doing wrong

1. **Chat-template empty-`<think></think>` leak across turns.** Qwen3.5/3.6 + Qwen3-Coder upstream Jinja templates emit empty thinking blocks for historical assistant turns when no reasoning_content exists, breaking prefix-cache byte-equality and growing token budgets. Fix: gate the `<think>...</think>` emission on `reasoning_content` truthiness (Qwen3.6 #131 one-liner). Atlas should patch the bundled chat templates and verify each multi-turn replay produces a byte-identical prefix.

2. **SSM/conv state rollback on rejected MTP drafts.** vLLM has 4 open bugs in this exact shape (#39273, #40880, #41190, #40831). SGLang fixed it via per-draft Mamba cache-isolation slots (`extra_buffer` strategy + page_size 64 + FLA). Atlas's MTP K=2 path needs an audit: confirm that on partial acceptance (n_acc < k), SSM recurrent state and conv1d state are checkpointed pre-draft and restored to the accepted boundary — not just the KV cache (the TRT-LLM v21 lesson applies here byte-for-byte).

3. **EBNF + XGrammar grammar enforcement is stricter than vLLM/SGLang and can fight the model.** Both engines use regex + state-machine tool parsers with permissive recovery (auto-close, multiple terminator candidates, normalize `<function=name>` vs `<function name=...>`). vLLM's newer `qwen3_xml` parser is a SAX state machine, not a grammar. Atlas's EBNF could be silently masking out valid tail tokens the model wants to emit (whitespace, attribute reordering, deferred closing) — manifesting as truncated args or hung tool calls. Worth a focused test: disable XGrammar enforcement on `<parameter>` body, keep it only on the outer `<tool_call>` / `<function=` / `</function>` skeleton, see if faithfulness improves.

4. **Sampler defaults likely diverge from Qwen3-Coder's recommended `temp=0.7, top_p=0.8, top_k=20, rep_penalty=1.05`.** Both vLLM and SGLang auto-honor `generation_config.json`. The Atlas memory at `project_qwen36_fp8_post_think_eos` already shows MODEL.toml `rep_pen=1.1` was dropped from prose categories — confirm `top_k=20` and the 1.05 penalty are active by default for Qwen3-Coder family, and that stop tokens stay exempt (already done per `feedback`). Wide top_k destabilizes the XML tail and produces the late-turn collapse pattern.

5. **Prefix-cache invalidation on hybrid first block.** vLLM bug #42948 (DSv4-Flash) shows hybrid groups losing the first-block cache hash entry on every reassignment, collapsing APC to 0%. The hybrid coordinator's intersection cascade propagates a single zero-hit through all SWA / SSM groups. Atlas's prefix-cache implementation for hybrid models should be tested: reissue the *same* prompt twice and confirm second-prefill is sub-100ms; if it isn't, the Mamba-state hash key or the first-block reassignment is the culprit. SGLang's `MambaRadixCache` with separate-LRU semantics (KV evicts leaf→root, Mamba evicts from any node) is the reference architecture.

---

Sources:
- vLLM main tree (gh api): `vllm/v1/engine/input_processor.py`, `vllm/entrypoints/openai/chat_completion/serving.py`, `vllm/tool_parsers/qwen3coder_tool_parser.py`, `vllm/tool_parsers/qwen3xml_tool_parser.py`, `vllm/reasoning/qwen3_reasoning_parser.py`, `rust/src/chat/src/parser/reasoning/mod.rs`, `rust/src/chat/src/parser/tool/mod.rs`, `rust/src/reasoning-parser/src/qwen3.rs`
- SGLang main tree (gh api): `python/sglang/srt/function_call/qwen3_coder_detector.py`, `python/sglang/srt/function_call/function_call_parser.py`, `python/sglang/srt/configs/qwen3_next.py`
- vLLM issues: #39056 (tool calls lost inside `<think>`), #39273 (ngram + GDN rollback), #40831 (TurboQuant × spec-decode), #40880 (MTP × CUDA-graph), #41190 (TP=2 MTP crash), #42948 (hybrid first-block APC), #34755 (Coder-Next TP freeze), #19051 (reasoning + tool_choice=required)
- SGLang issues: #8331 (qwen3 parser too eager, fixed #9023), #9654 (streaming multiple tool calls fail with auto), #20144 (limit Mamba states per rollout), #20415 (Unified Hybrid Radix Cache Refactor), #22949 (2026 Q2 roadmap)
- Qwen issues: Qwen3 #1831 (21-fix chat template), Qwen3.6 #131 (empty think blocks in history), Qwen3-VL #1812 (penalties not applied), Qwen3-Coder HF discussions
- Docs: vLLM Qwen3-Coder-480B-A35B recipe, vLLM Qwen3.5/3.6 recipe, SGLang Qwen3-Coder-Next cookbook, SGLang qwen3 basic usage, PyTorch blog "Hybrid Models Meet SGLang", NVIDIA DGX Spark/GB10 forum "Qwen3.5 Tool Calling finally fixed"
- Qwen3 reasoning + tool calling reference: docs.vllm.ai reasoning_outputs, qwen.readthedocs.io function_call
