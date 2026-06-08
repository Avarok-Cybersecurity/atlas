# TRT-LLM Tool-Call / Structured-Output / Grammar Pipeline — Deep Dive

**Audience:** Atlas (Rust, custom CUDA, GB10 / SM121) maintainers
**Target gap:** Qwen3.6-35B-A3B-FP8 multi-turn agentic coherence failures in opencode
**Sources:** TRT-LLM `main` (Oct–Nov 2025 commits, version line 1.0→1.3), XGrammar 0.1.25, NVIDIA dev forums

---

## 0. TL;DR — Executive Summary

TRT-LLM treats tool-calling as **two cooperating layers**: (1) an **XGrammar-driven hard constraint** during sampling — `_build_tool_strict_guided_decoding_params()` builds a *structural-tag* grammar from `tools[].function.parameters` and runs it through `GuidedDecoder` (`build`→`copy`→`apply_bitmask`), with per-sequence rollback for spec-decode; (2) a **per-model post-hoc tool parser** (`tensorrt_llm/serve/tool_parser/qwen3_tool_parser.py` for Qwen3.x) that re-parses the constrained text into `ToolCall` objects, with streaming buffering and partial-trigger anchoring (`_ends_with_partial_token`) and *graceful* JSON failure (warns, continues). The Qwen3 parser is **regex-based** on `<tool_call>\n…\n</tool_call>` blocks. The grammar layer uses an abstract `GrammarMatcher` (XGrammar or LLGuidance), wired into a `CapturableGuidedDecoder` variant that pipes bitmask uploads through CUDA-graph host-funcs. Speculative decoding rollback is first-class: `_rollback_rejected_tokens()` + `_rollback_draft_tokens()` track `num_advanced_tokens / num_guided_tokens / num_advanced_draft_tokens` per slot, and Mamba-2 rollback-replay landed in a recent release. None of the grammar layer is FP8/Blackwell-specific — XGrammar runs on CPU + a small bitmask kernel. Atlas already has XGrammar (commit context) and a Qwen3 byte-anchored sampler (F72); what's missing is the **strict-mode structural-tag pipeline** and **disciplined SSM/MoE rollback for speculative + grammar**, both of which are realistically portable.

---

## 1. Inventory of TRT-LLM Components

### 1.1 Sampler / Executor (`tensorrt_llm/_torch/pyexecutor/`)
- `sampler.py` (~5,261 lines): owns logprob mode, spec-decode draft accounting, tree managers. Grammar bitmask is *not* inline — sampler hands logits to `GuidedDecoder.execute()`.
- `guided_decoder.py`: orchestrates `add_batch` → `build` → `copy` (CPU→GPU async) → `apply_bitmask` (in-place masked logits). API: `add_batch / build / execute / apply_bitmask / rollback_rejected_tokens / rollback_draft_tokens`. Has a `CapturableGuidedDecoder` for CUDA-graph capture using host-func callbacks + a queue.
- `grammar_matcher.py`: abstract `GrammarMatcher` with `accept_token / rollback / fill_next_token_bitmask / is_terminated`. Two concrete impls: `XGrammarMatcher` (JSON, JSON-Schema, regex, EBNF, **structural tags**) and `LLGuidanceMatcher` (JSON, regex, Lark). No FP8/Blackwell-specific code anywhere in this layer.
- Spec-decode rollback counters: `num_advanced_tokens`, `num_guided_tokens`, `num_advanced_draft_tokens`. After verification, `_rollback_rejected_tokens` walks the matcher backward by `advanced − accepted`. `_rollback_draft_tokens` resets the drafting flag at end-of-draft.

### 1.2 OpenAI Compatibility (`tensorrt_llm/serve/`)
- `openai_protocol.py`: imports `xgrammar` directly. `ResponseFormat` types: `text | json | json_schema | json_object | regex | ebnf | structural_tag`. `_response_format_to_guided_decoding_params()` is the bridge.
- `openai_server.py`: holds `_build_tool_strict_guided_decoding_params()` which, when a tool has `strict=True` (or any tool when the parser declares `supports_structural_tag()`), synthesizes a **structural_tag** payload — triggered tags + per-tool JSON Schema — that becomes a hard grammar constraint at generation time. Non-strict tools fall through to free generation + post-hoc parser.
- `chat_utils.py`: Jinja chat template application + tool-segment re-parse for multi-turn history. **No Qwen3-specific normalization** — relies on the model's chat template for round-trip.

### 1.3 Tool Parsers (`tensorrt_llm/serve/tool_parser/`)
14 files, factory-dispatched on `config.json:model_type`:
- `qwen3_tool_parser.py`: **regex** `re.findall(r"<tool_call>\n(.*?)\n</tool_call>", DOTALL)`. Streaming via `_normal_text_buffer` + `_ends_with_partial_token` (byte-suffix watcher for `<`, `<t`, `<to`, …). JSON parse failures are *warned, not raised* — parser yields the surviving tool calls and the trailing prose.
- `qwen3_coder_parser.py`: XML-shaped — `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`. Streams parameter-by-parameter, types coerce defensively (failed `int` → string).
- `minimax_m2_parser.py`, `deepseekv3{,1,2}_parser.py`, `gemma4_parser.py`, `glm4{,7}_parser.py`, `kimi_k2_tool_parser.py`: format-specific siblings.
- `utils.py`: `partial_json_loads` (consumes incomplete JSON during streaming), `find_common_prefix` (computes delta for streaming arg deltas).
- `base_tool_parser.py`: contract = `detect_and_parse`, `parse_streaming_increment`, `has_tool_call`, `structure_info` (returns a fn that emits the grammar pattern for strict mode).

### 1.4 Speculative Decoding (`docs/source/features/speculative-decoding.md`)
8 mechanisms shipped: Draft/Target, EAGLE-3 (chain or dynamic tree), NGram, **MTP (Deepseek-only)**, PARD (parallel mask-token draft), DFlash (cross-attn from target hidden states), Suffix Automaton (model-free GPU pattern matcher), and user-pluggable `Drafter`. Recent release notes: **Mamba-2 rollback replay** + radix-based SWA cleanup — exactly the pattern Atlas had to hand-roll in v21 for NGram.

---

## 2. Answers to Specific Questions

### Q1. Does TRT-LLM use native FP8 MMA on Blackwell?
**Yes, and explicitly on SM120.** TRT-LLM 1.0 added FP8 support for SM120, and the PTX it lowers to is `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32` (non-block-scaled) plus the block-scaled `kind::mxf8f6f4.block_scale.scale_vec::1X` variant for NVFP4. This matches Atlas's confirmation that the instruction works on SM121. TRT-LLM ships these via CUTLASS 4.x kernels (with SM120's hardcoded Cooperative scheduler, ClusterShape 1×1×1, 4 tile shapes — same ceiling Atlas hit). **NVFP4 KV cache on SM120 is still gapped** (open issue #10241): cubins are SM10X-only, no SM120 build.

### Q2. Multi-turn agentic flow handling
TRT-LLM relies almost entirely on the model's own chat template for multi-turn assembly (`chat_utils.py:load_chat_template`). The only "agentic glue" is: (a) `_parse_assistant_message_content` normalises function-args JSON before re-injecting into history, (b) `make_tool_call_id` enforces stable per-model ID formats (`kimi_k2` style is `functions.{name}:{idx}`). There is **no model-specific multi-turn coherence mitigation** — strict-mode XGrammar is the only mechanism keeping agentic tool calls byte-correct turn over turn. No "post-think EOS guard" equivalent, no rep-penalty-exemption for stop tokens.

### Q3. Tool-call parser implementation
**Hybrid: hard-constrained generation + post-hoc regex.** Qwen3 path = regex on `<tool_call>…</tool_call>` with byte-suffix partial-trigger anchoring. *Not* an FSM. Strict-mode tools additionally compile a structural-tag grammar at request time and run it through XGrammar as a hard mask. The parser layer's job in strict mode is reduced to peeling the (already correct) JSON out of the (already correct) tags.

### Q4. Spec-decode + grammar rollback semantics
Per-slot counter triple (`num_advanced_tokens`, `num_guided_tokens`, `num_advanced_draft_tokens`). Verification computes accepted-prefix length `k`; matcher is rolled back by `advanced − k` calls to `matcher.rollback(n)`. End-of-draft does `rollback_draft_tokens` which only resets the drafting flag — the matcher state lives on the verified prefix. Mamba-2-specific replay is a separate executor-level path that re-runs the SSM forward on the verified prefix to restore conv1d/SSM state — the same trick Atlas built by hand for NGram v21.

### Q5. Recent NVIDIA blog / whitepapers on post-processing
NVIDIA's public material is thin on this. Perplexity's "Hosting Qwen on Blackwell" (May 2026) and the NV dev-blog Qwen3-Next post focus on NVLink-5 + expert routing. There is no NVIDIA post specifically on tool-call post-processing. The interesting material is **MLC's XGrammar-2 blog (May 4, 2026)**, which announces structural-tag as the unified protocol for OpenAI harmony, tool calling, and reasoning channels — and confirms TRT-LLM integration in strict mode.

### Q6. Known Qwen3-Coder / Qwen3.6 issues
- **vLLM #19056**: Hermes streaming parser breaks on a Qwen3 token. Maps directly to Atlas's F68/F72 byte-anchor problem.
- **vLLM #34755**: Qwen3-Coder-Next-FP8 with tools hard-freezes multi-GPU TP — grammar compilation deadlock under multi-rank.
- **TRT-LLM #12321**: Qwen3.5 transformers-v5.2+ support — config-version drift.
- **NV forum 366451**: Qwen3.5 tool-calling "finally fixed (possibly)" on DGX Spark/GB10 — the fix path is the structural-tag strict mode.
- **HF Qwen3-Coder-Next discussion #17**: community is still arguing which vLLM parser to use (hermes vs qwen3_coder vs xml) — i.e. the regex-only path is brittle by design.

---

## 3. Atlas Cross-Reference: What's Portable, What's NVIDIA-Proprietary

| TRT-LLM mechanism | Atlas status | Realistically portable? |
|---|---|---|
| XGrammar matcher + bitmask | XGrammar already integrated (F68, F70, F72 history) | **Already have it.** Gap = no structural-tag *strict-mode* wiring. |
| `GuidedDecoder` build/copy/apply pipeline w/ async bitmask upload | Atlas applies grammar inline in sampler (per F72) | **Yes, 1-2 days.** Async H2D + double-buffer would smooth CUDA-graph capture. |
| `CapturableGuidedDecoder` host-func callbacks | Atlas's sampler isn't fully graph-captured for grammar | **Yes, 3-5 days** — host-func is a CUDA driver feature, not NV-proprietary. |
| Per-slot rollback counter triple | Atlas has rollback in NGram v21 only | **Yes, 2-3 days.** Refactor to per-slot counters so rollback works for MTP, NGram, EAGLE uniformly. |
| Mamba-2 rollback-replay (executor-level) | Atlas hand-rolled in v21 NGram | **Already have the kernel infra.** Should generalise it (see #4 recommendation). |
| Structural-tag synthesis from `tools[].function.parameters` | Atlas only has free-gen + post-hoc parse | **Yes, 1 week** — purely Rust + XGrammar API. Biggest coherence win. |
| Per-model tool-parser factory | Atlas has Qwen3 / Qwen3-coder parsers | **Already have it.** Could be unified under a trait. |
| FP8 MMA `m16n8k32.f32.e4m3.e4m3.f32` | Atlas confirmed works, hasn't adopted | **Yes** — 2× theoretical vs BF16 for attention. Atlas should land. |
| SM120 NVFP4 KV cache cubins | TRT-LLM also missing | **Not portable** (NV cubin-gated). Atlas is on par. |
| CUTLASS SM120 MoE tile-shape ceiling | Same ceiling as Atlas | **Not Atlas's bug.** NV upstream issue. |
| NVLink-5 expert all-to-all | DGX Spark has no NVLink | **N/A.** GB10 uses ConnectX-7 RoCE. |

**Not portable** (NVIDIA-proprietary): trtllm-gen kernels (cubin-gated), nvjet GEMMs (closed PTX), TRT engine compiler, NVLink switch-fabric. **Portable** (anything Python/CUDA-C++ open in TRT-LLM): grammar pipeline, structural-tag synthesis, rollback counters, sampler integration patterns, host-func graph capture.

---

## 4. Diagnosis for Atlas's Qwen3.6-35B-A3B-FP8 Multi-Turn Failure

Cross-referencing the memory index (`feedback_no_workarounds.md`, `project_qwen36_fp8_post_think_eos.md`, `project_f72_byte_anchor.md`):

1. Atlas's tool layer = **free generation + post-hoc regex** (F72). Coherence in multi-turn opencode depends on the model emitting *exactly* `<tool_call>\n{…}\n</tool_call>`. Any byte drift (even one whitespace) breaks the regex anchor.
2. FP8 quant noise + small temperature variance is enough to wobble the tag literal occasionally — exactly the symptom seen.
3. TRT-LLM's escape hatch is **structural-tag strict-mode**: the begin/end tag literals + the args-JSON-Schema are *all* enforced by XGrammar. No drift possible, by construction.
4. Atlas already has the XGrammar matcher. The missing piece is `_build_tool_strict_guided_decoding_params()` equivalent — synthesise a structural-tag payload from `request.tools` at admission time.

This is the single largest expected win for multi-turn coherence and is purely Rust-side work (no kernel).

---

## 5. References

- TRT-LLM main: `tensorrt_llm/_torch/pyexecutor/{guided_decoder,grammar_matcher,sampler}.py`
- TRT-LLM main: `tensorrt_llm/serve/{openai_server,openai_protocol,chat_utils}.py`
- TRT-LLM main: `tensorrt_llm/serve/tool_parser/*` (14 files)
- TRT-LLM docs: `docs/source/features/{speculative-decoding,guided-decoding}.md`
- MLC XGrammar-2 blog, 2026-05-04
- NVIDIA forum SM120 PTX threads 329702 + 330254
- TRT-LLM issues #10241 (NVFP4 KV SM120), #11799 (FMHA SM120/121 cubins), #12321 (Qwen3.5)
- vLLM issues #19056 (hermes Qwen3), #34755 (Qwen3-Coder-Next FP8 TP freeze)
- HF `Qwen/Qwen3-Coder-Next` discussion #17
