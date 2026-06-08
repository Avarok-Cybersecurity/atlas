# Research 2 — Qwen3 Official Multi-Turn Agentic Guidance (as of 2026-05-26)

## Scope

Atlas ships a Qwen3.6-style chat template at
`/workspace/atlas-mtp/jinja-templates/qwen3_5_moe.jinja` plus an
OpenAI variant at `jinja-templates/openai/qwen3_5_moe.jinja`. This
note compares Atlas's snapshot against the **upstream Qwen3.6** chat
template, the **Qwen3-Coder / Qwen3-Coder-Next** model cards, the
official `qwen-agent` framework, and the May-2026 community fixes
(froggeric "Agentic Loop Cure" v19, allanchan339 vLLM template fix,
QwenLM/Qwen3.6 issue #131).

The relevant Atlas FP8 target model is
**Qwen/Qwen3.6-35B-A3B** (BF16, released 2026-04-16) and its
fine-grained block-128 FP8 sibling
**Qwen/Qwen3.6-35B-A3B-FP8**.

---

## 1. Chat-template diff: Atlas snapshot vs upstream Qwen3.6

Atlas snapshot (lines 100-104 of `qwen3_5_moe.jinja`):

```jinja
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

Upstream Qwen3.6 template (verified on
`Qwen/Qwen3.6-35B-A3B/blob/main/chat_template.jinja`):

```jinja
{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

**Delta:** Atlas's template has NO `preserve_thinking` branch. Atlas
silently strips every historical assistant `<think>` block before the
last user query, regardless of caller intent. Upstream Qwen3.6 lets the
caller opt-in via `chat_template_kwargs={"preserve_thinking": true}`.

Two additional upstream-only behaviours Atlas is missing:

| Behaviour | Upstream | Atlas |
|---|---|---|
| Generation prompt when `enable_thinking=false` | Emits `<|im_start|>assistant\n<think>\n\n</think>\n\n` (forces empty closed thought, prevents drift) | Emits bare `<|im_start|>assistant\n` |
| `reasoning_content` field on assistant messages | Read directly when caller supplies it (no `</think>` parse needed) | Same |
| `developer` role | Accepted (mapped to system) | Rejected as "Unexpected message role" |

---

## 2. The May-2026 "Agentic Loop Cure" (froggeric v19, 2026-05-18)

This third-party patched template — widely picked up by opencode,
LM Studio, Aider and Continue — is the most-deployed Qwen3.6 template
right now. It is *not* an official QwenLM commit but reflects
QwenLM/Qwen3.6 issue #131 (still open at time of writing) and was
echoed in the dev-forum discussion of the model card. Key claims:

1. **Empty-think poisoning.** When `preserve_thinking=false` the
   stock Atlas/Qwen template *strips* the body of `<think>...</think>`
   but still emits the opening tag, leaving `<think>\n</think>` stubs.
   In Atlas's template this happens implicitly because
   `reasoning_content` ends up trimmed-empty on lines 99-101 and the
   `<think>\n\n</think>\n\n` wrapper is still printed. v19 removes the
   wrapper entirely when `reasoning_content` is empty.
2. **`preserve_thinking` default = true.** v19 flips the default
   because dynamic history pruning causes KV-cache prefix-match
   failures on every turn. Retaining historical thoughts is now the
   "100% prefix-hit" config.
3. **System-prompt softening.** v19 rewrites the `<IMPORTANT>` block
   to allow `</think>` → conversational synthesis (previously, the
   model learned `</think>` ⇒ tool_call mandatory, causing premature
   `<|im_end|>` after a thinking step that didn't need a tool — the
   "amnesia stall").
4. **Native XML tool format preserved.** Upstream and v19 agree:
   `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`.
   No JSON detour. (Atlas already matches.)
5. **minijinja safety.** Replace `|items` with explicit
   `for k in arguments` (Atlas already does this on line 120).

---

## 3. Tokenizer / tool-call output

Both Qwen3.6 and Qwen3-Coder-Next ship with these special tokens in
`tokenizer_config.json`:
`<tool_call>`, `</tool_call>`, `<function`, `</function>`,
`<parameter`, `</parameter>`. There is **no `tool_call_start_token`
remap** between Qwen3.5 and Qwen3.6 — the tokens are identical. The
upstream vLLM `--tool-call-parser qwen3_coder` and SGLang
`--tool-call-parser qwen3_coder` both rely on those exact byte
sequences. Atlas's F72 byte-anchor work and `<minimax:_call>` bug-fix
are consistent with this contract.

Qwen3.6 introduces NO new tool tokens vs Qwen3.5. The chat template
**does** add a tightened `<IMPORTANT>` block in tools-on system
prompts — Atlas already mirrors this verbatim (line 53).

---

## 4. Sampling parameter recommendations

From the official model cards (verified 2026-05-26):

| Model | Mode | temperature | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (+ FP8) | thinking, general | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Qwen3.6-35B-A3B (+ FP8) | thinking, precise coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Qwen3.6-35B-A3B (+ FP8) | non-thinking (instruct) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |
| Qwen3-Coder-Next | (non-thinking only) | 1.0 | 0.95 | 40 | — | — | — |
| Qwen3-Coder-30B/480B | non-thinking | 0.7 | 0.8 | 20 | — | — | 1.05 |

Important asymmetries Atlas should know:

- **Qwen3.6 uses `presence_penalty=1.5`, not `repetition_penalty>1`.**
  The Atlas memory note `project_qwen36_fp8_post_think_eos.md`
  records dropping `rep_pen=1.1` from MODEL.toml prose categories,
  which aligns with upstream. Upstream goes further: presence_penalty
  is the recommended lever, NOT repetition_penalty.
- **Qwen3-Coder family is non-thinking only.** Qwen3-Coder-Next
  explicitly says: *"This model supports only non-thinking mode and
  does not generate `<think></think>` blocks. Specifying
  `enable_thinking=False` is no longer required."* Atlas's
  "template-forced thinking detection" needs to short-circuit on the
  Coder line.
- **Qwen3-Coder top_k=40** (not 20). Coder is the only Qwen3 family
  member with top_k=40.
- **Qwen3.6 NO LONGER supports `/think` and `/no_think` soft
  switches** that Qwen3 had. The toggle is solely via
  `chat_template_kwargs={"enable_thinking": ...}`.

---

## 5. Qwen-Agent multi-turn recipe

`qwen-agent` (the official framework, `github.com/QwenLM/qwen-agent`)
deploys Qwen3.6 with:

```python
llm_cfg = {
  'model': 'Qwen/Qwen3.6-35B-A3B',
  'model_type': 'qwenvl_oai',
  'model_server': 'http://localhost:8000/v1',
  'generate_cfg': {
    'use_raw_api': True,
    'extra_body': {
      'chat_template_kwargs': {
        'enable_thinking': True,
        'preserve_thinking': True,   # <-- key
      }
    },
  },
}
```

Key points:
- `use_raw_api=True` bypasses qwen-agent's own client-side prompt
  reformatting and lets the server template handle history. This is
  the upstream recommendation for Qwen3.6+.
- `preserve_thinking=True` is passed through `chat_template_kwargs`,
  which means *the server's Jinja template must read it*. Atlas's
  template currently ignores this kwarg entirely.
- Tools are registered via the standard OpenAI `tools=[…]` array OR
  via MCP `{'mcpServers': {…}}`. The qwen-agent README does not
  document the XML wire format because it's emitted by the server-side
  template, not the client.

---

## 6. Long-history (5–10+ turn) tool-call rounds — canonical behaviour

The reverse-walk `last_query_index` block (Atlas lines 67-77 and
identical upstream) decides where the "current reasoning window"
begins. Behaviour:

- Walks messages **right-to-left**, starting from the end.
- Treats `<tool_response>…</tool_response>`-only user messages as
  *not* a fresh query (they are tool returns wrapped in `user` role).
- Stops at the **first real user message**, sets
  `last_query_index = that index`.

Then on the forward render (lines 100-104 and upstream equivalent):

- Assistant turns **after** `last_query_index` → thinking preserved.
- Assistant turns **before** `last_query_index` → thinking dropped
  (default) OR preserved (if `preserve_thinking=true`).

What this means for the long-history question:

- **Tool-active turns inside the current user query window** (i.e.
  user asks → assistant thinks → tool_call → tool_response →
  assistant thinks → … all without an intervening fresh user
  question) always keep their `<think>` blocks.
- **Historical turns from earlier user queries** drop their thinks by
  default; with `preserve_thinking=true` they are kept
  chronologically.

So the canonical Qwen3.6 answer is: **preserve thinking inside the
current tool loop; drop historical pre-query thinking unless
`preserve_thinking=true`.** The v19 fix (and qwen-agent recipe)
recommends `preserve_thinking=true` as the production default
specifically because dynamic pruning blows the KV-cache prefix on
every new turn.

Atlas's current template implements only half of this contract: it
keeps current-window thinking but unconditionally drops historical
thinking with no override. For long agent loops past ~5 user-query
boundaries this can repeatedly invalidate the KV-cache prefix.

---

## 7. FP8 variant specifics

`Qwen/Qwen3.6-35B-A3B-FP8` ships an identical `chat_template.jinja`
to the BF16 version (verified via HF diff). FP8 model card prescribes:

- Quantization: fine-grained FP8, block size 128, mixed BF16/E4M3.
- Sampling: identical to BF16.
- vLLM/SGLang launch flags identical, except `--quantization fp8`
  is implicit from the safetensors metadata.

There are no FP8-only template differences. The Atlas
`project_qwen36_phase2b_softmax_expf.md` finding that late-layer
attention regresses under FP8 KV at deep layers is an inference-side
issue, not a template issue.

---

## Top-5 ranked patterns Atlas should adopt or correct

### 1. Add `preserve_thinking` support to both Atlas templates
**Impact:** unblocks 100% KV-cache prefix hits on long agent loops,
matches qwen-agent default, fixes the "amnesia stall" pattern
documented in v19. **Effort:** one-line predicate edit on lines
100-104 of both `qwen3_5_moe.jinja` files:
```jinja
{%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
```
**Default:** match upstream Qwen — `false` by default. Recommend
documenting `true` as the production default for opencode / agent
clients.

### 2. Skip the empty-`<think>` wrapper when `reasoning_content` is empty
**Impact:** removes the v19 "empty-think poisoning" bias. When
`reasoning_content|trim` is empty, do not emit `<think>\n\n</think>\n\n`
at all — emit just the content. Avoids teaching the model the toxic
correlation "empty think ⇒ I must call a tool." Atlas already saw a
symptom of this in `project_qwen36_fp8_post_think_eos.md` (short
answers force-truncated by EOS guard).

### 3. Emit closed-empty think on `enable_thinking=false` generation prompt
**Impact:** Atlas currently emits a bare `<|im_start|>assistant\n` when
`enable_thinking=false`. Upstream emits
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. This explicit closed
empty-think is what the model was trained to see in non-thinking
mode and prevents the model from spontaneously opening a `<think>`
later in generation (the bug Atlas already fixed in
`project_spontaneous_think_fix.md` lives in inference code; the
template fix complements it).

### 4. Switch FP8/BF16 Qwen3.6 production defaults to `presence_penalty=1.5`
**Impact:** Atlas's MODEL.toml currently leans on
`repetition_penalty`. Upstream's official recommendation is
`presence_penalty=1.5, repetition_penalty=1.0` for both thinking and
non-thinking general modes. Coding mode uses both at 0.0/1.0.
Aligning Atlas defaults to upstream eliminates a class of subtle
quality regressions and matches the exact numbers the model was
RL-tuned against.

### 5. Branch sampling defaults by model family in MODEL.toml
**Impact:** Qwen3.6 thinking-general (temp=1.0), Qwen3.6 coding (0.6),
Qwen3.6 non-thinking (0.7), Qwen3-Coder-Next (temp=1.0, top_k=40),
Qwen3-Coder-30B/480B (temp=0.7, top_p=0.8, top_k=20, rep_pen=1.05)
are five DIFFERENT recommended profiles. Atlas should ship per-family
defaults, and crucially the "template-forced-thinking detection"
should be **disabled** for the Qwen3-Coder family because those
models never emit `<think>`. This is a small but real correctness
improvement for any Qwen3-Coder serving Atlas does.

---

## Sources

- [github.com/QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6)
- [github.com/QwenLM/Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder)
- [github.com/QwenLM/qwen-agent](https://github.com/QwenLM/qwen-agent)
- [huggingface.co/Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/chat_template.jinja](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/chat_template.jinja)
- [huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
- [huggingface.co/Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [github.com/QwenLM/Qwen3.6/issues/131](https://github.com/QwenLM/Qwen3.6/issues/131) — empty-think KV invalidation
- [huggingface.co/froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) — v19 "Agentic Loop Cure" 2026-05-18
- [github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix](https://github.com/allanchan339/vLLM-Qwen3-3.5-3.6-chat-template-fix) — vLLM-specific template fixes
- [docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) — vLLM Qwen3.5/3.6 deployment recipe
- [qwen.readthedocs.io/en/latest/framework/qwen_agent.html](https://qwen.readthedocs.io/en/latest/framework/qwen_agent.html) — qwen-agent integration guide
- [qwen.ai/blog?id=qwen3.6-35b-a3b](https://qwen.ai/blog?id=qwen3.6-35b-a3b) — official launch post
