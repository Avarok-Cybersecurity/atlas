# Research-2: Sampling Parameters vs Qwen3-Coder Family Recipes

Scope: compare `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml [sampling.*]` against
upstream Qwen3-Coder recipes, vLLM/SGLang defaults, opencode/Cline/Roo Code, and
post-Oct-2025 community recipes. Audit rep-penalty exemption for stop / special
/ control / tool-grammar tokens. Audit logit-bias × rep-penalty interaction
inside the forced-token fast-path. Recommend ranked, concrete parameter changes.

## 1. Atlas current values (MODEL.toml, all 4 presets)

| Preset | temp | top_p | top_k | pres | freq | rep_pen | dry_mult |
|---|---|---|---|---|---|---|---|
| `thinking_text`    | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.10 | 0.50 |
| `thinking_coding`  | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.10 | 0.50 |
| `non_thinking`     | 0.7 | 0.80 | 20 | 0.0 | 0.0 | 1.10 | 0.50 |
| `tools`            | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.10 | 0.50 |

Dispatcher (`crates/spark-server/src/api/chat/sampling_setup.rs::build_sampling`)
only selects three of these:

```rust
let preset = if tools_active           { &state.sampling_presets.tools }
             else if enable_thinking   { &state.sampling_presets.thinking_text }
             else                      { &state.sampling_presets.non_thinking };
```

`thinking_coding` is parsed by `build.rs`, emitted into the codegen table
(`SamplingPresets.thinking_coding`), and then **never referenced anywhere in
spark-server** — confirmed by `grep -rn thinking_coding crates/spark-server`.
Dead preset.

## 2. Upstream recipes

### 2.a Qwen3-Coder-Next (HF model card, Feb 2026 — most recent Coder release)

> "To achieve optimal performance, we recommend the following sampling
> parameters: `temperature=1.0, top_p=0.95, top_k=40`."

Non-thinking-only (`<think>` blocks never emitted). No `repetition_penalty`,
`min_p`, or `presence_penalty` recommendation — silent on those.

### 2.b Qwen3-Coder-30B-A3B-Instruct (HF model card)

`temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05`.
Non-thinking only.

### 2.c Qwen3.6-35B-A3B (the actual model Atlas serves — `Qwen/Qwen3.6-35B-A3B` HF card "Best Practices")

| Mode | temp | top_p | top_k | min_p | pres | rep_pen |
|---|---|---|---|---|---|---|
| Thinking (general)    | **1.0** | 0.95 | 20 | 0.0 | **1.5** | **1.0** |
| Thinking (coding)     | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Instruct (non-think)  | 0.7 | 0.80 | 20 | 0.0 | **1.5** | 1.0 |

Card explicitly warns: `presence_penalty > 0` "may cause occasional language
mixing and slight decrease in performance" → set 0.0 for coding/precision.

### 2.d vLLM / SGLang defaults

Neither ships a Qwen3-Coder-specific override; both inherit
`generation_config.json` (`temp=1.0, top_p=0.95, top_k=20, rep_pen=1.0` —
the thinking-general values). No mode-switching — caller's responsibility.

### 2.e opencode / Cline / Roo Code defaults

* **opencode**: default `temp=0` most providers, `0.55` for Qwen. Never sets
  rep / pres / freq / min_p / top_k.
* **Cline / Roo Code**: `temp=0`, no penalties.

Implication: these clients **rely entirely on MODEL.toml** for penalty
values; mis-setting them server-side has no client override path.

### 2.f Post-Oct-2025 community recipes

* **Unsloth Qwen3-Coder-30B GGUF**: matches Qwen card exactly (`temp 0.7,
  min-p 0.0, top-p 0.80, top-k 20, repeat-penalty 1.05`).
* **Unsloth Qwen3.6-27B GGUF**: identical to the 35B-A3B table above.
* **DEV "Qwen3-Coder-Next 2026 Guide"**: `1.0/0.95/40, rep-penalty=1.1` —
  the only community source that recommends `rep_pen > 1.0`, justified by
  long-horizon agentic decode loops.
* **r/LocalLLaMA (Mar–May 2026)** consensus: DRY @ 0.8 (oobabooga default)
  is the community floor; DRY < 0.8 is "weakly observable" — Atlas's 0.5
  is below the noise threshold.
* **ollama#14493** (Mar 2026): Qwen3-Coder needs `presence_penalty=1.5`
  to prevent thinking loops; ollama silently dropped the param. Reinforces
  that card-recommended 1.5 is load-bearing for *thinking-general*.
* **llama.cpp#20164** (Apr 2026): Qwen3-Coder-Next long-ctx tool loops with
  one missing optional param — structural bug, not sampling.

## 3. Are Atlas's values inside the recommended bands?

| Atlas | Card / vendor | Verdict |
|---|---|---|
| `tools.temp = 0.6` | Qwen3-Coder-30B says 0.7; Qwen3.6-35B coding 0.6; Coder-Next 1.0 | **IN BAND** (matches Qwen3.6 coding) |
| `tools.top_p = 0.95` | All Qwen Coder cards: 0.8–0.95 | **IN BAND** |
| `tools.top_k = 20` | All Qwen cards: 20 (Coder-Next 40) | **IN BAND** |
| `tools.rep_pen = 1.1` | Qwen3.6-35B cards: **1.0**; Qwen3-Coder-30B: 1.05; Unsloth: 1.05 | **ABOVE BAND** by ~5 % |
| `tools.dry_mult = 0.5` | Community standard: 0.8 OR disabled (0.0) | **BELOW community-standard ON-band** |
| `tools.presence = 0.0` | Qwen3.6 coding: 0.0; general thinking: 1.5 | **IN BAND for coding** |
| `non_thinking.top_p = 0.80` | Qwen3.6 instruct: 0.80; Qwen3-Coder-30B: 0.80 | **IN BAND** |
| `non_thinking.temp = 0.7` | Qwen3.6 instruct: 0.70; Qwen3-Coder-30B: 0.70 | **IN BAND** |
| `non_thinking.presence = 0.0` | Qwen3.6 instruct card: **1.5** | **OUT OF BAND** (Atlas chose 0.0 deliberately, see TOML comment 2026-05-24) |
| `thinking_text.temp = 0.6` | Qwen3.6 thinking-general card: **1.0**; thinking-coding: 0.6 | **OUT OF BAND** for general thinking; Atlas implicitly treats all thinking as coding-style |
| `thinking_text.presence = 0.0` | Qwen3.6 thinking-general card: **1.5** | **OUT OF BAND** (empirical revert documented in TOML — Qwen's 1.5 caused max-token wandering in opencode A/B) |

The TOML comment block (lines 53-83) explains the deliberate Qwen-card
deviations: with `presence_penalty=1.5` the model walked off into rare-token
paths and burned the entire `max_tokens=8192` budget in two test runs. So
Atlas's deviations are *empirically calibrated*, not stale.

## 4. Is rep_penalty correctly excluded for stop / special / control / tool-grammar tokens?

**No exclusion exists in Atlas's sampler.**

`crates/spark-runtime/src/sampler/sample_impl.rs::sample_with_params_seeded`
applies `repetition_penalty`, `presence_penalty`, `frequency_penalty` to *every*
token in the history without filtering. The `stop_token_ids` field on
`SamplingParams` is carried only for the engine.rs termination check
(`spark-model/src/engine.rs:56,71,135,150`), never read by the penalty loop.

Practical impact:
* **EOS / `<|im_end|>` / `<|endoftext|>`**: not in `token_history` until they
  are *generated* — so penalizing them is a no-op on the first emission.
  Once emitted they would terminate the sequence. Safe in practice.
* **`<think>` / `</think>` / `<tool_call>` / `</tool_call>`**: these *do* appear
  in `output_tokens` (the token_history fed to the penalty step). On the
  *next* turn (or on multi-tool-call turns) their logits get the `/=1.1`
  penalty. For a 1.1 rep_pen on a logit ~+8.0 this knocks the score to ~+7.3
  — measurable, sometimes margin-flipping under FP8 drift.
* **Tool-grammar tokens (XGrammar bitmask survivors)**: the grammar bitmask
  runs *after* penalties (see `decode_logits_seq.rs::run_pipeline` — penalties
  inside `sample_with_params_history`, bitmask via
  `logit_processors::grammar_bitmask`). So a tool-grammar token already in
  history is penalized, then the bitmask either keeps or drops it. If
  grammar leaves *only* penalized tokens (common in nested JSON where `","`
  recurs hundreds of times), the surviving distribution is squashed — but
  argmax usually still wins. The greater risk is in *temperature>0* sampling
  where the penalty inverts margins.

**The in-tool-body workaround already neutralizes most of the damage.**
`decode_logits_seq.rs:450-466` sets `rep_pen=1.0`, `pres=0.0`, `freq=0.0`,
`dry=0.0`, `lz=0.0` whenever `a.inside_tool_body && !a.inside_thinking`.
Matches vLLM convention.

What is *missing* is the exemption for the **structural markers themselves**
— `<tool_call>`, `</tool_call>`, `<think>`, `</think>`, `<|im_end|>` —
which bracket the body but live *outside* `inside_tool_body`. Exactly the
tokens the model needs free to re-emit in multi-tool-call turns.

## 5. Logit-bias × rep_penalty interaction inside the fast-path

`sampling_setup.rs:97-108` appends an exponentially-decaying bias on the
`<tool_call>` opener:

```
repeat_count 0/1 → +3.0
            2   →  0.0
            3   → -5.0
            ≥4  → -10.0
```

Order of operations in `sample_with_params_seeded`:
1. rep_penalty (multiplicative, history-driven)
2. presence/frequency (additive, history-driven)
3. LZ / DRY
4. `logit_bias` (additive, **after** all penalties)
5. top-n-sigma → temperature → top-k → softmax → min-p → top-p → sample

So on the first tool call (`repeat=0`) the `<tool_call>` logit is
`(L / 1.1) + 3.0` (rep_pen has already touched it if `<tool_call>` is in
history; otherwise just `L + 3.0`). On the *second* identical tool call
(`repeat=2`, bias=0.0) the `<tool_call>` token is in history → rep_pen
divides by 1.1 with no countervailing bias. Then on repeat=3 the
`-5.0` bias plus rep_penalty produces an effective `L/1.1 - 5.0`, which is
strong enough to flip away from `<tool_call>` — *desired* — but at this
point the `dry_multiplier=0.5` and `lz_penalty` may also be firing on the
sequence `<tool_call>{"name":"…` n-gram, compounding the suppression.

This is a **mild loop-attractor risk** in the opposite direction: rep_pen +
dry both penalize the model away from a legitimate fourth tool call. The
forced-token fast-path (`ForcedTokenFastPath`, only fires when grammar
admits exactly one token) papers over this when grammar is active — when
grammar admits >1 tokens the penalty compounding is exposed.

## 6. Should we apply DIFFERENT sampling per token type?

| Context | Recommendation |
|---|---|
| **Inside `<think>`** | Qwen3.6 card: `1.0/0.95/pres=1.5/rep=1.0`. Wide exploration. Atlas uses `0.6` → too tight; can hide reasoning loops. |
| **Inside tool body** | Already scoped: penalties off, grammar enforces shape. Keep `temp=0.6` or drop to 0. |
| **Free prose between turns** | Card: `0.7/0.8/pres=1.5`. Atlas: `0.7/0.80/0.0` — only presence diverges (deliberately). |
| **Structural markers** | Exempt from rep_pen / presence / DRY. |
| **Stop tokens (EOS)** | Same — exempt. Non-issue today (they terminate), but breaks for Responses-API multi-turn. |

`thinking_coding` is the dead preset that *should* fire for thinking-in-code-
context. At minimum the dispatcher should let it activate via request hint
(`chat_template_kwargs.thinking_style="coding"`).

## 7. Top-5 ranked concrete sampling changes

1. **Exempt structural / stop / control tokens from rep_penalty,
   presence_penalty, frequency_penalty, DRY, and LZ across the board.**
   Current: all penalty stages iterate the full history with no filter.
   Proposed: pass a `penalty_exempt_ids: &[u32]` slice into
   `sample_with_params_seeded` containing `eos_token_id`, `think_start/end`,
   `tool_call_start/end`, `<|im_end|>`, `<|im_start|>`, plus any
   model-declared `special_tokens`. Skip the penalty loop body for those
   IDs. This is the single largest correctness fix and removes a measurable
   class of multi-tool-call regressions.

2. **Drop `dry_multiplier` 0.5 → 0.0 (or raise to 0.8) on all four presets.**
   Current: `dry_mult = 0.5` across the board.
   Proposed: `dry_mult = 0.0` for `tools` and `thinking_*`; `dry_mult = 0.8`
   only for `non_thinking` *if* free-prose loops are observed.
   Rationale: 0.5 is below community-noise floor (sources unanimous on 0.8)
   AND DRY is structurally redundant with rep_pen + LZ during tool body
   anyway. Half-strength gives the worst of both worlds: penalty noise
   without enough magnitude to break attractors.

3. **Drop `repetition_penalty` 1.10 → 1.05 on `tools` and
   `thinking_coding`; → 1.00 on `thinking_text` and `non_thinking`.**
   Current: 1.10 universally.
   Proposed: 1.05 for tool/coding contexts (matches Qwen3-Coder-30B card
   value, the closest official Coder recipe Atlas is shaped against); 1.00
   for prose contexts (matches Qwen3.6-35B card across all three modes).
   1.10 is industry-light but still above what Qwen explicitly says for
   *every* mode — and the 35B-A3B card is unambiguous that 1.0 is correct.

4. **Wire `thinking_coding` into the dispatcher and bump `thinking_text` to
   card values.**
   Current: `build_sampling` ignores `thinking_coding`; `thinking_text`
   uses `temp=0.6 / presence=0.0` (coding-style).
   Proposed:
   - `thinking_text` → `temp=1.0, top_p=0.95, top_k=20, presence=1.5,
     rep_pen=1.0` (Qwen3.6 thinking-general).
   - `thinking_coding` → `temp=0.6, top_p=0.95, top_k=20, presence=0.0,
     rep_pen=1.0` (Qwen3.6 thinking-coding) — selected when system prompt
     mentions code (`fn `, `def `, file extensions) OR when an opencode/
     Cline-style header is present, OR when `chat_template_kwargs.
     thinking_style="coding"`.
   Backward-compat: keep the empirical-revert `presence=0.0` block only
   for the `tools` preset (where the regression was observed); let prose
   thinking actually exercise the Qwen-card recipe.

5. **Decouple the `<tool_call>` exponential-bias schedule from rep_penalty.**
   Current: bias `+3 / +3 / 0 / -5 / -10` stacks on top of `rep_pen / dry /
   lz` penalties that *also* hit `<tool_call>` once it's in history.
   Proposed: while the bias schedule is active, force-add the
   `<tool_call>`, `</tool_call>`, `<tool_response>`, `</tool_response>`
   tokens to `penalty_exempt_ids` (change #1 above), so the bias is the
   *only* mechanism shaping the model's choice of repeating the opener.
   Eliminates double-penalty compounding that currently nudges the
   model away from legitimate 4th/5th parallel-tool-call attempts.

(Bonus, not in top-5 but worth noting: the *forced-token fast-path*
(`ForcedTokenFastPath` in `logit_processors/forced_token.rs`) is correct as
written — it short-circuits before sampling so neither rep_pen nor bias can
affect a grammar-mandated single-legal-token decision. No change there.)

## Sources

- [Qwen3.6-35B-A3B HF card "Best Practices"](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.6-27B HF card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Qwen3-Coder-Next HF card](https://huggingface.co/Qwen/Qwen3-Coder-Next)
- [Qwen3-Coder-30B-A3B-Instruct HF card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
- [Unsloth Qwen3-Coder tutorial](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally)
- [Unsloth Qwen3.6 tutorial](https://unsloth.ai/docs/models/qwen3.6)
- [DEV "Qwen3-Coder-Next Complete 2026 Guide"](https://dev.to/sienna/qwen3-coder-next-the-complete-2026-guide-to-running-powerful-ai-coding-agents-locally-1k95)
- [opencode agents docs](https://opencode.ai/docs/agents/)
- [ollama#14493 Qwen3-Coder presence_penalty silently dropped](https://github.com/ollama/ollama/issues/14493)
- [llama.cpp#20164 Qwen3-Coder long-context tool loop](https://github.com/ggml-org/llama.cpp/issues/20164)
- [HF Transformers RepetitionPenaltyLogitsProcessor exempt-list feature request #26902](https://github.com/huggingface/transformers/issues/26902)
- [vLLM SamplingParams docs](https://docs.vllm.ai/en/latest/design/logits_processors/)
- [oobabooga DRY reference (PR #5677)](https://github.com/oobabooga/text-generation-webui/pull/5677)
- Internal: `bench/qwen36_fp8_dequant_audit/per_model_sampler_recommendations.md`
- Internal: `crates/spark-server/src/api/chat/sampling_setup.rs`
- Internal: `crates/spark-server/src/scheduler/decode_logits_seq.rs:430-472`
- Internal: `crates/spark-runtime/src/sampler/sample_impl.rs`
- Internal: `crates/spark-server/src/scheduler/logit_processors/forced_token.rs`
- Internal: `kernels/gb10/qwen3.6-35b-a3b/MODEL.toml [sampling.*] + [behavior]`
