# Per-Model Sampler Recommendations

Curated from official model cards (HuggingFace, vendor docs, Unsloth). When an exact value isn't
documented, the row says **NOT DOCUMENTED** and a defensible fallback is suggested. **DRY sampling
is not mentioned by any model vendor** — defaults below come from the oobabooga DRY reference
implementation (dry_multiplier=0.8 when enabled, base=1.75, allowed_length=2). Treat DRY as
opt-in: leave `dry_multiplier=0` (disabled) unless you see verbatim loops.

---

## 1. Qwen3.6-35B-A3B  (Qwen 3.6 hybrid MoE, ~3B active, dual-mode Thinking/Instruct)

| Param | Thinking (general) | Thinking (coding) | Instruct (non-thinking) |
|---|---|---|---|
| temperature | 1.0 | 0.6 | 0.7 |
| top_p | 0.95 | 0.95 | 0.80 |
| top_k | 20 | 20 | 20 |
| min_p | 0.0 | 0.0 | 0.0 |
| presence_penalty | 1.5 | 0.0 | 1.5 |
| repetition_penalty | **1.0** | 1.0 | 1.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 | 0 | 0 |

**Your current rep_pen=1.1, dry_mult=0.5 is OFF-SPEC.** Qwen explicitly says
`repetition_penalty=1.0`; use `presence_penalty=1.5` for repetition control.

---

## 2. Qwen3.6-27B  (Qwen 3.6 dense, same family, dual-mode)

| Param | Thinking (general) | Thinking (coding) | Instruct (non-thinking) |
|---|---|---|---|
| temperature | 1.0 | 0.6 | 0.7 |
| top_p | 0.95 | 0.95 | 0.80 |
| top_k | 20 | 20 | 20 |
| min_p | 0.0 | 0.0 | 0.0 |
| presence_penalty | **0.0** (note: smaller model uses 0.0, not 1.5) | 0.0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 | 1.0 |

> Note: the 27B's thinking-general `presence_penalty` is 0.0 per its model card,
> versus 1.5 on 35B-A3B. This is intentional per Qwen.

---

## 3. Qwen3.5-35B-A3B / Qwen3.5-122B-A10B  (Qwen 3.5 MoE, dual-mode)

| Param | Thinking general | Thinking coding | Instruct general | Instruct reasoning |
|---|---|---|---|---|
| temperature | 1.0 | 0.6 | 0.7 | 1.0 |
| top_p | 0.95 | 0.95 | 0.8 | 1.0 |
| top_k | 20 | 20 | 20 | 40 |
| min_p | 0.0 | 0.0 | 0.0 | 0.0 |
| presence_penalty | 1.5 | 0.0 | 1.5 | 2.0 |
| repetition_penalty | 1.0 | 1.0 | 1.0 | 1.0 |

Same recipe for the 122B-A10B. Qwen3.5-27B (dense): use same table.

---

## 4. Qwen3-Next-80B-A3B-Instruct  (Qwen3-Next hybrid attention+gated-DeltaNet, 3B active)

| Param | Recommended |
|---|---|
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 0–2 range (1.5 is the Qwen-family convention) |
| repetition_penalty | NOT DOCUMENTED — Qwen-family convention is 1.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 |

---

## 5. Qwen3-VL-30B-A3B-Instruct  (Qwen3-VL multimodal MoE, 3B active)

Official model card does NOT contain a Best-Practices block. Community consensus
(Unsloth + Qwen3-VL discussions):

| Param | Instruct | Thinking variant |
|---|---|---|
| temperature | 0.7 (Qwen Instruct-family default) | 1.0 (community) / 0.6 (official GitHub) — **temp=1.0 has been reported to cause Chinese-char leakage; 0.6 is safer** |
| top_p | 0.8 | 0.95 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 1.5 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 | 0 |

---

## 6. Nemotron-3-Nano-30B-A3B  (NVIDIA Nemotron 3 Nano hybrid MoE)

| Param | Reasoning ON | Tool calling |
|---|---|---|
| temperature | 1.0 | 0.6 |
| top_p | 1.0 | 0.95 |
| top_k | NOT DOCUMENTED — leave unset (default 0 / disabled) | NOT DOCUMENTED |
| repetition_penalty | NOT DOCUMENTED — use 1.0 | 1.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 | 0 |

Reasoning-OFF mode: NVIDIA explicitly recommends **greedy** (`do_sample=False`).

---

## 7. Nemotron-3-Super-120B-A12B  (NVIDIA Nemotron 3 Super MoE, 12B active)

| Param | Recommended (all tasks) |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | NOT DOCUMENTED |
| repetition_penalty | NOT DOCUMENTED — use 1.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 |

NVIDIA's Super model card says use the same `temperature=1.0, top_p=0.95`
across reasoning / tool-calling / chat. (The Nano-style 0.6/0.95 split is for Nano only.)

---

## 8. MiniMax-M2.7-NVFP4  (MiniMax M2.7, ~229B MoE)

| Param | Recommended |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 40 |
| repetition_penalty | NOT DOCUMENTED — use 1.0 |
| presence_penalty | NOT DOCUMENTED |
| dry_multiplier | NOT DOCUMENTED — keep 0 |

---

## 9. Mistral-Small-4-119B-2603  (Mistral Small 4 dense)

Mistral docs only call out `temperature`:

| Param | reasoning_effort=high | reasoning_effort=none |
|---|---|---|
| temperature | 0.7 | 0.0–0.7 (task-dependent) |
| top_p | NOT DOCUMENTED — community uses 0.95 | 0.95 |
| top_k | NOT DOCUMENTED | — |
| repetition_penalty | NOT DOCUMENTED — use 1.0 | 1.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 | 0 |

Mistral historically recommends NO repetition penalty (1.0) and NO top_k cap.

---

## 10. Gemma-4-31B-IT  and  Gemma-4-26B-A4B-IT

Same defaults per Google / Unsloth for both 31B dense and 26B-A4B MoE:

| Param | Recommended |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |
| repetition_penalty | 1.0 (disabled; Unsloth says "keep disabled or 1.0 unless looping") |
| presence_penalty | 0.0 |
| dry_multiplier | NOT DOCUMENTED — keep 0 |

> Gemma 4 is documented to "work best with high temperature for coding" — do NOT lower
> to 0.6 for coding the way you would on Qwen.

---

## DRY sampling — universal note

No vendor (Qwen, NVIDIA, MiniMax, Mistral, Google) recommends DRY values in their
model cards. The oobabooga reference defaults — **dry_multiplier=0.8, dry_base=1.75,
dry_allowed_length=2** — are the community standard *when enabled*. Default state is
**disabled** (`dry_multiplier=0`). Recommendation: leave DRY off (mult=0) unless you
observe verbatim loops in production; then turn it on at 0.8 / 1.75 / 2 with
sequence-breakers covering `\n`, `:`, `"`, `*`, code-fence backticks, and EOS.

**Your current `dry_mult=0.5` is half-strength of community-standard and is not
documented by any of these vendors. Either go to 0.8 (oobabooga default) or 0.**

---

## Sources

- Qwen3.6-35B-A3B: https://huggingface.co/Qwen/Qwen3.6-35B-A3B  (Best Practices)
- Qwen3.6-27B: https://huggingface.co/Qwen/Qwen3.6-27B  (Best Practices)
- Qwen3.5-35B-A3B: https://huggingface.co/Qwen/Qwen3.5-35B-A3B  (Best Practices)
- Qwen3.5-122B-A10B: https://huggingface.co/Qwen/Qwen3.5-122B-A10B  (Best Practices)
- Qwen3-Next-80B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-VL-30B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct (no BP block) + community thread https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF/discussions/1
- Nemotron-3-Nano-30B-A3B: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  +  https://unsloth.ai/docs/models/tutorials/nemotron-3
- Nemotron-3-Super-120B-A12B: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16  +  https://unsloth.ai/docs/models/nemotron-3-super
- MiniMax-M2.7: https://huggingface.co/MiniMaxAI/MiniMax-M2.7  +  https://github.com/MiniMax-AI/MiniMax-M2
- Mistral Small 4: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603  +  https://docs.mistral.ai/models/model-cards/mistral-small-4-0-26-03
- Gemma 4 (31B + 26B-A4B): https://unsloth.ai/docs/models/gemma-4  +  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/discussions/21  +  https://ollama.com/library/gemma4:31b
- DRY reference defaults: https://github.com/oobabooga/text-generation-webui/pull/5677  +  https://github.com/oobabooga/textgen/wiki/03-%E2%80%90-Parameters-Tab
