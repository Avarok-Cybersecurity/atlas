# Speculative Decoding (MTP)

**MTP** = **Multi-Token Prediction**. The flagship feature that takes Qwen3.5-35B from ~70 tok/s to **131 tok/s** on a single GB10, and Qwen3-Next-80B from ~70 to **104**. Speculative decoding with a model-native draft head.

## The idea

Plain decode generates one token per forward pass. If the model is certain about the next few tokens, we could generate more than one per pass — the bottleneck is almost always memory bandwidth, not compute, so generating 2 or 3 tokens costs roughly the same as 1.

MTP does exactly that:

1. The main model forward produces logits for the next token (call it $t_{+1}$).
2. A small **MTP head** — a few transformer layers that hang off the main-model hidden state — predicts $t_{+2}$, $t_{+3}$, …, $t_{+k}$ as *drafts*.
3. The main model runs one extra forward pass on the drafted positions to *verify* them, producing the "true" logits at each drafted position.
4. We accept the longest prefix of draft tokens where the verify logits' argmax agrees with the draft. After the first disagreement (or the end), we take the main model's next token.

For K=2 (one draft), the best case is 2× throughput: one draft + one verify yields two tokens per verify pass. The expected speedup depends on draft acceptance rate — Qwen3.5 + NVFP4 MTP achieves ~85% acceptance on short-prompt benchmarks, which maps to ~1.8× throughput in practice.

Atlas's numbers:

| Model | No spec | MTP | Speedup |
|---|---:|---:|---:|
| Qwen3.5-35B-A3B | 70 tok/s | **131 tok/s** | 1.87× |
| Qwen3-Next-80B | 74 tok/s | 104 tok/s | 1.41× |
| Qwen3.5-122B-A10B (EP=2) | ~32 tok/s | **46 tok/s** | 1.44× |

## The MTP head

The MTP head lives in the checkpoint under the `mtp.*` prefix. For Qwen3.5, it's 1–2 transformer blocks (depending on K) fed from the penultimate layer's hidden state. It is trained jointly with the main model, so draft distributions match what the main model would actually emit.

`--num-drafts` controls K (the number of draft tokens). The per-model default comes from `MODEL.toml`. Most models cap at K=2; some (MiniMax-M2.7) support K=1/K=2 natively. The `MultiModuleMtpHead` caps naturally — more layers means more compute per verify, at some point not worth it.

`--mtp-quantization` must match the main-model checkpoint quantization. For an NVFP4 main checkpoint with an NVFP4 MTP head, `--mtp-quantization nvfp4`. Mixing is an error and will produce gibberish.

## The verify loop

Pseudocode for a K=2 step:

```
# One step produces up to 3 accepted tokens.
main_logits, main_hidden = main_model.forward(cur_tokens, kv_cache=shared)
t_1 = sample(main_logits[-1])

# Draft the next two tokens from the MTP head
t_2, t_3 = mtp_head.draft(main_hidden, prev=t_1)

# Verify both in one main-model forward
verify_logits = main_model.forward([t_1, t_2, t_3],
                                   kv_cache=shared,
                                   append_kv=True)
a_2 = argmax(verify_logits[1])   # true next token after t_1
a_3 = argmax(verify_logits[2])   # true next token after (t_1, t_2)

# Accept the longest matching prefix
if a_2 == t_2:
    accept(t_1, t_2)
    if a_3 == t_3:
        accept(t_3)
        next_seed = t_3
    else:
        accept(a_3)
        next_seed = a_3
else:
    accept(t_1, a_2)   # drop drafts
    next_seed = a_2
```

Three subtle correctness requirements that must hold throughout:

1. **KV cache must track accepted tokens, not drafts.** If a draft is rejected, its KV contribution must be unwound.
2. **The SSM state must track accepted tokens, not drafts.** For hybrid models, the SSM recurrence is stateful — verify passes update the state; rejects must roll back.
3. **Sampler state (running RNG, penalty counters) must track accepted tokens, not drafts.**

All three are implemented in `crates/spark-model/src/speculative.rs` and the paired `rewind_kv_cache` + `rewind_mamba_state` hooks in `spark-runtime`.

## The bug-sweep history

MTP is the single subsystem with the most documented bug-sweep history. Lessons that shaped the current code:

- **`seq_len += k - 1` off-by-one** (Pass-16). The MTP scheduler bootstrap violated the "tokens[0] already in seq.tokens" precondition; caused Fibonacci drift on 80B-MTP. Fixed to `seq_len += k`.
- **WY (whisperer/verifier) state desynchronisation** (Pass-16). State clamping across the draft/verify boundary was off by one step.
- **v_contiguous + MTP** (Pass-6). The `v_contiguous` optimisation broke MTP's KV append because it assumed one token per call. Fixed with an explicit K-aware KV-append path.
- **Sibling stride bugs** (Pass-22). `qwen3_attention` and `qwen3_ssm` layers assumed K=2 for stride arithmetic; corrected to handle K=1/2/3.
- **Slot-keyed `verify*_graph` caches** (Pass-22). CUDA graphs for verify were keyed by batch only; needed `(batch, k)` to avoid replaying a K=1 graph on a K=2 step.
- **NVFP4 MTP loader force-BF16** (Wave-6). When the checkpoint's `ignore_modules` listed `mtp.*`, the loader accidentally forced BF16 — disabled MTP on models where it should have worked. Fixed to fall through to the proper dequant.
- **MTP logit masking** (most recent). Masking MTP draft logits at propose time (disallowing tokens the main model can't produce from the current position) improved tool-call throughput by +37%.

## MTP with tools

A hidden value of MTP is tool-call throughput. Tool calls are structured (JSON or XML); most tokens in a well-formed call are predictable conditional on the opening delimiter. The logit mask at propose time filters draft tokens that would break the grammar; acceptance jumps from ~70% to ~95% inside a tool call.

On agentic workloads (Claude Code, OpenCode, Cline), this compounds because a large fraction of generated tokens are tool calls. The +37% throughput is measured end-to-end against a real agent.

## Self-speculative (no MTP weights)

`--self-speculative` is the fallback for models without an MTP head: layer-skipping drafts. The "drafter" runs the main model with some attention + FFN layers skipped, producing a fast-but-approximate draft; the full model verifies.

Acceptance rate is lower (~60%) than MTP (~85%), but it works on any model. Atlas ships this for coverage; operators typically use MTP when the checkpoint supports it.

## N-gram speculative (CPU-side)

`--ngram-speculative` is the other fallback: an n-gram pattern matcher on recent output. If the model is repeating a token pattern (e.g. verbatim quoting a document), the matcher predicts the continuation directly. Acceptance is binary (0 or 100%), and the average rate on open-ended generation is low, but on certain workloads (summarisation, re-ranking) it's free throughput.

N-gram speculative was experimented with heavily on TRT-LLM (see the `project_ngram_*` notes); Atlas's Rust implementation lives in `spark-server/src/ngram.rs` and is much simpler.

## Files to read

- `crates/spark-model/src/speculative.rs` — the verify + accept loop.
- `crates/spark-runtime/src/kv_cache.rs` — `rewind_kv_cache`.
- `kernels/gb10/<model>/<quant>/` — there isn't a "MTP kernel"; MTP reuses the main model's attention/MoE kernels with different shapes.
- `docs/SPEC-DECODING-TODO.md` — authoritative design + outstanding items.
- `docs/history/alpha-2-release.md` and subsequent pass notes — bug-sweep history.
