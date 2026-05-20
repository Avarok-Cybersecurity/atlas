# Atlas Debugging Methodology

A playbook for tracking down quality regressions in Atlas — distilled from the
Qwen3.6-27B-FP8 long-code degeneration investigation (commit `3ebc08a`, 2026-05-20)
that ultimately resolved two loader-side bugs producing a 14.2× improvement in
`tokens_to_first_degeneration` (1,196 → 16,968) and parity with vLLM behavior.

The investigation passed through several wrong hypotheses, multiple methodological
reversals, and finally landed on a 113-line, three-file fix. The *order* in which
those wrong hypotheses fell, and the diagnostics that let us discard each one, are
the durable lessons — more useful than the fix itself.

---

## 0. The general shape of the bug class

You're here because: the same model checkpoint produces clean output under a
reference framework (vLLM, HF Transformers) but degrades under Atlas. Same
tokenizer, same prompt, same sampler, same hardware. This is the hardest class of
bug because *nothing obvious is wrong* — the kernels run, the GEMMs return,
nothing crashes. The output just slowly drifts.

The trap: you start instrumenting at the kernel level too early, before ruling out
much cheaper explanations. Most "Atlas math is broken" reports turn out to be one
of: prompt template drift, tokenizer mismatch, sampler preset, or a loader-side
data-layout bug. **In this investigation, it was the last one. The kernels were
innocent.**

Memory rule that frames the whole game: *Never blame the model. Always find the
Atlas bug.* (See `feedback_never_blame_model.md` in the memory index.)

---

## 1. Cheapest-signal-first elimination ladder

Run these in order. Stop at the first one that explains the symptom.

| # | Hypothesis to refute | Cost to test | How to test |
|---|---|---|---|
| 1 | **Sampler/preset** is different from the reference | 5 min | Run both Atlas and reference under greedy (`temperature=0`). If the gap closes, it's sampling. If not, sampling is innocent — keep going. |
| 2 | **Prompt template render** differs | 10 min | Diff the rendered prompt strings (Atlas logs the Jinja-rendered output; vLLM has `print_chat_template_result`). One byte of drift is enough. |
| 3 | **Tokenization** differs | 5 min | Tokenize the rendered prompt in both engines and diff the token IDs. If even one ID differs, the engines aren't running the same input — fix that first. |
| 4 | **Termination logic** (stop tokens, length cap) differs | 5 min | Look at `finish_reason`. If reference says `stop` and Atlas says `length`, that's a stop-condition mismatch, not a math bug. |
| 5 | **Penalty / repetition / DRY** preset differs | 10 min | Build a "zero-penalty" image and re-run. If degeneration vanishes, the bug is a preset; if it persists, penalties are innocent. |
| 6 | **Quantization storage path** differs (FP8 vs NVFP4 dispatch) | 30 min | Inspect weight-loader logs; verify both engines load the same on-disk tensors at the same precision. |
| 7 | **Per-layer numerics** drift | hours | Only after 1-6 are clean. This is where the rest of this document lives. |

Most bugs die at step 1-3. The two real bugs in this investigation survived
through step 6 and required step 7.

---

## 2. Distinguish "completed cleanly" from "valid output"

Termination is not quality. A model can finish with `finish_reason=stop` and still
have emitted a lazy-stub loop, hallucinated identifiers, or repeated the same
6-line block 200 times.

Build a quality gate alongside the termination check. For code-output workloads,
the gate is typically: parses cleanly + contains expected entities + no duplicate
N-line blocks + a `tokens_to_first_degeneration` signal that fires when the model
starts repeating itself. The gate's *false-positive cost* (rejecting clean output)
matters less than its *false-negative cost* (accepting degenerate output).

In this investigation, `bench/longcode/harness.py` paired with `analyze.mjs`
provided the signal. The first version only checked `finish_reason` — and missed
the early bugs because Atlas was terminating "cleanly" on the lazy-stub loop.

> **Lesson:** Build the quality gate *before* you spend hours chasing numerics.
> Otherwise you'll declare wins on runs that didn't actually fix anything.

---

## 3. Build the byte-exact CPU oracle

Reference framework outputs (vLLM, sglang) are *behavior* baselines, not *math*
baselines — they have their own fused kernels with their own precision profiles.
For per-layer numerical comparison you need a reference whose math is canonical
and whose intermediate states you can read.

Use **HF Transformers on CPU, single-precision dequant**, with
`output_hidden_states=True`. It's slow (tens of minutes for a 27B forward pass on
CPU), but that's the *point*: it's the intended math without any custom kernels.

Critical gotchas this investigation hit:

1. **Feed the oracle the exact token IDs the SUT used — not the rendered prompt.**
   The oracle's own chat template may differ. Hooking `model(token_ids=...)` with
   `use_cache=False` bypasses the template entirely. Skip this step and you'll
   spend hours chasing "divergence" that's actually template drift.

2. **Don't hard-code the token-ID list in the oracle script.** Read it back from
   the SUT's `/tokenize` endpoint and pass it through. The investigation lost
   nearly a day to a 52-vs-54-token mismatch because the oracle was hard-coded to
   a 52-token sequence that no longer matched what Atlas was rendering.

3. **Verify oracle correctness independently.** Before trusting any per-layer
   divergence, confirm `oracle(token_ids) → expected_next_token` matches what the
   target model is *supposed* to do (e.g., compare against vLLM greedy output for
   the same IDs). If the oracle itself is wrong, every layer comparison is noise.

4. **Don't use an FP8 checkpoint for the BF16 reference.** HF Transformers may
   silently `Loading: ignoring all *_scale_inv tensors` and reinterpret FP8 bytes
   as BF16 — producing garbage values that mostly cross-correlate enough to look
   like "some" signal. If `|hf in_proj_qkv| = 800000` when you expected ~200,
   that's the smoking gun. Use the upstream BF16 checkpoint, not the FP8 quant.

5. **When comparing across SUT configurations (e.g., chunked-vs-unchunked
   prefill), guarantee the dumped position is the SAME** in every configuration.
   A dumper that fires on `first_call`'s last token will capture position
   `chunk_size - 1` under chunked prefill but position `L - 1` under single-chunk
   — a different token, different input, naturally low cosine. The "drift" you
   diagnose is then methodological noise, not a bug. Either dump on every call
   (overwriting) so the *last* chunk's last token wins, or thread an
   `is_last_chunk` flag through to the dumper.

---

## 4. Per-layer divergence comparator

Once oracle and SUT are emitting per-layer hidden-state dumps in a common format
(headerless bf16 `.bin` files works fine), compute:

- **Flat cosine** per layer
- **Flat relative L2** per layer (`||A−B|| / ||B||`)
- **Per-head cosine** (reshape to `[n_heads, head_dim]`, cosine per head)
- **Per-head magnitude ratio** (`||A_h|| / ||B_h||` per head)
- **Best-match permutation cosine** (greedy 1-1 head matching) — detects head-order
  rotation independent of magnitude

Plot all four against layer index. The **first-divergent layer** (where flat
cosine first drops materially from ~1.0) localizes the bug. The shape of the
*growth* curve from there reveals the bug class:

- Linear growth → BF16 truncation noise accumulating across layers
- Exponential growth → recurrence/state issue amplifying per token
- Sudden cliff → a specific kernel emitting garbage at one layer
- Per-head growth with low std → uniform precision noise (bf16 floor)
- Per-head growth with **high std** → ⚠️ pointer/layout bug, see §5

---

## 5. Per-head magnitude as a bug-class fingerprint

This is the diagnostic that ended the investigation. When you compute per-head
magnitude ratios (Atlas/oracle) across all heads of one layer:

| Pattern | Mean | Std | Interpretation |
|---|---|---|---|
| `~1.0 ± ULP` | 1.00 | <0.01 | Clean — no bug at this layer |
| `~0.98 ± 0.02` uniform | 0.98 | 0.02 | BF16 truncation noise; benign at low layer counts |
| `~1.05 ± 0.05` uniform | 1.05 | 0.05 | One-sided quant bias; suspect dequant precision |
| `mean ≈ 1.5, **std ≈ 1.7**, min 0.09, max 7.67` | high | **high** | ⚠️ **Pointer-type or layout aliasing** — kernel is reading the wrong bytes per head |

The high-std signature is unmistakable once you've seen it. It's not noise —
it's structure. Each head is being assigned a *different scrambled value* with
no relation to the intended one. That's not floating-point error; that's an
indexing or pointer-type mismatch.

> **Lesson:** Always report std/min/max alongside mean for per-head diagnostics.
> A "mean ≈ 1" report can hide std=1.7 — the fingerprint of the actual bug.

In this investigation, that single statistical observation collapsed weeks of
suspect chasing into a 2-line fix: `A_log` and `dt_bias` were stored on disk as
48-element BF16 (96 bytes) but consumed by the recurrence kernel via
`const float*` (48 × 4 = 192 bytes), so each head was reading
`⟨bf16[2h] ∥ bf16[2h+1]⟩` reinterpreted as IEEE-754 float. Std=1.7 is exactly
what you'd expect from random fp32-bit reinterpretation of an exp-distributed
bf16 array.

---

## 6. The sister-loader diff (the single highest-ROI lesson)

Atlas supports multiple model families through largely-parallel loader trees:

- `weight_loader/qwen35_dense.rs` ↔ `weight_map/ssm_qwen35.rs`
- `weight_loader/mistral.rs` ↔ ...
- per-architecture variants

When you've localized a bug to a specific model variant, **before building any
custom instrumentation, diff the loader against its sister loader for the
adjacent variant.** In this investigation:

```
$ diff weight_loader/qwen35_dense.rs weight_map/ssm_qwen35.rs
```

would have pointed straight at it. The MoE sister loader had this comment at
line 59, written *months earlier* by whoever first ported the MoE A3B variant:

> *"A_log and dt_bias MUST be FP32 — BF16 precision causes exponential error
> amplification in the GDR decay gate at 8k+ tokens."*

The dense loader was missing the corresponding `dense_keep_f32` call. The bug
existed *because* the MoE loader had been fixed and the dense loader hadn't.
The prescient comment was *already in the repo*, waiting to be diffed.

> **Lesson:** Diff the sister loader before instrumenting. Prescient comments
> often exist; they cost nothing to find and they collapse the search radius
> by orders of magnitude.

---

## 7. Reversal discipline

In this investigation, the following hypotheses were each held confidently for
hours, then disproven:

1. *"BF16 residual stream across 64 layers is the compounding error"* — disproven
   when fp32-residual builds still degenerated (and revealed the
   `scale_embeddings_fp32` general bug as a side-effect).
2. *"Triple-quantization FP8→BF16→NVFP4 is the source of all precision loss"* —
   partially correct (it was Bug #1), but the larger issue was elsewhere.
3. *"K-band attention is degenerating"* — this hypothesis flipped status three
   times: (a) suspected real, (b) refuted as an oracle alignment artifact, (c)
   re-confirmed as a real fingerprint but with a *different mechanism* than
   initially proposed (it was a downstream symptom of Bug #1's conv-k
   attenuation).
4. *"Per-head magnitude is a uniform deficit"* — refuted when std/min/max were
   actually computed and revealed the true high-std structure (§5).
5. *"Chunked SSM prefill at chunk_size=4096 introduces state-transition
   precision loss at long context"* (MoE 35B-A3B investigation, 2026-05-20) —
   the cosine appeared to collapse from 0.9999 to 0.95 at L=16k under chunked
   prefill but stay clean under single-chunk. This was traced to gotcha §3 #5:
   the dumper's Once-latch captured position 4095 of chunk 1 instead of
   position 16099 of the full prompt, comparing two different tokens against
   the same HF reference. After fixing the dumper to overwrite on every call,
   chunked and unchunked prefill produce identical math (cos 0.99989,
   per-head std 0.018). Refuted.

Each reversal cost hours. The discipline that limits the damage:

- **Maintain a reversal log.** Track each hypothesis, the diagnostic that
  promoted it, the diagnostic that refuted it, and the lesson. `tasks/lessons.md`
  in this repo is the canonical location.
- **Distrust hypotheses that fit too well.** If your favorite suspect explains
  the symptom but doesn't *predict* a falsifiable second observation, it's
  probably wrong.
- **Force a falsifiable prediction before committing to a fix.** "If Bug #2 is
  the cause, then `dense_keep_f32` should make per-head std collapse from 1.7
  to <0.01." That prediction was either going to confirm or kill the hypothesis
  in a single rebuild.

---

## 8. Tools to build (in order of leverage)

By rough leverage-per-hour-invested:

1. **Quality gate harness** (`bench/longcode/harness.py` + `analyze.mjs`) — pays
   for itself in the first day. Without this, you can't tell which builds
   improved anything.
2. **Per-layer dump pattern** (env-gated `ATLAS_DUMP_LAYERS` reading headerless
   bf16 `.bin` per `(layer, position, kind)`). Pattern in
   `crates/spark-model/src/layers/vision_encoder/enc_impl/utils.rs:38-64` —
   clone this, don't reinvent it. Zero-overhead when env unset (PCND/SSOT compliant).
3. **HF CPU oracle script** (model + hooks that dump intermediate tensors using
   the same bin format). One per family. Template in
   `bench/longcode/hang-forensics/hf_gdn_ref2.py`.
4. **Per-head diagnostic comparator** (cosine, relL2, magnitude ratio with
   std/min/max). Template in `bench/longcode/hang-forensics/gdn_chain_diff2.py`.
5. **CI numerical-divergence guard** — once you've shipped a fix, lock the
   per-layer cosines as a regression test. The next quality bug will be
   detected at PR time, not in production.

---

## 9. Polling discipline for multi-hour reproductions

Background reproductions on Atlas range from minutes (smoke tests) to hours (a
6000-token greedy completion through a 27B model with degeneration onset). Two
rules:

1. **Poll on ~60s intervals.** Anything longer and you'll spend half the day
   waiting; anything shorter and you'll burn the prompt cache without new
   information. (See `feedback_iterate_dont_tick_pace.md` in memory.)
2. **Fail-fast on the first failure signal.** Don't wait for the run to finish
   if you can detect from logs that it's already degenerating. `tail -f` the
   output and exit the polling loop on the first appearance of a degeneration
   marker. This converts hour-long waits into minute-long failure cycles.

---

## 10. Capture lessons, even when the fix is small

The final fix in this investigation was 113 lines across 3 files — surgical and
small. But the methodology that produced it took weeks. Don't lose that
methodology to the next person on the next bug.

After each correction (your own or a user's), update `tasks/lessons.md` with:

- **What you did wrong** (concretely — "I assumed per-head magnitude was uniform
  without checking std")
- **What signal would have caught it earlier** ("computing std/min/max alongside
  mean would have flagged this in the first hour")
- **The general rule it implies** ("always report std for per-head diagnostics")

The lesson file is the most-read file in the repo for anyone joining mid-investigation.

---

## Appendix: the specific bugs this investigation found

For grounding, the two bugs the methodology localized:

**Bug #1** — Dense Qwen3.6-27B-FP8's SSM `in_proj_qkv` and `out_proj` GEMMs went
through a triple-quant path (`FP8 → BF16 → NVFP4 → BF16'` at runtime) instead of
a native-FP8 GEMM. The intermediate BF16 truncation noise was uniform across
channels, but the depthwise conv weights for the *k* sub-segment were 18× smaller
than for the *v* sub-segment, so SNR for k channels collapsed to <1. Conv-k
cosine dropped to 0.55 against the HF oracle. Fix: route through native
`fp8_gemm_n128` (BF16 act × FP8 weight, FP32 accumulator). Originally
env-gated `ATLAS_FP8_SSM_PREFILL=1`; promoted to unconditional 2026-05-20
after a live-verified soak (and cross-ported to the MoE A3B sister loader
once an audit found the same triple-quant chain there).

**Bug #2** — `A_log` and `dt_bias` (per-head decay parameters for the GDR
recurrence, shape `[48]`) were loaded as BF16 (96 bytes) but consumed by the
recurrence kernel via `const float* A_log` (which reads 192 bytes). Each head's
4-byte fp32 read pulled
`⟨bf16[2h] ∥ bf16[2h+1]⟩` reinterpreted as IEEE-754 — producing scrambled per-head
decay gates with std=1.7. State magnitudes drifted exponentially over decode
tokens. Fix: `dense_keep_f32` loader path zero-extends BF16 to FP32 at load.
Already present in the MoE sister loader from an earlier port. Unconditional.

**Bug #3 (general)** — `scale_embeddings_fp32` was missing the no-embed-scale
guard its BF16 sibling had, so every non-Gemma model using fp32-residual mode
hard-failed at startup. Three-line fix.

None of the three bugs were in attention. All three were in the loader, the
SSM precision path, or the embedding step. The attention kernels and the
recurrence kernels themselves were correct — they were just being fed mis-laid
inputs. That's the most common failure mode in mature inference engines, and
the methodology above is built around finding it efficiently.

---

*Maintained alongside Atlas. Update when you learn something this document
doesn't already say.*
