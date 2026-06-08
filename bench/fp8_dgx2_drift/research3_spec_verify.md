# Research 3 — Speculative Verify of FP8 Drift via Higher-Precision Critic

**Hypothesis under test**: Qwen3.6-35B-A3B-FP8 produces "confidently wrong" low-entropy
tokens during tool-arg emission. Can a higher-precision (BF16) verifier — same model
or a small distilled critic — catch this drift cheaply via speculative-decoding-style
rejection sampling?

---

## 1. arXiv: FP8 (or low-bit) Draft + Higher-Precision Verify

### 1.1 QSpec — same model, two precisions (Oct 2024 / 2025)
- Architecture: **one** weight-quantized model, two activation paths. Draft runs
  **W4A4**, verify runs **W4A16** on the SAME weights. KV cache is reused.
- Acceptance: **74-89%** at γ=6 across ShareGPT/MATH; **HumanEval** kept at W4A16
  quality while pure W4A4 lost **38.73%** accuracy. Direct precedent for catching
  "confidently wrong" tokens from a low-precision activation path.
- Latency: 28-40% per-token improvement; memory parity with single W4A16.
- Why it matters to Atlas: this is the closest published analogue — *same weights,
  noisier activation path drafts, cleaner path verifies*. Maps to "FP8 weights
  drafted with FP8 activations, verified with BF16 activations" on shared MoE
  weights. Cheap to A/B because nothing needs retraining.

### 1.2 QuantSpec — INT4-upper draft, INT8 reconstructed verify (Feb 2025)
- Decomposes INT8 KV cache as INT4-upper + INT4-residual. Drafting loads only the
  upper 4 bits; verify loads both halves. 90-94% acceptance up to 128k context.
- Same conceptual primitive ("verify with strictly more bits than draft used") but
  applied to KV cache rather than weights/activations. Code-gen not benchmarked.

### 1.3 ML-SpecQD — MXFP4 (4-bit) drafts BF16 target (Mar 2025)
- MXFP4 weight-only quantization is direct-cast from BF16, so the draft is the
  **same model parameters, 4-bit format**. No training.
- **Code generation acceptance: ~91% on Qwen2.5-Coder-7B** (vs 62% for a 0.5B
  external draft). 2.22-2.72× over BF16 baseline. Best published evidence that
  a quantized-replica draft works *better than a small external draft on code*.
- Caveat: doesn't address tool-arg / structured-output drift directly.

### 1.4 SPEQ (BSFP) — bit-sharing FP draft extracted from FP16 (Oct 2025)
- Encodes target weights in a Bit-Sharing Floating Point format; extracts a 4-bit
  draft from the same parameters. 1.45-2.07× speedup across 15 LLMs. Confirms
  the "shared-weight, downcast-draft" pattern is generalizable beyond MX.

### 1.5 SubSpec — CPU-BF16 + GPU-4-bit, same model (vLLM issue #39427)
- Quantized substitute draft runs on GPU; BF16 master kept on CPU for verify.
  KV cache shared. Claims **9.1× on Qwen2.5-7B, 12.5× on Qwen2.5-32B**, KL
  divergence 0.1176 vs 0.5899 for a traditional external small draft. Active
  vLLM feature request; not merged.

### 1.6 SpecDec × Quant compatibility study (May 2025, 2505.22179)
- Negative finding worth noting: tree-style draft verification on W4 targets
  loses much of the quant memory benefit because the tree-attention pass is
  compute-bound. Suggests **chain (not tree) drafts** are the right shape when
  draft and verify share weights at different precision. Atlas-relevant.

### 1.7 SelfJudge (Oct 2025, 2510.02329)
- Trains a tiny logistic-regression "judge" head on top of target hidden states,
  to relax exact-distribution match into semantic-preservation. Doesn't address
  quantization drift directly, but the *idea* (sub-step verifier, near-zero
  overhead, trained from self-supervision) is structurally what a code-drift
  detector would look like.

**Code-gen-specific takeaway**: ML-SpecQD's 91% acceptance on Qwen2.5-Coder-7B
and QSpec's HumanEval parity are the strongest published evidence that a
same-weight quantized-draft + higher-precision-verify catches drift cheaply.

---

## 2. Production Engines: vLLM / SGLang / TRT-LLM

### 2.1 vLLM
- `--speculative-draft-model-quantization` exists; **can** combine quantized
  target with unquantized draft (and vice versa). No publicized FP8-draft +
  BF16-verify recipe is canonical, but the wiring exists.
- SubSpec (issue #39427) is the only *explicit* "FP8/INT4 draft verifies BF16
  master via shared weights" proposal — **not yet merged**.
- vLLM blog (Dec 2025) Speculators v0.3 explicitly mentions FP8 verifiers as
  valid; quantization of the verifier is supported.

### 2.2 SGLang
- `--speculative-draft-model-quantization unquant` is the canonical flag to
  force BF16 draft against FP8 target. So **the inverse of what we want** is
  natively supported (BF16 draft + FP8 target); the forward direction (FP8
  draft + BF16 verify) is achievable via custom model configs.
- DeepSeek-V3 MTP is wired through EAGLE backend in SGLang. **MTP head stays
  BF16 even when the body is FP8** — this is the production canon.
- No per-position acceptance-rate logging is surfaced by default; needs custom
  instrumentation.

### 2.3 TensorRT-LLM
- Native EAGLE/Medusa/MTP support; MTP head precision is configurable and
  Qwen3-Next ships BF16 MTP head against FP8 body in NVFP4 / FP8 deployments.
- No explicit "FP8-draft + BF16-verify" recipe; closest pattern is keeping the
  MTP head in BF16.

### 2.4 DFlash (z-lab, Dec 2025–May 2026)
- **0.5B BF16 block-diffusion drafter** for Qwen3-Coder-30B-A3B (both BF16 and
  FP8 targets). HumanEval acceptance length **8.09 tokens**, MBPP **7.23**.
  Direct precedent for a tiny BF16 external drafter on code.
- **Critical Atlas-relevant data point** (HF discussion z-lab/Qwen3.6-27B-DFlash
  #2): a DGX-Spark user reported **~0% acceptance at position ≥4 on the FP8
  version**, vs positive acceptance up to position 8-9 on BF16. **FP8 base
  weights compound the draft-verifier mismatch** — confirms our drift symptom
  from a third party.

---

## 3. EAGLE / Medusa / Hydra / variants & quantization noise

- EAGLE-3 (NeurIPS '25, 2503.01840): 3.0-6.5× speedup, fuses early/mid/late
  hidden states. **No explicit quantization-noise handling**. Acceptance
  reportedly within 1% across BF16 vs FP8 targets when MTP/EAGLE head stays
  BF16 (Qwen3 community reports).
- Medusa / Hydra: pre-FP8 era, no quant-aware features.
- **None of EAGLE / Medusa / Hydra were designed to catch "confidently wrong"
  tokens** — they reject by *distribution* match. If both quantized draft and
  quantized target prefer the same wrong token, the draft is accepted. This is
  exactly our failure mode.

---

## 4. MTP (Qwen3-Next / Llama-MTP / DeepSeek-V3)

- DeepSeek-V3 MTP1: ~80% acceptance, ~1.8× decode throughput. MTP module is
  **trained**, head is BF16 in practice.
- Qwen3-Next FP8 ships an MTP head that vendors (Qwen team, Unsloth, AEON-7
  DFlash repo) explicitly keep in BF16: per-position acceptance 87/72/61% for
  draft positions 1/2/3 with FP8 body. **Quality preserved by upcasting the
  prediction head.**
- **None of these mechanisms reject drafts that "look right but are wrong"**.
  They use modified rejection sampling matched to the joint draft+target dist;
  if the target itself is FP8 and prefers a wrong token, MTP confirms it.

**Implication**: stock MTP cannot catch FP8 drift in our setting because the
verify pass *is also FP8*. It would only catch drift if the verify pass were
strictly higher precision than the draft pass (QSpec-style).

---

## 5. DeepSeek-V3 inference engine — tool-calling fidelity

- DeepSeek-V3 uses FP8 mixed-precision training (E4M3 weights, finer-grained
  block scales), pre-conditioning the model to FP8 inference. Tool-call
  fidelity in the wild is high because the FP8 *was the training format*,
  not a post-hoc quantization. Our setting (Qwen3.6-35B FP8 post-quant of a
  BF16 model) is structurally noisier.
- The published inference engine uses standard rejection-sampled MTP — no
  drift-specific guardrails. Their fidelity is **training**-side, not
  decode-time critic.
- Operationally: DeepSeek-V3 emits tool calls inside dedicated channels via
  custom grammars + the FIM-style prompt structure. No mid-token critic.

---

## 6. Feasibility for Atlas — 1-3B BF16 critic verifying 35B FP8?

| Cost model | Estimate |
|---|---|
| 1B BF16 dense forward, 1 token, GB10 | ~3-6 ms (memory-bound, ~2 GB/tok) |
| 3B BF16 dense forward, 1 token, GB10 | ~10-15 ms |
| 35B-A3B FP8 MoE forward, 1 token | ~28-30 ms (current ceiling) |
| Critic overhead per *accepted* token | 10-50% relative wall-clock |

**Verdict**: token-by-token verification by a small external BF16 critic is
feasible (10-50% throughput cost) but the *acceptance criterion* is the hard
part. Pure logit-cosine against a 1B model is too noisy on tool-arg names.

What actually works (per the literature): the **critic must share weights**
with the draft for low-overhead, high-acceptance rejection (QSpec / ML-SpecQD).
Same-weights, BF16 activations as critic of FP8 activations is the right
abstraction — the critic is a 35B BF16 forward pass on a *single suspect
token*, not on every token. Trigger it only on tool-arg positions identified
by Atlas's existing parameter-body state machine.

---

## 7. Ranked Top-5 Atlas-Implementable Patterns

1. **QSpec-style same-weights mixed-activation verify on tool-arg tokens only**
   *(strongest evidence × smallest blast radius)*

   Run the FP8 MoE forward as today. At positions where the grammar /
   `inside_parameter_body` state machine says "we're inside a tool argument",
   re-execute **only the suspect token** through the **same MoE weights with
   BF16 activations** (or "upcast critical layers": gate, expert MLP, final
   logits) and accept the FP8 token iff cosine(FP8_logits, BF16_logits) > τ
   OR top-1 matches. Triggers ~5-10% of tokens, so wall-clock cost is bounded.
   QSpec precedent: 28-40% latency win in their setting; here we'd spend a
   modest 5-15% cost to *gain* correctness, not throughput.

2. **BF16 upcast of the LM head and last 2-4 transformer layers, always**
   *(zero new infrastructure)*

   The Qwen3-Next + DFlash community consistently keeps the **MTP head and
   final layers in BF16** even with FP8 body. The "confidently wrong" token
   bias is dominated by late-layer sensitivity (memory: see
   `project_qwen36_phase2b_softmax_expf.md` — L31-L39 FP8 KV regression). Make
   the last K layers and `lm_head` BF16-only and re-benchmark Phase-2c drift.
   Cost: small extra GPU memory; throughput likely <5% hit.

3. **EARS-style entropy-gated soft verify (no extra model)**
   *(Efficient Adaptive Rejection Sampling, arXiv 2512.13194)*

   Only invoke the critic (whatever form: re-execute, smaller model, or
   BF16-late-layer pass) when target entropy < τ_low AND we're inside a
   parameter body. "Low-entropy + structured context" is exactly the
   failure-mode signature. EARS-style adaptive threshold gives us a tunable
   knob without retraining anything. Pairs well with #1.

4. **DFlash-style 0.5B BF16 critic, but used inverted — as VERIFIER, not
   drafter, scoped to tool-arg positions**
   *(novel composition of two known pieces)*

   Train (or reuse z-lab's existing) 0.5B BF16 Qwen3-Coder DFlash drafter.
   Run it in *parallel* to the FP8 35B target, but use it as a *score-second-
   opinion* critic only inside parameter bodies: if its top-3 disagree with
   the FP8 top-1, force resample. ~3-6 ms per critic step × ~5% of tokens =
   <1% throughput cost. Risk: 0.5B may not be a strong enough critic for
   arbitrary tool args (works for code, less clear for free-form JSON).

5. **SubSpec-style FP8/BF16 shared-weight draft-verify, MoE-only**
   *(highest ceiling, highest complexity)*

   Mirror the SubSpec vLLM proposal (#39427) but in Atlas: same MoE weights
   loaded twice (FP8 + BF16 router/gate/expert subset), draft with FP8, verify
   with BF16 every K tokens or every suspect token. Higher memory cost (need
   BF16 copy of at least router + gate proj + final layers — call it 5-8 GB
   for 35B-A3B), but matches the production precedent that yielded 9-12×
   throughput in SubSpec's reports. Aligns with Atlas's existing
   `WeightQuantFormat` dispatch infrastructure (memory: `project_modelopt_nvfp4_fix.md`).

---

## Notes / sources

Key sources (arXiv IDs and engine docs):
- 2505.22179, 2502.10424, 2503.13565, 2410.11305, 2510.18525, 2510.02329,
  2512.13194 (EARS), 2503.01840 (EAGLE-3)
- vLLM issue #39427 (SubSpec), SGLang speculative_decoding.html, DeepSeek-V3
  technical report (2412.19437)
- HF: z-lab/Qwen3-Coder-30B-A3B-DFlash, z-lab/Qwen3.6-27B-DFlash#2
- Community: Qwen3-Next FP8 + BF16 MTP head reports (Unsloth docs, AEON-7
  DFlash repo)

No evidence found for: dedicated "tool-arg-token" verifier products in
production; FP8-draft + BF16-verify as a publicized vendor recipe; tool-call
fidelity claims tied to a decode-time critic (DeepSeek's fidelity is training-
side).
