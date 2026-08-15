---
name: measurement-discipline
description: Enforce the measurement discipline for any performance claim (tok/s, TTFT, TPOT, wall, accuracy). Invoke BEFORE measuring, comparing, or quoting a perf number, and before writing one into a commit message, PR comment, BENCH.toml, or report. Born from the 2026-08-15 decode-rate flip-flop (29→34→13 asserted in sequence while 22+ was correct); each rule traces to a root cause of that incident or a cited paper. Also invoke when tempted to call a prior number an artifact, declare "no regression", or declare a feature inert.
---

# /measurement-discipline — rules for performance claims

You are about to measure, compare, or publish a performance number. Work through
the checklist for the phase you are in. The rules exist because on 2026-08-15 a
session confidently asserted, in sequence: +15% real → +15% refuted → "no
regression; history was artifact" → "MTP inert on these weights" — every one
wrong, while an external engineer's controlled A/B read the truth (22+ tok/s).
The mechanism (thinking-gated speculative dispatch) was only found after four
public corrections. These rules would have blocked every wrong claim at the door.

## Phase 1 — before measuring

1. **Fingerprint the run (Rule 1: fingerprint or it didn't happen).** Record,
   next to the number, ALL of: build commit · serve binary + full flags
   (max-seq-len, batch, gpu-util, kv/lm-head dtypes, `--speculative` + K,
   scheduler policy, prefix-cache/ssm flags) · checkpoint id + revision ·
   harness + its params (ISL/OSL, prompt content class, temp/seed, thinking
   on/off/effort) · port · box · date. A number without its fingerprint may not
   be quoted later. If the harness cannot record this, write a sidecar file.
2. **Know what engages on this path.** On this engine, speculative decode is
   hard-gated OFF inside `<think>` (`scheduler/mod.rs` `!inside_thinking`), so a
   thinking-on request with a small OSL measures the SERIAL floor (~74 ms/step
   on dense-27B/GB10), not the engine. Decide thinking on/off *deliberately*:
   - decode headline → `reasoning_effort:"none"` (NOT
     `chat_template_kwargs.enable_thinking:false` — degenerates on qwen3_6_moe),
     `max_tokens ≥ 1024`, code-like prompt; expect MTP `mean_na ≈ 2–2.5`.
   - serial floor → thinking on, short OSL; label it as the serial floor.
3. **Prompt class matters.** Counting/enumeration prompts accept drafts at
   near-ceiling (~3.8 tok/step) and EOS early — they inflate tok/s. Natural/code
   text accepts ~2–2.5. Name the prompt class in the fingerprint.
4. **Plan n≥3 repeats** for anything that will headline. A single run may be
   reported only as "preliminary, single-shot".

## Phase 2 — before comparing two numbers

5. **One-variable rule (Rule 2).** A comparison is valid only if the two
   fingerprints differ in exactly the variable under test. If harness AND serve
   profile AND checkpoint changed, you have two *observations*, not an A/B.
   Say so explicitly.
6. **Verify engagement, don't assume it.** For speculative-decode comparisons,
   serve with `ATLAS_MTP_ACCEPT_DEBUG=1` and check the `MTP accept` lines
   (`mean_na`, `tok_step`). The 2-minute engagement test:
   one MinHeap-style code prompt, `temperature 0.0, max_tokens 1500,
   reasoning_effort:"none"` → expect `tok_step≈3`; the same request without
   `reasoning_effort` shows no accept lines until post-think. A feature that
   never engaged measures as inert (Rule 4: negative results need a positive
   control that provably exercises the feature).

## Phase 3 — before declaring "artifact", "no regression", or "impossible"

7. **Two-harness rule (Rule 3).** To reclassify a prior number as a measurement
   artifact you must BOTH (a) reproduce the artifact mechanism — make the
   suspect harness emit the wrong number and name the broken FIELD — and
   (b) confirm your competing number by an independent method (raw streamed-token
   timing via curl, nsys cadence, server-attested usage). Discredit fields, not
   whole records: one broken E2E column does not impeach the TPOT column beside
   it.
8. **Roofline arguments are hypotheses (Rule 7).** Any "physically impossible"
   claim must show its arithmetic inline (bytes/token × bandwidth, vs the
   documented roofline — dense-27B NVFP4 on GB10 ≈ 13.5 GB/step ÷ 273 GB/s ≈
   49.5 ms serial floor ⇒ ~20 tok/s serial, ~58+ with speculation) and STILL
   needs rule-7 empirical confirmation before it can dismiss data. If the
   arithmetic falls, every conclusion it supported re-opens automatically.
9. **Price the coincidence.** If your story requires many independent prior
   records to all be wrong in the same direction, that is evidence against the
   story, not for it.

## Phase 4 — before publishing (commit message, PR comment, BENCH.toml, report)

10. **Graduated claim language (Rule 5).** Single-harness/single-config ⇒
    "observed X under fingerprint F". Cross-validated per rule 7 ⇒ plain "X".
    Only cross-validated claims may use the words *real, artifact, no
    regression, impossible, inert*. "Not yet determined" is a first-class
    deliverable — prefer it over a confident guess.
11. **Commit-message gate (Rule 6).** No perf number enters a commit message,
    PR comment, or doc headline unless it satisfies rules 1, 5, and 7 and cites
    its run-record IDs (`~/.atlas/runs/...`). Otherwise label it
    "preliminary, single-harness". The retraction cost is paid at claim time.
12. **Ledger check (Rule 8).** Before asserting, grep prior session records and
    committed BENCH.toml notes for contradicting numbers. Contradicting a prior
    verified claim requires naming the MECHANISM of the discrepancy (as a
    supersede note), never silently overriding it. When an adversarial reviewer
    checks your claim, hand them raw run records, not your prose — a reviewer
    that reads only the narrative rubber-stamps it.

## The canonical worked example (why each rule exists)

One step cadence (~74 ms/step) explains every number this repo ever recorded on
the dense-27B; only MTP engagement varied:

| Reading | tok/s | What it actually was |
|---|---:|---|
| Historical sweeps 25.3–30.3 | real | 49-token counting bursts, near-ceiling accept, wall incl. TTFT |
| quick_bench 29.4–34.0 | real | same bursts, decode-only server rate |
| "Regression" 11.8–14.3 | real | thinking-on + osl 128 ⇒ output ~all inside `<think>` ⇒ serial floor |
| Healthy long-form 22.2–31.6 | real | thinking off/amortized, accept ~2–2.5/3 |

Every wrong intermediate claim ("+15% real", "history was artifact", "MTP
inert") violated rules 5–9. The correct output at each juncture was: "two
fingerprints disagree 4×; next action is a one-variable bisection."

## Citations

MLPerf Inference arXiv:1911.02549 · Curie arXiv:2502.16069 · CoVe
arXiv:2309.11495 · Self-correction limits arXiv:2310.01798 · Verbalized
overconfidence arXiv:2306.13063, arXiv:2305.14975 · R-Tuning arXiv:2311.09677 ·
Debate grounding arXiv:2402.06782 · Debate sycophancy arXiv:2509.23055,
arXiv:2509.05396 · Provenance arXiv:2606.04990 · Process supervision
arXiv:2305.20050 · CORE-Bench arXiv:2409.11363.
