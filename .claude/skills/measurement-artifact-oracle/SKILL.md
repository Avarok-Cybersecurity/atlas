---
name: measurement-artifact-oracle
description: "M.A.O. — the MEASUREMENT ARTIFACT ORACLE (Observation Reliability And Claim-Legitimacy Examiner). Adjudicate whether a benchmarked number is REAL or a MEASUREMENT ARTIFACT, before it is believed, quoted, ratcheted into a gate, put on a chart, or shown to anyone. Invoke whenever reading, comparing, or acting on measured values: tok/s, throughput, TTFT, TPOT, ITL, latency, wall time, J/token, tokens per watt, energy, power draw, watts, cost per token, accuracy, BFCL, acceptance rate, jitter, stability, p50/p90/p99. Invoke on any claim of the form faster, slower, cheaper, more efficient, regression, speedup, win, beats, N times better, no change, or inert. Invoke before setting a floor or ceiling from data, before comparing two engines or two commits, and whenever a number is surprising, too good, too flat, or the wrong sign. Takes JSONL of metrics with pointers to the code that measured them plus each metric's intent; returns an ARTIFACT / SOUND / UNDETERMINED verdict per metric with the failure class, the evidence, and the experiment that would settle it."
argument-hint: "<path/to/metrics.jsonl> [--strict]"
---

# M.A.O. — Measurement Artifact Oracle

You are adjudicating numbers that already exist. Not producing them — that is
`measurement-discipline`, which runs *before* measuring. This runs *after*, and
asks one question:

> **Does this number measure the thing it is being used to claim?**

## The standing posture

**The burden of proof is on the number.** Default verdict is `UNDETERMINED`.
`SOUND` is earned by checks that passed, and you must name them. Never certify
`SOUND` because you found nothing wrong — absence of evidence is
`UNDETERMINED`, and saying so is a *useful* answer, not a failure.

A wrong `SOUND` is the expensive outcome. It sends people optimizing against a
phantom, ratchets a gate onto noise, or puts a false claim in front of someone
who will repeat it.

## Input

JSONL, one object per metric. Minimum viable object:

```json
{
  "metric": "gpu_rail_j_per_token",
  "value": 3.653,
  "unit": "J/token",
  "intent": "prove Atlas costs less energy per token than vLLM",
  "measured_by": [{"path": "bench/ladder38/harness.py", "lines": "436-448"}],
  "window": {"declared": "the rep", "code_ref": "run_rep t0..t1"},
  "provenance": {"box": "dgx2", "date": "2026-08-18", "commit": "...",
                 "instrument": {"isl": 128, "osl": 1024}},
  "comparison": {"against": "vllm_mtp", "its_provenance": {}},
  "population": {"n": 3, "samples": [65.4, 52.2, 46.3]}
}
```

Fields may be missing. **A missing field is itself a finding** — record it as
`unverifiable: <field>` rather than assuming a default. `intent` and
`measured_by` are the two that matter; without them, say so and stop.

## Method

1. **Read the code that measured it.** Not the docs, not the field name, not the
   comment. Open `measured_by` and follow it to the sensor. *Comments lie*: a
   ladder harness once carried `# SM clock sampled INSIDE the rep window, not
   before it` directly above a synchronous `os.popen(...).read()` that completed
   **before** the rep began. The field name (`..._at_rep_start`) told the truth;
   the comment asserted its opposite.
2. **Restate the intent as a falsifiable claim.** "Prove we are cheaper" becomes
   "J/token(A) < J/token(B), same rail, same window, same box". If the intent
   cannot be made falsifiable, that is `UNDETERMINED` class **F2**, and it is
   the finding.
3. **Walk the taxonomy.** Every class has a mechanical detection test. Run what
   you can; list what you could not, and why.
4. **Emit a verdict per metric**, each with the experiment that would settle it.

## The taxonomy

Each entry: *what it is* → **how to detect it**.

### A — Window: does the measurement cover what it claims?

- **A1 Sample outside the window.** Reading taken before/after the interval it
  is attached to → **Locate the sample relative to the timed region in the
  source. A blocking read before the work starts is A1.** A trailing-average
  sensor sampled this way returns the *previous* interval.
- **A2 Warmup contamination.** → **Check whether warmup runs immediately before
  the first measured sample.**
- **A3 Edge dominance.** A sensor averaging over T, sampled across a measurement
  of duration D → **compute T/D. Above ~10%, the number is mostly edges.**
- **A4 Coverage.** → **`covered_s / window_s`. Below 0.9 the integral belongs to
  a different interval than the one named.**

### B — Instrument: is the sensor measuring the quantity?

- **B1 Load invariance.** ★ *Highest-yield test here.* A reading that barely
  moves while the workload moves by a large factor is not measuring the workload
  → **Correlate metric against work (throughput, batch, C). If physics demands
  correlation and |r| ≈ 0, it is B1.** Seen: power flat at 37–43 W across a
  **35× throughput range**.
- **B2 Inverse response.** Reading moves *opposite* to work → **same
  correlation, negative sign. Usually a cap or clamp is binding and the sensor
  reports the limiter, not the load.** Seen: power *falling* 80 W → 59 W while
  throughput rose 16×.
- **B3 Repeatability vs effect size.** → **Compare rep-to-rep spread with the
  claimed effect. If scatter ≳ effect, there is no measured effect.** Seen:
  65.4 / 52.2 / 46.3 W at one rung — ~30% scatter against a claimed 2× gap.
- **B4 Scope/rail coverage.** → **Ask what fraction of the system the sensor
  sees, and whether the omitted part differs between subjects.** A GPU rail
  excluding CPU and DRAM is ~half of system power, and the omitted half differs
  between a Python stack and a Rust one — biasing the *ratio*, not just the
  absolute.
- **B5 Resolution.** → **Sensor stated accuracy vs effect size.** ±5 W cannot
  resolve a 3 W difference.

### C — Comparison: are both sides the same experiment?

- **C1 Asymmetric instrument.** The same procedure captures different phases for
  different subjects → **Ask what each subject is doing at sample time.** Seen:
  one engine ends every request on one step, the other drains — an end-of-batch
  sample measured different things for each. *An asymmetry of the instrument,
  not of the workload.*
- **C2 Instrument mismatch.** → **Diff the instrument axes (ISL/OSL/context/
  dtype/batch). Any difference voids it.** Two numbers for "the same" metric
  once differed ~4× purely by instrument.
- **C3 Provenance mismatch.** → **Diff box, day, driver, thermal regime.
  Same-box, same-hour, interleaved A/B/A is the standard.** Arms 10 h apart
  inside a 16 h thermal ramp are not a matched regime.
- **C4 Absent on one side.** → **A field present for one subject and absent for
  the other is a DIFFERENCE, never a match.**
- **C5 Denominator drift.** → **Recompute with all attempted items in the
  denominator.** A score of 87.24 became 85.53 when 20 dropped samples returned.
- **C6 Definitional divergence.** Same name, different formula → **compare the
  arithmetic, not the label.** `/N` vs `/(N−1)`; numerator ending at last token
  vs last chunk.

### D — Population: is the summary statistic meaningful?

- **D1 Multimodality.** → **Sort or plot the raw samples before trusting any
  median. A central statistic over a bimodal draw is a coin flip.** Seen: one
  rung bimodal, another trimodal — both correctly refused for gating.
- **D2 Fake concurrency.** → **Compare `ttft_p99` to `e2e_p50`. If
  `ttft_p99 > 0.5 × e2e_p50`, later requests waited for earlier ones: sequential
  waves, not C concurrent streams.**
- **D3 Percentile of a rate.** → **Percentile the ratio, then invert:
  `1/p(x)`, never `p(1/x)`.** A p90 taken directly on a rate reports the *fast*
  tail — it flatters.
- **D4 Survivorship.** → **Check errors, timeouts and truncations are counted.**
- **D5 Draw dependence.** → **Require the ordered-sample hash / draw id beside
  any score.**

### E — Null results: did the measurement happen at all?

- **E1 Empty treated as a value.** → **Check the sample count. Zero samples must
  surface as a status, never as 0.0.** A sampler once produced zero rows because
  a CLI flag was rejected; a zero joule count reads as *free*.
- **E2 A green control.** → **Remove the cause and confirm the check goes red. A
  control that stays green proves the check measures nothing.** A control that
  fails by *not compiling* is not a control.
- **E3 Stale or shallow source.** → **Verify the ref.** A stale checkout answers
  "feature absent"; a shallow clone answers a diff wrongly, with no warning.
- **E4 Date/zone blindness.** → **Print full timestamps with date and zone.** A
  three-day-old log once read as live because the format omitted the date.
- **E5 Absent rendered as zero.** → **Distinguish missing from zero everywhere**
  — in records and in charts.

### F — Intent: does the number support the claim it is cited for?

- **F1 Intent/instrument mismatch.** → **Restate the intent, then ask whether
  this instrument can move if and only if the claimed thing changes.**
- **F2 Unfalsifiable claim.** → **Name the observation that would refute it. If
  none exists, the number cannot support it.**
- **F3 Moving denominator.** → **Hold it fixed, or report both terms.**

## Output

JSONL, one object per input metric:

```json
{
  "metric": "gpu_rail_j_per_token",
  "verdict": "ARTIFACT",
  "classes": ["A1", "B1", "B3", "C1"],
  "evidence": [
    {"class": "A1", "at": "harness.py:439-443",
     "finding": "os.popen().read() completes before await run_rep; value is the previous batch's 1s tail"},
    {"class": "B1", "finding": "comparison arm flat 37-43 W across a 35x throughput range"}
  ],
  "checks_run": ["A1","A2","A3","A4","B1","B2","B3","B4","C1","C2","C3","D1","D2"],
  "checks_not_run": [{"class": "B5", "why": "sensor accuracy not stated in input"}],
  "direction_of_bias": "unknown; could reverse the sign of the claim",
  "what_would_settle_it": "in-window integral, both engines, interleaved A/B/A, same box, same hour, >=100 samples per window",
  "may_be_quoted": false
}
```

Verdict rules:

- **`ARTIFACT`** — at least one class fires, with evidence. Say which.
- **`SOUND`** — the checks you ran passed, the intent is falsifiable, and both
  sides match on instrument and provenance. List `checks_run`.
- **`UNDETERMINED`** — you could not run the checks that matter. A legitimate,
  final answer. Name what is missing.
- **`may_be_quoted`** is `false` for anything not `SOUND`.
- **Every verdict carries `what_would_settle_it`** — a concrete experiment. A
  verdict without one is incomplete.

## Rules of engagement

- **Never repair the number.** Adjudicate only; fixing the instrument is
  separate work with its own review.
- **One artifact is enough to stop**, but list the others you saw.
- **Report the direction of the bias** when knowable. "Understates us" and
  "overstates us" are different findings, and the sign is often knowable when
  the magnitude is not.
- **Say when the conclusion could reverse.** If an artifact could flip the sign
  of the claim, that is the headline, not a footnote.
- **Do not grade on intent.** A careful process can still yield an artifact; a
  sloppy one can still yield a sound number.
- **`--strict`**: treat every `UNDETERMINED` as blocking.

## Worked verdict

*Claim:* "Atlas costs ~2× vLLM per token." *Intent:* prove we are cheaper.

`ARTIFACT`, classes **A1 · B1 · B2 · B3 · C1 · C3**.

A1 — the sample was taken before its own window. B1 — the comparison arm read
flat across a 35× throughput range. B2 — our arm read *inversely* to work. B3 —
rep-to-rep scatter ~30% against a claimed 2× effect. C1 — the sample caught
different batch phases per engine. C3 — arms ran 10 h apart in a thermal ramp.

**Direction:** unknown, and the sign could reverse — if both engines sit at the
same power cap under load, J/token collapses to `cap ÷ throughput`, and a
1.01–1.33× throughput lead becomes an energy *lead*. B4 compounds it: the
readable rail excludes the CPU side, where a Rust engine should beat a
multi-process Python stack.

**What would settle it:** in-window integral, both engines, interleaved A/B/A,
one box, one session, ≥100 samples per window, with the rail named.

*The lesson worth keeping: every one of those six classes was visible from the
committed data and the harness source. Nobody had to run a benchmark to find
them — only to read the code that produced the number.*
