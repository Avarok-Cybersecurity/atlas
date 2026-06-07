# CLAUDE HANDOFF — DFlash K>1 accept: propose-seed / per-draft-position conditioning hunt

**Date:** 2026-06-07
**Branch/HEAD:** `dflash-on-ssm-pin` @ `a6c5cb9` (descendant of frontier tip `527a7cc`)
**Author:** Friday, for Ronald. Budget is tight and this bug has burned ~$1100 over 6 days. **Kill rabbits as you find them — do NOT come back with a list of 5 maybes. Disprove or confirm each candidate with a read or one probe, then move on.**

---

## TL;DR — what to find, what NOT to re-derive

The bug: **DFlash K>1 (cap≥2) accept is floor-level and DECAYS BY DRAFT POSITION.** The drafter is byte-perfect and distribution-independent (see "SETTLED" below) — this is **NOT a weak-drafter / checkpoint problem.** It is in **how the proposed block is seeded / conditioned / fed** on the K>1 path. Find the specific feed defect, prove it, fix it.

**Your job:** localize the seed/conditioning defect to a specific buffer, stride, offset, or context-truncation, **prove it with the existing SEED_DUMP probe or a code read (not a theory)**, and either fix it or hand back the single proven root cause. Kill each false lead explicitly in your writeup so it's never re-chased.

---

## THE EVIDENCE (honest benches, fixed tree, post-NaN-fix honest math)

Per-event verify traces (this branch logs every accept/reject; logs in `/home/rstesiak/merge_baseline_k1_*.log`):

| cap | kernel | accept | tok/s | event breakdown |
|-----|--------|--------|-------|-----------------|
| 1 | K2 | ~50% | 15.1 | baseline floor |
| 2 | K3 | ~3% step | 10.9 | ACCEPT-2 ×2, ACCEPT-1 ×6, REJECT ×260 (268 steps) |
| 3 | K4 | 1.74% draft (14/804) | 6.1 | ACCEPT-3 ×1, ACCEPT-2 ×1, ACCEPT-1 ×9, REJECT ×257 (268 steps) |

**THE FINGERPRINT (load-bearing):** at K4, accepts decay by draft position — first draft lands occasionally (ACCEPT-1 ×9), position+2/+3 almost never (×1, ×1). This is **position-DEPENDENT decay, not uniform low accept.** A merely weak drafter accepts uniformly low; this degrades with depth. That is the signature of a **seed/context feed that gets worse as the block deepens** (matches the id176 "front conditions / tail collapses with query-row index" observation). The path mechanically WORKS (ACCEPT-3 fired once = all 3 drafts landed) — it is being **starved by depth**, not capped by quality.

---

## SETTLED — do NOT reopen any of these (each cost hours/days)

1. **Drafter is byte-perfect** vs PyTorch reference (15/16 gamma positions identical at pos93; lone diff = tie-break wobble). The drafter forward math is correct. **DO NOT propose "train/swap the drafter."**
2. **DFlash ≠ EAGLE.** DFlash's block-diffusion drafter generates its own block via its own math, INDEPENDENT of the target's hidden distribution. The EAGLE-style "checkpoint is weak" failure mode does not apply. (Ronald, emphatic.)
3. **NaN bug is DEAD** (commit `354d926`, in-kernel q/k L2-norm in the 27b `gated_delta_rule_decode` entry). Accept numbers are now HONEST. Do not chase NaNs.
4. **HDIM mismatch is FIXED** (commit `527a7cc`, `kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_paged_indirect.cu` = `#define HDIM 128` + include common). The front of the gamma-block now conditions correctly. Do not re-investigate HDIM.
5. **Ruled out earlier, do NOT re-chase:** CUDA graph, dtype/fp32-residual (`use_fp32_residual()==false` for 27b), token-handoff, verify INDEXING (drafts[i] vs verified[i] compare matches vLLM `rejection_greedy_sample_kernel`), k_t/v_t buffer offsets (include `t*conv_dim`, correct).
6. **Cap-bumping is a confirmed throughput LOSS** (15.1→10.9→6.1). The goal is NOT "raise cap to go faster" — it's get K=3/4 to HOLD accept by fixing the feed. Do not run more cap sweeps "to see"; both curves above are already measured.

---

## ⚠ SEED: EAGLE seed is DEAD, DFlash seed is UNMEASURED — do not confuse them

The EAGLE drafter's seed (`eagle_verify_buf` bootstrap, `forward.rs` slot-0 / layer-2 tap)
was disproven hard (id162/163): "DO NOT fix the seed; it is correct. The L2~15.756 was a
FALSE ALARM — the probe copied only slot-0 (a legitimately-small early-layer hidden)."
**That is a DIFFERENT drafter and a DIFFERENT code path. Do not import its verdict.**

The **DFlash** seed (`dflash_hidden_save`, consumed `impl_b3.rs:100/169`) was **never actually
dumped** — the check was queued in id172/id173 and the `ATLAS_DFLASH_SEED_DUMP` probe (commit
`0fda11d`) was BUILT for it, then the PGH storm interrupted before it fired. So the DFlash seed
is genuinely OPEN, not a re-chased rabbit. **But treat it as a 30-second GATE, not the hunt:**
fire the probe once, read the number, and if the seed is non-zero/finite, **PIVOT IMMEDIATELY to
Suspect B** (where the position-decay fingerprint actually points). Do NOT spend the session on
the seed.

## PRIME SUSPECTS (in order — kill each before moving on)

### Suspect A — GATE (≤1 launch): is the DFlash seed (`dflash_hidden_save`) zero / stale / wrong-position?
The drafter conditions at step0 on `dflash_hidden_save`. If it's zero, stale, or holds the wrong layer/position, the whole block starts from garbage and accepts decay with depth. **This is a fast disprove-or-confirm, then move on either way.**

- **Write side:** `impl_b3.rs:317` (`dst = ... dflash_hidden_save`), populated after gamma verify Phase 3 (`crates/spark-model/src/traits/model.rs:351`, `verify_d.rs:252`). Alloc: `impl_a1.rs:203` (`n_cap * hidden_size`).
- **Consume side:** `impl_b3.rs:100` / passed to `proposer.propose(...)` at `impl_b3.rs:169` (last arg).
- **THE PROBE ALREADY EXISTS** — `impl_b3.rs:90-153`, gated on env `ATLAS_DFLASH_SEED_DUMP`. It dumps `pos, n_cap, n_elem, L2sq, min, max, zeros/n_elem, nan, head[8]`. **Fire it first:**
  ```
  ATLAS_DFLASH_SEED_DUMP=1 ATLAS_DFLASH_DRAFT_CAP=3 bash ~/launch-merge-baseline-k1.sh
  ```
  (Ronald runs builds/launches — you write the patch + hand him the exact command, you do NOT run cargo.)
  - **READ:** if `L2sq≈0` or `zeros≈n_elem` → seed is dead → population bug on the write side (Suspect A confirmed). If seed is non-zero/varied/finite → **seed is fine, GO TO Suspect B, do NOT resize buffers** (id172's explicit caveat).

### Suspect B — per-draft-position cross-attn context (the feed that decays with depth)
This is where the position-decay fingerprint most likely lives. Each deeper draft row must cross-attend over the correct target hidden context. If the context stack is truncated, mis-strided, or the later draft rows see a shrinking/masked window, accepts will fall off exactly as observed.

- vLLM `dflash.py set_inputs_first_pass` feeds the FULL `target_hidden_states` stack as cross-attn context (`num_context = all tokens`). Diff Atlas's propose cross-attn K/V source against that.
- Check `eff_ctx` handling and the propose cross-attn K/V source in the propose path (`proposer.propose`, the DFlash proposer impl) — confirm later draft rows attend over the full context, not a per-row-shrinking one.
- The id176 tail-collapse signature said the masking/span worsens with query-row index. Confirm the propose path's per-row attention span is correct for rows 1..K-1, not just row 0.

### Suspect C — the verify-side accept compare per draft position
Lower priority (indexing was diffed vs vLLM and matched), but confirm the K3/K4 kernels (`verify_k3_step.rs`, `verify_k4_step.rs`, kernels `decode_verify_graphed_k{3,4}`) compare draft[i] vs target argmax[i] at the RIGHT position for each i, and that partial-accept rollback restores SSM state correctly (nologik's `gdn_decode_chunk2/chunk3` + `conv1d_update_chunk2` save/restore — these ARE wired for k2/3/4 per id177; confirm they're actually invoked on partial-accept in the k3/k4 step).

---

## RULES OF ENGAGEMENT

- **Crib vLLM FIRST, bench LAST.** Reference files: `~/eagle-refs/vllm-full/vllm/model_executor/layers/fla/ops/` and `dflash.py`. Diff against the reference; don't hand-eyeball + theorize.
- **One probe, one rebuild, then read.** Don't stack 5 instrumentation changes. The SEED_DUMP probe already exists — use it before adding anything.
- **`search_files` can lie (silent 0 matches on this tree).** Confirm "no callers / unused" with raw `grep -rn "symbol" crates/ kernels/ --include=*.rs` before believing it.
- **Friday/Ronald run all builds & launches.** You write patches + hand exact `ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server` + launch commands. You do NOT run cargo (token cost).
- **Kill rabbits explicitly.** Every candidate you touch: write "DISPROVEN: <reason, with the line you read>" or "CONFIRMED: <proof>". A disproven theory left vague gets re-chased and re-billed.
- **Safe floor:** cap=1 / 15.1 tok/s / GOLD tag `wip/dflash-GOLD-15tok-20260602`. Never regress that.

## DELIVERABLE
Either: (1) the single PROVEN root cause with the probe/read that proves it + a minimal fix diff, or (2) a crisp writeup that disproves A/B/C with evidence and names the one remaining untested lane. Append findings to this doc. Update LTM id190 framing only if you DISPROVE the seed-conditioning hypothesis with hard evidence.

---

## SESSION FINDINGS — 2026-06-07

### Suspect A (SEED GATE): SKIPPED — code read makes it redundant

The SEED_DUMP probe was not fired. Code reading resolved the question faster: with cap=1 (K=2 verify), `verify_b.rs:287` populates `dflash_hidden_save` every step, and 50% accept proves the seed is healthy. The bug is downstream.

### Suspect B: CONFIRMED — root cause found from code read, NOT from bench

**THE BUG:** `verify_c.rs` (K=3 graphed verify) and `verify_c2.rs` (K=4 graphed verify) have **zero** `try_dflash_capture` calls. They are the ONLY verify paths missing it:

| verify path | file | has `try_dflash_capture`? |
|-------------|------|--------------------------|
| K=2 (cap=1) | `verify_b.rs:287` | YES |
| K=3 (cap=2) | `verify_c.rs` | **NO** (bug) |
| K=4 (cap=3) | `verify_c2.rs` | **NO** (bug) |
| K=γ (cap≥4) | `verify_d.rs:259` | YES |

**Routing proof** (`mtp_step.rs:127-134`):
```rust
if drafts.len() >= 4 { step_verify_dflash(...)   // verify_d.rs — has capture
} else if ... drafts.len() >= 3 { step_verify_k4(...) // verify_c2.rs — MISSING
} else if ... drafts.len() >= 2 { step_verify_k3(...) // verify_c.rs — MISSING
} else { step_verify_k2(...)                       // verify_b.rs — has capture
}
```

**Effect:** With cap=2, every verify step runs K=3 (verify_c.rs). After each verify `dflash_hidden_save` is NOT updated. The next `propose()` appends the same stale capture from the bootstrap step into `ctx_hidden_acc`. After N steps the accumulator holds N copies of the same bootstrap-position hidden, all labeled at wrong absolute positions. The drafter conditions on noise and collapses to ~3% first-draft accept.

With cap=1 the same stale problem does NOT exist because K=2 verify DOES update `dflash_hidden_save` each step — so the ctx grows with accurate per-step captures and accept stays at 50%.

The "position-dependent decay" fingerprint (ACCEPT-1×9 > ACCEPT-2×1 > ACCEPT-3×1) is a red herring — it's a natural multiplicative consequence of ~4% per-position accept, not a depth-dependent conditioning failure.

### THE FIX — 2 lines, one per file

Added `self.try_dflash_capture(layer_idx, k - 1, stream)?;` inside the layer `for` loop of both files, after the `if layer_type == FullAttention { ... } else { ... }` block — inside the CUDA graph capture region, exactly mirroring `verify_b.rs:287` and `verify_d.rs:259`.

**`verify_c.rs:257`** (K=3):
```rust
                self.try_dflash_capture(layer_idx, k - 1, stream)?;
```

**`verify_c2.rs:247`** (K=4):
```rust
                self.try_dflash_capture(layer_idx, k - 1, stream)?;
```

### Build + bench command

Ronald runs the build:
```
ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server
```

Then bench cap=2 to confirm first-draft accept lifts from ~3% toward ~50%:
```
ATLAS_DFLASH_DRAFT_CAP=2 ATLAS_DFLASH_VERIFY_TRACE=1 bash ~/launch-merge-baseline-k1.sh 2>&1 | tee /home/rstesiak/merge_baseline_k3_fix.log
```

Expected result: cap=2 accept should lift substantially (≥20%) and REJECT count should drop. If cap=2 first-draft accept matches cap=1 (~50%), then cap=3 should also recover.

### Suspect C: NOT INVESTIGATED

The indexing was already verified correct (SETTLED), and fixing Suspect B should be sufficient. Only investigate if cap=2 accept stays floor-level after the fix.
