# CLAUDE HANDOFF — DFlash Option-B paged-γ propose: drafter emits degenerate drafts (~0% accept)

**Date:** 2026-06-06
**Author:** Friday (architect) → Claude Code (implementer)
**Branch/tree:** `~/code/atlas`, branch `dflash-on-ssm`, HEAD `0fda11d`
**Baseline safe tag:** `wip/dflash-l2norm-nan-fixed-20260606`; probe tag `wip/dflash-seed-dump-probe-20260606` (both committed + mirrored to bkpflash, atlas-may22)

### WORKING-TREE STATE — read before `git status`/`git clean` (DO NOT discard)
HEAD `0fda11d` is clean for the L2-norm NaN fix and the SEED_DUMP probe (both
committed). The following files are **intentionally modified and uncommitted** —
they are real work in progress, NOT junk. Do not `git checkout`/`git clean` them:
- `crates/spark-model/src/model/trait_impl/verify_d.rs` — per-layer hidden capture (Claude-prior, necessary-not-sufficient, keep)
- `crates/spark-server/src/scheduler/verify_dflash_step.rs` — verify trace/capture (keep)
- `crates/spark-model/src/layers/ops/ssm_gdn_b.rs` + `.../qwen3_ssm/trait_decode_batched_conv_gdn.rs` — `ATLAS_GDN_DUMP` instrumentation (orphaned from an older NaN theory; harmless, env-gated, leave parked)
If you need a clean tree, `git stash` — do NOT delete. Confirm with Ronald first.

### Reusable probe template (already committed at `0fda11d`)
The SEED_DUMP block in `crates/spark-model/src/model/impl_b3.rs` (just before the
`propose()` call, ~line 94, gated `ATLAS_DFLASH_SEED_DUMP`) is the **copy-paste
template** for the `q_buf`/`attn_out` probes you'll need in Suspect 1. It shows
the bf16→f32 readback + L2sq/min/max/zeros/nan/head log pattern. Reuse its shape.

---

## TL;DR for the implementer

The DFlash **batched γ-block proposer** running under `ATLAS_DFLASH_OPTION_B=1`
produces **degenerate draft tokens** — it cannot even reproduce the trivial
row-0 echo (input `last_token` → output should be `last_token`). Accept is
~0–1/15 (0.12% aggregate). The bug is **localized to the paged-γ FORWARD path**
(`forward_block_layer_paged.rs` + its attention call), NOT the verify kernel,
NOT context, NOT the seed, NOT the drafter weights.

Your job is **mechanical bisection against a known-good reference path**, not
re-derivation. A large list of hypotheses is **already disproven** (below) —
re-deriving any of them is the single most expensive recurring waste on this
project (two prior sessions burned ~100k tokens each in the RoPE/position/slot
lane). **Do not re-open the DEAD list. Bisect.**

---

## The symptom (live data, 2026-06-06)

Launch: `bash ~/launch-dflash-gamma-nograph.sh` (eager, OPTION_B=1, VERIFY_TRACE=1)

```
TRACE drafts: token_in=271 position=285 γ=16 drafts_pre_cap=[15, 760, 220, 220, 16, 16, 15, ...]
TRACE drafts: token_in=2218 position=286 γ=16 drafts_pre_cap=[11, 760, 11, 279, 279, ...]
K=γ verify: γ=15 accepted=0/15 (0%)   ← 164/167 steps are 0/15, 3 are 1/15
```

The drafts are the **unconditional frequency prior** (15, 279, 13, 16, 220, 264
= whitespace/punctuation/"the"-class token IDs). The drafter is drafting blind.

**Key diagnostic fact:** row-0 of `drafts_pre_cap` should be the trivial echo of
`token_in` (this is a BLOCK-DIFFUSION drafter — row 0's input is `last_token`
and it denoises trivially back to itself; that's why `propose.rs:433` drops row
0). Under Option-B it **never echoes**. token_in=271→draft[0]=15. The drafter
cannot reproduce its own input token.

---

## What is PROVEN (do NOT re-verify — these are settled)

1. **Seed buffer is healthy.** `ATLAS_DFLASH_SEED_DUMP=1` (probe at
   `impl_b3.rs` just before `propose()`) dumped `dflash_hidden_save` across
   positions: L2²=43k–178k, varied real floats, 9–22 zeros / 25600, **0 NaN**,
   head values change per position. The hidden the drafter conditions on is
   live and position-appropriate. **The drafter is NOT starved of a seed.**

2. **Context is NOT the issue.** Ran `ATLAS_DFLASH_OPTION_B_NO_CTX=1` (zeros the
   paged-cache context the layer body reads): drafts still degenerate, row-0
   still fails to echo. So the bug is in the **γ-block self-attention forward**,
   present even with zero context.

3. **Position-ids contract is CORRECT under Option B.** Verified by reading
   `forward_block.rs:285-289`: with `eff_ctx=0` (forced by Option B at
   `forward_block.rs:75`), `pos_host = [position, position+1, ..., position+γ-1]`
   — exactly the γ noise positions `forward_block_layer_paged.rs:240-241` RoPEs
   against. **RoPE position math is NOT the bug.** (This was the prime historical
   suspect — id117 hyp#1 — now disproven by static read. Do not re-open it.)

4. **Precompute math is bit-correct.** All 5 precompute stages pass cosine
   ≥0.9999 vs PyTorch reference (LTM id116): fc_proj, fc_proj_normed,
   fused_kv_out, layer0_k_post_rope, layer0_v.

5. **vLLM layout already line-by-line diffed.** `qwen3_dflash.py` was compared
   against Atlas's flow (LTM id118): math layout, RoPE convention (NeoX), fused
   KV layout (K-then-V interleaved) — all match. **Re-diffing vLLM is redundant.**

6. **NaN bug is dead** (separate, already fixed): in-kernel q/k L2-norm in the
   27b decode kernel. Verified |k|²=1.0, L2 finite. Committed at `354d926`.

7. **The drafter itself is sound.** Non-batched K2/K4 verify delivers >50%
   accept + 15 tok/s on this exact target+drafter (LTM id139), and the drafter
   forward was validated bit-for-bit (15/16 γ positions) vs a canonical PyTorch
   reference. The defect is SPECIFIC to the batched/paged-γ path.

8. **Verify kernel is NOT the first problem.** The drafts are already degenerate
   in the TRACE *before* verify runs. The wy16/wy17 batched-verify kernel is
   downstream of the bad drafts. Fix the propose first; verify is a separate
   later concern.

---

## DEAD hypotheses (disproven across LTM ids 116-139, 172, 173 — do NOT re-derive)

- ❌ Seed buffer zero/stale (disproven 2026-06-06, SEED_DUMP healthy)
- ❌ Drafter starved of full context stack / needs buffer resize (disproven —
   NO_CTX ablation didn't break echo; healthy seed present)
- ❌ RoPE/position-ids wrong under Option B (disproven by static read this session)
- ❌ Precompute math wrong (cos ≥0.9999, id116)
- ❌ vLLM layout mismatch (line-by-line diffed, id118)
- ❌ Drafter weights / forward wrong (bit-for-bit validated, id139)
- ❌ NaN / L2-norm (fixed, id172/173)
- ❌ "Regression vs historical baseline" (the id118 trap — baseline was broken;
   it is NOT broken now, K2 path gives >50%)

---

## The KNOWN-GOOD reference path (your bisection anchor)

Atlas has a drafter forward that produced the validated >50% numbers: the
**non-paged path** run at **cap=1 / K2 verify** (`forward_block_layer.rs` +
`ops::prefill_attention`, OPTION_B unset). That is the *clean* anchor (LTM
id139: >50% accept, drafter forward bit-for-bit vs PyTorch).

**IMPORTANT — honest caveat so the A/B doesn't mislead you:** when you run the
non-paged path at **cap=16** (`~/launch-dflash-gamma-legacy.sh`, OPTION_B off but
still γ-block), row-0 echoes only *intermittently* (observed 2026-06-06:
token_in=25→25, 2972→2972 echo; but 29→220, 10→220 miss). So legacy-γ is NOT a
perfect oracle either — the *clean* >50% reference is specifically the **cap=1/K2**
path, not cap=16 legacy. Do not expect legacy-γ to echo 100%; expect it to echo
*more often than paged* (which echoes ~never). The signal is the **relative**
delta: paged ≈ 0 echoes, legacy-γ partial, K2 cap=1 clean. If you want a hard
oracle, bench cap=1/K2 (`ATLAS_DFLASH_DRAFT_CAP=1`) and confirm >50% first.

A sibling launch script exists for A/B: `~/launch-dflash-gamma-legacy.sh`
(identical to the gamma-nograph launch but WITHOUT `ATLAS_DFLASH_OPTION_B=1`).

**The bug is the DELTA between the paged path and the working forward.** The
paged path differs in exactly three mechanisms — your suspects, priority order:

### Suspect 1 (PRIME): the paged-attention `kv_len`/`q_offset` indirect-args contract
`forward_block_layer_paged.rs:409-426` calls
`prefill_attention_paged_dflash_bf16_indirect`, which reads `kv_len` and
`q_offset` **from a device buffer** (`option_b_indirect_args_dev`) at kernel
entry, NOT as scalar args. That buffer is written host-side at
`forward_block.rs:409-417`:
```
let kv_len = option_b_ctx_count + self.gamma as u32;
let q_offset = option_b_ctx_count;
```
With NO_CTX, `option_b_ctx_count=0` → `kv_len=γ`, `q_offset=0`. Verify the kernel
actually reads these correctly from the indirect buffer and that the γ queries
attend over the γ keys with the right masking. **A wrong q_offset or kv_len here
makes every query attend to the wrong/empty key set → garbage logits → frequency
prior.** This is the least-validated seam and the one that differs most from the
legacy `prefill_attention` call.

### Suspect 2: slot_mapping for the γ K/V writes
`forward_block.rs:390-402` builds the γ slot mapping via
`fill_slots_from_block_table(... start=option_b_ctx_count, count=γ ...)`. The γ
K/V are written to cache slots `[ctx_count..ctx_count+γ]`
(`forward_block_layer_paged.rs:266-282`), then read back by the paged attention.
If the write slots and the read (block_table walk) disagree, attention reads
stale/zero K/V. Use `ATLAS_DFLASH_OPTION_B_DIAG=1` — there's a built-in one-shot
readback at `forward_block_layer_paged.rs:289-371` that compares the just-written
γ K against the source `k_buf`. **Run it first** — if src≠cached, the bug is the
slot mapping and you're done.

### Suspect 3: non-causal masking in the paged kernel
vLLM asserts the DFlash γ-block attention is **non-causal** (every query attends
to every key in the block, bidirectional — `dflash.py:71,189,286-293`). Confirm
`prefill_attention_paged_dflash_bf16_indirect` runs non-causal. If it applies a
causal mask, row 0 (the bonus/echo) can only attend to itself and the echo may
still work, but later rows degrade — though the legacy path being non-causal and
working points here. Lower priority than 1-2 but cheap to check.

---

## Recommended bisection procedure (mechanical — bench, don't theorize)

1. **A/B the two paths** with identical everything else:
   - `bash ~/launch-dflash-gamma-legacy.sh` (OPTION_B off) → confirm row-0
     echoes `token_in` and accept is healthy. This re-establishes the anchor.
   - `bash ~/launch-dflash-gamma-nograph.sh` (OPTION_B on) → row-0 fails. Confirmed.
   - The delta is isolated to the 3 paged mechanisms above.

2. **Run the cache-write DIAG** (Suspect 2 — but expect PASS, this is confirm-not-discover):
   `ATLAS_DFLASH_OPTION_B_DIAG=1 ATLAS_DFLASH_OPTION_B_NO_CTX=1 bash ~/launch-dflash-gamma-nograph.sh`
   Read `DFLASH OPTION_B DIAG: γ K layer0 ... src[..] cached[..]`. **Note:** this
   DIAG was run historically and PASSED bit-perfect (LTM id118, id129) — so a PASS
   here only re-confirms the cache write survived the spec_ssm merge; it is NOT a
   new discovery. If src≠cached now → a merge regressed the slot mapping, fix that.
   If equal (likely) → cache write is fine, go straight to Suspect 1.

3. **Instrument the paged attention** (Suspect 1): dump `q_buf` (post-RoPE),
   the resolved `kv_len`/`q_offset` the kernel actually read from
   `option_b_indirect_args_dev`, and `attn_out` for layer 0, row 0, with NO_CTX.
   Compare against the legacy path's layer-0 row-0 `attn_out` for the same input.
   The first layer where they diverge is the bug. (One probe, one rebuild, read
   the numbers — do NOT add five probes.)

4. Once row-0 echoes under Option B, re-check accept. Only THEN look at the
   batched verify kernel (separate concern, separate doc if needed).

---

## Guardrails (Ronald's standing rules — enforced)

- **Ronald runs ALL builds/benches/launches.** You write code + hand exact
  commands. Never run `cargo build`/bench yourself.
- Build cmd: `ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server`
- **Do NOT touch CUDA kernels** unless the bug is provably in-kernel — vLLM runs
  the same kernels with positive DFlash perf; the bug is Rust-side orchestration
  (id121 scope rule). Stay in `crates/spark-{server,model}/`.
- **Do NOT re-open any DEAD hypothesis.** Bisect against the working K2/legacy
  path. One probe → one rebuild → read the number → smallest edit.
- Tag before any risky change. GOLD safety net: `wip/dflash-GOLD-15tok-20260602`.

## Files (paths + WHAT to read — do not read whole files, reads cost $$)
- **PRIME suspect** `crates/spark-model/src/layers/dflash_head/forward_block_layer_paged.rs` — read `forward_block_layer_attention` (~:384-429, the paged-attn call) + `forward_block_layer_pre_attn` RoPE/reshape (~:240-282). Skip the post_attn MLP half unless attention is clean.
- **Orchestrator** `crates/spark-model/src/layers/dflash_head/forward_block.rs` — read ONLY :75 (eff_ctx=0 gate), :285-289 (position_ids, already verified clean), :390-426 (slot_mapping + indirect-args write). ~37KB file; do not read it all.
- **Propose entry** `crates/spark-model/src/layers/dflash_head/propose.rs` — :248-374 (Option-B setup + ctx precompute) only.
- **Working anchor** `crates/spark-model/src/layers/dflash_head/forward_block_layer.rs` — the legacy `prefill_attention` call is the diff target for Suspect 1/3.

## Existing design docs (reference SPECIFIC sections — these are large, do not read whole)
- `docs/design/dflash_option_b.md` (26KB) — the authoritative Option-B architecture spec. Read **§6 (risk register, esp. risk #2 rollback)** and the paged-attention/slot-mapping section. This is the design contract the code should match; check Suspect 1/2 against it.
- `docs/design/dflash_propose_cuda_graph.md` (120KB — DO NOT READ WHOLE) — only if you touch the indirect-args buffer; it documents WHY `kv_len`/`q_offset` are read from device memory (CUDA-graph replay). Grep it for `option_b_indirect_args_dev` if needed; otherwise skip.
- `docs/design/dflash_gamma_on_ssm.md` (5KB) — context on how γ-batched maps onto the SSM target; read only if the verify kernel becomes relevant (later phase).
- IGNORE for this bug: `dflash_eagle_batch_port*.md`, `dflash_gdn_dump_handoff.md`, `dflash_verify_offset_findings.md` — EAGLE/verify-era, not the propose path.

## vLLM reference (already diffed in LTM id118 by a prior session — for orientation only, NOT a fresh task)
- `~/eagle-refs/vllm-full/vllm/v1/spec_decode/dflash.py` — proposer (query=[bonus,mask×γ], non-causal assert lines 71/189/286-293).
- `~/eagle-refs/vllm-full/vllm/model_executor/models/qwen3_dflash.py` — model forward; layout/RoPE/fused-KV were line-by-line matched to Atlas in id118. **Do not re-diff unless a specific suspect contradicts it.**

## LTM (Postgres `friday_memory.memories` @ 10.0.0.50, query by id)
Read these for full history if a suspect needs deeper context — but the relevant findings are already distilled above:
- **id116** precompute cos≥0.9999 · **id117** the 5-hypothesis Option-B collapse list (1=position, the one that ate a session) · **id118** vLLM diff done + "baseline was broken" trap · **id121** Option-B was 44.9% at cap=1 (scope rule: Rust-side, not kernels) · **id124** apply regressed to 0.9%, suspect forward_block_layer_paged · **id139** K2/K4 sound >50%, batched-γ is the lone failure · **id172/173** NaN fix + this session's checkpoint.
Pull cmd: `PGPASSWORD=… psql -U rstesiak -d friday_memory -h 10.0.0.50 -At -c "SELECT content FROM memories WHERE id=NNN;"`

## Session checkpoint
`~/dflash-session-checkpoint-20260605.md` — the prior session's full state (NaN fix detail + accept-bug localization).
