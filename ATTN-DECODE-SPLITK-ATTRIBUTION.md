# Paged decode attention split-K on H100 (#928)

**Headline: split-K was disabled on Hopper by a constant, not by a kernel.**
`atlas-core/src/device.rs:16` declares `NUM_SMS = 48` (GB10) in a module named
`sm121`; `run_paged_decode.rs` imported it, so on a 132-SM H100 the split count
was 1 at every batch size and paged decode attention ran 24 CTAs for the whole
campaign. The split-K kernels existed and were wired; nothing selected them.

## Receipt (nsys, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8, round 13)

Cell T1N, C=1, 510 steps, ctx ≈ 4847, median step **16.692 ms**:

| kernel | nodes | µs/step | grid | µs/launch | bytes | achieved | % HBM |
|---|---:|---:|---|---:|---:|---:|---:|
| `paged_decode_attn_fp8` | 12 | 2778.2 (16.6%) | **(24,1,1)** | 231.51 | 9.93 MB | 42.9 GB/s | **1.28%** |
| `paged_decode_attn` (bf16 KV) | 4 | 1013.4 (6.1%) | **(24,1,1)** | 253.36 | 19.86 MB | 78.4 GB/s | **2.34%** |

**3.79 ms of a 16.69 ms C=1 step — 22.7% — moving 198.6 MB.** At n=16 (cell V,
ctx ≈ 1335) the same two kernels are **1.301 ms/step at 19.4% of HBM**; the
prefill twin `inferspark_prefill_paged_fp8` has the identical `(24,1,1)` grid
and costs 9353.4 µs = 2.03% of the 4593-token prefill at 12 GB/s. Byte model:
`n · L · num_kv_heads · head_dim · elem`, K and V. Full tables:
`scratchpad/h100-r13-attribution.md` §C.3, §C.5, §E lever 1.

## Root cause, in two lines

```rust
let current_ctas = num_q_heads * split_ref_seqs(num_seqs, max_decode_seqs);
let num_splits = if current_ctas >= NUM_SMS { 1 } else { NUM_SMS / current_ctas };
```

At `--max-batch-size 16`, `24 × 16 = 384 ≥ 48` → 1 split, **including at C=1**.
Two independent faults: the SM count was another card's, and the occupancy the
rule sized for was the PINNED max batch, not the one sequence in flight.
`KvCacheDtype::Bf16` separately took an explicit "no Split-K (not implemented
for BF16 yet)" branch.

## The policy

```
num_splits = clamp(ceil(SPLITK_TARGET_WAVES × sm_count / num_q_heads), 1, MAX_DECODE_SPLITS)
             with SPLITK_TARGET_WAVES = 2, MAX_DECODE_SPLITS = 16
```

A pure function of `(sm_count, num_q_heads)` — never of the runtime co-batched
count — so the non-associative online-softmax reduction tree is **fixed for the
life of a serve**, the invariant `split_ref_seqs` exists to protect
(`tasks/determinism_investigation.md`). `sm_count` is now the compiled target's
(`[hardware] sm_count`), cross-checked at boot against the driver's own count.
Short contexts are handled INSIDE the kernel (`PD_MIN_KV_PER_SPLIT = 256`) from
each sequence's own `seq_len`: the host may not read `seq_lens` (device memory,
behind a captured graph), and a per-sequence rule stays co-batch invariant.

| target | sm_count | q heads | policy | num_splits | CTAs at C=1 | CTAs at n=16 |
|---|---:|---:|---|---:|---:|---:|
| hopper | 132 | 24 | `auto` | **11** | **264** (2.0 waves) | 4224 |
| b200 | 148 | 24 | `legacy` | 1 | 24 | 384 |
| gb10 | 48 | 24 | `legacy` | 1 | 24 | 384 |
| hopper (`=0` control) | 132 | 24 | pinned 1 | 1 | 24 | 384 |

Active splits also follow context, via the kernel's floor: at L=16384 and
L=4847 all 11 carry work (1490 / 441 positions); at L=1335 only **6** do (256
each) and five are empty — those write `l=0` and the reduce skips them.

## Kernels

Hopper-owned ADDITIONS (new stems, new entry names; gb10's `.cu` untouched, per
the 2026-09-11 placement rule), declared in that target's `[kernels] overrides`
and NOT symlinked into `kernels/b200`:

* `paged_decode_fp8_splitk_hopper.cu` — `adds`. gb10 has an FP8 split-K pair,
  but its inner loop is the SCALAR remainder path; a split count that fills 132
  SMs multiplies that dependency chain rather than hides it, so this restores
  the non-split kernel's `PD_BC=4` batched loads. A same-stem `replaces` would
  have forked the non-split `paged_decode_attn_fp8` entry beside it — which
  five targets compile and six KV dtypes route through — to retune two.
* `paged_decode_bf16_splitk_hopper.cu` — `adds`. Split-K BF16 KV never had.
* `paged_decode_splitk_hopper.cuh` — `adds`. Shared partition, merge, workspace
  format, reduce body.

**Prefill is NOT addressed here.** `inferspark_prefill_paged_fp8` has no
split-K variant: it is a flash-attention prefill with `grid = (nq,
ceil(q_len/BR), 1)`, so splitting its KV range needs a new kernel and a
BR-row-wise reduce, not a launch-geometry change. Left for its own lever — the
9.1 ms it is worth at T=4593 is unclaimed.

## What round 15 measured (1xH100 80GB HBM3, tip `8a6f50b61`, cells A15/S0/S4/S6)

**Landed, C=1.** `ATLAS_ATTN_DECODE_SPLITK=0` (S0) against `auto` (A15), same
binary, one variable:

| rung | S0 (1 split) | **A15 (11, `auto`)** | Δ | rep spread |
|---|---:|---:|---:|---|
| `4096x512` C=1 tok/s | 56.46 | **68.54** | **+21.4%** | 0.02–0.03% |
| `4096x512` C=1 TPOT | 16.79 ms | **13.66 ms** | **−18.6%** | — |
| `1024x256` C=1 tok/s | 68.19 | **71.53** | **+4.9%** | 0.02–0.04% |
| `1024x256` C=1 TPOT | 14.09 ms | **13.40 ms** | −4.9% | — |
| C=1 TTFT, both shapes | 489.2 / 160.7 | 489.7 / 160.5 | ±0.1% | — |

The C=1 gains are 500–900× their rung's rep spread. TTFT is untouched, as it
must be — this is a decode kernel. The curve, C=1 only:

| splits | `1024x256` tok/s | TPOT | `4096x512` tok/s | TPOT |
|---|---:|---:|---:|---:|
| 1 (S0) | 68.19 | 14.09 | 56.46 | 16.79 |
| 4 (S4) | 71.22 | 13.47 | 65.58 | 14.31 |
| 6 (S6) | 71.67 | 13.38 | 67.20 | 13.95 |
| **11 (`auto`)** | **71.53** | **13.40** | **68.54** | **13.66** |

Monotone in splits on the long shape, saturating by 4–6 on the short. The
policy's own choice is the best measured point on the long shape and ties with
6 on the short. Nothing here argues for pinning a smaller count.

**Geometry receipt (nsys, cell A15N, C=1, 4593/512, 510 decode steps).**
`paged_decode_attn_splitk_fp8_hopper` at **`grid=(24,11,1)`**, block (256,1,1),
12 launches/step, **32.70 µs/launch = 303.7 GB/s** against round 13's
`(24,1,1)` **231.51 µs = 42.9 GB/s** — **7.08×** — with the BF16 twin at
35.01 µs (7.24×) and the two reduces at `(24,1,1)`, 6.4 µs each. 24 q heads ×
11 splits = 264 CTAs on 132 SMs = the policy's 2.0 waves. The attention pair
went from 22.7% of the step to **4.38%**, and the median step 16.692 → 14.455 ms.

**REFUTED: the C=16 half, and the microtest already said so.** This doc
predicted long C=16 aggregate 405.9 → ≈453 tok/s and TPOT 31.31 → ≈27.9.
Measured: **401.28 and 31.56** — a null inside a 2% rep spread — and the SHORT
C=16 is a real **−1.6%** (522.15 → 513.86, 7–9× its rep spread).

`native_attn_decode_splitk_hopper_microtest` measured the cause before the
ladder ran, at n=16 (FP8 KV, ms / GB/s):

| L | splits=0 | 4 | best ÷ splits0 |
|---:|---:|---:|---:|
| 1335 | **0.086 / 507.3** | 0.095 | **0.90× — a loss** |
| 4847 | 0.280 / 566.4 | **0.270 / 588.8** | 1.04× |
| 16384 | 0.904 / 594.1 | **0.846 / 634.7** | 1.07× |

**At n=16 the non-split kernel is already at 15–31% of HBM** (507–1025 GB/s)
against 1.3–2.3% at n=1, so there is nothing for split-K to recover; splits buy
**0–7%**, and at L=1335 eleven of them cost more reduce than they save. The
policy cannot back off there: it is a pure function of `(sm_count,
num_q_heads)` **by design**, because co-batch invariance is what keeps the
determinism pin (`tasks/determinism_investigation.md`), and a rule that read
the co-batched count would give one sequence a different reduction tree alone
than beside fifteen others. **Co-batch invariance forecloses a C=16 win.** That
is a deliberate trade, not an oversight, and this lever should not be sold on
C=16.

**The determinism question the split raised is answered.** All seven coherency
exchanges are md5-identical across split counts 1, 4, 6 and 11, with the BA
twin on and off and varlen on and off, and identical to round 14's D14 and
round 13's T1; determinism is 8/8 × 3 and byte-identical to two prior rounds.
The microtest's worst cross-split `rel_rms` over 90 arms is **9.417e-5** with
`cos ≥ 0.999999996` — far below what an argmax over a 248 077-entry vocabulary
would need to flip.

**Two observability defects round 15 had to use nsys to work around, now
closed.** There was no dispatch-side line naming the chosen split count at all
— the kernel-selection table proves the entry RESOLVED, not that eleven splits
reached the launch — so `splitk_dispatch::route_line` now emits, once per KV
dtype on that arm's first decode dispatch:

```
paged decode attention: paged_decode_attn_splitk_fp8_hopper num_splits=11 sm_count=132 policy=auto (ATLAS_ATTN_DECODE_SPLITK)
```

And `ATLAS_ATTN_DECODE_SPLITK=0` boots as `attn_decode_splitk=1 (env)`: the
resolver prints the RESOLVED count and `0`/`off` is one split. That is correct
and is now stated in `kernels/hopper/HARDWARE.toml`'s own comment; the route
line above names the kernel that then ran, which is the unambiguous receipt.

Microtest: `native_attn_decode_splitk_hopper_microtest` — `num_splits ∈
{0,1,2,4,6}` × `L ∈ {1335, 4847, 16384}` × `n ∈ {1,4,16}`, FP8 and BF16 KV; row 0
byte-identical between n=1 and n=16 (an equality), split counts graded against
the non-split kernel at `rel_rms ≤ 2e-3` with a KNOWN_BAD control that fires.
90 arms, all green, round 15.
