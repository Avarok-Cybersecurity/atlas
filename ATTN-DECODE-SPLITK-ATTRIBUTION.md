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

## What round 14 must measure
1. **`4096x512` C=1 TPOT 16.86 → ≈13.1 ms** (vLLM 12.45) — 3.72 ms/step × 512
   ≈ 1.9 s of a 9.11 s e2e. The prediction this lever rests on.
2. **`4096x512` C=16 TPOT 31.31 → ≈27.9**, aggregate 405.9 → ≈453 tok/s;
   `1024x256` C=1 TPOT 14.16 → ≈13.2, C=16 25.38 → ≈24.4.
3. **Determinism 8/8 × 3** (`r13_cell.sh … DET=1`) and coherency 4/4 — the
   split count moved, so the pin's invariant must still hold end to end.
4. **The A/B**: `ATLAS_ATTN_DECODE_SPLITK=0` is the pre-#928 geometry on the
   same binary; `=2/4/6` walks the curve. nsys must show
   `paged_decode_attn_splitk_fp8_hopper` at `grid=(24,11,1)` plus its reduce,
   and its GB/s against the 42.9 above. The boot line must read
   `attn_decode_splitk=auto` with no environment, and no SM-count warning.

Microtest: `native_attn_decode_splitk_hopper_microtest` — `num_splits ∈
{1,2,4,6}` × `L ∈ {1335, 4847, 16384}` × `n ∈ {1,4,16}`, FP8 and BF16 KV; row 0
byte-identical between n=1 and n=16 (an equality), split counts graded against
the non-split kernel at `rel_rms ≤ 2e-3` with a KNOWN_BAD control that fires.
