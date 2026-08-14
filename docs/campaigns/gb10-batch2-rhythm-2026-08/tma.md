# PR: TMA weight prefetch for `w4a16_gemv_batch2`

Status: **design only**. Do not implement until:

1. The bandwidth oracle (`w4a16_batch2_bw_oracle`) is on the branch you measure.
2. Dual-issue (`w4a16_gemv_batch2_dualissue`) has a GB10 number. If dual-issue
   already wins ≥3% and is bit-exact, TMA is optional, not next.

## Why this is not `#497`

`#497` used **`cp.async` from the SM** into shared memory. On K=2048 that
mailbox cost more than the wait it hid (195 vs 227 GB/s).

**TMA** is a different truck: a copy engine. The SM writes a descriptor
(“this tile of packed weights”) and keeps doing FMAs. Issue-slot tax is
the thing `cp.async` paid and lost on.

It can still lose. One next box + expensive setup = `#497` again. The
oracle is the judge, not the story.

## What to build

- New module `kernels/gb10/common/w4a16_gemv_batch2_tma.cu`
  (do not grow `w4a16_gemv.cu`).
- Same grid/block/args as `w4a16_gemv_batch2`.
- Same FMA helper / virtual-lane reduce as template batch2 (bit-exact).
- TMA only the **packed weight** stream. Scales and activations stay
  `ld.global` (tiny / L1).
- Alignment: 16-byte TMA loads are illegal when `K/2 % 16 == 8` on odd
  `n` (the `#497` K-tail bug). Gate wide copies on gmem alignment or
  pad the packed row stride. Document which.
- Opt-in only: `ATLAS_GEMV_BATCH2_TMA=1`. Unset or `=0` is template
  batch2. Missing handle falls back.
- Do not default-on in this PR.

## Kill gates (must all pass on GB10)

```
ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-35b-a3b ATLAS_TARGET_QUANT=nvfp4

# bit-exact vs template batch2
cargo run -p spark-model --release --features cuda,gpu-examples \
  --example w4a16_batch2_dualissue_microtest
  # (or a tma-specific twin — same 3 shapes, 3 seeds, raw BF16 bytes)

# bandwidth
ATLAS_GEMV_BATCH2_CANDIDATE=w4a16_gemv_batch2_tma:w4a16_gemv_batch2_tma \
  cargo run -p spark-model --release --features cuda,gpu-examples \
  --example w4a16_batch2_bw_oracle
```

Fail if any production shape is **>3% slower** than template batch2.
Print GB/s vs STREAM **230** and datasheet **273**.

Default-on is a **follow-up PR** after those numbers, not this one.

## Out of scope

- Turning on `#497` `ATLAS_GEMV_BATCH2_CPASYNC=1`
- Changing the echolp draw / n=1004
- SW GEMV at K=2
- More speculative drafts
- Wiring TMA into M=1 / batch4+ in the same PR

## Suggested commit

`kernels: TMA prefetch for w4a16_gemv_batch2 (opt-in)`
