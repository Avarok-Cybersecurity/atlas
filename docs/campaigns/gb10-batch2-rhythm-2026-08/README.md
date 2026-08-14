# GB10 batch2 rhythm (2026-08)

K=2 verify streams NVFP4 weights through `w4a16_gemv_batch2`. The bytes
are already minimal (one pass, 4-bit packed). The leftover time is
**load latency**: ncu shows ~72% long-scoreboard stalls. Production
Qwen3.6-35B projections are K=2048 — one wave per phase — so a
shared-memory `cp.async` mailbox lost on this box.

## Already tried (do not merge as default)

| Attempt | Result on GDN `in_proj` 12288×2048 |
|---|---|
| template `w4a16_gemv_batch2` | **227 GB/s** |
| `#497` `w4a16_gemv_batch2_cpasync` | **195 GB/s** (1.16× slower) |

`#497` stays opt-in. `ATLAS_GEMV_BATCH2_CPASYNC=1` is not a fix.

## Denominator

Datasheet LPDDR5X is 273 GB/s. STREAM read on this GB10 is **~230 GB/s**.
Oracles must print both. The kill bar is vs **template batch2**, not vs 273.

## PR split

1. **Oracle** — cold-DRAM GB/s, STREAM-230, 3% kill bar, candidate kernel name.
2. **Dual-issue** — hoist phase-1 `ld.global` before phase-0 compute. No smem mailbox.
3. **TMA** — hardware copy engine. Only after (1) exists and (2) is not enough.

Do not default-on (2) or (3) until the oracle says the candidate is ≥3%
faster than template batch2 on both `12288×2048` and `8192×2048`, and
bit-identical to `w4a16_gemv` × 2.
