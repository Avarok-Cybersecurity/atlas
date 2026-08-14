# GB10 batch2 rhythm (2026-08)

K=2 verify streams NVFP4 weights through `w4a16_gemv_batch2`. The bytes
are already minimal (one pass, 4-bit packed). The leftover time is
**load latency**: ncu shows ~72% long-scoreboard stalls. Production
Qwen3.6-35B projections are K=2048 — one wave per phase — so a
shared-memory `cp.async` mailbox lost on this box.

## Already tried (do not merge as default)

| Attempt | Result on GDN `in_proj` 12288×2048 |
|---|---|
| template `w4a16_gemv_batch2` (2026-08-13, #497) | **227 GB/s** |
| template `w4a16_gemv_batch2` (2026-08-14, this oracle) | **224.2 GB/s** (63.14 us, 97.5% STREAM) |
| `#497` `w4a16_gemv_batch2_cpasync` | **195 GB/s** (1.16× slower) |

2026-08-14 baseline (`w4a16_batch2_bw_oracle`, 40 iters, cold DRAM):

| Shape | us | GB/s | % STREAM 230 | % peak 273 |
|---|---|---|---|---|
| GDN in_proj 12288×2048 | 63.14 | 224.2 | 97.5% | 82.1% |
| attn Q 8192×2048 | 42.91 | 219.9 | 95.6% | 80.6% |
| GDN out 2048×4096 | 23.91 | 197.4 | 85.8% | 72.3% |

in_proj is already 97.5% of STREAM. A candidate needs ≥3% vs batch2 to `Win`; that shape has ~2.5% headroom to the measured ceiling.

## Dual-issue (#500) — measured, leave unwired

Two cold-DRAM runs on this box (candidate not in the committed tree; `.cu` copied for the oracle only):

| Shape | Run 1 vs batch2 | Run 2 vs batch2 | Verdict |
|---|---|---|---|
| GDN in_proj 12288×2048 | 0.983× (228 vs 224 GB/s) | 0.985× (226 vs 223) | Neutral |
| attn Q 8192×2048 | 0.989× | 0.971× | Neutral |
| GDN out 2048×4096 | 0.721× (batch2 noisy 138 GB/s) | 0.958× (194 vs 186) | Win on the small shape only |

Default-on needs `Win` on **both** 12288×2048 and 8192×2048. Dual-issue does not. TMA (#501) is optional, not next.

`#497` stays opt-in. `ATLAS_GEMV_BATCH2_CPASYNC=1` is not a fix.

## Denominator

Datasheet LPDDR5X is 273 GB/s. STREAM read on this GB10 is **~230 GB/s**.
Oracles must print both. The kill bar is vs **template batch2**, not vs 273.

## PR split

1. **Oracle** — cold-DRAM GB/s, STREAM-230, 3% kill bar, candidate kernel name.
2. **Dual-issue** — hoist phase-1 `ld.global` before phase-0 compute. No smem mailbox.
3. **TMA** — hardware copy engine. Only after (1) exists and (2) is not enough.

Do not default-on (2) or (3) until the oracle prints `Win` (≥3% faster)
on both `12288×2048` and `8192×2048`, and a separate microtest is
bit-identical to `w4a16_gemv` × 2. `Fail` (>3% slower) is a hard no;
`Neutral` is not a default-on.

The 3% bar lives in `spark_model::layers::ops::gemv_batch2_oracle` so a
candidate PR cannot silently move it.
