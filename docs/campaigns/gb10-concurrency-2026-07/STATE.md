# GB10 concurrency campaign — STATE (durable; survives session crashes)

**Goal:** beat vLLM at C=[1,2,4,8,16] — aggregate tok/s at every C AND TTFT/TPOT p50/p99 not losing.
**Plan:** `/workspace/.claude/plans/validated-zooming-bentley.md` (approved 2026-07-25).
**Runtime dir:** `/workspace/.wt-golden/conc_sweep/` (driver logs + per-leg results json).

## How to resume after a session crash
1. Read this file — the Log below is appended by the DRIVER after every leg (not by the session).
2. `tail /workspace/.wt-golden/conc_sweep/phaseA.log` — look for `LEG_DONE <name>` / `PHASEA_DONE` /
   `SERVE_DIED`.
3. Drivers are resumable: re-running `bench/phaseA_c_sweep.sh` skips any leg whose
   `conc_sweep/results/<leg>.json` already exists.
4. Re-arm the convenience monitor on `conc_sweep/*.log` (filter:
   `LEG_DONE|_DONE|SERVE_DIED|Traceback|CUDA error|out of memory`).

## Configuration of record
- Box: dgx1. Model: 27B (Atlas: `centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`; vLLM: same checkpoint if it
  loads, else `nvidia/Qwen3.6-27B-NVFP4` — the leg json records which).
- Atlas serve: golden flags + `--max-batch-size 16`, fifo scheduling (SLAI starves prefill at load),
  env incl. `ATLAS_MTP_GATE_FORCE=1`. Binary: pushed tip of PR #369.
- vLLM serve: `sparkrun-eugr-vllm:latest` (vLLM 0.23.1rc1.dev207), `--max-num-seqs 128`,
  `--max-model-len 32768`, util 0.85.
- Synthetic scoreboard: `bench/bench-atlas-concurrency.py`, C=[1,2,4,8,16], default 4 ISL/OSL
  regimes (≤4096); agentic-harness `target_concurrency` sweep follows as a second driver.

## Log (appended by drivers)
5048fa13d69c2870420a8b9050f54221  conc_sweep/spark_phaseA_baseline
- 2026-07-25T18:13:50Z LEG atlas_synth SERVE_DIED
- CONFIG CHANGE after first atlas_synth SERVE_DIED: --max-batch-size 16 + slots 128 + nd=3 needs
  ~52G of SSM reservations (seq-state 14.2G + rollback ring 18.9G + Marconi 18.9G) + 17.5G weights
  before ANY KV — preflight refusal territory at util 0.70. New atlas C-config: --max-batch-size 20
  (headroom over C=16; pool-boundary exhaustion KILLS requests) + --ssm-cache-slots 32 (synthetic
  sweep has no multi-turn reuse; 4.7G) = 46.2G SSM. Driver now captures a deathlog on serve failure.
- 2026-07-25T19:15:28Z LEG vllm_synth DONE on centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf -> results/vllm_synth.json
- 2026-07-25T19:15:34Z PHASEA_DONE
- 2026-07-25T19:17:44Z LEG atlas_synth SERVE_DIED (deathlog: conc_sweep/atlas_synth.deathlog)
- 2026-07-25T19:17:47Z PHASEA_DONE
- SECOND atlas_synth death, deathlog decisive: "39.7 GB consumed + 50.5 GB inference reserve =
  90.2 GB committed" vs 85.2 budget at bs=20/nd=3. Fix: bs=16 (saves ~8.3G: seq-state 3.6 + ring
  4.7) AND --max-seq-len 4096 (the sweep's regimes cap at ISL+OSL=2048; 32768 was inflating
  max_blocks_per_seq metadata and KV expectations for no benefit). nd=3 kept for C=1 K=4 fairness.
51fc31d43a6e59aec8e9eaced56a02b2  conc_sweep/spark_phaseB
5048fa13d69c2870420a8b9050f54221  conc_sweep/spark_phaseA_baseline
- 2026-07-25T22:10:42Z LEG atlas_synth DONE -> results/atlas_synth.json
- 2026-07-25T22:10:46Z PHASE A compare written -> results/compare.txt
- 2026-07-25T22:10:46Z PHASEA_DONE
- 2026-07-26T00:40:49Z LEG atlasB_nographs DONE -> results/atlasB_nographs.json
- 2026-07-26T02:57:30Z LEG atlasB_graphs DONE -> results/atlasB_graphs.json
- 2026-07-26T02:57:34Z PHASEB_DONE
- 2026-07-26T18:21:44Z LEG atlasC_perseq DONE -> results/atlasC_perseq.json
- 2026-07-26T21:07:11Z LEG atlasC_batched DONE -> results/atlasC_batched.json
- 2026-07-26T21:07:15Z PHASEC_DONE
- 2026-07-27T02:57:15Z LEG atlasD_kmarm DONE -> results/atlasD_kmarm.json
- 2026-07-27T05:32:37Z LEG atlasD_kmarm_graphs DONE -> results/atlasD_kmarm_graphs.json
- 2026-07-27T05:32:41Z PHASED_DONE

## 2026-07-27 — the n=16 step decomposed (this is where the gap lives)

Instruments: `ATLAS_MS_PROFILE` (branch split), `ATLAS_SSM_MS_PROFILE` (mixer vs FFN inside the SSM
layers), `ATLAS_SSM_DETAIL_PROFILE` (mixer stages). Config: phase-D binary, bs=16, fifo, slots 32.

**Step at n=16 = 264.9 ms** (vLLM's is 94 ms):

| block | ms | share |
|---|---|---|
| FFN inside the 48 SSM layers | 97 | 37% |
| SSM mixer (qkvz 33% / recurrent 52% / out_proj 14%) | 93 | 35% |
| Attention branch (16 layers, incl. their FFN) | 55 | 21% |
| LM head | 20 | 8% |

Head is FLAT in n (19.8 → 20.3 from n=4 → 16): it is properly batched. Everything else scales.

**The per-seq projections are NOT the problem — that path is already ACTIVE.** The batched-projection
mixer (`try_decode_multi_seq_ssm_batched`) engages for this config and reads QKVZ/out_proj once per
step. Confirmed by a new one-shot log; it used to decline silently, which is why an earlier phase
measured the symptom (SSM time linear in n) without being able to name the cause.

**The recurrent inner is bandwidth-bound, not launch-bound.** 43 us per sequence per layer for
6 MB of FP32 h_state traffic (3 MB read + 3 MB write) = ~140 GB/s, about half of LPDDR5X peak.
Batching its launches therefore cannot help much, and measurement agrees:
`ATLAS_SSM_BATCHED_RECURRENT` + `ATLAS_GDN_FUSED_NORM` = **+2.6% at C=16** (53.7 → 55.1 tok/s),
coherence preserved. `ATLAS_GDN_FUSED_CONV` adds nothing on top. This confirms the older
"batched-recurrent +1-2%" null was not an artifact of the FFN masking it.

**Both FFN and GDN run at ~2x their own bandwidth floor**, and the whole step is ~4x the roofline
(weights 17.5 GB / 273 GB/s = 64 ms, + ~17 ms of GDN state at n=16). vLLM at 94 ms is ~1.2x that
floor. So the remaining 2.8x is kernel bandwidth efficiency at M=16, spread across FFN (37%),
mixer (35%) and attention (21%) — not one hotspot.

**Levers measured this session** (C=16, decode-style 192-token requests):
| lever | C=16 tok/s | note |
|---|---|---|
| phase-D tip | 53.7 | |
| + batched recurrent + fused norm | 55.1 | +2.6%, coherence OK |
| + FFN NVFP4 MMQ (drop `ATLAS_NO_FFN_NVFP4_MMQ`) | 61.2 | +11.3%, C=1 neutral, output identical |

MMQ re-measured 3x per leg (the 11% was N=1): MMQ off 55.0 / 54.8 / 54.8 (mean 54.9), MMQ on
61.4 / 59.1 / 61.2 (mean 60.6) = **+10.4%, ranges do not overlap**. The MMQ legs also completed the
full 16x192 tokens twice, where the frozen config always truncated to 2977.

`ATLAS_NO_FFN_NVFP4_MMQ` is a PRESENCE flag: `=0` does NOT enable MMQ, the variable must be absent.

**Not a km-arm regression:** the balanced/prefill regime failures are the pre-existing KV
pool-exhaustion wedge tracked in open PR #373 ("decode alloc fails, scheduler livelocks in
decode-ckpt SAVE"), which is also why `balanced_long` is excluded from the sweep. `decode_short` is
clean (0 errors at every C in every leg), so the scoreboard above is valid for that regime only.

## 2026-07-27 — the C=2 "regression" is SOLVED, and it reframes the whole campaign

Probe: same serve config, one leg WITH `--speculative --num-drafts 3`, one leg with it removed
entirely (`conc_sweep/c2_probe.log`, 192-token requests).

| C | with --speculative | without --speculative |
|---|---|---|
| 1 | **25.5** | 14.1 |
| 2 | 20.6 | 20.6 |
| 3 | 27.8 | 27.7 |
| 4 | 36.4 | 36.6 |

**At C>=2 the two legs are identical.** Speculative decode is completely inert above C=1 (the gate
is `active.len()==1`), so C=2 is not a regression in batching — it is the cliff of losing MTP. The
apparent "C=2 slower than C=1" is entirely 25.5 -> 20.6 from spec going away.

**Two consequences that should steer everything after this:**

1. **Atlas non-spec at C=1 is 14.1 tok/s; vLLM at C=1 is 14.2.** Identical. At batch 1 both engines
   sit on the same bandwidth-bound floor, and 100% of our 1.93x C=1 win is MTP speculative decoding.
   We have no baseline decode advantage to fall back on.
2. **Scaling, normalised to each engine's own C=1 non-spec throughput:**
   vLLM: 1.0x -> 1.96x -> 3.75x -> 6.96x -> 11.9x  (C=1,2,4,8,16)
   Atlas: 1.0x -> 1.46x -> 2.60x -> ~3.9x -> ~4.3x
   vLLM scales nearly linearly to C=8; Atlas saturates around 4.3x. The gap is a BATCHING-EFFICIENCY
   gap, and it is already visible at C=2 (1.46x where 2.0x is available).

**Therefore the two levers with real headroom are:**
- **Speculative decode at C>=2** (currently structurally disabled). Worth 1.81x at C=1. Even a
  fraction of that at C=8/16 is worth more than every kernel tweak measured so far combined.
- **Batching efficiency itself** (1.46x at n=2 where 2.0x is on the table) — the 11.2 ms/seq marginal.

## 2026-07-27 — CONFIRMED on hardware: the SSM-layer FFN reads its weights TWICE above n=8

Zero-edit probe (`ATLAS_SSM_MS_PROFILE=1`, one serve, drove C=4/8/16, 9168 samples per n):

| n | mixer us/layer | FFN us/layer | FFN us per seq |
|---|---|---|---|
| 4 | 584 | 751 | 187.9 |
| 8 | 935 | **1023** | 127.9 |
| 16 | 1916 | **2022** | 126.3 |

**n=16 costs 1.98x n=8** — and FFN-per-sequence is FLAT from 8 to 16 (127.9 -> 126.3). That is the
exact signature of two chunked batch-8 passes: the ~7.2 GB of FFN weights are streamed twice per
step at n=16 instead of once. A correctly batched FFN is weight-bandwidth-bound, so n=16 should cost
about the same as n=8 (~1030 us/layer), not double it.

Prediction was made from code alone (`trait_decode_multi_seq.rs:173-204`, the `4..` arm chunking
through `forward_km`/batch-8 GEMV) at ~1010 us vs ~2020 us. Measured 1023 vs 2022. Mechanism
CONFIRMED without touching a line of source.

**Size of the prize:** ~1000 us/layer x 48 layers = **~48 ms off a 264.9 ms step (~18%)**, i.e.
C=16 roughly 60 -> 73 tok/s, stacking with the MMQ lever. The fix is an added dispatch arm routing
`n>8 && ffn.is_dense()` to `forward_prefill` (the NVFP4 MMQ path the ATTENTION layers already use —
`multi_seq/ffn.rs:135`), behind an `ATLAS_NO_SSM_FFN_PREFILL` kill switch. n<=8 must keep
`forward_km`: the recorded crossover says GEMV still wins at M=4. C=1 cannot be affected (n=1 never
enters this arm).

## 2026-07-27 — WIN: wide-batch dense FFN arm for the SSM stack, +30% at C=16

`trait_decode_multi_seq.rs`: added an `n > 8 && ffn.is_dense()` arm routing to `forward_prefill`
(weights read ONCE) above the chunked batch-8 GEMV arm. Direct twin of the attention ladder's
"WIDE-VERIFY BATCHED DENSE FFN" branch. Default ON, kill switch `ATLAS_NO_SSM_FFN_PREFILL=1`
(strict `== "1"`, not a presence check).

3 reps per cell, stacked on the Tier-1 env set (MMQ on + batched recurrent + fused norm):

| C | OLD chunked GEMV | NEW batched FFN |
|---|---|---|
| 1 | 25.4 | 25.4 — untouched, n=1 never enters the arm |
| 8 | 54.4 / 54.0 / 54.2 | 54.3 / 52.6 / 54.4 — unchanged, arm fires only at n>8 |
| 16 | 57.3 / 61.1 / 61.1 | **79.3 / 79.4 / 79.3** |

**+30% at C=16**, above the ~18% predicted, because the batched path also beats the chunked one on
its first chunk, not just the second. Coherence byte-identical. C=8 confirms the gate: no change
where the arm does not fire, which is the control this A/B needed.

### Cumulative C=16 progress this session
phase-D tip 53.7 -> +batched recurrent/fused norm 55.1 -> +MMQ 61.2 -> **+wide FFN 79.4** (+48%).
vLLM 168.9. Ratio 0.35x -> **0.47x**.

## 2026-07-27 — speculation re-enabled at n=2 (+19% at C=2)

`step_mtp` already takes `&mut [ActiveSeq]` and is index-correct over it, and no `active[0]`
assumption survives in the MTP verify path — so the `active.len() == 1` gate was the ONLY thing
stopping multi-seq speculation. Replaced with `active.len() <= mtp_max_seqs()`
(`ATLAS_MTP_MAX_SEQS`, default 2).

This runs MTP PER SEQUENCE: n verify forwards of M=K+1 each, i.e. n weight sweeps per step instead
of one. It therefore only pays where the extra accepted tokens outweigh the extra sweeps.
2 reps/cell, coherence preserved:

| cap | C=2 | C=4 |
|---|---|---|
| 1 (old) | 21.2 / 21.0 | 36.1 / 38.4 |
| **2 (new default)** | **25.3 / 25.1** | 38.4 / 38.3 (inert) |
| 4 | 25.7 / 24.5 | 25.9 / 25.2 (**-34%**) |

n=2 wins, n=4 collapses — the crossover is exactly where the sweep arithmetic put it. C=1 and C>=4
are untouched. **Owed before this ships anywhere near a submission: the BFCL subset accuracy gate.**
MLPerf-edge runs target_concurrency=1, so the golden submission path is unaffected either way.

## Scoreboard after this session

| C | session start | now | vLLM | ratio |
|---|---|---|---|---|
| 1 | 27.4 | 25.4 | 14.2 | **1.79x WIN** |
| 2 | 21.3 | 25.2 | 27.8 | 0.91x |
| 4 | 38.6 | 38.4 | 53.3 | 0.72x |
| 8 | 55.4 | 54.4 | 98.8 | 0.55x |
| 16 | 59.9 | **79.4** | 168.9 | 0.47x |

## What is left, in order
1. **Batched verify** — one fused forward of M = n*(K+1). Needs a new `decode_verify_batch` on the
   Model trait. NOTE the shape already exists: `prefill_batch_chunk(&mut [PrefillSlice])` does n
   sequences x variable tokens. Batched verify is that shape plus (a) per-seq SSM state in, not
   fresh, (b) logits at EVERY position, (c) per-seq rollback. Build on it rather than from scratch.
2. Mixer tensor-core projections (`ATLAS_SSM_TC_PROJ`, ssm_batched.rs) — qkvz/out_proj still run
   scalar batch-16 GEMV at 2.3-3.0 TFLOP/s, 5-7x off the weight-stream floor. Est +10%.
3. LM head at n>=2 off the base `w4a16_gemm` (floor ~2.6 ms vs 20 ms today). Est +5-6%.
4. Host sampling: b1_margin gate-after-scan, f2 softmax when inert, batch-wide argmax poison.
