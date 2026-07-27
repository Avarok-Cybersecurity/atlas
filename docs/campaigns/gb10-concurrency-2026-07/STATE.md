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

## 2026-07-27 — correctness fix (self-inflicted) + FFN crossover is n=5, not n=9

### BUG I INTRODUCED, now fixed
Flipping `ATLAS_MTP_MAX_SEQS` to 2 exposed that the spec-eligibility predicate reads
`inside_thinking`, `post_think_emitted`, `suppress_tool_call` and `disable_mtp` from **`active[0]`
only** (`scheduler/mod.rs`). These are PER-SEQUENCE properties: at n=2, sequence 1 would be
speculated even when its own `suppress_tool_call`/`disable_mtp` said it must not be. Now
`active.iter().all(..)`. At n==1 `all()` over one element is exactly the old predicate, so the C=1
path is unchanged by construction. Verified: tool calls still emit correctly with n=2 speculation on.

Same commit fixes the MTP gate's throughput accounting: `emitted` was
`active[0].seq_len - before`, counting ONE sequence's tokens while timing a step that produced n
sequences' worth — under-reporting MTP throughput by ~n and biasing the gate toward serial decode.
Now summed over all active sequences. (Inert under `ATLAS_MTP_GATE_FORCE=1`, which the benchmarks
set, but wrong for anyone who doesn't.)

### The FFN tile-GEMM crossover is n=5
2 reps/cell, coherence held:
| MIN_N | C=4 | C=8 |
|---|---|---|
| 9 | 37.7 | 53.4 |
| **5** | 37.8 | **57.8 (+8%)** |
| 4 | 36.2 (regresses) | 57.8 |
Default is now 5. So eliminating the double weight read was only PART of the C=16 win — the tile
GEMM is simply better per pass from n=5 up. n=4 regresses, so the GEMV genuinely wins at 4.

### Sweep with everything landed
| C | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Atlas | 25.5 | 25.3 | 37.9 | **57.9** | **79.5** |
| vLLM | 14.2 | 27.8 | 53.3 | 98.8 | 168.9 |
| ratio | **1.80x** | 0.91x | 0.71x | 0.59x | 0.47x |

## 2026-07-27 — WIN: tensor-core mixer projections (+9.2% at C=16)

The five background agents converged on ONE root cause, from vLLM's installed source:
**vLLM never lets M influence kernel selection.** Marlin runs `mma` tensor cores even at M=1
(`gptq_marlin.cu`: `thread_m_blocks = min(ceil(M/16), 4)`, no GEMV path at all), and the CUTLASS
FP4 SM120 path has one fixed 128x128x128 tile. So its weight cost is FLAT from M=1 to M=16.
Atlas instead dispatches BY M into a scalar-FMA GEMV ladder whose runtime is proportional to M.
That single design difference is the marginal-cost gap.

`ATLAS_SSM_TC_PROJ` routes the mixer's qkvz/out_proj onto `w4a16_gemm_t` (M64/N128 FP8-MMA tile
GEMM). Cost to implement: two dispatch arms. The transposed NVFP4 twins `qkvz_nvfp4_t` /
`out_proj_nvfp4_t` are ALREADY built at load and already used by the SSM PREFILL path — no repack,
no new kernel, no new buffer, no extra VRAM.

2 reps/cell, coherence identical:
| leg | C=8 | C=16 |
|---|---|---|
| GEMV (old) | 57.8 / 57.7 | 79.7 / 79.4 |
| **TC n>=9 (new default)** | 57.5 / 57.6 | **86.9 / 86.8** |
| TC n>=5 | 54.9 / 54.8 (**regresses**) | 86.4 / 86.5 |

The mixer's crossover is **9**, not the FFN's 5 — different shapes, different crossover. Do not
assume one transfers.

**ACCURACY DEBT (tracked, not yet paid — per the standing "no gates until parity" directive):**
`w4a16_gemm_t` is W4A8 (E4M3 activations) where the GEMV is W4A16, so it CAN move a greedy token.
It is the production SSM prefill path for these same two weights and the coherence smoke is
identical, but a BFCL gate is owed before merge. Same debt applies to `ATLAS_MTP_MAX_SEQS=2`.

## Scoreboard
| C | session start | now | vLLM | ratio |
|---|---|---|---|---|
| 1 | 27.4 | 25.5 | 14.2 | **1.80x WIN** |
| 2 | 21.3 | 25.3 | 27.8 | 0.91x |
| 4 | 38.6 | 37.9 | 53.3 | 0.71x |
| 8 | 55.4 | 57.9 | 98.8 | 0.59x |
| 16 | 59.9 | **86.9** | 168.9 | **0.51x** |

## Next, from the agents (ranked, all with file:line in their reports)
1. **LM head kernel** — `decode_a2.rs:429` calls `w4a16_gemm` unconditionally; its M64 tile wastes
   75% of the MMA at M=16, giving a FLAT ~20 ms/step against a 2.65 ms roofline. Atlas ALREADY owns
   `w4a16_gemv_batch4/8` and the MTP verify path (`impl_a3.rs:160-192`) already routes M<=8 there
   with a comment measuring the same 19.3 ms. Est **10-17 ms**, ~10 lines. Cheapest item on the board.
2. **GDN third h_state pass** — `_f32_norm`/`_f32_conv_norm`/`_f32_strided*` re-read all of H after
   writing it, purely to compute a Frobenius norm the update loop already had in registers.
   ~8.4 ms/step at n=16, bit-identical to remove.
3. **GDN double read** — the decode kernel reads H twice (hk_dot, then update). An algebraic identity
   (`out = g*(H_old^T q) + vnew*(k.q)`) collapses it to one pass. 9 MiB -> 6 MiB per seq per layer.
4. **Batched verify** — full design with trait signatures, the M<=32 metadata cap, and the
   intermediate-stride kernel bug is in the agent report; the pool's batch stride ALREADY matches
   `h_bytes`, so the wy kernels need one added `inter_batch_stride_floats` parameter.

## 2026-07-27 — WIN: batched-GEMV decode lm_head (+22% C=4, +12% C=8, +6% C=16)

Same root cause as the mixer and the FFN, third instance: the decode head called the base M64-tile
`w4a16_gemm` unconditionally. On the [248320, 5120] NVFP4 head only 16 of 64 MMA tile-rows carry
data at padded_n=16, so it ran at ~1/7 of the weight-stream floor — and being FLAT in n, it sat in
the FIXED term at every batch size, which is why C=4 gained most.

Atlas already owned the fix: the MTP verify path (`impl_a3.rs`) routes M<=8 to the batched GEMV
with an nsys note measuring **19.3 ms for the GEMM vs ~2.5 ms** for the GEMV streaming the same
636 MB once. The decode head never dispatched there, and the MODEL level had no batch16 handle at
all (the SSM mixer carries one) — so there was no arm above 8 even if it had.

2 reps/cell, coherence identical:
| C | old M64 GEMM | new batched GEMV |
|---|---|---|
| 1 | 25.4 | 25.5 |
| 4 | 38.0 | **46.2 (+21.6%)** |
| 8 | 58.0 | **65.2 (+12.4%)** |
| 16 | 86.9 | **91.8 (+5.6%)** |

## SCOREBOARD — full sweep, everything landed

| C | session start | now | vLLM | ratio | was |
|---|---|---|---|---|---|
| 1 | 27.4 | 25.5 | 14.2 | **1.80x WIN** | 1.93x |
| 2 | 21.3 | 24.4 | 27.8 | 0.88x | 0.77x |
| 4 | 38.6 | **46.1** | 53.3 | **0.87x** | 0.72x |
| 8 | 55.4 | **65.2** | 98.8 | **0.66x** | 0.56x |
| 16 | 59.9 | **92.1** | 168.9 | **0.55x** | 0.35x |

C=16 has gone 59.9 -> 92.1 tok/s (**+54%**) this session; the vLLM ratio 0.35x -> 0.55x.
C=4 is now within 13% of vLLM.

## The pattern, stated plainly
Every win this session is the SAME bug in a different place: **Atlas dispatches by M into a
scalar-FMA GEMV/chunked path where a tensor-core tile GEMM was already available and already used
elsewhere in the tree.** FFN (+30%), mixer projections (+9%), lm_head (+22/12/6%). vLLM never makes
this choice — Marlin issues mma at M=1. Anywhere Atlas still selects a kernel BY M is a suspect.

## Remaining, ranked
1. GDN third h_state pass (norm clamp re-reads H after writing it) — ~8.4 ms/step at n=16,
   bit-identical to remove. Agent gave exact file:line for all 4 kernel variants.
2. GDN double read — algebraic identity `out = g*(H_old^T q) + vnew*(k.q)` collapses two passes to
   one. 9 MiB -> 6 MiB per seq per layer.
3. Batched verify (full design in the agent report; the wy kernels need one
   `inter_batch_stride_floats` parameter — today's hardcoded stride is wrong by a factor of `ni`
   and is dead code only because every call site passes batch_size=1).
4. Attention branch RoPE / KV-write are still per-sequence loops (`multi_seq/attn.rs`).

## 2026-07-27 — GDN third pass removed (bit-identical, +1.1%) and a RE-MEASURED map

### The third-pass fix landed, and under-delivered — informatively
7 kernel variants re-read all of H after writing it, to accumulate a Frobenius norm the update loop
already had in registers. Now accumulated in-loop, one add at a time in ascending j so the summation
order is unchanged. **Emitted-text SHA identical across a pre/post binary A/B** (`981ca44911471b59`),
on a real kernel rebuild (158 kernels, 0 cache hits).

Measured **+1.1% at C=16** (91.65 -> 92.65) and +0.6% at C=8, against a ~3% prediction. **That
settles the analysis's open question: the re-read was mostly absorbed by L2, so h_state traffic is
much less DRAM-bound than the roofline model assumed.** Size the remaining GDN passes off THIS datum,
not the model — the "collapse the double read via the algebraic identity" item should be expected to
return ~1-2%, not the ~5% its traffic arithmetic suggests, and it is token-equal-not-bit-identical,
so it is now a poor trade. DEPRIORITISED.

### Re-measured decomposition at n=16 (eager, ATLAS_MS_PROFILE, 190 samples)
| block | before all fixes | now | change |
|---|---|---|---|
| TOTAL | 264.9 ms | **150.4 ms** | -43% |
| ssm (48L) | 189.8 | 102.2 (68%) | -46% |
| attn (16L) | 54.7 | 38.4 (26%) | -30% |
| head | 20.3 | 9.7 (6%) | -52% |

Inside the SSM block (per layer x48): **FFN 42.0 ms | GDN recurrence ~30 ms | qkvz 15.0 ms |
out_proj 13.5 ms**. qkvz fell 678 -> 313 us/layer from the tensor-core arm, as intended.

### TWO NEW FINDINGS from that profile
1. **`out_proj` did NOT improve** (288 -> 282 us/layer) even though its tensor-core arm shipped in
   the same commit as qkvz's, which DID improve. The transposed twin is built by the loader
   (`weight_loader/qwen35_dense.rs:686`), so either the arm is not firing on this checkpoint's
   loader branch, or the tile GEMM is under-filled at N=5120 (5120/128 = 40 CTAs on ~48 SMs — a
   single partial wave, which the analysis flagged as a range rather than a point estimate).
   **Worth 13.5 ms and an hour of investigation.** Settle it with a one-shot log in the arm.
2. **The batched-recurrent path is silently falling back part of the time.** The same profile shows
   BOTH `recurrent_batched_gdn_norm` (618 us) AND the per-seq `recurrent_gdn`/`_ba`/`_conv`
   (567+159+127 = 853 us) in one run. `ssm_batched_recurrent.rs:66-89` requires the n slots to be
   EXACTLY contiguous and returns `None` silently otherwise; pool slots fragment as sequences
   finish. This is the analysis's predicted failure and it means **the +2.6% batched-recurrent
   datum was partly measuring the fallback.** Per-seq costs 853 us/layer vs 618 batched — a 28%
   penalty on ~30 ms whenever it fires. Add the one-line diagnostic FIRST, then decide.

### Scoreboard
| C | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Atlas | 25.5 | 24.4 | 46.1 | 65.5 | **93.2** |
| vLLM | 14.2 | 27.8 | 53.3 | 98.8 | 168.9 |
| ratio | **1.80x** | 0.88x | 0.87x | 0.66x | **0.55x** |

## 2026-07-27 — NULL RESULT that reprices the whole FFN block: M-sized MMQ tiles

Analysis said the FFN's MMQ tile is hard-wired to `mmq_x=128`, so at M=16 it issues MMAs for all
128 tile columns and discards 112 in the write-back predicate — 87.5% of MMA slots. The padded-issue
arithmetic predicted 41.1 ms against a 42.0 ms measurement, an almost exact fit, and therefore
+7-12% at C=16 from sizing the tile to the batch.

Implemented: `atlas_nvfp4_mmq{16,32}_{nc,wc}` instantiations of the SAME template (mmq_x is a free
template parameter; the vendored MMA path's granularity is 8), `nvfp4_mmq_gemm_tiled` with the smem
size DERIVED from the vendor layout (the derivation reproduces the previously-hardcoded 57856 at
mmq_x=128, which is the check that it matches), dispatch by m in dense_ffn, kill switch
`ATLAS_NO_MMQ_SMALL_TILE=1`. Verified the new entries really compiled (present in t0__nvfp4_mmq.ptx).

**MEASURED FLAT.** 2 reps/cell, output SHA identical (`981ca449...`):
| C | 128 tile | M-sized tile |
|---|---|---|
| 4 | 45.8 / 46.2 | 46.2 / 46.3 |
| 8 | 65.8 / 65.8 | 66.0 / 65.9 |
| 16 | 93.1 / 89.1 | 92.5 / 92.6 |

**THE CONCLUSION MATTERS MORE THAN THE CHANGE: the FFN at M=16 is WEIGHT-BANDWIDTH-bound, not
MMA-issue-bound.** The padded MMAs are free because they hide behind the 7.22 GB weight stream
(26-31 ms). The 41.1-vs-42.0 fit was a coincidence. Kept the code (bit-identical, strictly less
wasted issue, and the instantiations are reusable) but it is NOT a win.

### This is the second time a traffic/compute model over-predicted by ~3x
First: the GDN third-pass removal predicted ~3%, delivered 1.1% (L2 absorbed it).
Now: the FFN tile predicted 7-12%, delivered ~0.
**Rule for this campaign: an analytical model that says "X% of the work is wasted" is a hypothesis
about the BOTTLENECK, not a prediction of speedup. Wasted work behind a bandwidth wall is free.**
Both blocks are now known to be bandwidth-bound and within ~1.5x of their floors:
- FFN: ~42 ms actual vs 26-31 ms weight-stream floor
- GDN: ~30 ms actual vs 17-20 ms state-traffic floor
Neither has a 2x left in it. The remaining gap to vLLM is NOT in these two blocks.

### Where that leaves the search
The attention branch (38.4 ms, 16 layers) is the least-examined block and the analysis found ~2,300
per-sequence launches/memcpys per step there (RoPE, KV-write, q/k norms, gate-mul, plus a
scatter/re-gather round trip that batchn makes unnecessary), est. 4-9 ms — and unlike the two blocks
above, that is LAUNCH overhead, which is not hidden behind a bandwidth wall. `sigmoid_gate_mul_batched`
already exists and is unused. That is the next thing to try.

## 2026-07-27 — multi-seq CUDA graphs DEFAULT-ON (+3.2%), and it RETIRES the attention rewrite

The attention branch had ~2,300 per-sequence launches/step (RoPE, KV-write, q/k norms, gate-mul,
plus a scatter the batched QKV then re-gathers), analysed at 4-9 ms if hand-batched. Before building
that, two cheaper things settled it:

1. **Hand-batched the gate-mul** (16 launches/layer -> 1, using `sigmoid_gate_mul_batched` which
   already existed and which the PREFILL path already drives on these same buffers). Removed ~240
   launches/step. **MEASURED FLAT** — consistent with the estimate (240 x ~2-4 us is inside noise),
   not a refutation. Landed anyway: strictly less work, identical output.
2. **CUDA graphs — a ZERO-CODE test of the whole hypothesis**, since graphs capture every launch
   wholesale. **C=8 65.75 -> 67.6 (+2.8%), C=16 92.6 -> 95.6 (+3.2%)**, emitted-text SHA unchanged.

**So the launch overhead is real but worth ~3%, and a flag already captures ALL of it.** Hand-batching
the remaining ~2,000 calls cannot beat what graphs get for free. **The attention pipeline rewrite is
retired** — do not re-open it without new evidence.

`ATLAS_DECODE_GRAPHS_MULTISEQ` is now DEFAULT-ON (`ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1` disables). Its
own comment had said "opt-in until soaked; flip the default once validated" — this is that
validation, and it is exactly the pattern `feedback_good_defaults_not_flags` exists to catch.

## SCOREBOARD — end of session

| C | session start | **now** | vLLM | ratio | start ratio |
|---|---|---|---|---|---|
| 1 | 27.4 | 25.6 | 14.2 | **1.80x WIN** | 1.93x |
| 2 | 21.3 | 25.3 | 27.8 | 0.91x | 0.77x |
| 4 | 38.6 | **47.7** | 53.3 | **0.90x** | 0.72x |
| 8 | 55.4 | **67.7** | 98.8 | **0.69x** | 0.56x |
| 16 | 59.9 | **95.9** | 168.9 | **0.57x** | 0.35x |

**C=16 +60% this session. C=4 is within 10% of vLLM.**

## What is now KNOWN to be near its floor (do not re-open without new evidence)
- **FFN** (~42 ms): weight-bandwidth-bound. M-sized MMQ tiles measured FLAT.
- **GDN recurrence** (~30 ms): state-bandwidth-bound, ~1.5x floor. Third-pass removal 1.1%,
  batched-recurrent 2.6%, double-read deprioritised by the same L2 calibration.
- **Launch overhead** (~3%): fully captured by CUDA graphs, now default-on.

## What is left
1. **out_proj occupancy** — 40 CTAs on 48 SMs at N=5120, a single under-filled wave; both the GEMV
   and the tile GEMM bottom out at ~60 GB/s for different reasons. Split-K or the already-compiled
   `w4a16_gemm_t_k64`. Est ~8 ms.
2. **Batched speculative verify** — still the only structural lever with a >2x shape. Spec is worth
   1.8x at C=1 and is inert above n=2. Full design + the latent `inter_batch_stride_floats` kernel
   bug are recorded above.

## 2026-07-27 — out_proj: K_STEP_T=64 is a REGRESSION, so the diagnosis narrows to occupancy

Analysis proposed a zero-new-kernel first test for out_proj's poor efficiency: route it (and qkvz)
to `w4a16_gemm_t_k64`, the same M64/N128 kernel with K_STEP_T=64, halving the sync-bound outer-loop
count 192 -> 96. Both projections qualify (qkvz K=5120, out_proj K=6144, both multiples of 64) and
the handle was already compiled and bound.

**MEASURED WORSE**, 2 reps/cell, SHA identical: C=8 68.0 -> 67.5, C=16 96.0 -> 95.5. **Reverted.**

That is informative rather than merely negative: it rules out barrier/iteration count as out_proj's
limiter and leaves ONLY the under-filled wave. At N=5120 the grid is 40 CTAs on ~48 SMs — 8 SMs idle
and 1 CTA/SM on the rest, so there are no co-resident CTAs to hide any stall, and a deeper K-step
just makes each of the 40 CTAs do more work serially. The remaining fix is the one that ADDS CTAs:
split-K over `gridDim.z` (S=4 -> 160 CTAs -> 3.3 waves, the regime where qkvz reaches ~55% of peak),
accumulating partials into an FP32 workspace. That is a real new kernel, est. ~8 ms of a ~150 ms
step, and it is now the best-understood remaining kernel lever.

### Running tally of measured-flat/negative levers (all with SHA-identical output)
| lever | predicted | measured |
|---|---|---|
| GDN third h_state pass removed | ~3% | **+1.1%** (L2 absorbed it) |
| M-sized MMQ tiles (mmq_x=16/32) | +7-12% | **0%** (FFN is weight-bound) |
| Hand-batched attention gate-mul | ~0.3-0.6% | **0%** (inside noise, as predicted) |
| Slot-sort in the mixed paths | recovers 28% of a block | **0%** on this workload |
| out_proj K_STEP_T=64 | ~1.5-2x on the block | **-0.5%** |
| **CUDA graphs default-on** | — | **+3.2%** ✓ |
The one that worked is the one that removed a whole CLASS of overhead rather than a slice of it.

## 2026-07-27 — DECISION: batched speculative verify is KILLED (for now). Fix batch scaling first.

Stage 1a ran in ~1 hour instead of the projected 1.5 days and produced two gates, both
pre-registered before measuring.

**PASSED — byte identity.** Batched wy4 on the pointer-table pattern is bit-exact against n
sequential single-sequence launches at n=2/4/8, across h_state, all three rollback intermediates
and the output. **The confirmed cross-sequence corruption bug is fixed with a permanent regression
test** — banked regardless of this decision.

**FAILED — cost.** Fused wy4 at n=8/K=4 costs 723.9 us vs 214.6 us for a plain 1-token n=8 decode
= 3.37x, past the 2.00x stop line; batching the launches is worth only 1.03x over 8 sequential.

**The measurement that reframed it.** Verify step cost vs draft width at n=1:
  K=2 (2 verify rows): 97.1 ms | K=4 (4 verify rows): 97.0 ms — IDENTICAL.
So the +26 ms gap between the plain step (70.9 ms) and the verify step (97 ms) is NOT row cost — it
is the FIXED drafter overhead of entering the spec path. Verify rows are FREE at n=1 because the
whole n=1 step is weight-streaming bound (17.5 GB / 273 GB/s = 64 ms of a 70.9 ms step) AND the GDN
kernel runs at ~8% occupancy with spare memory-level parallelism. **At n=8 the GPU is full and that
slack is gone** — which is exactly why the same kernel measures 3.37x there. The property that makes
speculation cheap at C=1 does not transfer.

Budget at n=8 against a pre-registered 60/82 ms build/kill band: GDN +24.4 (measured), attention
+31 (fitted), drafter +26 (measured at n=1, ASSUMED flat in n) = ~81 ms, i.e. 99.5 tok/s vs vLLM's
98.8 — dead even, one millisecond off the kill line.

### Why KILLED anyway — the strategic ground, which does not depend on that arithmetic
The adjudication argued the attention leg is OVER-counted (verify rows share a sequence's KV, so
n=8xK=4 streams 8 KVs not 32 => +5-15 ms, not +31), which would put the budget in the BUILD band —
and still killed, because:

**Atlas C=1 non-spec is 14.1 tok/s; vLLM C=1 is 14.2. PER-SEQUENCE PARITY on identical silicon.
Yet vLLM scales 11.9x to C=16 where Atlas scales 6.8x.** The whole C=8/C=16 deficit is
batch-scaling efficiency — pure software, un-diagnosed. Fixing it lifts EVERY cell. Fused verify
even optimistically flips C=4 and maybe C=8 while C=16 stays lost (~131 vs 168.9), and it would be
built against a substrate the scaling fix moves (base step, attention batch behaviour, GDN
saturation point), forcing a re-measure anyway.

**Correct order: scaling first, then fused verify becomes the lever that WINS C=16 instead of one
that tops out at C=8.** The verify work is shelved WITH its gate intact and its kernel fix landed.

### The diagnosis is already started — per-sequence marginal, measured
| n | total | ssm | attn | head |
|---|---|---|---|---|
| 4 | 81.3 | 58.6 | 19.5 | 3.2 |
| 8 | 111.8 | 79.4 | 27.9 | 4.4 |
| 16 | 152.2 | 104.1 | 38.4 | 9.7 |

**Marginal per added sequence: 5.91 ms = ssm 3.79 + attn 1.58 + head 0.55.** vLLM's is ~1.5 ms/seq.
The 4.4 ms/seq difference x16 = ~70 ms of a 152 ms step IS the C=16 gap.

**The SSM leg owns 64% of it at 3.6x its physics floor** (1.05 ms/seq for 144 MB of h_state read+
write at 273 GB/s). And that leg is mixer 61.7 (qkvz 15.0, out_proj 13.5, GDN recurrence ~30) +
FFN 42.0 — i.e. mostly WEIGHT-bound work that should be FLAT in n but is scaling with rows. Same
pattern as every win today. If the SSM marginal reached floor: total 3.18 ms/seq, step at n=16
~119 ms, **~134 tok/s with no speculation involved.**

### Carried forward
- The drafter's +26 ms is measured at n=1 and ASSUMED flat in n. Same assumption class that went
  1-for-8 this session. MEASURE IT at n=8 before any re-adjudication.
- OVERTURNING MEASUREMENT: if a roofline decomposition shows the plain n=16 step is already
  near-irreducible traffic, the scaling gap is not a days-scale fix and the verify build becomes
  the best available use of the time. Flip back to BUILD if so.

## 2026-07-27 — ★ THE BANDWIDTH CEILING IS 230 GB/s, NOT 273 — AND NOT 155

Measured with a STREAM microbenchmark on GB10 (48 SMs, 256-bit LPDDR5X), grid 48..384, 2 GiB
buffers, float4 vectorized (scratchpad `stream.cu`):

    READ  230 GB/s        COPY (read+write)  215 GB/s

That is 84% / 79% of the 273 GB/s nominal — a normal STREAM efficiency. Use **230 (read-only
streams) / 215 (read+write streams)** as the floor denominator from now on. Do NOT use 273.

### ★ CORRECTION: "the FFN is at floor, weight-bandwidth-bound" was WRONG
That verdict (recorded 2026-07-27 earlier, and in the m_dispatch memory) was derived against a
floor computed from NOMINAL bandwidth and from an ASSUMED intermediate_size. Real config:
hidden 5120, intermediate **17408**, 64 layers, head_dim **256**, kv_heads 4, vocab **248320**,
full_attention_interval 4 (=> 16 attn + 48 GDN layers), GDN nv48/kd128/vd128 fp32.
FFN weights = 3 x 5120 x 17408 x 0.5 B = 133.7 MB/layer = 8.56 GB over 64 layers.
Floor at 230 GB/s = 37.2 ms. Measured ~57 ms => **1.53x over floor, ~20 ms recoverable.**
The FFN block is RE-OPENED. (The earlier MMQ-tile null result stands as a null for THAT lever,
not as proof the block is at its floor.)

### Full step budget at C=16, eager, 190 samples/point (ATLAS_MS_PROFILE + ATLAS_SSM_DETAIL)
Instrumented forward 152.1 ms; actual step ~167 ms (graphs on) => host leg ~20 ms.

| block                | measured | floor @achieved | ratio | recoverable |
|----------------------|----------|-----------------|-------|-------------|
| GDN projections      | 28.7 ms  | 12.0            | 2.39x | 16.7 ms     |
| FFN (64 L)           | ~57 ms   | 37.2            | 1.53x | 20 ms       |
| host leg             | 20.5 ms  | ~0 overlappable | --    | 20 ms       |
| attention mixer      | ~24 ms   | 4.6 (KV 1.05GB) | 5.2x  | 19 ms       |
| GDN state (48 L r+w) | 29.8 ms  | 22.5            | 1.33x | 7.3 ms      |
| lm_head              | 9.7 ms   | 2.8             | 3.5x  | 6.9 ms      |

### ★ THE ROOFLINE, AND WHAT IT SAYS ABOUT vLLM
Total traffic per decode step at n=16 = **18.4 GB** (FFN 8.56 + GDN state r+w 4.83 + GDN proj
2.77 + KV 1.05 + attn weights 0.59 + lm_head 0.64). At achieved bandwidth that is **~82 ms/step
= a 195 tok/s roofline at C=16**.
- vLLM 168.9 tok/s = 94.7 ms/step = **87% of roofline** -> vLLM is essentially AT the memory wall.
- Atlas 95.9 tok/s = 167 ms/step = **49% of roofline**.
**To beat vLLM we need <=94 ms/step.** The six prizes above sum to more than the 73 ms required,
so the target is arithmetically reachable without inventing a new algorithm. No single lever
does it; this is a six-front grind, and NONE of the six is at its floor.

### Instrument trap (cost one full measurement cycle)
`ATLAS_MS_PROFILE` forces eager, but `ATLAS_SSM_MS_PROFILE` / `ATLAS_SSM_DETAIL_PROFILE` only
skip during graph CAPTURE. Once multi-seq CUDA graphs became default-on, the step REPLAYS the
graph and the Rust-side SSM timers never execute => zero profile lines, silently. Always pass
`ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1` with the SSM profilers. (Same class as the ATLAS_MTP_TIMING
K=2-only trap: an instrument existing != an instrument covering your config.)

### Weighting trap
The detail profile emits both batched and per-seq recurrent stage names in one run. At n=16 the
per-seq rows had **48 samples vs 9120** for the batched rows (one warmup step, 0.2% share) --
the batched path carries everything. Always print sample COUNTS before scaling a stage mean by
the layer count.

## 2026-07-27 — ★ NSYS KERNEL PROFILE (C=16, 154 decode steps) — THE REAL RANKING

`nsys profile --trace=cuda --cuda-graph-trace=node` on a native serve, real C=16 drive at
97.5 tok/s (matches the committed baseline, so the profiled run is representative).
Report: /tmp/atlas_prof3.nsys-rep. Invocation that WORKS: plain `nsys profile` + SIGTERM to
the spark PID. `--delay/--duration` produced no report; `--cpuctxsw` is invalid on
`nsys launch`; the driver script's port must match the serve's.

| kernel                                   | %GPU | inst/step | avg     | ms/step | vs floor |
|------------------------------------------|------|-----------|---------|---------|----------|
| atlas_nvfp4_mmq16_nc (FFN)               | 35.5 | 154       | 283 us  | 43.6    | 1.29x    |
| **w4a16_gemm_t_k64 (projections)**       | 26.6 | 129       | 255 us  | **32.9**| **2.1x** |
| gated_delta_rule_decode_f32_strided_norm | 19.2 | 48        | 613 us  | 29.4    | 1.31x    |
| **w4a16_gemv_batch16 (lm_head)**         | 6.3  | 1         | 9.68 ms | **9.7** | **3.5x** |
| atlas_nvfp4_mmq32_nc                     | 3.6  | 16        | 276 us  | 4.4     | --       |
| rope_forward                             | 0.8  | 256       | 4.5 us  | 1.2     | fan-out  |
| rms_norm                                 | 0.5  | 414       | 1.5 us  | 0.6     | fan-out  |
| **paged_decode_attn**                    | 0.4  | 16        | 42.6 us | **0.68**| --       |

### ★ THE ATTENTION LEVER IS DEAD — DO NOT RE-OPEN
`paged_decode_attn` is **0.68 ms/step, 0.4% of GPU time**. The GQA 6x-KV-re-read fold, the
per-position shuffle-chain rewrite, the split-KV work — ALL of it targets 0.4% of the GPU.
The 38.5 ms that ATLAS_MS_PROFILE attributes to "attention layers" is those layers' FFN
(~12 ms) + their projections (~16 ms) + fan-out; the attention kernel itself is noise.
This killed three successive sizings of that lever (19 ms -> 3 ms -> 1 ms -> 0).

### ★ THE REAL #1: PROJECTIONS AT 2.1x FLOOR (even after the k64 fix)
32.9 ms/step for 3.6 GB of projection weights (SSM qkvz 2.01 + SSM out_proj 0.755 + attn
qkv/o 0.84). Floor at 230 GB/s = 15.7 ms. **~17 ms available** -- the largest single prize.

### ★ #2: lm_head IS AN M-DISPATCH INSTANCE, STILL
ONE launch per step at **9.68 ms**. 636 MB of weights (5120 x 248320 x 0.5) = **66 GB/s =
29% of achievable**. It runs `w4a16_gemv_batch16`. Earlier this campaign lm_head was moved
ONTO that GEMV for +22% C=4 -- but the alternative then was `w4a16_gemm` (N64), which the
bench shows is **4.7x slower than w4a16_gemm_t_k64**. Route it to the k64 tile GEMM
(needs a transposed weight twin): 9.7 ms -> ~3 ms.

### ★ METHOD CORRECTION: THE ISOLATED BENCH OVERSTATES BY ~1.5x
`w4a16_m17_bench` reports 166 us for a ~50 MB weight = ~300 GB/s, ABOVE the 230 GB/s STREAM
ceiling -- impossible. Its 100 back-to-back iterations over ONE weight get L2 reuse the
in-model path never sees (in-model k64 avg is 255 us, not 166). This is exactly why k64's
predicted 1.30x delivered +1.6% e2e. **Size levers from in-model nsys numbers, not from the
microbench.**

### Revised prize table (step ~166 ms at C=16, 195 tok/s roofline)
| lever                        | ms/step | floor | prize   |
|------------------------------|---------|-------|---------|
| projections -> better tiling | 32.9    | 15.7  | ~17 ms  |
| FFN mmq16                    | 43.6    | 37.2* | ~11 ms  |
| lm_head -> k64 tile GEMM     | 9.7     | 2.8   | ~7 ms   |
| GDN state                    | 29.4    | 22.5  | ~7 ms   |
| host leg                     | ~20     | ~0    | ~16 ms  |
| rope/rms_norm fan-out        | 1.8     | ~0.2  | ~1.6 ms |
(*FFN floor is for the full 64-layer weight stream.)
