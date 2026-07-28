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

## 2026-07-27 — ★ PROJECTION UNDER-FILL CONFIRMED: k/v ARE THE WORST KERNELS IN THE MODEL

`w4a16_gemm_t_k64`: M_TILE 64, N_TILE_LG 128, 128 threads (4 warps), **38.19 KiB smem/CTA
=> only 2 CTAs/SM = 96 resident slots on 48 SMs**. Grid is `(ceil(N/128), ceil(M/64), 1)` --
`gridDim.z` is UNUSED and at M=16 `gridDim.y == 1`, so the z axis is free for a K split.

Per-shape CTA counts at M=16 (decode):
| shape        | N     | CTAs | fill                        | calls/step |
|--------------|-------|------|-----------------------------|-----------|
| ssm qkvz     | 16384 | 128  | 1.33 waves (tail 32/96)     | 48 |
| ssm out_proj | 5120  | 40   | 0.83 -- 8 SMs IDLE          | 48 |
| attn q       | 12288 | 96   | exactly one full wave       | 16 |
| **attn k**   | 1024  | **8**| **40 of 48 SMs IDLE**       | 16 |
| **attn v**   | 1024  | **8**| same                        | 16 |
| attn o_proj  | 5120  | 40   | 0.83                        | 16 |
Byte inventory: 48*(41.9+15.7) + 16*(31.5+2.6+2.6+15.7) = **3603 MB = exactly the profiled
3.6 GB**. Shape mix independently confirmed.

### Measured standalone at M=16 (w4a16_m17_bench, 230 GB/s denominator)
| shape                    | time    | achieved   | vs floor |
|--------------------------|---------|------------|----------|
| **attn_k / attn_v N=1024**| 125 us | **23.6 GB/s** | **9.75x** |
| attn_o_proj N=5120       | 166 us  | 106.6 GB/s | 2.16x |
| ssm_out_proj N=5120      | 155 us  | 113.8 GB/s | 2.02x |
| ssm_qkvz N=16384         | 343 us  | 137.7 GB/s | 1.67x |
| **attn_qkv FUSED N=14336**| **332 us** | 124.3 GB/s | 1.85x |
k/v move 2.9 MB in 125 us. Those weights fit ENTIRELY in L2, so the bench's usual ~1.5x
optimism does not apply -- this is pure occupancy starvation, not bandwidth.
**k+v alone = 250 us; the FUSED q+k+v = 332 us total.**

### ★ WHY THE TWO PRIOR NULLS WERE NULLS
`K_STEP_T -> 64` for out_proj (-0.5%) and M-sized MMQ tiles (0%) both re-partition work
INSIDE the CTA. Neither changes the CTA count. The under-fill model predicts ~0 for both --
they are evidence FOR the diagnosis, not against it. It also means occupancy tricks
(smem/reg cuts) are provably useless for out_proj (40 CTAs) and k/v (8 CTAs): when
CTAs < 48 you cannot fill one per SM no matter the residency.

### Ranked remedies
1. **Fuse q+k+v into one N=14336 GEMM — BIT-IDENTICAL.** 3 launches (96/8/8 CTAs) -> 1 at
   112 CTAs. ~2.7 ms/step and -32 launches/step. Needs a fused transposed twin at load
   (row-wise interleave: the `_t` layout is [K/2, N], so it is NOT a flat concat).
2. **Split-K on `gridDim.z`** (ksplits=2 for out_proj/o_proj, 8 for k/v; K%(64*ksplits)==0
   holds: 6144/2=3072, 5120/8=640). ~9.3 ms. **NOT bit-identical** -- one FP32 accumulator
   chain becomes 2-8 chains summed in a reduce; FP32 add is non-associative. Template
   ALREADY IN THE SAME FILE: `int8_gemm_splitk`/`int8_splitk_reduce` at
   `w4a16_gemm.cu:2533/:2652`, whose rationale block states the identical diagnosis.
   ★ Pin ksplits to the WEIGHT SHAPE, never to runtime concurrency -- mirroring the
   `split_ref_seqs` determinism pin (`qwen3_attention/mod.rs:92`), or a sequence's output
   would depend on who else is in its batch.
3. M_TILE=16 + warp-over-N repartition: smem 39.1 -> 25.3 KiB => 3 CTAs/SM. ~3.6 ms,
   bit-identical, qkvz only. Does nothing for out_proj/k/v.
4. Persistent/stream-K: ~10-12 ms but high risk; split-K first. Note `mul_mat_q_stream_k_fixup`
   exists (`q4k_vendor/mmq.cuh:3789`) but Atlas bypasses it with `fixup=false`, justified as
   "prefill has thousands of tiles >> 48 SMs" -- that rationale is decode-blind and INVERTS
   at M=16.

DECISION: take the bit-identical fusion (1) first -- the standing accuracy-gate directive
blocks the BFCL run needed to discharge split-K's numerical debt.

## 2026-07-27 (late) — SHIPPED: fused q/k/v; RE-ADJUDICATED spec decode: NO-GO (measured)

### Shipped since the nsys profile
- `b98ce911` w4a16_gemm_t_k64 dispatch at K>=4096 + wire into SSM projections. +1.6% C=16.
- `4b1b9fa7` batched KV-cache write (kernel was already strided; caller passed 1 in a loop). +0.5%.
- `2db1b349` **fused q|k|v into ONE N=14336 GEMM writing qkv_buf DIRECTLY.** 3 GEMMs
  (96/8/8 CTAs) -> 1 (112 CTAs), AND the 48-copy per-layer scatter deleted (`per_seq_qkv`
  already equalled the fused row width). 4 reps/leg, byte-identical: **97.80 -> 99.38 tok/s
  (+1.6%), sigma 0.09, distributions disjoint.** Kill switch ATLAS_NO_FUSED_QKV=1.
  ★ `n > 8` is REQUIRED: `wide_verify_gemm` early-returns on the batched-GEMV arms for m<=8
  using the BASE weight and ignoring `w_t`, so a fused N reads past q_proj. An earlier build
  without the gate produced truncated output + HTTP 500s — caught by BYTE-IDENTITY, not by
  throughput.
- `8f85418b` **fail-fast guards on GDN contiguous state addressing at batch>1.** wy2/wy3/wyN
  still hardcode `(b*num_v_heads+vh)*hv` for the intermediates whose pool stride is
  `ni*h_bytes`; wy4's only protection was a `debug_assert!` that COMPILES OUT IN RELEASE.
  One call-site change from silent cross-sequence rollback corruption.

### ★★ SPEC-DECODE RE-ADJUDICATION: NO-GO. Four gates, two failed on measurement.
| gate | threshold | measured | verdict |
|---|---|---|---|
| acceptance epsilon at n=16 | >= 2.3 | **~2.6** (mean accepted 1.61 over six 100-step windows) | PASS |
| batched wy4 byte-identity | exact | n=2/4/8/16 all byte-identical | PASS |
| fused GDN cost | <= 2.5x plain | **3.92x**; batching the layer saves only 1.7% (726.6 vs 739.4 us) | **FAIL** |
| drafter propose | <= ~2 ms/seq | **16.08 ms/seq median (n=572)** | **FAIL 8x** |

**The decisive number is propose.** At n=8: verify 8x80.2 = 640 ms + propose 8x16.1 = 129 ms
= 769 ms/step vs the 794 ms/step implied by the observed 26.1 tok/s — the model reconciles
end to end. Fusing verify collapses the 640 -> ~225-259 ms, but **propose stays per-sequence:
256 ms/step at n=16, on its own larger than the entire fused-verify budget.** Batching the
drafter is NOT in the ~1-1.5k-line estimate.

### What I got wrong (recorded so it is not repeated)
I argued the failing GDN gate "measured the wrong quantity" — that the win comes from the 73%
of the step that is weight-bound and flat in M. **That physics is CORRECT**: an adversarial
re-run of `w4a16_m17_bench` at M=16 vs 64 shows +/-12%, with ssm_qkvz 16% FASTER at M=64.
But the arithmetic omitted two terms the KILL had explicitly carried forward as "MEASURE IT
before any re-adjudication": the **+26 ms drafter overhead** (measured at n=1, assumed flat —
now measured at 16 ms/seq) and **host sampling over 64 logit rows instead of 16** (~+12 ms).
Restoring them: ~259 ms -> 6.2 ms/tok -> 1.51x -> **~145 tok/s vs vLLM 168.9.**
Also wrong: "head unchanged" — lm_head has NO M=64 arm, so verify would run 4x
`w4a16_gemv_batch16` = 4 weight sweeps (~39 ms, not 9.7).
Also wrong: epsilon 2.6 is the SYNTHETIC probe. MTP eligibility is
`active.iter().all(...)` over thinking/tool-suppression (`scheduler/mod.rs:551`), so at n=16
ONE sequence in a think block de-speculates the WHOLE batch. Agentic epsilon = 2.6 x an
unmeasured eligible-step fraction.
★ RULE: re-deriving a budget minus its flagged terms is goalpost-moving. The KILL had already
run this exact framework and pre-registered a DIFFERENT flip-back condition, which tested FALSE.

### ★ PRE-REGISTERED RE-OPEN KEY (so it cannot drift)
Flip to BUILD only if ALL THREE hold: drafter+sampling <= ~12 ms combined at n=16 AND
agentic epsilon >= 2.4 (measured on the agentic harness, not the synthetic probe, with the
`all()` predicate live) AND eligible-step fraction >= 80%.

### Next: the non-spec board (ranked, and each ms also lowers the future verify numerator)
1. split-K on `gridDim.z` for the narrow-N projections (~9 ms). Template `int8_gemm_splitk`
   at `w4a16_gemm.cu:2533`. NOT bit-identical — pin ksplits to the WEIGHT SHAPE.
2. host-leg: split-row GPU argmax (~4-8 ms) + pin DECODE_LOGITS_HOST_SCRATCH (~1-2 ms).
   ★ `--disable-thinking` sets `think_ended=true` for EVERY sequence at birth
   (`prefill_a_step.rs:229` +3 siblings), and the gate at `decode_logits_step.rs:76` then
   forces ALL rows onto the host path — so the GPU argmax fast path is DEAD CODE in this
   benchmark AND in MLPerf-edge. The rows only need a 2-token ban mask.
3. lm_head -> k64 tile GEMM (~7 ms; needs a ~715 MB transposed twin; also required before any
   future fused verify).

## 2026-07-27 (late) — SHIPPED: GPU argmax for think_ended. lm_head tile GEMM: REVERTED (CUDA 716)

### `9daec5b9` — admit `think_ended` rows to the GPU argmax fast path. **+2.4% C=16.**
`--disable-thinking` sets `think_ended = true` for EVERY sequence at birth
(`prefill_a_step.rs:229` + 3 siblings). The batch-wide gate at `decode_logits_step.rs:76` then
forced the WHOLE batch onto the host path whenever any row had it — i.e. always. **The GPU
argmax fast path was unreachable dead code in this benchmark AND in the MLPerf-edge config**,
so every step paid a 7.95 MB D2H + n full-vocab host passes.
Such a row needs only `PostCloseThinkMask` = TWO ids. A4's bias floor is gated on
`inside_thinking` (`sample_step.rs:164`) so it is inert. With request penalties exactly
neutral the pipeline reduces to the raw argmax modulo those two ids — so: run `argmax_batch`
(64-byte D2H), and if a returned token lands on a masked id, fall THROUGH and redo the step on
the host. `THINK_MASK_FALLBACKS` counts those.
Measured 3 reps/leg, byte-identical: **99.60 -> 102.00 tok/s, sigma 0.26 -> 0.17, disjoint.**
Kill switch `ATLAS_NO_THINKENDED_GPU_ARGMAX=1`.
★ TRAP hit while writing it: the first version returned `Vec::new()` from the fall-back arm,
which emits NO tokens and stalls every sequence. The fast path must yield an `Option` so
"needs the host pipeline after all" genuinely falls through.

### lm_head -> tile GEMM: BUILT TWICE, FAILED TWICE, REVERTED. ★ NOT a kernel constraint.
★ CORRECTION to the earlier entry below: I recorded this as "a real constraint inside
`w4a16_gemm_t_k64` at N=248320". **That was WRONG.** Adding the lm_head shape to
`w4a16_m17_bench` runs ALL FOUR variants clean at N=248320, M=1..64:
| kernel | M=16 | achieved | vs floor |
|---|---|---|---|
| **`w4a16_gemm_t`** | **3672 us** | **194.8 GB/s (84.7% of ceiling)** | **1.18x** |
| `w4a16_gemm_t_m128` | 3865 us | 185.0 GB/s | 1.24x |
| `w4a16_gemm_t_k64` | 4062 us | 176.0 GB/s | 1.31x |
| `w4a16_gemm` (N64) | 17304 us | 41.3 GB/s | 5.57x |
| *in-model GEMV today* | *9680 us* | *66 GB/s* | *3.1x* |
715 MB cannot sit in L2, so 84.7% is honest, not microbench optimism. **~6 ms/step is real
and available** — the largest remaining prize.
★ ALSO: the `K >= 4096` k64 threshold picks the WORSE kernel here (`_t` is 10% faster at
N=248320). That threshold was derived from projection shapes; it does not generalise.

**What is actually known about the fault** (three attempts, three hypotheses eliminated):
- Fails with `w4a16_gemm_t_k64` AND plain `w4a16_gemm_t` => not kernel-specific.
- The single-request identity probe PASSES BYTE-IDENTICALLY every time; only CONCURRENCY
  faults. First error is always decode-side: `argmax_batch: cuMemcpyDtoHAsync_v2 failed:
  status 716` (CUDA_ERROR_MISALIGNED_ADDRESS, sticky — the memcpy is just the first sync
  point after the faulting kernel).
- Dims verified against the checkpoint: `lm_head.weight [248320, 2560]`,
  `weight_scale [248320, 320]`, K=5120. N = 1940x128 exactly. Twin layout [K/2, N] correct.
- Alignment RULED OUT: every arena buffer is its own `gpu.alloc()` (`buffers.rs:134-147`),
  i.e. a separate cuMemAlloc at 256-B alignment. The twin's weight and scale are likewise
  fresh allocations.
- The remaining discriminator is that concurrency runs inside the CAPTURED multi-seq CUDA
  graph and the single request does not.
**compute-sanitizer HAS NOW BEEN RUN. Result: ZERO invalid memory accesses, and under the
sanitizer the error changes from 716 (MISALIGNED_ADDRESS) to 719 (LAUNCH_FAILED).** So this
is NOT an out-of-bounds or misaligned access at all — it is a kernel that fails to launch or
execute, which memcheck cannot attribute to an address.

SIX hypotheses eliminated, all by measurement:
1. kernel-specific constraint at N=248320 — NO: all four variants run clean standalone.
2. wrong dims — NO: verified against the checkpoint (`lm_head.weight [248320, 2560]`,
   `weight_scale [248320, 320]`, K=5120, N = 1940x128 exactly).
3. pointer misalignment — NO: every arena buffer is its own `gpu.alloc()`
   (`buffers.rs:134-147`), i.e. a separate cuMemAlloc at 256-B alignment.
4. CUDA-graph interaction — NO: faults identically with `ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1`.
5. unguarded epilogue store overrunning `logits` at M_TILE=64 (48 spare rows x 496,640 B =
   23.8 MB, which WOULD explain why only the widest N faults) — NO: the store is explicitly
   guarded, `if (r0 < M && c0 < N)`.
6. an addressable memory error — NO: memcheck is clean.

★ NEXT TOOL, not next guess: inspect the LAUNCH itself (shared-memory request vs the 48 KB
default without `cudaFuncSetAttribute`, register/occupancy limits at grid.x=1940, or
`--tool synccheck`). The wiring has been written THREE times and looked correct each time;
the defect is in launch configuration or a resource limit, not in the pointers.

### (superseded) original entry
Motivation was sound and stands: nsys puts lm_head at **9.68 ms/step in ONE launch**, 715 MB at
**~66 GB/s = 29% of the 230 GB/s ceiling**, the largest non-GEMM kernel in the step. At
N=248320 the tile GEMM launches ~1940 CTAs, so unlike the narrow-N projections it is
well-occupied; expected ~5 ms.
- The transposed twin builds fine: **1.8 s, ~605 MB** (`transpose_for_gemm(gpu, vocab, hidden)`).
- The single-request identity probe **PASSED** (byte-identical).
- Concurrent drives died: first error `argmax_batch: cuMemcpyDtoHAsync_v2 failed: status 716`
  (CUDA_ERROR_MISALIGNED_ADDRESS, sticky — the real fault is the preceding tile GEMM), then
  cascading 716s from `cuMemsetD8Async` in prefill.
- Ruled OUT as causes: N=248320 = 1940x128 exactly; K=5120 = 80x64; twin layout [K/2, N]
  matches what the kernel expects; cuMemAlloc base pointers are 256-B aligned.
- Therefore this is a real constraint inside `w4a16_gemm_t_k64` at N=248320 (suspect the
  epilogue's store vectorisation, or a cp.async alignment assumption that holds for the
  projection shapes but not this one). **Read the kernel's epilogue before retrying.**
- REVERTED rather than shipped default-off: shipping known-broken code behind a flag is not a
  compromise, it is a landmine.

### Board after this
1. split-K on `gridDim.z` (~9 ms) — template `int8_gemm_splitk` at `w4a16_gemm.cu:2533`.
   NOT bit-identical; pin ksplits to the WEIGHT SHAPE, never runtime concurrency.
2. lm_head tile GEMM — blocked on the 716 above.
3. host-leg residue: pin `DECODE_LOGITS_HOST_SCRATCH` (it is a pageable Vec, so the "async"
   D2H is a staged sync copy), and the two O(output_len) `rposition` scans per seq per step.

## 2026-07-27 (late) — FFN MMQ BLOCK OPENED: under-fill REFUTED, occupancy hint NULL

The FFN (`atlas_nvfp4_mmq16_nc`) is the largest single block: **192 inst/step, 54.3 ms,
35.5% of GPU time**. Grid is `[div_ceil(N,128), div_ceil(M,16), 1]`, block 256, so at decode
M=16 `gridDim.y == 1` and gate/up (N=17408) launch **136 CTAs** while down (N=5120) launches
**40 CTAs on 48 SMs**. That looked like the same under-fill as the projections.

### ★ IT IS NOT. Splitting the 29,661 FFN launches by gridX in the nsys capture:
| shape | gridX | count | avg | GB/s |
|---|---|---|---|---|
| gate/up | 136 | 19,774 | 282.3 us | 167.7 |
| **down** | **40** | 9,887 | **284.3 us** | 166.6 |
**Within 0.7%.** Using 40 of 48 SMs costs essentially nothing here — 40 CTAs already saturate
the memory system. => **stream-K would buy ~nothing at decode**, so the vendored header's
"prefill shapes have thousands of tiles >> 48 SMs so stream-k buys ~nothing" bypass
(`fixup=false`) happens to be RIGHT at decode too, for a different reason than it states.
A complete stream-K path (`mul_mat_q_stream_k_fixup`, `mmq.cuh:3789`, nsm-sized grid,
`tmp_fixup` partials, and a launcher that picks it below 90% tile efficiency at `:4003`) is
sitting unused — do NOT integrate it on under-fill grounds; the measurement says no.

### Occupancy hint: NULL, reverted
Dynamic smem is `4*(mmq_x + pad256(mmq_x*36) + 128*76)` = **41.06 KiB at mmq_x=16**, 43.1 KiB
at 32, vs 100 KiB/SM on sm_121 — so TWO CTAs fit, yet every Atlas entry carried
`__launch_bounds__(256, 1)`. Raised the four small-M entries to `(256, 2)` (mmq128 needs
56.5 KiB and must stay at 1). Measured C=16, 4 reps, byte-identical:
control 102.4/102.0 (mean 102.20) vs 102.8/102.4/102.5/102.2 (mean 102.48) — **+0.27%, ranges
OVERLAP, indistinguishable from noise.** REVERTED: unmeasurable benefit, and `mmq32` is a
PREFILL kernel whose register budget would tighten for an unproven decode gain.

### Verdict on the FFN block
Uniform **~167 GB/s = ~77% of the 230 GB/s achievable** across all three shapes (counting
block_nvfp4's real 36 bytes per 64 weights). There is **no structural defect** here — no dead
capability, no wrong dispatch, no under-fill — unlike every other win this session. Closing
the remaining ~23% means real inner-loop work inside vendored llama MMQ (dequant/scale ALU
overlap with the cp.async weight stream), worth ~6.7 ms. That is a genuine project, not a
dispatch fix, and it is the code path Atlas owns least.

## 2026-07-27 (night) — OVERNIGHT BASELINE + a measurement artifact worth knowing

**Baseline on HEAD (`b0a248f1`), settled GPU, 6 reps, identity sha `bf3a0b07`:**
`99.8 | 102.8 103.0 102.7 103.0 102.9` => **rep1 is a WARMUP OUTLIER**; steady state is
**102.88 tok/s, sigma 0.13** over reps 2-6.
★ Every A/B run today included rep1 in the mean. At ~3% low it is large enough to hide or
fake a 1% effect. The canonical harness now runs a **discarded warmup drive** before the
measured reps (`scratchpad/ab_template.sh`).

★ ALSO: a `compute-sanitizer` serve survived its own script's `kill -TERM` and held **74.6 GB**
for ~20 minutes, starving the next container and producing 500s that looked exactly like "HEAD
is broken after the reverts". `compute-sanitizer` runs the app under a `TreeLauncherSubreaper`,
so the TERM went to the wrapper, not the process, while the script printed its DONE marker.
**Verify with `nvidia-smi --query-compute-apps` — do not trust a script's completion message.**

## 2026-07-28 — k64 THRESHOLD FIX (+3.4%, shipped) · GDN register-resident (REGRESSION, reverted)

### `140be0e6` — k64 threshold 4096 -> 6144. **+3.4% C=16. Biggest single win of the session.**
★ This fixed a regression I introduced EARLIER THE SAME DAY in `b98ce911`, which lowered the
threshold to 4096 based on the ffn/out_proj shapes without ever benchmarking K=5120.
Measured at M=16 on the real decode shapes (230 GB/s denominator):
| shape | `_t` | `_k64` | `_m128` |
|---|---|---|---|
| ssm_qkvz     N=16384 **K=5120** | 281.9 | **341.6 (was selected)** | 272.4 |
| attn qkv     N=14336 **K=5120** | 273.9 | **328.5 (was selected)** | 262.8 |
| ssm_out_proj N=5120  **K=6144** | 237.7 | **163.3 (correct)** | 240.7 |
`_k64` is the WORST variant at K=5120 and the best only at K>=6144. 48 qkvz + 16 fused-qkv
launches/step were on the slowest kernel available for a full day.
Measured, 4 reps/leg, warmup discarded, byte-identical:
OLD 103.4/103.1/102.6 = **103.03** -> NEW 106.7/106.5/106.8/106.1 = **106.53**, disjoint.
★ RULE: a threshold measured on two shapes does NOT generalise to a third. Added
`ATLAS_W4A16_K64_MIN_K=<n>` so any A/B can pin a prior threshold exactly.
★ `_m128` is faster still at K=5120 (272.4 / 262.8) — a further ~0.6 ms is available.

### GDN single-pass register-resident decode: **REGRESSION -11.6%, REVERTED**
The diagnosis was right: `gated_delta_rule_decode_f32_strided_norm` reads H for `hk_dot` and
then RE-READS the identical values for the update; at batch 16 the live state is ~49 MB so the
second read partially misses L2. A standalone prototype holding the H column in registers
measured **927 -> 542 us (1.71x), byte-identical**.
**In production it is 11.6% SLOWER end-to-end**: 107.13 -> 94.73 tok/s (4 reps/leg,
byte-identical, disjoint). GDN is ~29.7 ms of a ~140 ms step, so the kernel roughly DOUBLED.
Cause: `hreg[128]` (512 B/thread) spills to local memory. The production kernel carries far
more register pressure than the prototype — the Frobenius norm clamp, the two-stage RMS
reduction, and the packed-BF16 epilogue all live in the same function — so it spills where the
isolated version did not.
★ LESSON: an isolated kernel prototype does NOT transfer to a kernel with a larger epilogue.
Register-residency wins are contingent on the WHOLE function's register budget, not the loop
being optimised. Retry only with the epilogue split into a second kernel (so the hot loop's
budget is its own), or with a smaller tile (e.g. hreg[64] and two passes over half-columns).

## 2026-07-28 (night) — ★ THE NEXT LEVER, FULLY SPECIFIED: out_proj/o_proj -> NVFP4 MMQ

### Per-shape truth (nsys of HEAD, split by gridX — do NOT reason from blended averages)
| kernel | gridX | shape | avg | **GB/s** |
|---|---|---|---|---|
| `w4a16_gemm_t` | 128 | ssm_qkvz N=16384 K=5120 | 311.6 us | **151.4** |
| `w4a16_gemm_t` | 112 | attn qkv fused N=14336 K=5120 | 296.5 us | **139.2** |
| **`w4a16_gemm_t_k64`** | **40** | **out_proj / o_proj N=5120 K=6144** | 210.7 us | **84.0** |
| `atlas_nvfp4_mmq16_nc` | 40 | ffn_down N=5120 K=17408 | 285.5 us | **175.6** |
| `atlas_nvfp4_mmq16_nc` | 136 | ffn gate/up N=17408 K=5120 | 289.5 us | **173.2** |

★ A blended "projections run at 102 GB/s" figure is WRONG and cost an agent-hour: qkvz and
fused-qkv are already efficient (151/139). **The entire projection deficit is out_proj/o_proj
at 84 GB/s.**
★ MMQ hits ~174 GB/s at gridX=40 AND gridX=136, and at K=5120 AND K=17408 — so there is no
shallow-K cliff between those endpoints and **out_proj's K=6144 should transfer**. That is
2.09x `_k64` at the SAME 40-CTA occupancy, i.e. the gap is the KERNEL, not the tiling.

### Prize: 64 launches x 210.7 us = 13.5 ms/step -> ~6.5 ms at 174 GB/s
Minus 64 activation-quantize launches (~0.75 ms) and 64 `nvfp4_scale_bf16` launches
(~0.75 ms) => **~5.4 ms net, ~+3.8%**. Roughly 4x split-K's honest prize (1.3-1.5 ms) for the
identical shapes.

### Implementation (NO SSM MMQ plumbing exists today — grep confirms zero hits)
Mirror `dense_ffn.rs`: handles (`nvfp4_mmq16_nc/_wc`, `nvfp4_quant_act`, `nvfp4_repack`,
`nvfp4_scale`) -> repacked twin via `ops::nvfp4_mmq_repack` (ops/nvfp4_mmq.rs:56) ->
`ops::nvfp4_mmq_quantize_act` (:80) -> `ops::nvfp4_mmq_gemm_tiled` (:147, tile=16 at m<=16)
-> `ops::nvfp4_scale_bf16` (:222) for the scale2 fold (the GEMM output is documented
"missing x scale2" at :103).
★ HAZARD: the repack MUST be eager at LOAD time, before KV sizing and before CUDA-graph
capture — the FFN does exactly this in `finalize_nvfp4_mmq_load` (dense_ffn.rs:445). A lazy
OnceCell repack inside the decode path would allocate during graph capture.
★ VRAM: out_proj twin is 17.7 MB x 48 + 17.7 x 16 = **1.13 GB**. Unlike the FFN, the `_t`
copy CANNOT be freed — it is still the SSM prefill path (`ssm_batched.rs:17-19`).
Constraints all pass: N=5120 %128, K=6144 %64, m=16 <= mmq_x=16.

### ★ ACCURACY ORDERING IS THE INVERSE OF THE OBVIOUS ONE
MMQ is **W4A4** (activations quantized to FP4).
- **out_proj / o_proj = LOW risk.** Their input is the post-GDN gated-norm output, so the
  error is feed-forward-shaped and does not re-enter the recurrence. THIS is the one to build.
- **qkvz = HIGH risk. DO NOT convert.** It feeds conv1d -> the FP32 GDN recurrent state, where
  per-token error persists across the sequence. No cosine measurement for SSM-projection W4A4
  exists anywhere in the repo; dense_ffn's down-proj 0.9961 does NOT transfer. Memory records
  FP16 h_state causing ~25% trajectory divergence, so the recurrence is precision-sensitive.
- Debt is UNDISCHARGEABLE while [[feedback_no_accuracy_gate_until_vllm_parity]] stands, and it
  STACKS on the existing `ATLAS_SSM_TC_PROJ` W4A8 debt (ssm_batched.rs:28-32 already records
  "a BFCL gate is owed before this merges").

### Also unexplained, worth 1.4 ms: 11.3 extra `w4a16_gemm_t` launches/step
The capture shows `w4a16_gemm_t` at gridX=8 (N=1024, 512 inst) and gridX=96 (N=12288, 256
inst) — i.e. the fused-qkv path splitting back into separate q/k/v launches during ramp/drain
when n<=8 (the `n > 8` gate from `2db1b349`). ~1.44 ms/step. Lowering that gate is NOT safe
(see the gate's comment), but batching the n<=8 case differently might be.

### `_m128` for qkvz: NULL, reverted (and rebuilt)
The bench ranks `_m128` fastest at K=5120 (272.4 us vs `_t` 281.9 / `_k64` 341.6 at M=16), and
it affects the 48 qkvz launches/step. But qkvz is only ~14.9 ms of a ~140 ms step, so a 3.4%
kernel gain is ~0.36% e2e — below what the harness resolves.
Measured 4 reps/leg, warmup discarded, byte-identical:
OLD 106.1/106.9/106.4/106.5 = **106.48** vs NEW 106.3/106.8/106.7/106.6 = **106.60**.
+0.11%, ranges fully OVERLAP => NULL. Reverted **and rebuilt** (a `git checkout` alone leaves
the old binary in place — that bit me earlier tonight with the GDN kernel).
★ RULE OF THUMB now calibrated: this harness resolves ~>=0.8% reliably. A kernel-level gain
only matters if (kernel share of step) x (kernel gain) clears that. qkvz is 10.6% of the step,
so it needs a >7% kernel win to be worth measuring at all.

### Fused q/k/v at ALL n (removing the `n > 8` gate): NULL on this benchmark, reverted
The gate exists only because `wide_verify_gemm` early-returns on its GEMV arms for m<=8 and
ignores `w_t`; calling `ops::w4a16_gemm_n128` DIRECTLY removes the need for it. The work
reduction is real — nsys prices the split path at 279 us (q, gridX 96) + 218.5 + 218.5 (k/v,
gridX 8) = **716 us vs ~273 us fused**.
Measured 4 reps/leg, byte-identical, control pinned via a new `ATLAS_FUSED_QKV_MIN_N=9`:
OLD **106.60** vs NEW **106.45** => -0.14%, ranges OVERLAP. NULL, reverted and rebuilt.
★ WHY, and the sizing error to avoid repeating: I derived "~1.4 ms/step" by dividing 1024
split-path instances by ~175 steps. Those instances are NOT spread across steps — they are
concentrated in the brief ramp/drain tail, because `prof_drive` fires all 16 requests at once
with identical `max_tokens`, so n stays 16 for nearly the whole run.
**An "instances over the run / total steps" average is meaningless when the instances are
concentrated in a few steps.** Check the step DISTRIBUTION before sizing.
★ Worth revisiting under STAGGERED arrivals (real serving), where small-n steps are common —
the change is strictly less work and byte-identical. It needs a benchmark with arrival jitter,
which `prof_drive` does not model.

### GDN register retention, attempt 2 (HALF-width, `hreg[64]`): -4.6%, reverted
Better than full-width's -11.6% but still a regression: **106.68 -> 101.78 tok/s**, 4 reps/leg,
byte-identical, disjoint.
★ ROOT CAUSE IS AN IMPLEMENTATION ERROR, NOT THE IDEA. Pass 2 was left as
`#pragma unroll 4` over `j < k_dim` (a RUNTIME bound) and indexes `hreg[j]` — a dynamic index,
so `hreg` is placed in LOCAL memory. The conditional
`(j + 0 < GDN_HALF_KD) ? hreg[j + 0] : H[...]` guarantees it. I avoided this trap in pass 1
(full `#pragma unroll` over a compile-time bound) and then reintroduced it in pass 2.

**The correct shape, for whoever retries:**
```
// pass 2a — retained half, FULLY unrolled, static indices
#pragma unroll
for (unsigned int j = 0; j < GDN_HALF_KD; j += 4) { h0 = hreg[j+0]; ... }
// pass 2b — remainder, re-read from H
#pragma unroll 4
for (unsigned int j = GDN_HALF_KD; j < k_dim; j += 4) { h0 = H[(j+0)*v_dim + tid]; ... }
```
Both loops must write H and accumulate `q_dot`/`norm_acc` in ascending j so the summation order
matches the original exactly (the Frobenius comment in the production kernel already relies on
this for bit-identity).

### ★ GDN REGISTER RETENTION: 3 attempts, all regressions. Do not retry without the above.
| attempt | shape | e2e | mechanism |
|---|---|---|---|
| full-width `hreg[128]` | 512 B/thread | **-11.6%** | spills; budget shared with Frobenius clamp + RMS reduction + packed-BF16 epilogue |
| half-width `hreg[64]` | 256 B/thread | **-4.6%** | pass 2 dynamic index -> local memory |
| (standalone prototype) | 512 B/thread | *+71%* | carried NONE of the epilogue |
★ The prototype's 1.71x is real and irrelevant: it measured a loop, not the function. Any
retry should FIRST split the epilogue (Frobenius clamp + RMS reduction + BF16 pack) into a
second kernel so the hot loop owns its register budget, THEN retain.

## 2026-07-28 (night) — ★★ GDN HALF-WIDTH REGISTER RETENTION: +5.4%, SHIPPED (`5aada944`)

`gated_delta_rule_decode_f32_strided_norm` reads all of H for `hk_dot` then RE-READS it for
the update: 2R+1W over the state each step. At batch 16 the live state is ~49 MB, past L2, so
the second read partially reaches DRAM. Retaining the first 64 H columns makes it 1.5R+1W.

**Measured 4 reps/leg, warmup discarded, byte-identical, disjoint:**
OLD 107.1/106.9/106.2/106.5 = **106.68** -> NEW 112.8/112.0/112.4/112.7 = **112.48**. **+5.4%.**

### ★ THREE ATTEMPTS — the failures were REGISTER BUDGET, not the idea
| # | config | e2e | cause |
|---|---|---|---|
| 1 | `hreg[128]`, 512 B/thread, static indices | **-11.6%** | genuine spill — budget shared with the Frobenius clamp, the two-stage RMS reduction and the packed-BF16 epilogue |
| 2 | `hreg[64]`, 256 B/thread, RUNTIME index in pass 2 | **-4.6%** | dynamic index puts `hreg` in LOCAL memory |
| **3** | **`hreg[64]`, 256 B/thread, static throughout** | **+5.4%** | fits |
★ I nearly stopped after #2, having written "register retention in this kernel is closed".
What rescued it was re-reading my own summary and finding an error in it: I had blamed #1 on
dynamic indexing, but #1 was ALREADY static. That left half-width + static as the one untested
cell — and it was the winner. **Check your own postmortem before you accept its conclusion.**

### Sweet spot is 64 — 96 buys nothing
KD=96 (384 B/thread, 1.25R+1W = 25% traffic cut vs 64's 17%) measured **112.45** against the
same two-pass control, i.e. IDENTICAL to KD=64's 112.48. The extra retention is exactly offset
by register pressure. Do not chase larger tiles.

### The required code shape
```
// pass 1: retain first GDN_HALF_KD (full unroll, compile-time bound), stream the rest
// pass 2a: retained half from hreg — #pragma unroll, STATIC indices
// pass 2b: remainder — re-read from H
```
Ascending j across both pass-2 loops keeps the `q_dot`/`norm_acc` summation order identical, so
the Frobenius bit-identity argument in the two-pass kernel still holds.
★ The standalone prototype of the FULL-width variant measured 1.71x on the loop alone and
still lost 11.6% in production. **Size register-retention changes against the WHOLE function's
budget, never the loop being optimised.**

### ★ out_proj -> MMQ has a VRAM BLOCKER at the benchmark's util (checked before building)
The repacked block_nvfp4 twin is 17.7 MB x 48 layers = **850 MB**, and unlike the FFN the `_t`
copy CANNOT be freed (SSM prefill still uses it, `ssm_batched.rs:17-19`).
Arithmetic against the observed pool: KV is 4735 blocks (4.6 GB, 65536 B/block); batch 16 at
`--max-seq-len 4096` needs 256 blocks/seq x 16 = **4096**. 850 MB is ~875 blocks, leaving
~3860 — **below the requirement**, so the serve fails to build with
"KV cache can hold at most 15 concurrent sequence(s)".
=> The lever needs `--gpu-memory-utilization >= 0.75`, i.e. a BENCHMARK CONFIG CHANGE. Do not
fold it in silently alongside a throughput claim; measure the config change separately first.
Options: (a) raise util and re-baseline everything, (b) convert only attn o_proj (16 layers,
283 MB — fits) for ~1.4 ms, (c) make the SSM prefill path use the MMQ weight too so the `_t`
copy can be freed, which is the FFN's own solution (`finalize_nvfp4_mmq_load`, dense_ffn.rs:445).
(c) is the right long-term shape and removes the VRAM cost entirely.

### Post-GDN-win budget (nsys, 160 steps, C=16 at 112.5 tok/s)
| block | ms/step | share | state |
|---|---|---|---|
| FFN `mmq16` | 55.2 | 42% | ~174 GB/s, NO structural defect — vendored inner-loop work |
| projections `_t` (qkvz, fused qkv) | 22.9 | 17% | 139-151 GB/s, near floor |
| GDN `_half` | 22.1 | 17% | **1.18x floor — DONE** |
| projections `_k64` (out_proj, o_proj) | 13.8 | 10% | **84 GB/s** — the MMQ lever, VRAM-blocked above |
| lm_head | 9.7 | 7% | 29% of achievable — launch failure, 6 hypotheses dead, memcheck clean |

## 2026-07-28 (night) — ★ PREFIX CACHING IS A NET LOSS AT SHORT PROMPTS (−7% at C=1)

Every C-sweep tonight showed C=1 drifting DOWN within a run (25.4 -> 23.6 -> 23.6 and then
flat). Cause isolated with a direct A/B, 5 reps each, same binary, only `--enable-prefix-caching`
differing:

```
prefix caching ON  : 25.4  25.4  23.6  23.6  23.6   <- degrades once the cache warms
prefix caching OFF : 25.3  25.3  25.3  25.3  25.2   <- flat
```

**A warm prefix-cache hit costs ~0.5 s per request** here (7.6 s -> 8.1 s for a 192-token
generation off a **26-token** prompt). Steady-state C=1 is **25.3 with caching OFF vs 23.6 with
it ON — prefix caching is costing 7%.**

### Why: no minimum-match threshold
`prefill_a.rs:170-215` takes the snapshot-restore path on ANY match. Blocks are 16 tokens, so a
26-token prompt matches ~1 block: it restores GDN state (3 MB x 48 layers) to avoid recomputing
**16 tokens** of prefill. For SSM models a hit WITHOUT a usable snapshot is even worse — it
forces `kv_write_start = 0` (full KV rewrite), so the lookup cost is paid for zero benefit.
Same mechanism as the recorded 9-20 s snapshot-miss spikes, at small scale.

### Scope / caveats before anyone "fixes" this by disabling caching
- This benchmark uses 26-token prompts. With LONG shared prefixes the cache is surely a win —
  the defect is the ABSENCE OF A THRESHOLD, not the feature.
- **C=16 shows NO drift** (112.0/112.9/112.7 across reps), so the cost is hidden or amortised
  at concurrency. It is a low-C effect on this workload.
- MLPerf-edge runs WITH prefix caching and short-ish prompts — worth measuring there before
  assuming the golden config is unaffected.

### Suggested fix (unbuilt)
Gate the snapshot-restore path on matched-prefix length: take it only when the tokens saved
exceed the restore cost. Needs the restore cost measured per layer-count first — the ~0.5 s
observed here is far above the naive 144 MB / 215 GB/s = 0.67 ms, so **something other than raw
state bandwidth dominates it** and should be profiled before a threshold is chosen.

### Prefix-cache penalty: the two hit types differ, and it is NOT the state copy
Serve log at C=1, consecutive reps:
```
rep2  "Prefix cache hit: 16 tokens (1 blocks) but no SSM snapshot"  -> 24.9 tok/s  FAST
rep3  "Marconi SSM cache hit: 26 tokens skipped (2 blocks)"         -> 23.1 tok/s  SLOW
```
**The slow path is exactly the one that USES the snapshot to skip prefill.** Skipping 26 tokens
of work makes the request 0.6 s SLOWER.

nsys of that run (3 reps, ~24 s) rules out the copy:
- memcpy **375 ms total** (93,376 copies, 28.7 GB) — nowhere near 0.6 s/request
- memset 33.7 ms
- dominated instead by `w4a16_gemv_batch4` (66,051 launches, 13.7 s) = the MTP verify path
=> The penalty is NOT snapshot-restore bandwidth. The most likely mechanism is DOWNSTREAM:
spec-decode throughput is trajectory-dependent ([[reference_spec_decode_tokps_is_trajectory_dependent]]),
so resuming from a restored state can change draft acceptance and therefore verify cost.
★ NEXT STEP is an acceptance measurement, not a copy optimisation: run the same C=1 reps with
`k4_record_positional` and compare mean-accepted on snapshot-hit vs cold reps. If acceptance
drops after a restore, the fix is in the drafter/state handoff, not in the cache.
★ Do NOT "fix" this by adding a size threshold until that is checked — a threshold would hide
the symptom while leaving a spec-decode state-handoff bug in place.

### ★★ ROOT CAUSE: a snapshot restore degrades MTP DRAFT ACCEPTANCE (not the cache, not the copy)
C=1, prefix caching on, 4 consecutive reps, hit type from the serve log:
| rep | hit type | tok/s |
|---|---|---|
| 1 | cold | **25.3** |
| 2 | `Prefix cache hit: 16 tokens` (KV-only, NO snapshot) | **25.3** |
| 3 | `Marconi SSM cache hit: 26 tokens` (**snapshot used**) | **23.6** |
| 4 | `Marconi SSM cache hit: 26 tokens` | 23.5 |

`k4_record_outcome` summaries, chronological:
```
mean accepted = 1.45, 1.52   <- cold / KV-only reps
mean accepted = 1.33         <- after snapshot restore
```
**Acceptance falls ~10%** (1.485 -> 1.33) => epsilon 2.485 -> 2.33 => **-6.2% predicted**.
Measured **-6.7%**. The acceptance drop accounts for essentially the whole penalty.

=> The defect is a STATE-HANDOFF GAP: the main model resumes warm from the restored SSM
state, but the MTP drafter does not — it effectively starts cold, drafts worse, and the extra
rejected drafts cost more than the skipped prefill saved. Ruled out along the way: the state
copy (memcpy is 375 ms across a 24 s capture) and the prefill skip itself (the KV-only hit,
which skips less work, is FAST).

**Where to look:** the drafter consumes hidden states saved during decode
(`save_hidden_for_mtp`, `trait_impl/speculative.rs`). A snapshot restore reinstates SSM
h_state/conv_state but there is no corresponding restore of the drafter's hidden history, so
the first drafts after a resume are made from a cold proposer.
**Fix shapes:** (a) include the drafter's hidden state in the Marconi snapshot, (b) suppress
MTP for the first few steps after a restore so a cold proposer does not waste verify slots, or
(c) warm the proposer from the restored state before drafting.
★ Do NOT paper over this with a minimum-prefix-size threshold — that hides the symptom on
short prompts while leaving the handoff bug live for every long-prefix resume, which is
exactly where prefix caching is supposed to pay.

### ★★★ EXACT LOCATION: `speculative.rs:194` — the snapshot restore disables the drafter prefill
```rust
let cold_prefill_ok = p >= 2 && captured >= p && seq_tokens.len() >= p;
```
`captured` is `mtp_prefill_capture_len`: positions whose hidden states were captured DURING THE
MAIN MODEL'S PREFILL. A Marconi snapshot restore SKIPS that prefill, so nothing is captured,
`captured >= p` is false, `cold_prefill_ok` is false, and **`prefill_drafter` never runs**. If
the cross-turn carry (`ATLAS_MTP_CARRY_DRAFTER`) does not also apply, the proposer starts empty.

**Complete causal chain, every link measured:**
snapshot restore -> prefill skipped -> hidden-state capture skipped -> drafter prefill disabled
(`speculative.rs:194`) -> cold proposer -> mean accepted 1.485 -> 1.33 (-10%) -> epsilon 2.485
-> 2.33 -> **-6.2% predicted / -6.7% measured**.

**Correct fix: extend the Marconi snapshot to cover the DRAFTER state**, so a resume restores
the proposer alongside the SSM h_state/conv_state. The drafter is ONE layer against the model's
64, so the alternative — replaying only the drafter over the skipped prefix — is also cheap
(~1.5% of a full prefill) and needs no snapshot-format change. Either removes the penalty
without giving up the prefill saving.
Rejected: a minimum-prefix-size threshold. It hides the symptom on short prompts and leaves
the handoff broken for long-prefix resumes, which is exactly where prefix caching should pay
and where the lost prefill is largest.
★ This also means the cross-turn carry path (`try_carry_drafter`) is what keeps multi-turn
conversations fast; single-turn resumes off a cold cache get no drafter state at all.

### ★ CORRECTION to the fix options above: "replay just the drafter" DOES NOT WORK
`prefill_drafter(prompt_tokens, hiddens, ...)` (`speculative.rs:362`,
`mtp_head/draft_proposer.rs:86`) consumes `hiddens` = `mtp_prefill_hidden`, the MAIN MODEL's
per-position hidden states captured during ITS prefill. After a Marconi restore those were
never computed, so there is nothing to feed the drafter. The drafter cannot be replayed
independently — it is a function of the target's hidden states, not of the tokens.

**So there are exactly two viable fixes:**
1. **Extend the Marconi snapshot to carry the drafter's own state** (its KV rows / proposer
   state), so a resume restores target AND drafter together. Correct, and preserves the whole
   prefill saving. Snapshot-format change.
2. **Capture hidden states for the skipped span anyway** — i.e. do not skip the target prefill
   when MTP is active and the drafter would be left cold. This gives up the prefill saving,
   which is the thing the cache exists to provide, so it is only sensible as a stopgap.
Option 1 is the real fix. (An earlier note here suggested a cheap drafter-only replay; that was
wrong and is retracted.)

### ★ THE PENALTY PEAKS AT C=2 (-9.2%) AND VANISHES BY C=4 — the scoreboard understates low-C
Same binary, 3 reps/point after a discarded warmup, only `--enable-prefix-caching` differing:
| C | caching ON | caching OFF | cost | corrected ratio vs vLLM |
|---|---|---|---|---|
| 1 | 23.6 | **25.2** | **-6.8%** | 25.2 / 14.2 = **1.77x WIN** |
| 2 | 23.05 | **25.17** | **-9.2%** | 25.2 / 27.8 = **0.91x** (reported 0.85x) |
| 4 | 48.4 | 48.07 | -0.7% (noise) | 0.90x |
=> **C=2, the weakest cell in the sweep, is ~half explained by this bug rather than by an
architectural gap.** Fixing the drafter snapshot should move C=1 and C=2 up ~7-9% with no
kernel work at all.
★ Do NOT respond by disabling prefix caching. It is inert at C>=4 here and REQUIRED for real
multi-turn workloads with long shared prefixes; the defect is the cold drafter after a resume
(`speculative.rs:194`), not the cache.
★ Every headline number in this campaign was measured at C=16 WITH caching on, where the
penalty does not appear (no within-run drift at C=16), so the shipped wins are unaffected.

### ★★ CORRECTION: the defect is KNOWN, and my "reject the threshold" advice above was WRONG
`crates/spark-model/src/model/mtp_carry.rs` documents this exact mechanism from a prior
session: on a warm turn `mtp_prefill_capture_len` stays 0, the `captured >= prompt_len` guard
fails, `prefill_drafter` is skipped and the drafter starts EMPTY — measured there at
**"+10% accepted tokens per verify step"**, matching tonight's -10% acceptance measured
independently via throughput drift -> A/B -> k4 stats -> `speculative.rs:194`.
It also PRE-REFUTES the obvious remedy with numbers: a full warm-turn `prefill_drafter` costs
**1136 ms** against a **1134 ms** warm TTFT — it doubles TTFT to buy ~10% of decode, a
wall-clock LOSS. Do not propose the rebuild.

**What is NEW tonight is the SCOPE.** The shipped fix (`ATLAS_MTP_CARRY_DRAFTER`, on by
default) carries the drafter's KV across turns OF THE SAME SESSION, and its premise is that
"a turn's prompt is a strict extension of the previous turn's full sequence". That covers
multi-turn resumes. It does NOT cover a **preamble-only hit**: a fresh request matching the
shared chat-template prefix of a DIFFERENT conversation. There is no previous turn to carry
from, so carry cannot fire and the drafter is cold. That is precisely what `prof_drive`
produces, and it costs **-6.8% at C=1 and -9.2% at C=2** (inert by C=4).

### => The size threshold I rejected earlier IS the right fix here. That rejection was wrong.
- long-prefix hit that is a turn extension -> carry fires -> drafter warm -> NO penalty
- short preamble-only hit -> carry cannot fire -> drafter cold -> penalty, while the prefill
  saved is only ~16-26 tokens (~30 ms) against 0.5-1.7 s of lost acceptance
So gating the Marconi skip on matched-prefix length does not hide a handoff bug for long
prefixes — **carry already handles those** — it declines a trade that is measurably bad.
**Proposed rule:** take the Marconi SSM skip only when the carry will fire (same session,
strict extension) OR the matched prefix is long enough that the saved prefill exceeds the
acceptance cost. Calibrate the threshold from the two measured points (26 tokens => -7 to -9%;
inert at C>=4) before choosing a constant; do not guess it.

### ★★ CALIBRATION: the crossover is between ~32 and ~629 MATCHED tokens
Identical prompt each rep (so reps 2-3 are FULL-prompt hits), C=1, 128 max tokens:
| prompt tokens | caching ON (warm reps) | caching OFF | delta |
|---|---|---|---|
| 35 | 24.8 | 24.87 | **-0.3%** (neutral) |
| 629 | 21.85 | 20.4 | **+7.1%** |
| 2709 | 21.55 | 15.6 | **+38%** |
(Cold rep-1 with caching ON matches the OFF number exactly at every size — 18.9 vs 20.4 and
15.6 vs 15.6 — confirming the benefit is entirely in the warm reps.)

**Reconciles with the -6.8%/-9.2% measured earlier:** that used `prof_drive`, whose prompts
DIFFER per rep, so the hit covered only the ~16-26-token shared chat-template preamble — a
negligible prefill saving bought with a cold drafter. When the hit covers a real prefix the
saving dominates and prefix caching wins decisively.

=> **The threshold rule is sound and now bracketed by measurement: gate the Marconi SSM skip
on MATCHED-prefix length, crossover between 32 and 629 tokens.** A conservative 256 sits
inside the bracket; narrowing it further needs points at ~128/256/384 (~15 min of the same
harness). Do NOT set it from the prompt length — set it from the MATCHED length, which is what
determines both the saving and whether carry can fire.
★ This also means the headline C=1/C=2 sweep numbers are pessimistic for real workloads:
`prof_drive`'s per-rep prompt variation produces the worst case (preamble-only hits). Real
multi-turn traffic hits long prefixes, where caching is worth +7% to +38%.

### ★★★ CROSSOVER NARROWED: between ~99 and ~219 matched tokens => **threshold 256**
Identical-prompt reps (full-prompt hits), C=1, warm reps vs caching-off:
| matched tokens | ON (warm) | OFF | delta |
|---|---|---|---|
| 99 | 23.85 | 26.4 | **-9.7%** LOSS |
| **219** | **24.15** | 22.0 | **+9.8%** WIN |
| 349 | 23.55 | 21.9 | +7.5% |
| 629 | 22.45 | 20.3 | +10.6% |
Sharp crossover, not gradual. **Recommended constant: 256 matched tokens** — inside the
measured win region, comfortably above the 99-token loss point, and block-aligned
(16 x 16-token blocks).

**The fix, now fully specified:** gate the Marconi SSM skip on `matched_tokens >= 256`. Below
that, take the KV-only path (which measured FAST — it is the snapshot restore that costs, not
the cache) so the drafter keeps its prefill and acceptance stays high. Expected: +6.8% at C=1
and +9.2% at C=2 on preamble-only traffic, with the +7-38% long-prefix win untouched.

Caveat on the data: output lengths differ between ON/OFF at some sizes (62 vs 78, 128 vs 92)
because a restored state changes the greedy trajectory — the known spec-decode trajectory
dependence. tok/s is a RATE so the comparison holds, but do not compare wall times directly.

## 2026-07-28 — ★★ SHIPPED `7ba11dc5`: Marconi skip floored at 256 matched tokens
**C=1 23.65 -> 25.27 (+6.9%, predicted +6.8%) · C=2 23.10 -> 25.30 (+9.5%, predicted +9.2%) ·
C=16 112.6 -> 112.37 (unchanged, as expected).** C=1 is now STABLE across reps
(25.3/25.3/25.2) — the within-run drift present in every sweep tonight is gone.

★ NOT byte-identical, deliberately: short-match hits now take the KV-only path instead of
restoring a snapshot, which changes the greedy trajectory. The new path is the one that
matches full recompute, so it is the more faithful of the two.
`ATLAS_MARCONI_MIN_TOKENS=<n>` overrides; 0 restores always-restore.

### ★ BEFORE THE NEXT MLPerf-edge RUN: check this interacts as expected
MLPerf-edge runs WITH prefix caching, and `mtp_carry.rs` records **987 of 1007 scored samples
are WARM turns**. Those are turn extensions with long prefixes, so they sit above the 256-token
floor and keep the snapshot skip — the expected impact is nil-to-positive. But:
- any turn whose MATCHED prefix is < 256 now recomputes it (a little more TTFT) in exchange
  for a warm drafter (better acceptance). Given the recorded wall split (decode 59.6% /
  fixed TTFT 21.1% / marginal prefill 18.8%), that trade should be net-positive, but it is
  UNMEASURED on the golden workload.
- the golden leg runs `target_concurrency=1`, which is exactly where this fix is largest
  (+6.9%), so the MLPerf wall may move more than the C=16 number suggests.
**Measure the golden leg before folding this into a submission**, and quote the harness, not
the serve log.

### Final scoreboard (3 reps/point, warmup discarded)
| C | start | end | vLLM | ratio |
|---|---|---|---|---|
| 1 | 27.4 | 25.3 | 14.2 | **1.78x WIN** |
| 2 | 21.3 | 25.3 | 27.8 | 0.85x -> **0.91x** |
| 4 | 38.6 | 48.7 | 53.3 | 0.91x |
| 8 | 55.4 | 70.7 | 98.8 | 0.72x |
| 16 | 59.9 | **112.4** | 168.9 | 0.35x -> **0.67x** |

## 2026-07-28 — ★ ROLLBACK VERIFIED, and the accounting closes exactly
All seven kill switches engaged simultaneously (`ATLAS_NO_W4A16_K64=1`,
`ATLAS_NO_ATTN_BATCH_CACHE_WRITE=1`, `ATLAS_NO_FUSED_QKV=1`,
`ATLAS_NO_THINKENDED_GPU_ARGMAX=1`, `ATLAS_NO_ARGMAX_BATCH=1`, `ATLAS_NO_GDN_HALF_REG=1`,
`ATLAS_MARCONI_MIN_TOKENS=0`):
```
ALL WINS ON   112.6 tok/s   sha bf3a0b07...   coherent
ALL OFF        95.4 tok/s   sha bf3a0b07...   coherent
```
1. **The escape hatch works.** All switches fire together without conflict; one env block
   restores pre-session behaviour with no rebuild and no revert.
2. **The wins compose to +18% jointly** (95.4 -> 112.6), measured together rather than summed
   from individual A/Bs — they neither cancel nor double-count.
3. **Byte-identical across the full rollback** (same sha both ways), as expected: seven wins
   are bit-exact and the eighth (the Marconi floor) only diverges on a SHORT-MATCH prompt,
   which this probe is not.
4. **95.4 ~= the 95.9 this session started from**, so the decomposition is exact:
   - campaign start -> session start: 59.9 -> 95.9 (+60%, earlier phases)
   - **this session: 95.9 -> 112.2 (+17%)**
   - campaign total: **59.9 -> 112.2 (+87%)**
