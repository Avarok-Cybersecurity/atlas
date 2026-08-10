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

## 2026-07-28 — SYSTEMATIC SWEEP for the campaign's dominant pattern (dead kernel handles)
Seven of the eight wins were "a capability exists and the dispatch never reaches it", all found
by accident. Mechanical sweep: every `*_k: KernelHandle` struct field vs. any `self.<field>`
read outside init/mod. **294 handles declared; 38 with no dispatch site.** Most are legitimately
inert (MoE handles on a dense model, MLA/prefill variants, LoRA). The notable ones:

| handle | verdict |
|---|---|
| `fused_k_norm_rope_cache_write_bf16_k` | REAL dead capability — see below |
| `gemm_splitk_partial_k` / `gemm_splitk_reduce_k` | the BF16 split-K toy; already recorded dead |
| `moe_*` (8 handles) | inert — dense model |
| `mla_*`, `prefill_attn_*` | inert — other attention regimes |

### `fused_k_norm_rope_cache_write_bf16`: real, batched, never dispatched in decode — but only 0.3%
`kernels/gb10/common/fused_k_norm_rope_cache.cu:53`. Fuses K-side RMSNorm + RoPE + paged cache
write, grid `(num_tokens, num_kv_heads)` so it is BATCHED OVER TOKENS. Loaded at
`qwen3_attention/init.rs:223`; the only dispatch is the **mrope** variant in
`qwen3_attention/prefill/paged.rs:450`. Decode never uses it.
**Sized before building:** per layer at n=16 decode issues 16 q-norms + 16 k-norms + 16 ropes
+ 1 (already-batched) cache write. The fused kernel covers only K, so it removes 256 k-norm
launches/step at ~1.5 us = **~0.4 ms = ~0.3%** — Q still needs per-sequence norm and rope, and
strided Q variants do not exist. **Below the harness's measured ~0.8% resolution floor; not
worth an A/B alone.**
★ It becomes worthwhile only as part of collapsing the WHOLE attention fan-out (rope 1.2 ms +
rms_norm 0.8 ms/step), which additionally needs strided q-norm and q-rope kernels. That bundle
is ~1.5 ms (~1.1%) and is the right shape for a future session.

## 2026-07-28 — SHIPPED: attention per-sequence fan-out collapsed (rope +0.87%, q/k-norm sub-floor)

A mechanical sweep of decode paths for `for _ in 0..n` loops containing kernel launches found 17
sites. Profiling separated per-LAYER fan-out (~70/step = 64 layers, expected) from genuinely
per-SEQUENCE fan-out, which is the wasteful kind:

| kernel | instances/step | ms/step | after |
|---|---|---|---|
| `rope_forward` | 258 (16 layers x 16 seqs) | 1.18 | 16 |
| `rms_norm` (q_norm + k_norm) | 516 (16 x 16 x 2) | 0.76 | 32 |
| **total** | **774** | **1.94 (~1.5%)** | **48** |

Both needed a row-stride parameter: after the fused QKV GEMM, q/k rows sit `per_seq_qkv` apart
inside each sequence's interleaved [Q|K|V|gate] block, NOT packed at `num_*_heads*head_dim` as the
packed kernels assume. Same shape as the strided cache-write that already shipped.

- `rope_forward_strided` (kernels/gb10/common/rope.cu) — explicit q_row_stride/k_row_stride.
- `rms_norm_strided` (kernels/gb10/common/rms_norm.cu) — grid (rows_per_group, num_groups),
  groups `row_stride` elements apart. One block per row either way.

Both are bit-identical by construction: identical per-block math and reduction, only the base
address differs, and no cross-row interaction is introduced.

### Measured (C=16, 4 reps/leg, warmup discarded, kill switch as control)
| lever | ON | OFF | delta | identity |
|---|---|---|---|---|
| `rope_forward_strided` | 113.20 (113.0-113.5) | 112.22 (112.0-112.6) | **+0.87%, ranges DISJOINT** | byte-identical |
| `rms_norm_strided` | 113.28 (113.0-113.4) | 112.92 (112.8-113.0) | +0.31%, ranges TOUCH | byte-identical |

RoPE clears the measured 0.8% harness floor and matches its 1.18 ms prediction (0.9%) almost
exactly. **The q/k-norm result is SUB-FLOOR and is NOT claimed as a win** — kept default-on only
because it is byte-identical, strictly less work (484 fewer launches/step), and the point estimate
is positive with 7 of 8 reps favouring. Kill switches: `ATLAS_NO_ROPE_STRIDED=1`,
`ATLAS_NO_QK_NORM_STRIDED=1`.

**C=16 now ~113.2 tok/s** (from 112.2), ratio 0.67x vs vLLM 168.9.

★ RULE CONFIRMED: per-SEQUENCE launch fan-out is a real and cheap lever class; per-LAYER fan-out
is not (it is inherent). Separate the two before costing any "too many launches" hypothesis.

## 2026-07-28 — lm_head 716: THE TILE GEMM IS EXONERATED. The fault is in the WIRING.

A standalone repro (`crates/spark-model/examples/w4a16_lmhead_716_repro.rs`) reproduces the serve's
decode step exactly — created (non-NULL) streams, the SSM `gemm_t`/`_k64` sequence, `lm_head` gemm
alternating `_t`/`_t_k64`, cross-stream `argmax_bf16_batch` (mirroring `meta.rs:298`, which launches
on `default_stream()`), 4*M-byte D2H per step, M cycling the 2/4/8/12/16 ladder, a concurrent
prefill `w4a16_gemm_t_m128` + `memset_async` on a third stream, and the serve-exact 32-row logits
buffer. **3000 iterations of the full concurrent leg: PASS.**

Hypotheses 7-12 now dead, all by measurement:
7. launch-config/resource mismatch — NO. `w4a16_gemm_t` static smem = 19,584 B, `_t_k64` = 39,104 B,
   both under 48 KB with no `cudaFuncSetAttribute` needed; no `__launch_bounds__`; block (128,1,1)
   matches what the kernel indexes by threadIdx.
8. index-type overflow at N=248320 — NO. Largest intermediate is B_packed `(u64)(gk>>1)*N+gn`
   ~= 635.7 M, in u64 and under 2^31 anyway. grid.x=1940 is far under limits. All cp.async offsets
   are 16-B aligned given N = 15520x16.
9. logits row mismatch — NO. `buffers/sizes.rs:113` sizes logits at `min(m,32) x vocab` = 32 rows;
   the ladder tops at 16; the store is guarded `r0 < M`.
10. non-NULL streams / cross-stream argmax race / concurrent prefill GEMM+memset / M changing
    between launches — NO, 3000 iterations clean.
11. the `ATLAS_NO_DECODE_GRAPHS_MULTISEQ=1` elimination — VALID (the check is value-based,
    `decode_a2.rs:30`, not presence-based), so hypothesis 4 stands eliminated.
12. an output overrun masquerading as 716 — NO. A DELIBERATE 5.7 MB overrun (undersized C) yields a
    sticky **700 ILLEGAL_ADDRESS, not 716**. So the serve's 716 cannot be an epilogue overrun.

★ CONCLUSION: everything attributable to the tile GEMM's launch is measured clean at serve shape.
The faulting kernel is NOT the lm_head GEMM. The defect is in the wiring that was written (and
reverted) three times, or in another kernel in the step.
★ NEXT: one run with `CUDA_LAUNCH_BLOCKING=1` — the API call that returns 716 then names the
ACTUAL faulting kernel instead of the first downstream sync point. Cheap, and it ends the search.

### Side finding (latent, not currently a bug)
`crates/spark-model/src/model/meta.rs:296-298` `argmax_batch_dispatch` IGNORES its `_stream`
parameter and launches on `default_stream()`. Benign today because decode runs on the default
stream; a cross-stream race the moment that changes.

### Open question that reframes the whole lever
The reverted patch replaced `w4a16_gemv_batch16` with a tile GEMM needing a **715 MB transposed
twin**. But the GEMV itself moves 715 MB in 9.68 ms = **74 GB/s = 32% of the 230 GB/s ceiling**, and
a GEMV that reads each weight byte once should be bandwidth-bound near the ceiling. If the GEMV's
own bandwidth is fixable, the ~6 ms/step prize needs NO twin, NO extra VRAM, and NO 716. Under
investigation before the wiring is rebuilt a fourth time.

## 2026-07-28 — lm_head MICROBENCH: two plan assumptions REFUTED before any wiring

Real shape N=248320 K=5120, 100 iters after 20 warmup, floor 3109 us @ 230 GB/s
(`cargo run -p spark-model --release --example w4a16_m17_bench --features cuda,gpu-examples`;
the GEMV family and the two BF16 tile variants were ADDED to that bench for this).

| kernel | M=1 | M=4 | M=8 | M=16 | numerics |
|---|---|---|---|---|---|
| w4a16_gemm_t | 3620 | 3727 | 3661 | **3801** | FP8 E4M3 downcast of B AND activations |
| w4a16_gemm_t_m128 | 3821 | 3909 | 3844 | 3991 | FP8 |
| w4a16_gemm_t_k64 | 4074 | 4152 | 4062 | 4142 | FP8 |
| w4a16_gemm_t_m128_bf16_v2 | 5137 | 5206 | 5107 | **5192** | LOSSLESS |
| w4a16_gemm_t_m64_bf16 | 9435 | 9427 | 9435 | **9465** | LOSSLESS |
| w4a16_gemv_batch4 | 3174 | 3208 | -- | -- | incumbent |
| w4a16_gemv_batch8 | -- | -- | 4465 | -- | incumbent |
| w4a16_gemv_batch16 | -- | -- | -- | **9845** | incumbent |

★ REFUTATION 1 — `w4a16_gemm_t_m64_bf16` is the WORST of the four tile GEMMs here, 1.86x slower
than `m128_bf16_v2` and NO BETTER than the incumbent GEMV. It was chosen as primary on the theory
that its higher occupancy (4-5 vs 3 CTAs/SM) wins at low M. That occupancy edge was tuned for
PREFILL's large M; at N=248320 with M<=64 the grid is (1940,1) either way, so it buys nothing and
pays for the halved tile. **The lossless option is `m128_bf16_v2` at ~5150 us, not ~4000.** That
moves the FP8-vs-lossless gap from an assumed 0.25% of step to a real **1.05%**.

★ REFUTATION 2 — the incumbent GEMV is ALREADY OPTIMAL at low M. **`batch4` runs at 3174 us =
226 GB/s = 98.3% of peak = 1.02x the roofline floor.** It is not a broken kernel at low M; there is
NOTHING to win there, and BOTH tile GEMMs REGRESS it (lossless by ~1960 us/call, FP8 by ~450).
A C=1 step is ~39 ms, so the lossless kernel with no threshold is a several-percent regression at
the ONE concurrency where Atlas already beats vLLM 1.79x (25.4 vs 14.2). Crossover is M ~= 8.
=> "repack-and-replace, no threshold, all M" would trade C=1 away to win C=16.

★★ HOW TO READ THE GEMV COLUMNS — `w4a16_gemv_batchm_impl<MAX_M>` bounds its row loop
`for (int t = 0; t < MAX_M; t++)` at COMPILE TIME with no runtime clamp
(kernels/gb10/common/w4a16_gemv.cu:490). A `batch4` timing at M=16 therefore computes only 4 rows
and is INVALID WORK — it looks like a 98%-of-peak win at M=16 and is nothing of the kind. Each
batchN kernel is meaningful ONLY at M<=N; the other cells are struck out above. Generalises: for
any template-bounded kernel, a sweep past the bound measures a DIFFERENT, SMALLER problem.

### FP32-logits concern: CLOSED, non-issue for this migration
`w4a16_gemv_logits` (w4a16_gemv.cu:270) writes FP32 with the comment "FP32 logits are critical for
sampling quality", and every tile GEMM writes BF16 — so this looked like a numerics regression
beyond accumulation order. It is not. That kernel is dispatched ONLY at `impl_a3.rs:260`, under an
`fp32` flag, on the single-row `w4a16_gemv` path (added to close a 0.125-logit BF16 tiebreak flip
behind Gemma-4-31B's creative-collapse stop-word loop). **The multi-seq decode head being migrated
(`decode_a2.rs:437-499`) already writes BF16** via `w4a16_gemv_batchm`. The FP32 path is a
different kernel on a different code path and is not touched.

### Consequence
A threshold is MANDATORY, which forces one of:
(a) resident twin (+715 MB, ~1% of the 84 GB budget at util 0.70) + M-dependent dispatch between
    two tensors — precisely the configuration identified as the prime 716 suspect;
(b) a NEW transposed-layout GEMV (`w4a16_gemv_t_batchm`) so ONE layout serves all M. Transposed B
    is [K/2, N], so at fixed k a warp's 32 lanes read 32 CONSECUTIVE bytes across n — naturally
    coalesced, which a row-major per-n GEMV cannot be. Would allow repack-and-replace (no twin, no
    716 class) AND keep 98%-of-peak at low M AND get the tile GEMM at M>=8.
Decision pending.

## 2026-07-28 — ★★★ ROOT CAUSE OF THE CUDA 716, AFTER THREE FAILED ATTEMPTS

**The transposed lm_head weight's ROW STRIDE IS THE VOCAB SIZE, and this checkpoint's vocab is
248077 — an ODD number. The tile GEMM loads B with 16-byte `cp.async`, which REQUIRES a
16-byte-aligned source address. Only 1 k-row in 16 is aligned; the rest fault with
CUDA_ERROR_MISALIGNED_ADDRESS.**

```c
// kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:392
cp_async_pred_16(&smem_Bp[(buf)][kp][ns],
    &B_packed[(unsigned long long)(gke >> 1) * N + gns], (gke + 1 <= K) && (gns + 15 < N));
```
`gns` is ALWAYS a multiple of 16 (`ns = (threadIdx.x & 7) << 4`, `cta_n = blockIdx.x * 128`), so
alignment reduces entirely to the row stride `N`:

| N | source | N mod 16 | k-row byte offsets mod 16 | verdict |
|---|---|---|---|---|
| **248077** | centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf (**what the benchmark actually serves**) | **13** | 0,13,10,7,4,1,... | **15 of 16 rows MISALIGNED** |
| 248320 | nvidia/Qwen3.6-27B-NVFP4 (the bench + the standalone repro) | 0 | 0,0,0,0,... | cannot fault |

### Why this defeated twelve hypotheses and three rebuild attempts
- **The standalone repro passed 3000 iterations** because it used N=248320, on which the bug is
  STRUCTURALLY IMPOSSIBLE. The kernel was never exonerated — it was tested on a shape that
  excludes the defect. ★ The "dims verified against the checkpoint, `lm_head.weight [248320,2560]`"
  note came from the WRONG CHECKPOINT.
- **All three attempts failed identically across kernel variants** because every variant loads B
  through the same `cp_async_pred_16` with `N` as the stride.
- **Single requests always passed byte-identically** because `padded_n <= 4` dispatches the
  row-major GEMV and never touches the transposed tensor. Not "M=1 is aligned" — the faulting
  kernel never ran at all.
- **compute-sanitizer was CLEAN and the error became 719 under it** because there is NO
  out-of-bounds access. Every address is inside a valid allocation; they are merely unaligned.
  That is exactly the class memcheck cannot attribute to an address.
- It was invisible to code review because NOTHING IN THE CODE IS WRONG. The defect is an UNSTATED
  INVARIANT between a checkpoint's vocab size and a kernel's load width.

★★ RULE: when a microbenchmark and a serve disagree, DIFF THE SHAPES FIRST — not the wiring. A
benchmark constant that "matches the model" from a different checkpoint of the same family will
silently make the reproducer immune to the bug being chased. Assert real runtime dims IN the
repro; never hardcode them from docs.
★★ RULE: any kernel using `cp.async`/vector loads carries an ALIGNMENT PRECONDITION ON ITS
STRIDES. Odd/unaligned N is a legal tensor shape; the kernel must pad or the caller must.

### THE FIX (required for ANY lm_head tile-GEMM path, twin or repack-and-replace)
Give the transposed weight a row stride padded to a multiple of 16 (128 for tiling):
`align_up(248077, 128) = 248192`, zero-filling the pad columns. Then either
(a) add an explicit `ldb` parameter so B uses the padded stride while C keeps the true vocab
    stride (C stores are SCALAR 2-byte and guarded — `C[r0*N+c0] = ...` at :539-542 — so C is NOT
    the problem and needs no padding), or
(b) pass the padded N and pad the logits buffer + argmax count as well (wider; risks argmax
    selecting a pad column, whose zeroed logit can beat real negative logits).
(a) is the smaller and safer change.

### Twin is DEAD on VRAM — measured, not estimated
The 681 MB twin drops the KV pool **4757 -> 2957 blocks** (~1800 blocks, far more than the twin's
own size) and the serve HARD-FAILS preflight:
```
KV cache can hold at most 11 concurrent sequence(s) at --max-seq-len=4096, but --max-batch-size=16
was requested. KV pool has 2957 block(s) of 16 tokens each; each sequence needs 256 block(s).
```
★ My interim estimate of "4042 blocks, 1.3% short, recoverable with a util bump" was WRONG and too
generous. The original KV objection stands: NO RESIDENT TWIN ON THIS BOX. The lm_head tile GEMM
must go via repack-and-REPLACE (single layout), which also needs a transposed-layout GEMV for
`padded_n <= 4` (where the row-major GEMV measures 98.3% of the roofline and must not be lost).

### Status
Twin implementation is BUILT and reverted-in-place (kill switch `ATLAS_NO_LMHEAD_TGEMM=1`
defaults ON => must be left OFF/removed). Not shipped. Next: the padded-stride fix + the
transposed GEMV, per the repack-and-replace plan.

## 2026-07-28 — ★ SHIPPED: lm_head tile GEMM, +5.50% at C=16. The 716 was an ALIGNMENT bug.

The padded stride ELIMINATES the 716 outright. Root cause confirmed by experiment, not just analysis:

```
lm_head transposed twin: vocab=248077 -> padded stride=248192 (vocab%16=13), tile GEMM active
CUDA 716 / misaligned count: 0
```

### Measured (C=16, 4 reps/leg, warmup discarded, kill switch as control)
| leg | mean | range |
|---|---|---|
| tile GEMM ON | **119.32** | 119.0-119.6 |
| OFF (GEMV) | 113.10 | 112.9-113.3 |
| **delta** | **+5.50%** | **ranges DISJOINT** |

Full sweep, default-ON: C=1 **25.4** · C=2 25.2 · C=4 48.25 · C=8 **71.2** (+0.85%) · C=16 **119.1** (+5.2%).
NO REGRESSION anywhere. C=1/2/4 are BYTE-IDENTICAL (`bf3a0b07...`, same hash both legs) because
`padded_n <= 4` stays on the GEMV — which measures 98.3% of the memory roofline and must not be lost.
Coherence + tool-call smoke PASS on the tile-GEMM path.

vs vLLM: C=1 **1.79x WIN** · C=2 0.91x · C=4 0.91x · C=8 0.72x · C=16 **0.71x** (from 0.67x).

### What shipped
- `w4a16_gemm_t` gains an `ldb` (transposed-B row stride) parameter; B loads and their guards use
  `ldb`, stores still use `N`. Bounding loads by `ldb` also FIXES a latent bug: bounding by `N`
  silently dropped the real columns in the final 16-wide group whenever N % 16 != 0 (13 columns here).
- `w4a16_gemm_n128_ldb` wrapper; the existing `w4a16_gemm_n128` DELEGATES to it with `ldb = n`, so
  all 21 existing call sites are untouched (SSOT: one launch site).
- `transpose_concat_for_gemm_padded` + a shared `transpose_impl` (both public helpers now route
  through one implementation).
- `lm_head_nvfp4_t: Option<(QuantizedWeight, u32)>` — ADDITIVE twin, never replaces or aliases the
  original, so `draft_lm_head_nvfp4`'s copy stays valid. Built once, immutable.
- Dispatch at `padded_n >= 5` only. Default ON, kill switch `ATLAS_NO_LMHEAD_TGEMM=1`.

### ★★ CORRECTION: the twin does NOT cost KV. My earlier "4757 -> 2957 blocks" was CONTAMINATED.
With the twin active the pool reads **4759 blocks vs 4757 without it** — no measurable impact, and
preflight passes at `--max-seq-len 4096 --max-batch-size 16`. The 2957-block reading came from a
serve that computed its self-relative KV budget (`baseline-free - free-now`) while a STRAY
CONTAINER still held ~94 GB. I built two successive wrong conclusions on that number ("twin is
dead on VRAM", "repack-and-replace is mandatory"). BOTH ARE WITHDRAWN.
★ RULE: `nvidia-smi --query-compute-apps` BEFORE trusting any self-relative memory budget. A
stray serve does not just slow things down — it silently corrupts the KV sizing arithmetic, and
that number then looks like a hard architectural constraint.
★ Also withdrawn: an intermediate estimate of "4042 blocks, 1.3% short, recoverable with a util
bump" — that was arithmetic on the contaminated figure.

### Accuracy debt (recorded, NOT yet measured — accuracy embargo still in force)
`w4a16_gemm_t` dequants B to FP8 E4M3 AND downcasts activations BF16->FP8 for
`mma.m16n8k32.e4m3`. So at `padded_n >= 5` the lm_head logits carry FP8 activation precision and a
different accumulation order. `padded_n <= 4` is unaffected and byte-identical.
- Coherence + tool-call smoke PASS; no BFCL/IoU run (embargoed until vLLM parity).
- The lossless alternative `w4a16_gemm_t_m128_bf16_v2` measures 5192 us vs 3801 us at M=16, i.e.
  ~1.05% of step — the fallback if a flip-gate later rejects FP8.
- BFCL + IoU re-validation MANDATORY before any accuracy claim or external quote.

### lm_head FP8 accuracy debt — CHARACTERIZED (not a substitute for BFCL)
8 concurrent prompts (padded_n >= 5, so the tile GEMM IS active), temp 0.0 seed 42, tile GEMM vs
`ATLAS_NO_LMHEAD_TGEMM=1`, same binary:

| | result |
|---|---|
| fully byte-identical | **2 / 8** |
| diverged | 6 / 8 |
| divergence point | 1%-71% into the response |
| quality of divergences | **benign paraphrase in every case** |

Examples: "employees sorted by salary" vs "students sorted by name" (both valid illustrations);
"i.e., two keys hash to the same index" vs "two keys hash to the same index"; "0.1 + 0.2 != 0.3"
rendered with vs without code markup. Both branches stay coherent, correct and complete — no
repetition, no collapse, no truncation, no degradation. This is the expected signature of temp-0
tiebreak flips propagating (cf. `spec_not_output_neutral`).

★ This does NOT establish accuracy parity — only BFCL/IoU can, and those stay embargoed until
Atlas >= vLLM at C=1..16. It bounds the RISK, it does not discharge the DEBT.
★ MEASUREMENT TRAP hit while doing this: the first comparison parsed the capture line-by-line, so
it only compared FIRST LINES and reported "7/8 identical". The hashes (computed over full
responses) said 6/8 DIFFER. When a hash comparison and a text diff disagree, the DIFF is the one
that is probably broken.

## 2026-07-28 — ★ REGRESSION I INTRODUCED, CAUGHT BY AUDIT: strix-hip missed the `ldb` fix

The lm_head twin is built by PLATFORM-AGNOSTIC Rust and is DEFAULT-ON, but only the GB10 CUDA
`w4a16_gemm_t` had gained `ldb`. `kernels/strix/` is a SYMLINK to the GB10 file and inherited the
fix; **`kernels/strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu` is a real separate copy and did not.**
On the Windows/HIP gfx1151 build (PR#353), which serves this exact NVFP4 checkpoint, decode at
padded_n>=5 would launch `w4a16_gemm_t` against a twin built with stride 248192 while the kernel
strides by N=248077 — shearing every row past the first and producing GARBAGE LOGITS, **silently,
with no fault**, because RDNA tolerates the misalignment that traps on CUDA as 716.

FIXED: `ldb` ported to the strix-hip copy, with a comment binding it to the CUDA original.
★ NOT COMPILE-VERIFIED on this box (no ROCm here) — the HIP build must be run before that
platform is trusted. Flagged, not assumed.

Added a load-time WARN when `stride != vocab`, naming the `ldb` requirement — this is the
signal that would have caught the drift immediately. Verified firing:
`lm_head twin uses a PADDED stride (248192 != vocab 248077): this target's w4a16_gemm_t MUST
accept the `ldb` argument...`. Post-fix C=16 119.9 tok/s, 0 errors.

★★ RULE: a kernel change made for ONE target is not done until every OTHER target's copy of that
file is checked. Symlinked targets inherit; whole-file copies DRIFT. This is the same shadow-drift
failure mode that left 27B's multi-seq GDN kernels uncompiled (`shadowed_kernels_null`), except
here the drift produced SILENT WRONG ANSWERS on the other platform rather than a null handle.
★ Fleet check: every other served vocab (248320, 200064, 151936, 131072, 262144, 129280, 128896,
100352) is a multiple of 128, so stride == N and `ldb` is a no-op — 248077 is the fleet's ONLY
unaligned N, and only lm_head carries it. Other model dirs' 9-arg `w4a16_gemm_t` receiving a 10th
argument is benign (the launch API reads only declared params).

## 2026-07-28 — FFN MMQ: my 69% figure was WRONG. It is 76%, and the prize is ~5-6 ms not ~18 ms.

★ MY ARITHMETIC ERROR: I divided (mmq16 + mmq32) TIME by mmq16-ONLY BYTES. mmq32 is not overhead
on the same 9.63 GB — it runs on steps where m is 17..32 (MTP-verify steps at C=16) and moves its
OWN ~50 MB/call. Per-CALL is the artifact-free metric: 54.8 ms / 192 calls = 285.4 us/call over
50.14 MB = **175.7 GB/s = 76.4% of achievable**, which matches the earlier gridX-split nsys
(175.6 down / 173.2 gate-up) exactly.

Shape audit (the lm_head-class check): `nvfp4_mmq.cu` exists ONLY in the 27B dir (no common/ twin
to shadow), and N/K are RUNTIME-derived from the served tensors, not hardcoded — the `_nc`
variant is selected only when `n % 128 == 0`, and nsys gridX 136/40 confirms N=17408/5120 exactly.
**So the sibling-checkpoint failure mode that invalidated the lm_head verdict does NOT apply here.**

★ CORRECTION to the earlier entry's mechanism: STATE.md:955-956 attributed the gap to "dequant/
scale ALU overlap with the cp.async weight stream". **There is no cp.async in this kernel at all**
— zero hits for `cp.async|memcpy_async|__pipeline` in the vendored `q4k_vendor/mmq.cuh`. Weight
loads are scalar 4-byte `ld.global`. The real mechanism is that the inner loop is PHASE-SERIAL:
`[load] -> sync -> vec_dot(MMA) -> sync -> [load] -> sync -> vec_dot -> sync`, four barriers per
512-K iteration, with DRAM IDLE during both vec_dot phases and the epilogue.

Confirmed dead (do not re-litigate): stream-K/split-K for the down under-fill (down gridX=40 at
284.3 us vs gate/up gridX=136 at 282.3 us — within 0.7% at identical bytes, so 40 CTAs already
saturate LPDDR5X); occupancy hints (+0.27%, null); M-tile size (flat — kernel is weight-bound).

Remedies, ranked: (1) cp.async 2-stage double-buffer in `mul_mat_q_process_tile` — smem
2x(38.9+3) KiB ~= 84 KiB <= 99 KiB/CTA, numerics UNCHANGED, est. 4-5 ms/step; (2) SoA weight
relayout in `atlas_nvfp4_repack` (Atlas owns the layout) so the x-tile fill vectorizes 16-B,
bit-identical, ~1-2 ms and the natural enabler of (1); (3) fuse gate+up into one N=34816 launch,
bit-identical, ~0.5-1.5 ms. Upstream llama.cpp has NO faster variant to pull — mainline MMQ has
the identical phase-serial loop and no cp.async either, so this would be Atlas-original work.

## 2026-07-28 — ★ W4A4 IS DEAD WEIGHT AT DECODE (external evidence, converging)
- QServe/QoQ (arXiv 2405.04532): W4A4 beats W4A8 only above ~78 concurrent sequences. We run 1-16.
- APEX4 (arXiv 2606.08761): all W4A4 wins at M>=64; concedes Marlin W4A16 is fastest at small batch.
- QuaRot: its decode win is KV4 memory, not A4 GEMM.
- ★ MEASURED ON GB10 ITSELF — llama.cpp PR #22196 (Blackwell native NVFP4, benchmarked on a DGX
  Spark by NVIDIA's own engineer): prefill 506->611 t/s (+21%), decode 12.01->**11.91** t/s
  (UNCHANGED/slightly worse), and the W4A4 tensor-core path had WORSE perplexity (4.6577) than the
  W4Q8 fallback (4.6283).
At decode we move ~9.63 GB of weights and ~0.5 MB of activations per step — activation precision
is INVISIBLE in the traffic. Its only decode "win" is FP4-MMA issue rate, irrelevant when
bandwidth-bound. => Candidate lever: keep A4 for PREFILL, run DECODE with A8/A16 activations
against the same NVFP4 weights — same bandwidth, same speed, strictly better numerics. Must be
gated by a measured iso-speed A/B before folding.

## 2026-07-28 — m_dispatch sweep: NONE REMAIN on the 27B C=1..16 decode path
All four fixed instances verified in place (FFN, attention projections, GDN mixer, lm_head).
Every M>8 GEMM on the campaign model now rides a tile kernel. Residuals are in OTHER configs:
(1) `impl_a3.rs::lm_head_batched` has no twin arm at M>8 (falls to base `w4a16_gemm`, ~19.3 ms at
    M=16) — reached only by DFlash wide-verify (gamma=16 => M=17) and `prompt_logprobs`, NOT the
    MTP K=4 production path. Fix is mechanical: mirror decode_a2.rs:470-486.
(2) FP8 lm_head is still a per-row GEMV loop at every M (decode_a2.rs:441-452) — FP8 flagship
    configs only (dgx3 256k, 80B FP8), not this campaign's NVFP4 model.
Landmines recorded (no live exposure, all currently aligned): `mamba2_in_proj_size` adds a raw
head count so a checkpoint with `mamba_num_heads % 16 != 0` would go unaligned into a 9-arg
kernel; `tp_shard.rs` never checks `local_out % 16`; the FP8 tile family guards B with
`(kb)+31 < K` so K % 32 != 0 would silently DROP the last K-chunk. The correct defensive pattern
already exists in-tree at `w8a16_gemm_t_m128.cu:156-165` (runtime `(N & 15) == 0` check + scalar
fallback).

## 2026-07-28 — ★ RETRACTION: "W4A4 is dead weight at decode" is WRONG AT OUR M
I recorded (above, same night) that 4-bit activations buy nothing at decode and proposed A8/A16
decode activations as a free accuracy reclaim. **That lever is already REFUTED by an A/B in this
very file** (2026-07-27, STATE.md:95-101): the bf16-activation path against the same NVFP4 weights
is `ATLAS_NO_FFN_NVFP4_MMQ`, and it measures **MMQ off 55.0/54.8/54.8 (mean 54.9) vs MMQ on
61.4/59.1/61.2 (mean 60.6) = +10.4% for MMQ, ranges NON-OVERLAPPING, output identical.**
So dropping W4A4 at decode is a **10.4% REGRESSION**, not a free accuracy reclaim.

★★ WHY I GOT IT WRONG — "DECODE" IN THE LITERATURE MEANS M=1. QServe's ~78-sequence crossover,
APEX4's M>=64 framing, and llama.cpp PR #22196's GB10 measurement (12.01 -> 11.91 t/s) are ALL
single-stream decode. Our C=16 decode is **M=16**, a different regime, where the FP4 MMA path
wins on this hardware. A batched-serving campaign must translate every "decode" claim in a paper
into the M it was measured at BEFORE importing its conclusion.
★ Lesson compounding: I ALSO nearly imported the same class of error from the m17 bench, whose
per-kernel numbers overstate by ~1.5x from L2 reuse (STATE.md:685-690). External and microbench
evidence both need their regime checked against the in-model measurement before they move a
decision.
The accuracy question stands but is now a TRADE, not a freebie: W4A4 decode activations are
throughput-justified at +10.4%; revisit only at vLLM parity, in the accuracy-debt ledger.

## 2026-07-28 — ★ SHIPPED: SoA weight layout for the FFN MMQ tile load (+4.6% C=16)

`atlas_nvfp4_repack` emitted an array of 36-byte `block_nvfp4`, which INTERLEAVES 4 scale bytes
with 32 nibble bytes. The tile loader therefore issued **NINE 4-byte global loads per block**
(8 qs + 1 d). Splitting each row into `[qs: bpr*32][d: bpr*4]` makes the 32 qs bytes contiguous:
two 16-byte loads + one 4-byte load. **9 global ops -> 3 at identical total bytes.**

The SHARED tile layout is unchanged, so `vec_dot`/MMA are untouched and the output is bit-identical
(VERIFIED: identity sha `bf3a0b07...` unchanged, not assumed).

| C | before | after | delta |
|---|---|---|---|
| 1 | 25.4 | 25.45 | flat |
| 2 | 25.2 | 25.3 | flat |
| 4 | 48.25 | 48.55 | +0.6% |
| 8 | 71.2 | **73.85** | **+3.7%** |
| 16 | 119.3-119.9 | **125.4** (125.8/125.4/125.1) | **+4.6%, DISJOINT** |

vs vLLM: C=1 **1.79x WIN** · C=2 0.91x · C=4 0.92x · C=8 **0.75x** · C=16 **0.74x** (from 0.71x).

★ THE BUG THAT BROKE THE FIRST ATTEMPT (produced garbage `!</think>`): **`kbx0` is a LINEAR block
index that ALREADY FOLDS IN THE CTA'S ROW OFFSET** (`offset_x = ... + it*mmq_y*stride_row_x`,
mmq.cuh:3640) — it is NOT a pure k-offset. A per-ROW layout must decompose it back:
`row0 = kbx0 / bpr`, `kb0 = kbx0 - row0*bpr`, one div+mod hoisted out of the i-loop.
★ RULE: before changing a memory layout, determine whether the consumer's index is LINEAR or
already DECOMPOSED. A linear index silently mixes the dimensions you are trying to separate.

Alignment: row stride `bpr*36` is 16-B aligned iff `bpr % 4 == 0` i.e. `K % 256 == 0`; both live K
qualify (5120->80, 17408->272) and MMQ already requires K % 512 == 0. Safe because Atlas exposes
NO MMVQ entry point, so the MMQ kernels are the buffer's only consumer (`vecdotq.cuh`'s nvfp4 path
is unreachable). No independent kill switch — the layout is internal and bit-identical; the escape
hatch is the existing `ATLAS_NO_FFN_NVFP4_MMQ`.

★ THE ESTIMATE WAS 4x LOW: this was sized at "~1-2 ms, mostly subsumed by the cp.async pipeline".
It delivered ~5.5 ms ALONE — i.e. the load-ISSUE path, not load LATENCY, was the dominant limiter.
Implied post-fix MMQ bandwidth ~195 GB/s = ~86% of the 226 GB/s achievable at a 9.6 GB working set,
i.e. already at w4a16 parity. **A1 (cp.async double-buffer) must be RE-SIZED against a fresh
profile before it is built — its remaining headroom is far smaller than the original 4-5 ms.**

## 2026-07-28 — NEXT LEVER, SPECIFIED AND READY: deepen the `w4a16_gemm_t_k64` pipeline

NOT STARTED — specified deliberately rather than half-built (a partial vendored-kernel refactor is
worse than nothing). Everything below is measured or computed; it should be directly executable.

### The target
`ssm_out_proj` (x48/step) + `attn_o_proj` (x16/step), both **N=5120 K=6144**, on
`w4a16_gemm_t_k64`. 13.4 ms/step, 10.7% of GPU, and the WORST efficiency of any live kernel.
Per call: B_packed 15.73 MB + scales 1.97 MB + A/C ~0.36 MB = **18.05 MB => 79.9 us floor at
226 GB/s**; 64 calls => **floor 5.1 ms/step vs 13.4 measured = 38% efficient in-model.**
Reaching 75% saves **~6.6 ms/step = ~5.3% at C=16** — larger than the MMQ cp.async lever's entire
remaining prize.

### ★ THE MECHANISM (efficiency is MONOTONE IN grid.x across the SAME kernel family)
| grid.x | shape | % of achievable |
|---|---|---|
| 40 | out_proj / o_proj (N=5120) | 47-50 |
| 112 | attn_qkv (N=8192) | 66 |
| 128 | ssm_qkvz (N=16384) | 73 |
| 1938 | lm_head (N=248077) | 83 |
Same kernel, same K-loop, same scale layout — ONLY occupancy varies. At N=5120 the grid is
(40, 1) = **40 CTAs on 48 SMs = exactly 1 CTA/SM** (128 threads resident, ~8% occupancy). The k64
main loop is `ISSUE(nxt) -> commit -> MMA(cur) -> cp_async_wait_all -> sync -> DEQUANT(nxt) -> sync`:
only ONE load group is ever in flight and `wait_all` drains it every step, so during DEQUANT
(32 LUT+cvt iterations between two barriers) **every active SM has ZERO outstanding loads**.
lm_head has the identical serial phase but a co-resident CTA covers it. That is the whole gap.
Pure SM under-fill only accounts for 48/40 = 1.20x; the other ~1.65x is exposed latency.

★ The campaign's two prior refutations DO NOT TRANSFER: (a) the 40-vs-136-CTA under-fill null was
measured on MMQ, whose 256-thread/2-resident pipeline self-hides latency; (b) split-K was refuted
for projections AND would break bit-identity (FP32 accumulation order). Deepening the pipeline
attacks the same mechanism intra-CTA, BIT-IDENTICALLY.

### ★ SMEM CORRECTION — a naive 3-stage does NOT fit
With M_TILE 64, K_STEP_T64 64, PAD_T64 8, N_TILE_LG 128, BP_PAD 16:
- 2 stages (today): A 18432 + Bp 8704 + Bs 1088 + B_fp8 10240 + LUT 64 = **37.6 KB**
- naive 3 stages: **51.4 KB — EXCEEDS the 48 KB static limit**, would need `cudaFuncSetAttribute`
  + `extern __shared__`, which the launch wrapper does NOT do.
- ★ **A[2] + Bp/Bs[3] = 43.2 KB — FITS.** Only the WEIGHT tiles need the third stage: `A[cur]` is
  free the moment `MMA(cur)` clears its barrier, so A can stay double-buffered.

### Schedule (bit-identical: same MMA order, same operands, only load timing moves)
per step i: `MMA(cur)` -> sync -> `ISSUE_LOADS(A[cur], Bp/Bs[(i+2)%3] <- kb+2)` -> commit ->
`cp_async_wait_group(1)` -> sync -> `DEQUANT((i+1)%3)` -> sync.
Loads for kb+2 stay in flight THROUGH DEQUANT — the memory pipe never drains.
The existing `K64_ISSUE_LOADS(buf,kb)` / `K64_DEQUANT(buf)` / `K64_COMPUTE_MMA` macros already take
`(buf)` as a parameter, so the change is mostly the smem declarations + loop reordering; `[2]` ->
`[3]` on `smem_Bp_k64`/`smem_Bs_k64` only. If only `cp_async_wait_all` is wrapped in-file, add the
`cp.async.wait_group 1` asm one-liner.

### Plan
1. Add `w4a16_gemm_t_k64_p3` BELOW the existing kernel (ADDITIVE — the old one keeps working if
   this is abandoned). Move the macro `#undef`s below it.
2. Resolve the new handle at `qwen3_ssm/init.rs:122` and `qwen3_attention/init.rs:419`, falling
   back to `w4a16_gemm_t_k64` under **`ATLAS_NO_K64_PIPELINE3`** (PRESENCE check — `=0` is NOT off).
   Call sites untouched.
3. Gates: build with `ATLAS_TARGET_MODEL=qwen3.6-27b` and **md5 the binary vs previous** or the A/B
   is fake; add the `_p3` row to `examples/w4a16_parity_microtest.rs` and ASSERT bit-identity;
   `w4a16_m17_bench` on the two out_proj shapes (relative only — that bench overstates ~1.5x from
   L2 reuse) expecting 153/164 us -> ~110; then coherence smoke + C=16/C=8 >=3 reps per leg vs the
   kill switch, ranges must not overlap; confirm with nsys that k64 ms/step drops 13.4 -> ~8.
4. Stretch only after D ships: A3, fuse FFN gate+up into one N=34816 MMQ launch (bit-identical,
   ~0.5-1.5 ms).

### Also re-sized tonight
MMQ cp.async (A1) is now a ~2.0-4.6 ms lever, NOT 4-5: post-SoA `mmq16_nc` runs 257.0 us/call =
**195.1 GB/s = 86% of achievable**, and `mmq32_nc` already hits 203.4 GB/s = 90% on the SAME
weights with more work per byte — so what remains there is largely FIXED PER-CALL overhead, which
a load pipeline does not fix. Do D first.

## 2026-07-28 — ★ SHIPPED: 3-deep weight pipeline in `w4a16_gemm_t_k64` (+1.81% C=16)
C=16 **124.62 -> 126.88** (4 reps/leg, 126.6-127.1 vs 124.5-124.7, DISJOINT), byte-identical
(`bf3a0b07...` on both legs). Default ON, kill `ATLAS_NO_K64_PIPELINE3` (PRESENCE, not value).
All three handle sites resolve through ONE `layers::k64_kernel` helper (SSOT).

★ MEASURED +1.81%, vs the ~5.3% the drain model predicted. The MECHANISM was right (disjoint,
byte-identical, and it moved the needle) but the SIZING was optimistic — the pipeline drain is a
real component of the 40-CTA shapes' inefficiency, NOT the whole of it. Remaining out_proj gap is
still open; do not assume another pipeline stage recovers it.

Implementation notes for whoever touches this next:
- smem forced the design: naive all-3-stage = 51.4 KB, OVER the 48 KB static limit (would need
  `cudaFuncSetAttribute` + `extern __shared__`, which the launch wrapper does NOT do). Only the
  WEIGHT tiles are tripled — `A[i&1]` is free once `MMA(i)` clears its barrier — giving 43.2 KB.
- needed a `cp.async.wait_group 1` helper; only `wait_group 0` (wait-all) existed in-tree.
- TRAP hit twice while generating the variant: the file defines the DEQUANT macro TWICE (a
  `__SCALE__`-guarded pair). A blanket rename fixed only the first and the build failed on the
  second. Grep for ALL definitions of a macro before transforming it.

### Running total for the night
C=16 **113.1 -> 126.88 tok/s (+12.2%)**, ratio 0.71x -> **0.751x** vs vLLM 168.9.
Shipped: strided RoPE +0.87% · lm_head tile GEMM +5.50% · FFN MMQ SoA +4.6% · k64 pipeline +1.81%.
Plus one correctness fix (strix-hip `ldb`, silent garbage logits at C>=5 on that platform).

## 2026-07-28 — verification gap closed + a harness breakage I introduced

★ **The `ldb` parameter silently broke two dev harnesses.** `w4a16_parity_microtest.rs` and
`w4a16_m17_bench.rs` build their OWN `KernelLaunch` with the old 8-argument list, so after
`w4a16_gemm_t` gained `ldb` they passed an uninitialized 9th param. PRODUCTION was unaffected (it
routes through `w4a16_gemm_n128`, which delegates with `ldb = n`), but every future run of those
tools would have been garbage — and the bench is the campaign's main per-shape instrument.
Both fixed by passing `n` as the packed-case stride.
★ RULE: when adding a kernel parameter, grep for `KernelLaunch::new` on that kernel — the ops
wrapper is NOT the only launcher. Test harnesses bypass it by design.

★ **`w4a16_gemm_t_k64_p3` added to the parity oracle and PASSES**, with cos/max|Δ| IDENTICAL to its
`_k64` parent on every shape (attn_o 0.9991119/0.09961, attn_q 0.9991262/0.10156, ffn_down
0.9991187/0.17969). That is kernel-level bit-identity evidence, strictly stronger than the
single-prompt output hash the lever originally shipped on. Full gate: PASS.

## 2026-07-28 — ★ SHIPPED: 3-deep weight pipeline for `w4a16_gemm_t` (+2.57% C=16)
Same change as the k64 sibling, applied to TWICE the surface (26.5 ms/step vs 13.4).
C=16 **126.47 -> 129.72** (4 reps/leg, 129.6-129.9 vs 126.0-126.8, DISJOINT), byte-identical
(`bf3a0b07...` both legs). Default ON, kill `ATLAS_NO_TGEMM_PIPELINE3` (PRESENCE). Three qwen3
handle sites route through one `layers::tgemm_kernel` helper, which falls back automatically on
targets that do not ship `_p3`.

Sweep: C=1 25.45 · C=2 25.45 · C=4 48.55 · C=8 **75.6 (0.764x)** · C=16 **129.35-130.2 (0.768x)**.

### ★ NULL (with a mechanism): FOUR stages is a 5% REGRESSION — depth 3 is the optimum
Extending the same kernel to a 4-deep pipeline measured **120.3 vs 126.5 for the 2-stage parent**,
i.e. WORSE than doing nothing, and far worse than 3-deep's 129.7. Cause: a 4-deep weight pipeline
also forces A to >=3 deep (step i+3 would otherwise overwrite an A buffer that MMA(i) has not read
— caught BEFORE measuring, by index analysis). Making A 4-deep takes the tile from 5,120 to 20,480
B and total smem from ~22 KB to ~37 KB, which drops the LARGER-grid shapes (qkvz 128 CTAs, fused
QKV 112) from 2 CTAs/SM to 1 — destroying exactly the co-residency that was covering their latency.
★ RULE: pipeline depth trades in-flight bytes against OCCUPANCY. On a kernel whose bigger shapes
rely on co-residency, past ~3 stages the occupancy loss dominates. Do not assume deeper is better;
it is a tunable with an interior optimum, and here the optimum is 3.

### Running total for the night
C=16 **113.1 -> 129.7-130.2 tok/s (+15%)**, ratio 0.71x -> **0.768x**. C=8 71.2 -> 75.6 (0.764x).
Shipped, ALL byte-identical except lm_head: strided RoPE +0.87% · lm_head tile GEMM +5.50% ·
FFN MMQ SoA +4.6% · k64 3-deep pipeline +1.81% · tgemm 3-deep pipeline +2.57%.

### lm_head routed through `tgemm_kernel` — MEASURED NEUTRAL (kept for consistency)
lm_head was the last site resolving `w4a16_gemm_t` directly, so it stayed on the 2-stage parent.
Routing it through the resolver measures **+2.55% vs +2.57% before** — i.e. contributes NOTHING,
which is the predicted result and a check on the grid.x model: at 1938 CTAs lm_head already has the
co-residency that covers the parent's drain. Kept because it is byte-identical, cost-free, and
removes the last bypass of the resolver. ★ A neutral result that CONFIRMS a model is worth keeping
and recording; it is evidence, not a wasted experiment.

## 2026-07-28 — ★★★ THE STRUCTURAL LEVER, CONFIRMED: batched speculative decoding

External scan (vLLM V1 source in the vendored checkout + measured literature) confirms plainly:
**everyone runs spec decode across the FULL continuous batch; nobody gates it to one sequence.**
The industry answer to "spec decode at batch" is SHRINK K AS BATCH GROWS, not turn it off.
vLLM's documented EAGLE3 ladder is **K=5 for batch 1-16** — C=16 is "full speculation on" territory
even on H100. The "batch size 1 only" folklore is TRT-LLM's LEGACY ENGINE FLOW, a dead code path.

vLLM V1 shape (all in `scratchpad/vllm-src/`):
- verification IS the normal decode forward — draft tokens appended, ragged varlen batch
  (`gpu_model_runner.py:2743` `_calc_spec_decode_metadata`, mixed `[3,0,2,0,1]` draft lengths)
- batched Triton rejection sampler over `[batch, k]` (`v1/sample/rejection_sampler.py`)
- per-request rewind is scheduler arithmetic (`scheduler.py:1547-1571`)
- ★ **GDN ROLLBACK IS SOLVED UPSTREAM**: `gdn_attn.py` allocates **num_spec+1 recurrent-state slots
  per sequence**; the FLA kernel writes a checkpoint per draft position INLINE
  (`INPLACE_FINAL_STATE`, `fla/ops/fused_recurrent.py:104-166`); next step loads slot
  `num_accepted-1`. **Rollback is an index lookup.** This eliminates EXACTLY the per-token FLA
  overhead that killed the TRT-LLM ngram v21 experiment — checkpoints are a byproduct of one fused
  kernel, not extra passes.
- full CUDA graph on the uniform 1+k step; dynamic K-vs-batch ladder in the scheduler.

Measured elsewhere: EAGLE-3/SGLang H100 **B=32: 1.30x TPOT, 1.70x aggregate**; EAGLE-3 paper 1.38x
at B=64; MagicDec 1.18-1.91x at B=32 and **speedup GROWS with batch in the memory-bound regime** —
GB10 is squarely in that regime (FFN at 87% of achievable bandwidth). Against our 130 tok/s that
projects to **169-221 at C=16**, i.e. meeting or beating vLLM's 169.
★ Published GB10 envelope: NO source exceeds ~170 agg tok/s at C=16 for any model >=27B, and NONE
of those runs used speculation at batch.

### ★ MEASURED: `ATLAS_MTP_MAX_SEQS` is a GUARD, NOT A KNOB — raising it HALVES throughput
| cap | C=2 | C=4 |
|---|---|---|
| **2 (default)** | 25.3 | **48.5** |
| 4 | 25.7 | **25.8** |
| 8 | 25.3 | **25.4** |
Enabling MTP at C=4 collapses aggregate throughput to single-sequence levels (**-47%**). Output is
COHERENT and IDENTICAL in all three legs, so this is NOT correctness — it is SERIALIZATION: the
multi-seq MTP path runs but aggregate throughput degenerates to ~C=1's number, consistent with
`mtp_carry.rs:37`'s `active.len()==1` assumption and the single-slot design at `types.rs:187`.
=> **Batched MTP requires the real vLLM-shaped implementation (ragged 1+k varlen verify, batched
rejection, per-sequence k+1 GDN state slots, per-request rewind). It is NOT a constant change.**
★ A 20-minute experiment correctly scoped a multi-day project — run the cheap disambiguating test
BEFORE estimating the expensive one.

### Recommended order when this is picked up
1. Batched MTP, vLLM's exact shape (expected **1.3-1.7x at C=16** — THE lever; everything else is
   a rounding error beside it). Blueprint is in the vendored checkout.
2. Dynamic K-vs-batch ladder shipped WITH it (prevents the fixed-K regression that produced vLLM's
   historical 1.4-1.8x slowdowns).
3. Overlapped scheduling + full-graph 1+k step (hides the growing host-side spec bookkeeping).
4. GDN batched-recurrent as a STANDALONE lever: SKIP (our own +1-2% measurement stands, and vLLM's
   design agrees) — but adopt its inline-checkpoint slot scheme as part of (1).

## 2026-07-28 — ★ BATCHED-MTP GATE: PASS (fixer round 1) — C=4 cap=4 25.8 → 49.0

The eb85ce41 batched verify left the gate 10% short (43.9 vs >48.5). nsys at C=4 showed the gap
was NOT the per-seq GDN loop the failure report guessed — it was the PROPOSE: 12 per-seq drafter
forwards x ~5 ms/step (~62 ms of a ~180 ms step; the BF16 drafter reads ~850 MB of weights per
forward, `--mtp-quantization bf16`). Three fixes on binary 4d01c9a4:

1. **Batched cross-seq propose** — one M=n drafter forward per draft position (drafts chain
   WITHIN a seq, are independent ACROSS seqs). Weight-bearing GEMMs on
   `dense_gemm_bf16_pipelined` (microbenched 2.7x the 4x-GEMV loop at M=4: 5.1 vs 14.4 ms per
   position; scalar `dense_gemm_bf16` only 1.8x — measure before wiring), LM head on
   `w4a16_gemv_batch4`, everything small looped per row with `forward_one`'s exact kernels.
   A/B alone: 46.3-46.8 vs 42.9-43.6 kill-switch (`ATLAS_NO_MTP_BATCH_PROPOSE`), disjoint.
2. **out_proj M>8 dispatch** (m_dispatch class strikes again, in reverse): R=16 verify rows fell
   into the pre-dequanted-FP8 PREFILL arm — fp8_fp8_gemm_ldmab 379 us vs 182 us for the SAME
   shape on w4a16_gemm_t_k64_p3 (2x weight bytes, bandwidth-bound at M=16) — x48 layers =
   ~9.5 ms/step. Attention o_proj at the same M was ALREADY on the k64 tile GEMM, which is what
   fingered the SSM arm. New arm via `deep_k_gemm`, kill `ATLAS_NO_VERIFY_OUTPROJ_TGEMM`.
3. **Batched argmax** in R-row verify + batched propose (~2 ms/step; the single-row argmax is a
   one-CTA scan, R serial calls = R single-SM passes).

GATE (3 scored reps, warmup discarded): **48.8 / 48.5 / 49.6 (mean 49.0)** vs bar >48.5; from
25.8 serialized = +90%. C=1 cap=4 25.5-25.6; C=1 default-cap unchanged (batched paths dead at
cap=1). Coherence + tool-call smokes PASS (finish=stop / tool_calls, well-formed args). MTP
sustained all legs (16 K4 summaries), zero ERROR/panic/716. Accept telemetry healthy at n=4:
p1 0.70-0.82, mean accepted 1.19-1.55 — the batched drafter's drafts are as good as per-seq.

★ Batched MTP at C=4 is now at PARITY-to-slightly-ahead of MTP-off (49.0 vs 48.5). The lever is
REAL but not yet decisive at C=4; per the spec the C=8/16 regime is where the weight-read
amortization grows. Remaining sized levers before re-raising the cap: per-seq GDN conv/WY loop
(~5 ms/step small kernels + share of ~22 ms/step eager gaps), K-vs-batch ladder / D-Cut.

★★ SELF-INTERFERENCE TRAP (cost ~1.5 h): a serve failed its KV-budget check because page cache
from the PREVIOUS leg inflated "pre-KV" (fix: `sysctl vm.drop_caches=3` + settle before every
serve of this config); its driver script kept polling :8888, and when the NEXT serve came up it
fired a second C=4 drive → 8 active > cap 4 → the scheduler correctly disabled MTP
(`active.len() <= mtp_max_seqs()`) → an entire nsys profile of the wrong regime that presented
as "batched propose killed MTP after one round" (36.5 tok/s, zero K4 lines, batch-8 graphs).
`pgrep -af prof_drive` before EVERY benchmark serve; a clean-looking profile of the wrong regime
is worse than no profile.

## 2026-07-28 — ★★★ BATCHED MTP LANDED (ultracode workflow, 11 agents). C=2 BEATS vLLM — first parity crossing beyond C=1.

Workflow wf_2149fd7e-81d built the full lever: batched K=4 verify forward (model side, `291bdc0b`),
scheduler dispatch (`eb85ce41`), batched cross-sequence propose + an out_proj M>8 dispatch fix
(`865d580a`), gate-validated (C=4 cap=4: 25.8 serialized -> ~48-49.6 batched, coherent, zero CUDA
errors, though statistically at PARITY with the 48.5 MTP-off bar at N=8, mean 47.65).

### Validated sweep (cap default = 2, commit `f49e00c9`)
| C | Atlas | vLLM | ratio |
|---|---|---|---|
| 1 | 25.60 | 14.2 | **1.803x MET** |
| 2 | **29.40** | 27.8 | **1.058x MET** — first time, +13% over the 26.4 MTP-off number |
| 4 | 48.70 | 53.3 | 0.914x |
| 8 | 74.85 | 98.8 | 0.758x |
| 16 | **131.75** | 168.9 | 0.780x (best ever) |
No regression anywhere vs the MTP-off baselines.

### ★ cap=16 sweep — the fixed-K collapse, MEASURED on our own engine
C=2 29.95 but **C=8 49.0 (vs 74.65 off) and C=16 50.1 (vs 131.25 off)** — throughput PLATEAUS at
~50 regardless of C. n*(K+1) verify rows of SUPERLINEAR GDN + ~22 ms/step of ungraphed eager verify
gaps. Exactly the fixed-large-K failure the literature warned about (vLLM's 1.4-1.8x slowdowns;
D-Cut's DFlash bs=64 collapse). The cap defaults to the measured crossover (2).

### What takes C=4..16 past the bar (in order)
1. CUDA-graph the batched verify step (validator: ~22 ms/step eager gaps at cap>1).
2. Spec step 2 remainder: per-seq GDN conv/WY batching (~5 ms/step sized by the validator).
3. K-vs-batch ladder, then D-Cut-style per-request depth pruning (task #35) — our GDN K-superlinearity
   makes depth cuts worth more here than on dense models.
4. Accept-rate lift: batched accept p1 0.59-0.71 vs single-seq 0.70-0.82 — find the divergence.

### ★ Trap hit AGAIN this session: a regex replaced the WRONG unwrap_or
Blind `.unwrap_or(N)` regex edit changed `shadow_topk` default 0->2 instead of the cap. Caught by
re-reading the file after the sweep stayed collapsed. Reverted. RULE: anchor numeric-default edits
to the FUNCTION, never to the literal.

## 2026-07-28 — ★★★ FINALIZED: adaptive policy = cap 4 DEFAULT (`dafd990d`). C=4 crosses vLLM — three of five levels MET.

The per-C policy needs no new mechanism: `active.len() <= mtp_max_seqs()` already IS the adaptive
gate. Batched K=4 MTP at C<=4, bit-for-bit MTP-off fallback at C>4. `ATLAS_MTP_MAX_SEQS` default
2 -> 4 (`speculative.rs`, anchored to the function); `=1` restores single-seq-only.

### Validated sweep at PURE DEFAULTS (binary 472ed410 = 7f4ffd6c + the default constant; no env
overrides beyond the standard non-MTP flag set; scored drives after one discarded warmup per
serve; two independent serves, caches dropped before each)
| C | reps (tok/s) | mean | vLLM | ratio | floor (never-regress) | verdict |
|---|---|---|---|---|---|---|
| 1 | 25.6, 25.5 | 25.55 | 14.2 | **1.80x** | 25.60 | **MET** |
| 2 | 35.3, 35.4 | 35.35 | 27.8 | **1.27x** | 29.40 | **MET** (+20% over floor) |
| 4 | 52.6, 56.6, 53.6, 53.5, 54.1 | 54.1 | 53.3 | **1.01x** | 48.70 | **MET** (+11% over floor) |
| 8 | 72.1, 74.6, 72.1, 71.4, 74.4, 71.7, 75.2, 75.1 | 73.5 | 98.8 | 0.74x | 74.85 | not met |
| 16 | 131.4, 129.9, 131.6 | 131.0 | 168.9 | 0.78x | 131.75 | not met |

- C=8/16 regime is PROVABLY the floor's own code path (8 > cap ⇒ MTP off, same scheduler branch),
  so no policy adjustment can move them. Fast C=8 reps (75.1-75.2) reproduce the floor exactly;
  the ~72 mode (both serves) carries ~1 s extra wall on the prefill leg — serve-to-serve /
  prefix-cache variance, not decode. C=16 130.-131.6 sits inside the historical 129.35-132.5 spread.
- cap=8/16 remain strictly dominated (fixed K=4 over n>=8 plateaus ~55 tok/s at EVERY C — worse
  than MTP-off). NEVER raise the cap past 4 until the K-vs-batch ladder / D-Cut (task #35) lands.
- Coherence smoke (temp 0, seed 42): fluent Rayleigh-scattering answer, finish=stop. Tool-call
  smoke: well-formed get_weather({"location":"Paris"}), finish=tool_calls. Zero CUDA errors /
  panics in either serve log (one benign content-loop watchdog, known).
- C=2 at defaults measured 35.35 — well above the 29.40 recorded at cap=2 default; the graphed
  batched verify + accept fix (`2070cdd6`/`684657e5`) lifted it further since that row was taken.

### Campaign position
Goal ">=1.0x at ALL of C=4,8,16": **C=4 MET (first time at defaults); C=8 0.74x and C=16 0.78x
remain open** and are unchanged-by-construction from the MTP-off baselines. The remaining gap is
exactly the fixed-K collapse at n>=8: n*(K+1) verify rows of superlinear GDN. Next lever, in
order: K-vs-batch ladder / D-Cut per-request depth pruning (task #35), then re-raise the cap.

## 2026-07-28 — ★★★ FINALIZED: K ladder ends at n=8 (4:3,8:1), cap 8 (`27ca65ca`). C=8 +10.5%; FOUR of five floors beaten, C=16 preserved.

The 24913dda ladder (4:3,8:2,16:1, cap 16) was matrix-tested (binary 4b92a774, 3-4 reps/cell):
C=8 81.1 with the default steps, **82.4 with 8:1** (vs 75.1 MTP-off control = +10-12%), but C=16
**114.6-117.3 vs 131.5 control in BOTH configs** — speculation at n=16 loses at even minimum
depth. Arithmetic: at p1~0.72 a win needs the K=1 verify step to cost <1.72x a plain batch-16
decode step; measured 117.3/131.5 implies ~1.9x. Suspects (all deliberately left eager): per-seq
GDN conv/WY loop at k<4 (the cross-seq 2-launch fast path is K=4-only), batched propose chunked
to groups of <=4, Phase-A per-seq M=1 bootstrap decodes.

**Fix (`27ca65ca`)**: default ladder 4:3,8:1 (the 8:1 step measured BEST at C=8); cap default
16 -> 8, so n>8 is MTP-off by construction and the C=16 floor returns. Both anchored in
`speculative/ladder.rs` with the measurements in the comments.

### Validated sweep at PURE DEFAULTS (binary 7a99488d, two fresh serves, caches dropped,
scored drives after one discarded warmup per serve)
| C | reps (tok/s) | mean | floor | vLLM | ratio | verdict |
|---|---|---|---|---|---|---|
| 1 | 25.6, 25.7, 25.6, 25.6, 25.6 | 25.62 | 25.55 | 14.2 | **1.80x** | **MET** |
| 2 | 34.9, 35.4, 35.4, 35.4, 34.8 | 35.18 | 35.35 | 27.8 | **1.27x** | **MET** (floor within noise, see below) |
| 4 | 56.4, 57.4, 54.3, 53.6, 56.5 | 55.6 | 54.1 | 53.3 | **1.04x** | **MET** (+2.8% over floor) |
| 8 | 80.9, 81.5, 82.5, 80.3, 81.0 | 81.2 | 73.5 | 98.8 | 0.82x | floor **+10.5%**; bar not met |
| 16 | 131.6, 131.9, 132.0, 131.8, 131.7 | 131.8 | 131.0 | 168.9 | 0.78x | floor MET (MTP-off by cap) |

- Ladder engagement PROVEN from graph captures: batched K=4 verify graphs at n=2..4, K=2 at
  n=5..8, NONE above 8; zero graph churn/evictions.
- Coherence smoke (temp 0, seed 42): fluent Rayleigh answer, finish=stop. Tool-call smoke:
  well-formed get_weather({"location":"Paris"}), finish=tool_calls. Zero CUDA errors/panics in
  all serve logs.
- ★ C=2 slow-mode finding: after ~4+ drives on one serve, C=2 bimodally drops to ~32-33 tok/s
  (+1.1 s wall; log shows "Prefix cache hit ... but no SSM snapshot — recomputing all KV").
  REPRODUCED EXACTLY with `ATLAS_NO_MTP_K_LADDER=1` (control serve: 35.7/35.0/35.4/35.4 then
  32.1/32.0) — pre-existing SSM-snapshot/prefix-anchor drift (the ssm_miss_anchor class), NOT
  the ladder; at n=2 the ladder is behaviorally identical to the floor config anyway (3 drafts
  either way). Fresh-serve C=2 mode is 35.2-35.4 = the floor.

### Campaign position
Goal ">=1.0x at ALL of C=4,8,16": **C=1/2/4 MET at defaults; C=8 0.82x (was 0.74x), C=16 0.78x
open.** The C=8 bar (98.8) needs ~+22% more; the C=16 bar (168.9) needs the n=16 verify-step
cost cut below ~1.7x a plain decode step before ANY spec depth can win there. Next levers, in
order: (1) k<4 GDN table-form port (extend the cross-seq conv/WY 2-launch fast path beyond
K=4), (2) wider batched propose (drop the <=4 chunking), (3) graph the Phase-A bootstrap
decodes — then re-test 16:1 and re-raise the cap.

## 2026-07-28 — ★★★ FINALIZED (round 2): three eager-cost fixes + ladder `4:3,8:2` (cap 8). C=8 +9.6%, FOUR floors beaten, C=16 preserved.

The three costs named by the cap-16 matrix were all fixed and all verifiably engage:
`b93982d9` K-parameterizes the cross-seq GDN conv/WY 2-launch fast path (was K=4-only),
`a83627a2` widens the batched propose to n=16 (was chunked to groups of <=4), `fa373bf4`
batches the Phase-A bootstrap (was per-seq M=1 decodes). Serve-log proof, this session:
`batched-verify GDN conv+WY ENGAGED (n=8, k=3)` and `(n=3, k=4)` — zero DECLINED for that
path; `propose_batch active: n=8 ... lm_head_batchm=0x...4120` (ONE group, width-selected
handle, non-zero ⇒ no `try_kernel` handle-0 fallback); `MTP batched bootstrap ENGAGED (n=7)`.

★ Their measured value is NOT a direct speedup — at the shipped `8:1` ladder they were worth
only +2.1% (82.9 vs the 81.2 floor). They pay by MOVING THE OPTIMAL LADDER POINT: pre-fix
`8:1` (82.4) beat `8:2` (81.1); post-fix `8:2` (89.25) beats `8:1` (82.9) by +7.7%. The
verify step got cheap enough that a deeper draft finally pays at n=8. **Default ladder is now
`4:3,8:2`** (anchored in the `mtp_ladder_steps` function body, per the burned-twice regex
trap); cap unchanged at 8.

### C=16 still loses — the arithmetic (why the cap does NOT move)
| config | C=16 tok/s | vs MTP-off control 131.9 |
|---|---|---|
| MTP-off (cap 8, pure defaults) | **131.9** | — |
| 16:1 (cap 16) | 128.4 | 0.973x |
| 16:2 (cap 16) | 94.1 | collapse |
At p1~0.72 (~1.72 tok/step) the 0.973x implies a verify-step cost of ~1.77x a plain batch-16
decode step, against the <1.72x break-even. Pre-fix the matrix implied ~1.9x: the three fixes
bought ~0.13x of the ~0.18x needed (~70% of the way). The named remaining cost is the piece
deliberately left undone — **the Phase-A bootstrap forward is not graph-captured** (it routes
through `decode_batch`, which disables graphs at n>=2), plus n small argmax D2H syncs. That is
the single next lever if C=16 is attacked again; depth at n=16 is decisively dead (16:2 = 94.1).

### Validated sweep at PURE DEFAULTS (binary md5 `e65c232d49732d409339a1dccad00ae8`,
`ATLAS_TARGET_MODEL=qwen3.6-27b cargo build -p spark-server --release --features cuda`;
ONE FRESH SERVE PER C, `vm.drop_caches=3` + all containers removed + `:8888` asserted dead
before each, first drive per serve discarded as warmup)
| C | scored reps (tok/s) | mean | floor | vLLM | ratio | verdict |
|---|---|---|---|---|---|---|
| 1 | 25.7, 25.6, 25.6 | **25.63** | 25.62 | 14.2 | **1.80x** | **MET** |
| 2 | 35.8, 35.2, 35.6 | **35.53** | 35.18 | 27.8 | **1.28x** | **MET** (floor +1.0%) |
| 4 | 59.1, 58.1, 56.7 | **57.97** | 55.6 | 53.3 | **1.09x** | **MET** (floor +4.3%) |
| 8 | 88.5, 90.4, 88.8, 88.6 · 88.5, 89.1, 89.0 (2 serves) | **89.0** | 81.2 | 98.8 | 0.90x | floor **+9.6%**; bar not met |
| 16 | 132.0, 131.7, 131.6 | **131.77** | 131.8 | 168.9 | 0.78x | floor MET (MTP-off by cap) |

- Smoke at final defaults (temp 0, seed 42, run on two separate serves): fluent Rayleigh
  answer `finish=stop`; `get_weather({"location":"Paris"})` `finish=tool_calls`. PASS.
- **Zero** ERROR/panic/illegal-access/error-700/716 lines in all six serve logs, including a
  full C=8 leg's log dumped AFTER its drives. Accept telemetry healthy at n=8, k=3:
  ~33-41 accept-3 / 32-37 accept-2 / 22-35 reject per 100 steps, mean accepted 0.98-1.19.
- One benign `SSM batched recurrent DECLINED (n=4): pool slots are not contiguous` — the known
  slot-fragmentation notice, present in the floor config too.
- ★ Harness trap re-hit and hardened: the serve helper must remove EVERY container (not just
  its own label) and assert `:8888` is both un-listened and un-answering before launching, or
  the readiness probe passes against the PREVIOUS engine.

### Campaign position
**C=1/2/4 MET at pure defaults (1.80x / 1.28x / 1.09x). C=8 0.90x (was 0.82x, floor +9.6%),
C=16 0.78x.** C=8 needs a further ~+11%; C=16 needs the n=16 verify-step cost below ~1.72x a
plain decode step before any depth can win there. Next lever, singular and named: graph-capture
the Phase-A bootstrap forward, then re-test 16:1 and re-raise the cap.

## 2026-07-29 — ★★★ FINALIZED (round 3): chunk-cap fix + ladder `4:3,8:3` (cap 8). C=8 95.68 (floor +7.6%), FOUR floors held, C=16 preserved.

The wave's finding is a MEASUREMENT ARTIFACT, not a new optimization: **the `8:2` ladder step was
compensating for a bug, not for GDN depth cost.** `mtp_step`'s verify chunk cap for `rows=4` was
pinned to 4 SEQUENCES ("keeps the proven 4-seq/R=16 envelope byte-identical"), so an 8-wide K=4
batch ran TWO serialized 4-wide verify forwards — 2x the weight reads per step. Every historical
"8:3 collapses at n=8" number recorded that chunking, never depth-3 at width 8 (57.9 in the
round-2 matrix; 62.6 on re-measure). The real bound is the verify row buffer, which
`can_batch_verify` already enforces as `n*k <= 32`: 8 seqs x 4 rows fits EXACTLY.
Raising the cap: **62.58 -> 95.68 (+53%)**, and the GDN path engages ONCE at `(n=8, k=4)` instead
of twice at `(n=4, k=4)`.

★ It was caught by INSTRUMENT ABSENCE, not by a throughput reading: the width-attributed accept
telemetry printed ONLY `n=4` lines during a C=8 drive. The corroborating arithmetic — 62.58 is
BELOW the 73.5 MTP-off floor, and a depth cost cannot take you below the no-speculation path
while a doubled forward can.

### Landed
- `48fb8a11` fix(mtp): chunk cap -> row-buffer bound (`rows=4` 4 -> 8 seqs). Provably a NO-OP at
  the pre-existing default ladder, where `rows=4` occurred only at n<=4 and every chunk was <= 4.
- `3313a733` perf(mtp): default ladder `4:3,8:2` -> `4:3,8:3`. Kill switch
  `ATLAS_MTP_K_LADDER=4:3,8:2`. Unit tests updated (3 pass), rustfmt clean.
- `d4633f09` docs: the stale `8:2` prose in `ladder.rs` / `mtp_step.rs` (three comments still
  described the superseded default). Doc-only.

★ **The diff is inert everywhere except 5 <= n <= 8, by construction**: at n<=4 both ladders
return 3 drafts AND the chunk cap was already <= 4; at n>8 the cap (8) makes the path MTP-off.
So C=1/2/4/16 are unchanged code paths and their rows below are variance measurements, not
regression risk.

### Validated sweep at PURE DEFAULTS (binary md5 `f134f6fa267cdc257197d092f113089a`, built with
`ATLAS_TARGET_MODEL=qwen3.6-27b cargo build -p spark-server --release --features cuda`; ONE FRESH
SERVE PER C, all containers removed + `:8888` asserted dead + `vm.drop_caches=3` + a MemAvailable
>= 108GB settle gate before each; first drive per serve DISCARDED as warmup; 5 scored drives)
| C | scored reps (tok/s) | mean | floor | vLLM | ratio | verdict |
|---|---|---|---|---|---|---|
| 1 | 25.6, 25.6, 25.6, 25.6, 25.6 | **25.60** | 25.62 | 14.2 | **1.80x** | **MET** |
| 2 | 35.2, 35.6, 35.6, 35.4, 35.4 | **35.44** | 35.18 | 27.8 | **1.27x** | **MET** (floor +0.7%) |
| 4 | 60.5, 59.9, 58.2, 59.7, 59.9 | **59.64** | 55.6 | 53.3 | **1.12x** | **MET** (floor +7.3%) |
| 8 | 95.7, 94.8, 95.7, 96.1, 96.1 | **95.68** | 88.9 | 98.8 | 0.968x | floor **+7.6%**; bar not met |
| 16 | 131.4, 131.5, 131.0, 131.1, 131.1 | **131.22** | 131.9 | 168.9 | 0.777x | MTP-off by cap |

Confirming second serves for the two levels that read at/under their reference floor (both on
code paths this diff does not touch): **C=16 131.7, 132.0, 131.3, 131.5, 131.0 = 131.50** and
**C=1 25.5, 25.6, 25.6, 25.6, 25.5 = 25.56**. Pooled two-serve: C=16 **131.36** (10 reps,
131.0-132.0, inside the documented 129.35-132.5 C=16 spread), C=1 **25.58** (wall 7.5s and 192
tokens on every single rep — the spread is 1-decimal print quantization on a constant wall).
Both levels are serve-to-serve variance around their floors, not movement.

### Hygiene
- Smoke at final defaults (temp 0, seed 42) on the C=1 and C=8 serves: fluent Rayleigh-scattering
  answer `finish=stop`; `get_weather({"location":"Paris"})` `finish=tool_calls`. PASS.
- **Zero** ERROR/panic/illegal-memory/error-700/716 lines in all seven serve logs, each dumped
  while its container was still alive (`conc_sweep/fin2_c{1,2,4,8,16}.log`,
  `fin2b_c{1,16}.log`).
- Engagement PROVEN from the logs: `batched-verify GDN conv+WY ENGAGED (n=8, k=4)` at C=8 (56
  launches/layer -> batched) and `(n=2, k=4)` at C=2/C=4; **zero DECLINED on the verify path**;
  `MTP batched bootstrap ENGAGED`; zero `graphs OFF`; zero graph evictions. At C=16 there is NO
  batched verify at all (only a ramp/tail bootstrap at n=2) — MTP-off by cap, as designed.
- The one recurring `SSM batched recurrent DECLINED (n=...): pool slots are not contiguous` is
  the known benign slot-fragmentation notice, present in the floor config too.

### Multiplier arithmetic, with measured numbers
- **(B) ACCEPT was the real lever and is now largely banked.** Implementer-2's per-sequence
  drafter-prefill fix (`36d340a0`) reproduces exactly: p1 **0.740 -> 0.780** at n=8, and the
  throughput ratio equals the tok_step ratio to 0.1% (1.0529 vs 1.0516, disjoint rep ranges
  88.4-90.8 vs 92.8-96.9).
- **Depth at n=8 is worth only ~+2%**: tok_step 2.301 -> 2.606 (+13.3%) buys +2.3% throughput,
  so the K=4 verify step costs ~10.7% more than the K=3 one. Consistent in sign across 5 serves
  (8:3 = 95.84/95.68/94.93 vs 8:2 = 93.40/93.30) but rep ranges overlap at the tails — treat it
  as a ~2% effect, NOT a step change. Shipped as a default because the sign is consistent and the
  mechanism is measured.
- **C=8 remaining gap: 98.8/95.68 = 1.033x.** At the current verify cost that needs tok_step 2.69
  (mean_na 1.69 from 1.606), i.e. p1 ~0.85 at K=4 — or a ~3% cheaper verify step.
- **C=16 is now PARITY, not a loss**: 16:1 = 131.93 vs a same-session MTP-off control of 131.42
  (was 128.4 = 0.973x). Parity buys nothing, so the cap stays 8. p1 at n=16 = 0.797 => break-even
  cost 1.797x and the measured implied cost is ~1.79x — sitting exactly on it. Clearing 168.9
  needs <= 1.40x. Depth at n=16 remains dead (16:2 = 94.1).

### Accept telemetry by width (`ATLAS_MTP_ACCEPT_DEBUG`)
| width | p1 | mean_na | tok_step |
|---|---|---|---|
| n=4, k_drafts=3 | 0.776 | 1.533 | 2.533 |
| n=8, k_drafts=2 (old default) | 0.780 | 1.301 | 2.301 |
| n=8, k_drafts=3 (new default) | 0.793 | 1.606 | 2.606 |
| n=16, k_drafts=1 | 0.797 | 0.797 | 1.797 |

### Traps re-burned this wave (all previously documented)
1. `pkill -f prof_drive.py` SELF-MATCHES the caller's own command line and killed the harness
   mid-run. Match only real python interpreters.
2. **Stale concurrent driver**: a backgrounded drive loop still firing C=8 during a C=8 leg =>
   16 active > cap 8 => MTP correctly off => a flat 65.5 tok/s plateau that would have been
   reported as "the 8:2 control". Detected by the 65.5 signature; leg discarded and re-run.
3. **KV-budget knife edge (new)**: the engine sizes its KV pool from system MemAvailable at
   startup and a just-removed container's memory is still being reclaimed. Pools measured
   3746-5056 blocks across IDENTICAL launches; `--max-batch-size 16` needs 4096, so two serves
   DIED at build. Worse than dying is coming up just under and silently running a different
   regime. The harness now double `sync`+`drop_caches`, gates on MemAvailable >= 108GB, and
   RETRIES until the pool is >= 4300 blocks (one retry fired this session at 4225 blocks). Every
   reported serve got 4225-5056; all measured ones 4387-4923.
4. **Log loss**: each serve removes ALL containers, so `docker logs` after the NEXT serve returns
   "No such container". Dump logs while the container is alive.

### Campaign position
**C=1/2/4 MET at pure defaults (1.80x / 1.27x / 1.12x). C=8 0.968x (floor +7.6%, was 0.90x),
C=16 0.777x (MTP-off by cap).** C=8 needs a further ~3.3%.
Next levers, in value order:
1. **L3 refeed made slot-keyed** — powered A/B measured +0.016 p1 / +2.3% tokens/step.
2. **L1 carry is silently OFF at EVERY concurrency including C=1**, because `mtp_multi_seq_mode()`
   is cap-based. This is a LIVE REGRESSION on multi-turn/MLPerf traffic that `prof_drive`'s
   single-turn prompts cannot see.
3. **L6 drafter-pool sizing** (`mtp_head/new.rs:228` still sizes for ONE sequence) is now MORE
   urgent: every sequence allocates drafter KV at prefill. Fine at this bench's 26-token prompts,
   a silent accept->0 collapse at C=8 with long contexts.
4. The row buffer (`n*k <= 32`) is now the binding constraint at C=8: K=4 at n=8 is the deepest
   legal point. Going deeper requires enlarging the verify row buffer — an untested regime.
- Still red and pre-existing: `decode_a2.rs` 598 LoC and `trait_impl/mod.rs` 800 LoC, both over
  the 500 cap and absent from `.github/workflows/file-size-cap.yml`'s allow-list.

## 2026-07-29 — ★★★ FINALIZED (round 4): cap 8 → 16 + ladder rung `16:1`, D-Cut default-ON @ 0.75. C=16 +15.0%, C=8 CROSSES vLLM. FOUR of five levels MET, ALL FIVE floors beaten.

Two independent defaults landed, each measured with its own kill switch by the wave-6 matrix and
then re-validated together at pure defaults:

- **`16:1` ladder rung + cap 16** — the big one. **C=16 131.4 → 151.14 (+15.0%)**.
- **D-Cut default-ON at ratio 0.75** — **C=8 95.84 → 107.40 (+12.1%)**, which **crosses vLLM's
  98.8 bar (1.087x)**. This is the fifth level to be attacked and the fourth to fall.

### Why the cap finally moved, after three rounds of refusing to move it
The n=16 history is a record of a COST curve, not a depth curve, and each round moved the cost:

| round | verify cost @n=16 | break-even (p1) | 16:1 vs MTP-off control | verdict |
|---|---|---|---|---|
| after the 3 eager-cost fixes | ~1.90 → ~1.77x | 1.72x (p1 0.72) | 128.4 vs 131.9 | LOSS, cap stays 8 |
| after the accept lift `36d340a0` | ~1.79x | 1.797x (p1 0.797) | 131.93 vs 131.42 | PARITY, cap stays 8 |
| after `296b9674`'s 3 per-row cuts | **~1.55x** | 1.83x (p1 0.83) | **152.38 vs 131.40** | **WIN, cap → 16** |

`296b9674` (wide LM-head arm at 9..32 rows, one-launch gated RMS norm, fused BA-projection+gates)
bought ~0.20x of verify-step cost. That is what took 16:1 off break-even. ★ Note the cost cut is
LARGER than the throughput gain, because the fused BA+gates arm is not bit-identical and costs
accept: at C=8 its kill-switch A/B measured tok_step 2.58 → 2.50 (−3.1%) for +9.2% throughput,
i.e. a −11.6% verify step — dead inside its predicted −11.5..−13%. The numerics change is real,
measurable, and worth paying.

### The raise is inert at n ≤ 8 BY CONSTRUCTION
The cap only gates dispatch above 8 (`active.len() <= mtp_max_seqs()`) and the `16:1` rung only
matches above 8 (the `n<=8` rung is found first). So C=1/2/4/8 run unchanged code paths; their
rows below are variance measurements, not regression risk. Kill switch `ATLAS_MTP_MAX_SEQS=8`.

### D-Cut: the full bucket sweep, and the one bucket that wins
Ranked prefix-product survival scores across the batch, top-`ratio` retained (arXiv 2607.14647).
All at C=8, one fresh serve per leg, 5 scored reps:

| ratio | tok/s | kept_frac | tok_step | verdict |
|---|---|---|---|---|
| OFF (kill switch) | 105.56 | — | 2.50 | control |
| 1.0 (wiring live, 0 pruned) | 105.80 | 1.000 | 2.52 | **inert, as designed** |
| **0.75** | **108.57** (pooled 2 serves) | 0.876 | 2.54 | **+2.6% — SHIPPED** |
| 0.5 | 107.43 | 0.750 | 2.43 | worse |
| 0.25 | 101.56 | 0.626 | 2.16 | clear loss |

The mechanism is visible in telemetry: tok_step degrades monotonically as rows are pruned while
rows fall, and 0.75 is the ONLY point where the row saving outruns the token loss. The ratio-1.0
control being statistically identical to OFF is the proof that the ragged-row plumbing itself is
free. Kill switch `ATLAS_NO_MTP_DCUT` (presence).

★ **D-Cut contributes exactly NOTHING at C=16, and the logs prove it rather than infer it.** Its
v1 floor of one mandatory draft per sequence means every sequence gets ≥2 rows; 16 × 2 = 32 = the
whole `VERIFY_ROW_BUDGET`. There is zero slack, so `select()` returns retained=1 for all 16 at
every ratio. In this sweep the C=16 leg emitted NO D-Cut telemetry line at all (`plan` returns
early at `ladder_nd < 2`). Running D-Cut over a 16:3 ladder is strictly WORSE (120.76): it is
forced back to k=1 anyway while the propose still pays for 3 drafts.

### Landed
- `mtp_max_seqs` default 8 → 16, anchored in the function body (`speculative/ladder.rs`).
  Kill switch `ATLAS_MTP_MAX_SEQS=8`.
- Default ladder `4:3,8:3` → `4:3,8:3,16:1`, anchored in `mtp_ladder_steps`'s body (NOT a bare
  literal regex — the burned-twice trap). Kill switch `ATLAS_MTP_K_LADDER=4:3,8:3`.
- D-Cut `dcut_enabled` OFF-by-presence → ON-by-default with `ATLAS_NO_MTP_DCUT`; `dcut_ratio`
  default 0.5 → 0.75.
- Ladder unit test updated for the new rung (9 → 1 draft, 16 → 1, 32 → 1). 3 ladder + 9 D-Cut
  tests pass; both edited crates rustfmt-clean.

### Validated sweep at PURE DEFAULTS (binary md5 `fae54de95627212898f51d5a0303d61d`, built with
`ATLAS_TARGET_MODEL=qwen3.6-27b cargo build -p spark-server --release --features cuda`)
ONE FRESH SERVE PER C; no `ATLAS_MTP_DCUT` / `ATLAS_MTP_MAX_SEQS` / `ATLAS_MTP_K_LADDER` anywhere
in the environment (only `ATLAS_MTP_ACCEPT_DEBUG=1` for the telemetry columns) — the binary's own
defaults produce these numbers. All containers removed + `:8888` asserted dead + double
`vm.drop_caches=3` + MemAvailable ≥ 108GB settle + a ≥4300-block KV-pool gate before each; first
drive per serve DISCARDED as warmup; 5 scored drives.

| C | scored reps (tok/s) | mean | floor | vs floor | vLLM | ratio | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 27.0, 27.0, 27.0, 27.0, 27.0 | **27.00** | 25.53 | +5.8% | 14.2 | **1.90x** | **MET** |
| 2 | 38.4, 38.2, 38.1, 37.7, 38.3 | **38.14** | 35.37 | +7.8% | 27.8 | **1.37x** | **MET** |
| 4 | 71.2, 67.4, 71.1, 68.0, 71.2 | **69.78** | 60.40 | +15.5% | 53.3 | **1.31x** | **MET** |
| 8 | 107.2, 103.7, 108.8, 107.5, 109.8 | **107.40** | 95.84 | +12.1% | 98.8 | **1.087x** | **MET** |
| 16 | 152.1, 150.6, 150.1, 151.4, 151.5 | **151.14** | 131.40 | +15.0% | 168.9 | 0.895x | floor +15.0% |

**ALL FIVE floors beaten. FOUR of five levels now beat vLLM.** Only C=16 remains, needing +11.8%.

### Engagement PROVEN from the serve logs, not assumed
- C=16: `batched-verify GDN conv+WY ENGAGED (n=16, k=2)` — the cap raise AND the `16:1` rung are
  both live at pure defaults. Accept telemetry `n=16 k_drafts=1 p1=0.820-0.845 tok_step=1.82-1.85`.
- C=8: D-Cut ragged and live — `kept_frac=0.875-0.876`, `last_ks=[3,4,4,4,4,4,2,3]` and
  `[3,4,4,3,3,4,3,4]` (genuinely ragged, not a uniform shape).
- C=4/C=2: `kept_frac=0.875`, `last_ks=[3,3,4,4]` / `[4,3]`.
- C=1: NO D-Cut line and NO batched verify — n=1 never reaches the batched partition
  (`verify_idxs.len() >= 2`), so C=1 is provably untouched by both defaults.
- KV pools 4372-4819 blocks on all five serves; every serve came up on attempt 1.

### Hygiene
- Coherence + tool-call smoke on ALL FIVE serves (temp 0, seed 42): fluent Rayleigh-scattering
  answer `finish=stop`; `get_weather({"location":"Paris"})` `finish=tool_calls`. PASS.
- **Zero** ERROR/panic/illegal-memory/error-700/716/719 lines in all five serve logs
  (`conc_sweep/w7_c{1,2,4,8,16}.log`), each dumped while its container was alive. The twin-tile
  GEMM's sticky-716 risk flagged for `296b9674` did NOT reproduce at any concurrency.
- All containers removed at the end; `nvidia-smi` compute-apps empty; `:8888` dead.

### Traps and caveats for the record
- ★ **Graph-key churn is real and D-Cut wins DESPITE it.** Ragged `ks` takes captures from ~15 to
  **407** per serve at C=8 against `VERIFY_BATCHED_GRAPH_CAP = 32`, and `batched-verify GDN
  conv+WY DECLINED` lines appear at ragged widths (the `(k desc, slot)` sort makes the
  consecutive-ssm-slot precondition hold less often). The +2.6% is a FLOOR, not a ceiling.
- **C=16 spec-on reps emit 3009 tokens, not 3072**, deterministically in every rep — the
  documented `spec_not_output_neutral` trajectory shift (one request hits a natural stop 63 tokens
  early), not the content-loop watchdog. Token-normalized it changes nothing: 19.9s × 3072/3009 =
  20.3s ⇒ 151.3 tok/s.
- **C=4 is bimodal within one serve** (71.2/71.1/71.2 vs 67.4/68.0, identical 768 tokens) — the
  known SSM-snapshot/prefix-anchor drift, present in the floor config too. Reported as the mean of
  all five, not the fast mode.

### Campaign position
**C=1/2/4/8 MET at pure defaults (1.90x / 1.37x / 1.31x / 1.087x). C=16 0.895x.**
Next levers, in value order:
1. **Lift D-Cut's `rows_i >= 2` floor** so a sequence can drop out of speculation entirely. It is
   the ONLY route by which D-Cut can ever attack C=16, where it is currently budget-locked.
2. **Raise `VERIFY_BATCHED_GRAPH_CAP`** and re-sort by slot-then-depth — D-Cut currently pays ~27x
   the graph captures and loses the GDN fast path at some widths. Unmeasured headroom on a win.
3. **Prune the propose, not just the verify** (implementer-1's deviation #4): the drafter still
   runs at full ladder width even when D-Cut throws the rows away.
4. **L1 carry is silently OFF at every concurrency** because `mtp_multi_seq_mode()` is cap-based —
   a LIVE REGRESSION on multi-turn/MLPerf traffic that `prof_drive`'s single-turn prompts cannot
   see. Unchanged from round 3, and the cap raise does not fix it.
- Still red and pre-existing: `decode_a2.rs` 598 LoC and `trait_impl/mod.rs` 800 LoC, both over the
  500 cap and absent from `.github/workflows/file-size-cap.yml`'s allow-list.

---

## Wave 55 — INTEGRATION (2026-08-02): merge main, CLI flags, close the campaign

The campaign's last wave. Three jobs: land `origin/main` on the branch, replace the
ten-variable environment recipe with command-line flags, and prove the merge cost nothing.

### 1. Merge — `origin/main` @ `c19481aa` into `perf/enterprise-concurrency-v2` @ `8c15fe6c`

Merge base `bc08abb5` (#377, the ratatui TUI, already an ancestor). Three commits ahead:
#385 (TUI Benchmarks + teardownable model state), #384 and #380 (DeepSeek-V4 E8M0 K2/K3
verify → per-token GS32).

**The two `fix(v4)` commits do NOT touch the Qwen3.6-27B NVFP4 path** — as expected; they
are DeepSeek-V4 E8M0 MoE only, and no file they touch is on the 27B's dispatch.

**#385 is the whole conflict surface.** It converts ~105 model-dependent statics into two
carried structs (`ModelLevers` on `ForwardContext`, `SchedLevers` on `SchedCtx`) plus a run
mailbox for counters. The campaign had been ADDING statics at those exact sites. 19 files,
24 hunks. The general resolution was: take main's carrier, keep our lever.

Two resolutions worth remembering:

- **`decode_logits_step.rs` — rayon host sampling vs a `RefCell` run scratch.** Main moved
  the per-thread dequant scratch onto `SchedCtx` as a `RefCell`, which makes `SchedCtx`
  `!Sync`; our parallel arm captures it in a rayon closure. Neither "take theirs" nor "take
  ours" works. Resolved by giving each worker its own `DecodeScratch` from a `thread_local`
  and hoisting the `Copy`/`Arc` context pieces OUT of the closure, so nothing `!Sync` is
  captured. Both levers survive.
- **`serve.rs`** was restructured by both sides at once — main split it 1197 → 397 LoC into
  `serve_load.rs`; we had grown it to 1244. Resolved by taking main's `serve.rs` wholesale
  and re-applying our two hunks (the shadowed-kernel fail-fast, the `DECODE_META_MAX_ROWS`
  ceiling) into `serve_load.rs`, where that code now lives.

### 2. Env → CLI

`ATLAS_SSM_H_FP16` was decoded independently in TWO places (the kernel accessor and the
preflight check), which is how a preflight could pass on a reading the kernels did not
share. It is now one accessor behind one flag. The five knobs the best config needs:

| was | is | default |
|---|---|---|
| `ATLAS_SSM_H_FP16` | `--ssm-h-dtype {f32,f16}` | `f32` |
| `ATLAS_GDN_FUSED_NORM=1` | `--gdn-fused-norm` | off |
| `ATLAS_SSM_BATCHED_RECURRENT=1` | `--ssm-batched-recurrent` | off |
| `ATLAS_SSM_TAIL_MIDCHUNK=0` | `--ssm-tail-midchunk <bool>` | on |
| `ATLAS_MTP_GATE_FORCE=1` | `--mtp-gate {auto,force}` | `auto` |

★ **The clap defaults sealed all five under `spark serve`. That is fixed now — but every
number this campaign recorded was measured before the fix.** Each knob is read once
through a `OnceLock` whose fallback closure reads the environment, and
`publish_kernel_flags()` published the clap DEFAULT into all five cells on every boot.
Sealing a cell with a default makes the closure unreachable, so the flag did not merely
win when it was passed — it won **always**, including when the operator never passed it.

The fix: the five flags are `Option`s, an absent flag publishes NOTHING, and the
environment decides again exactly as this table says. Check it by content, not by date —
`spark_runtime::set_ssm_tail_midchunk` takes an `Option<bool>` and returns without writing
on `None`, `scheduler::levers::set_mtp_gate_force` does the same, and the three GDN flags
are published together only when at least one of them is given
(`main_modules/serve_flags.rs`, which is where `publish_kernel_flags` now lives). The bare
switches still mean on; `--gdn-fused-norm false` is the newly expressible explicit off.

Consequence for this campaign's frozen configs, which the fix does NOT undo: every launch
script that set `-e ATLAS_SSM_TAIL_MIDCHUNK=0` **without** also passing
`--ssm-tail-midchunk false` ran with mid-chunk capture **ON**, and every ladder measured
before the fix ran with the other four at their clap defaults whatever the environment
said. `grep -rn 'ssm-tail-midchunk'` over `scripts/`, `docker/`, `docs/` and the root
`*.md` returns this table and nothing else — the flag is used nowhere.

`--ssm-h-dtype f16` without `--gdn-fused-norm` is now a startup ERROR (it used to be a
silent FP32-kernel-over-FP16-pool, i.e. fluent garbage).

**The other six of the ten did nothing.** `ATLAS_MTP_CATCHUP=0`, `ATLAS_MTP_DRAFT_CONF=0.0`,
`ATLAS_SSM_TAIL_PROTECT=1`, `ATLAS_SSM_TAIL_LEASE_TTL=128`, `ATLAS_BF16_TC_PREFILL=1`
(shadowed by the MMQ arm) and `ATLAS_MTP_ACCEPT_DEBUG=1` (a log line) were each already the
compiled default, inert, or unread. Every ladder from wave 17 on carried all ten.

### 3. `ATLAS_GDN_FUSED_NORM` / `ATLAS_SSM_BATCHED_RECURRENT` were NOT promoted to defaults

Wave 53's bitwise legs could not certify output-equivalence, because **the CONTROL failed**:
two identical serves differed on 7 of 42 completions (C=4 and C=16). The flag legs differ on
5-6 — inside the control's own noise, so the comparison cannot separate the flags from
run-to-run nondeterminism. Under PCND an unproven numerics change is explicit configuration,
so both became CLI flags rather than defaults. **A control leg that is not reproducible
retires the whole gate, not just the failing arm.**

## 2026-08-03 — C=1..128 sweep re-verification on the final image (8c #4, CLOSED)

Leg results land in `/workspace/w55_sweep/results/` (driver `/workspace/w55_sweep/w55_conc_ladder.py`,
sha256 6412b12d). Image `avarok/atlas-gb10:7241a95` = gate image = this branch modulo bench-only
harness deltas, so this IS the #388 binary's sweep. Recipe-derived serve
(`serve_atlas.sh` ← `recipes/qwen3.6/qwen3.6-27b-w55-sweep-dev.yaml`): util 0.85, bs 128, bf16 KV,
spec-on num-drafts 3, ssm-h f16 + fused-norm, thinking OFF on BOTH engines
(`chat_template_kwargs:{"enable_thinking":false}`), prompt parity 200=200, temp 0.

| C | Atlas | vLLM | ratio | prev(2026-08-02) |
|---|---|---|---|---|
| 1 | 24.34 | 14.69 | **1.656x** | 1.694x |
| 2 | 35.79 | 28.63 | **1.250x** | 1.281x |
| 4 | 71.61 | 54.76 | **1.308x** | 1.368x |
| 8 | 113.13 | 100.50 | **1.126x** | 1.167x |
| 16 | 199.51 | 169.07 | **1.180x** | 1.205x |
| 32 | 290.16 | 260.76 | **1.113x** | 1.105x |
| 64 | 360.82 | 355.04 | **1.016x** | 1.021x |
| 128 | 429.52 | 423.49 | **1.014x** | 1.040x |

**8/8 rungs reconfirmed** — Atlas wins every rung on tok/s. Atlas absolute tok/s
within ±0.6% at C≤2, −1.5..−2.7% at C=4..32, −5.3% at C=64, −9.0% at C=128 vs the
2026-08-02 ladder. Spread: vLLM 0.05–0.63%; Atlas 1.0–4.1% (C=1 rep spread 9%).
Two confounds on the original ladder are now explained, not denied:
1. The C=128 "timeout" finish-reason in the Atlas leg traces to `--request-timeout 300s`
   (Atlas default; vLLM runs no comparable deadline). Control leg reran C=128 with
   REQ_TIMEOUT=0: 439.43 tok/s vs spec-on 429.52 — the deadline was costing ~2%, not the engine.
2. Clock probe healthy on every rung (2236–2457 MHz Atlas, 2463–2483 vLLM) — no 513 MHz clamp.
