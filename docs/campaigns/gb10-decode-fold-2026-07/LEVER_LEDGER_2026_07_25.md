# Lever ledger — 2026-07-25 sweep on the shipping GB10 golden config

Every entry is a measurement or a source-level proof, not an estimate. Config throughout is the
frozen c2final golden config at K=4 (`--num-drafts 3`), model
`centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf`. Wall references are the shipped golden run
(`chainK_golden_e2e_20260724_131209`, 4104.0 s, 1007 samples).

## The budget everything is scored against

From `WALL_DECOMPOSITION.md` (harness event stream, perf phase):

| slice | seconds | % |
|---|---|---|
| decode | 2447.6 | 59.6% |
| fixed per-turn TTFT | 867 | 21.1% |
| marginal prefill | 771 | 18.8% |
| client gap | 0.1 | 0.0% |

Median output is only **45 tokens**, so per-request overhead competes directly with decode.

## Verdicts

| lever | verdict | evidence |
|---|---|---|
| **fp8 KV cache** at K=4 | **DO NOT FOLD** | +8.1% TPOT p50 / +13.4% p90; token match 0.9954, mean KL 0.0526. Frees **0 GB** — the KV budget is derived, so fp8 only buys 2x KV *tokens* (323k→651k) that 32k/batch-1 never uses. Hypothesis dies on its own premise: **bf16 TPOT is flat 3k→13k context (42.58→42.91 ms), so decode is not KV-bandwidth-bound.** Closes the ledger's last `[pending A/B]`. |
| **`--ssm-cache-slots 192`** | **NULL — keep 128** | Full e2e: wall +11.4 s, TTFT p90/p99 −0.8%/−1.6%, IoU −0.0021, BFCL −0.20. All inside the noise floor. See `SSM_SLOTS_AB.md`. |
| **`ATLAS_DECODE_GRAPHS_MULTISEQ`** | **DEAD — not applicable** | Advertises "the dominant lever for n>=2 decode (~1500 kernel launches/step)", but `decode_a2.rs` iterates over concurrent *sequences*, not the K verify tokens; at `--max-batch-size 1` with MTP gated to `active.len()==1` the path is never entered. Independently, the serve log shows `Captured CUDA graph for K=4 verify (slot=0)` — the verify path already graphs by default (`verify_c.rs:170`). Rejected without spending a leg. |
| **`ATLAS_SSM_TAIL_MIDCHUNK` re-enable** | **DO NOT FOLD (cost/benefit)** | The `=0` in the frozen config IS a stale workaround — the 2026-07-16 fix is present verbatim in `snapshot.rs::lookup`. But N=3: warm TTFT **median unchanged** (894.1 both), mean −2.7%, sd 107.5→72.5. ~26 s = 0.6% of wall. Not worth re-opening a silent cross-request corruption path. |
| **GEMM tile padding at M≈187** | **NOT WORTH IT** | At the real delta distribution (p50 210 / mean 331 tok), M128→M64 cuts padded rows only 7.4% = 57 s = **1.4% of wall**. |
| **W4A4 prefill + decode** | **DEAD (both axes)** | Prior: 0.995-1.011x speed, 70.6% token match, 7.64 mean KL. See `W4A4_PREFILL_AB.md`. |
| **`ATLAS_BF16_TC_PROJ`** | **no speed case** | Monotonic warm TTFT 990.8 → 1016.5 ms (+2.6%), TPOT ~neutral, tool call fine. Its motivation is accuracy (removes FP8 E4M3 activation crushing on attention QKV/o, which we already avoid on the FFN via `ATLAS_BF16_TC_PREFILL=1`), so it would need an accuracy run to justify — but it does not pay for itself on speed. |
| **SSM snapshot storage dtype** (h_state FP32→BF16/FP16) | **DO NOT FOLD** | Implemented on dgx3 with compute left FP32. Both narrowings FAIL the >=0.99 token-match bar: bf16 0.880, fp16 0.889, ~19% of responses diverge, mean KL ~3. Coherence and tool-call smoke PASS on both, so it is silent trajectory drift. **fp16's two extra mantissa bits buy nothing (0.880→0.889) — the sensitivity is not mantissa-limited**: `h_state` is a recurrent accumulator, so any rounding at restore perturbs everything downstream. And the payoff was already null, since more slots are worth nothing here (see `SSM_SLOTS_AB.md`) — both sides of the trade are dead. |
| **`ATLAS_GDN_REGRESIDENT`** | **FOLD — the one win of the sweep** | Full e2e: **wall 4134.01 → 3834.44 s (−7.25%)**, TTFT p50/p90/p99 −11.4%/−18.0%/−15.8%, **BFCL identical at 87.24**, IoU −0.0048 (exactly on the noise floor). Attribution shows decode is provably untouched, so the defensible win is ~219 s = **5.3% of wall, all TTFT**. Full detail in the RESULT section at the end of this file. |

## The biggest claimed lever, measured and refuted

A parallel investigation on dgx2 established a server-side law,
`server_prefill_ms ~= 170 + 2.0 x tokens_actually_prefilled`, and found that Marconi restores at the
previous turn's PROMPT end, leaving its generated response uncheckpointed and re-prefilled once it
becomes prompt — measured 172-348 tokens in CHAT mode. Projected to this workload that read as
"checkpoint at end-of-generation ⇒ ~660 ms/turn ⇒ ~655 s ≈ 16% of wall, bigger than the decode gap."

**It does not hold here.** Rather than extrapolate, `run_replay_distance.sh` replays the REAL prompt
sequence of a real golden-run conversation (`django__django-16899`, 25 turns, straight out of
`events.jsonl`) and reads the server's own accounting:

```
Marconi intermediate hit: restored from checkpoint at token 1376
  (skipping 1376 tokens, replaying 308 SSM tokens to reach 1684;
   16 of those are the anchor->match gap to 1392)
```

Over 22 intermediate hits:

| quantity | p50 | mean | max |
|---|---|---|---|
| **anchor->match gap (the only waste)** | **16** | **16.0** | **16** |
| new tokens genuinely needed | 342 | 334 | 691 |
| tokens skipped (cache win) | 4696 | — | 8240 |

Total replayed 7691 tokens, of which waste = 352 = **4.6%**; the other 95.4% is content that must be
prefilled regardless. The gap is **exactly 16 on every hit** — one block at `block_size=16`, i.e.
pure block-granularity rounding. **Worth ~32 s = 0.8% of wall, not 16%.**

Why the chat-mode result does not transfer: the harness never echoes the assistant turn back. It
drives a flat prefix-extended prompt and substitutes its own
`[{"id":"functions.bash:0",...}]` JSON, so the model's generated tokens never become prompt and
there is nothing uncheckpointed to replay. This also independently corroborates the
`WALL_DECOMPOSITION.md` finding that the warm fit predicts cold turn-1 within 3% — only possible if
wasted replay is ~0.

**Lesson: a root cause measured in one request-shaping regime does not automatically hold in
another. Verify the mechanism in the target regime before sizing a lever off it.** This one was 20x
over-sized and would have been the largest item in the queue.

## Two defects found in our own instruments

**`kl_coherence_gate.py` could only ever return FAIL.** `kl()` renormalized P over the truncated
top-k support but compared it against raw Q, adding a constant −log(sum p) ≈ **0.061** at every
position. `KL(p,p)` returned 0.061 against a documented `mean_kl < 1e-3` PASS threshold, so a
byte-identical config could not pass its own fold gate. Fixed (`6fdf6f88`), verified exactly 0.0 on
two live controls. No past verdict flips — W4A4 failed at 7.64 KL, far outside the offset — but
every future output-neutral fold would have been wrongly rejected.

**Probe cache invalidation.** Appending a per-rep marker AFTER the delta leaves `base + delta`
identical across reps, so it is itself cached from rep 0 and reps 1..n-1 measure only the marker.
The cell then goes flat in delta (241 ms at 288 chars vs 245 ms at 4320 — +4 ms for 15x the input),
and the p50 is meaningless. The marker must PRECEDE the delta. This invalidated the first
regresident magnitude and produced a bogus "+6 ms for 310 new tokens" on dgx2.

## Method lessons

1. **Check the target workload's issue order before building a probe.** The interleaved slots probe
   predicted a 3.8x TTFT-tail collapse and over-predicted by ~50x, because this benchmark runs one
   conversation at a time (max 0 interleaving) and therefore has no cross-session eviction at all.
2. **N=1 over-sells.** Midchunk looked like mean 991→921 ms with a stable 882 ms floor at N=1; at
   N=3 the median did not move at all.
3. **Match the probe to the mechanism.** A probe that alternates back to the base prompt thrashes
   the single per-session tail slot, so it is the wrong instrument for anything touching tail
   retention — it reported midchunk at 0.76x while the monotonic probe reported a small win.
4. **Trajectory divergence invalidates TPOT, not TTFT.** Whenever legs emit different token counts
   (5.6% here), TPOT is confounded and must not be quoted as a speed result.

## Rig traps

- `inference-endpoint --mode` takes **`acc`**, not `accuracy`. The wrong value errors with a bare
  `Required: --mode`, which reads as a *missing* argument — both accuracy legs failed silently while
  the latency legs completed.
- The GDN path banners (`FLA chunked` / `REGISTER-RESIDENT`) fire on the first prefill or replay,
  **not at startup**, so they must be read after the probes. They are the only proof a flag engaged.
- `ATLAS_*` presence-flags treat `=0` as ENABLED; only `ATLAS_SSM_TAIL_MIDCHUNK` and
  `ATLAS_DECODE_GRAPHS_MULTISEQ` are strict-string tests. Control legs must OMIT the variable.

## RESULT: ATLAS_GDN_REGRESIDENT — FOLD (full e2e, 2026-07-25)

`results/regres_e2e_20260725_071238`, frozen golden config + `ATLAS_GDN_REGRESIDENT=1` only,
both phases 1007/1007 + 995/995, MLPerf runtime seed lock, Gate C2 passed, traceback count 2
(identical to the reference run's benign `apply_chat_template` fallback).

| metric | tip (control) | regresident | delta |
|---|---|---|---|
| **perf wall** | 4134.01 s | **3834.44 s** | **−299.57 s (−7.25%)** |
| tps | 19.25 | 20.08 | +4.3% |
| TTFT p50 / p90 / p99 | 1301.6 / 2544.9 / 5415.6 ms | 1153.7 / 2085.9 / 4560.1 | **−11.4% / −18.0% / −15.8%** |
| TPOT p50 | 32.62 ms | 32.90 | +0.8% |
| **BFCL** | 87.24 | **87.24** | **identical** |
| IoU | 0.6279 | 0.6231 | −0.0048 |
| output tokens | 79576 | 77005 | −3.2% |

### Attribution — the win is TTFT, and decode is provably untouched
| slice | tip | regresident | delta |
|---|---|---|---|
| TTFT total | 1659.8 s | 1440.7 s | **−219.1 s (−13.2%)** |
| decode total | 2474.1 s | 2393.7 s | −80.5 s |

The decode delta is **entirely** the 3.2% drop in emitted tokens: at the tip's 31.09 ms/token the
expected decode at the new token count is 2394.2 s against 2393.7 s actual — residual **−0.5 s**.
Per-token decode cost is unchanged, which is exactly right for a kernel that only touches the warm
replay path, and is a strong internal consistency check.

**Discounting the token-count effect entirely as trajectory divergence, the defensible win is
~219 s = 5.3% of wall, all TTFT.** The probe had projected 1.6-2.2%; the e2e beat it because the
gain is not confined to the median — p90/p99 fell 16-18%, i.e. the kernel also relieves the deep
replays that dominate the tail, which a fixed-delta probe cannot see.

### Caveats carried forward
- **IoU −0.0048 sits exactly ON the measured noise floor** (two identical runs differ by 0.0048).
  Not a demonstrable regression, but not demonstrably clean either — it is the one number that is
  undecidable at N=1. BFCL, the gated metric, is bit-identical at 87.24 against floors 83.64/85.32.
- Not byte-identical at long replay (the 4320-char probe cell differed), so this is a numerics
  change, not an output-neutral one. The accuracy evidence is the subset gate (0.8494 vs 0.8456)
  plus this full 995-sample tie.

**Verdict: FOLD, and flip the default with a kill switch** — it has been default-OFF since
2026-06-28 labelled "until serve-validated", which is precisely the pattern
`feedback_good_defaults_not_flags` exists to stop.

