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
| **`ATLAS_GDN_REGRESIDENT`** | **strongest candidate; re-measuring** | Token-equal to WY4 by construction (cos 1.0, max\|dH\|~1e-8) and confirmed MATCH on the probe. First-round monotonic warm TTFT 990.8 → 895.1 ms (**−9.7%**). Both rounds are N=1 with diverged trajectories, so the magnitude is not yet trustworthy — re-run in flight. |

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
