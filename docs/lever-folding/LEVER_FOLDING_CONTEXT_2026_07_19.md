# Atlas Lever-Folding Context — 2026-07-19

Handoff + corrected understanding. The TTFT + decode levers that matter were all
BUILT on feature branches but NOT folded into the running e2e baseline `53543f55`
(`perf/mtp-reverify-dgx1`). The overnight e2e ran without them, so TTFT and
apples-to-apples decode looked like they "disappeared." This doc is the ground truth.

## TL;DR
- Levers built but not folded: **dense-FFN TC prefill GEMM** (cold TTFT), **GDN tail
  capture** (warm TTFT, beats llama), **MTP K=1** (apples-to-apples BFCL winner).
- What WAS folded: MTP decode, ldmatrix GDN-*projection* GEMM (non-dominant),
  in-place K=4 verify (+0.35% only), fp8kv alignment fix.
- Gate = **ST-995 (2.5h perf) + ST-996 (BFCL tool-calling) TOGETHER**. Run **from
  main, mode = BOTH** — NOT separate ST-995 / `--6` invocations.
- Open box right now: **dgx2 (spark-43fa)**. dgx1 busy (perf_ab_base run), dgx3 serving 35B.

## The gate (corrected)
- Full agentic gate = **ST-995 + ST-996 together**. ST-996 = BFCL tool-calling, 368 steps.
- **Run methodology: from main, mode = BOTH.** Do not run ST-995 and ST-996 separately.
- Baselines to beat:
  - llama.cpp GB10: wall 2h19m, TTFT p50 0.68s, TPOT 84ms, acc 86.55.
  - Prior Atlas apples-to-apples **MTP K=1 ≈ 21.8 tok/s** on ST-996 BFCL
    (`DFLASH_VALIDATION_2026_07_16.md`). THIS is the "mid-20s we had."
  - vLLM 27B-NVFP4: ~20 decode / 14.6 TPS e2e.
- 34.7 decode is unrealistic. Real target: mid-20s decode + TTFT better than llama.

## Levers — BUILT but NOT folded (the gap)
| Lever | Branch + SHA | Effect | In 53543f55? |
|---|---|---|---|
| Dense-FFN TC prefill GEMM (cold-TTFT bottleneck) | `perf/qwen36-dense-ffn-tc-prefill` `7e1ffef9` (w4a16 NVFP4 TC) + `20fbd6ec` loader + `78729bfa` ci; `fix/dense-gemm-tc-atile` `8281eb09` (A-tile coop M>8); `perf/agentic-2.5h-prefill` `ccc1f808` (BF16 TC) + `00a1a59f` (+2% occ) + `3adf30dc` (FP8 k32) | FFN-GEMM-bound 1.74:1 vs attn, 8-12% TC peak → cold TTFT | ❌ |
| GDN tail capture (warm TTFT, beats llama) | `perf/mtp-reverify-midchunk` `312af030` + `f5aa9848` v2 + `2974ec58` session-gate + `d1c70cc0` rebase + `e6566c0b` adaptive-K + ldmatrix A-operand | warm TTFT 2958→1784ms | ❌ (on midchunk, default-OFF) |
| MTP K=1 (BFCL apples-to-apples winner) | shipped default `num_drafts=1` | ST-996 BFCL ~21.8 tok/s | ❌ swapped to K=3 (regressed BFCL) |
| Grammar-constrained drafting | `feat/grammar-masked-mtp-experimental`, `fix/first-token-grammar`, `feat/xgrammar-perf` | could raise ST-996's 19.8% MTP acceptance | ❌ (`--disable-tool-grammar true`) |

## Already folded in 53543f55 (confirmed)
- 9 MTP/ldmatrix cherry-picks (MTP decode, drafter prefill, K=3/K=4 batched verify,
  ldmatrix GDN-projection GEMM `a80b68d7`, F16 loader, MTP-proj dequant).
- in-place GDN K=4 verify `73f331da` — byte-identical, but +0.35% (the ~120MB D2D copy
  is 0.44ms vs 127ms verify step; compute-bound, not copy-bound).
- fp8 cp.async pipelining + 16-byte alignment fix `53543f55` — fp8 KV 1/8→8/8, prefill
  parity, 2× KV capacity.
- #278 SSM tail-protect; #229 W4A4 MMQ FFN + FlashInfer-GDN + Marconi tail-checkpoints.

## Why the overnight e2e looked flat
- TTFT levers (FFN TC prefill + tail capture) NOT in baseline → cold TTFT stayed
  ~5.3s (BFCL online_edge) / p99 4s (ST-995).
- Decode: raw microbench still 21-23 tok/s (mid-20s ✓), but e2e on BFCL tool-calling
  regressed because K=1→K=3 swap + 19.8% acceptance → verify overhead dominates short
  tool-call outputs. K=3 did NOT beat K=1 on BFCL.
- ldmatrix (folded) hit the GDN-*projection* GEMM (non-dominant); the dominant
  dense-FFN GEMM sat on its own branch.

## Fold plan (CORRECTED 2026-07-19 — NO code surgery / NO cherry-pick needed)
The levers are already IMPLEMENTED in baseline `53543f55` / midchunk — they are
**env-gated OFF** and NO run script enables them. That is the entire "why is this
not on???" The "fold" = build from midchunk (tail capture already folded at
`e6566c0b`) + **enable the env vars in the serve command**.

**DO NOT cherry-pick `7e1ffef9`.** Its FP8 `w8a16_gemm_t_m128` path was deliberately
DISABLED in HEAD for accuracy ("perturbs generation, length-truncations / accuracy
risk on Qwen3.6-27B") and replaced by the bit-identical `ATLAS_BF16_TC_PREFILL`.
Re-introducing it would REGRESS accuracy. The dense-FFN TC prefill is already in the
baseline via newer, safer paths.

Run config (dgx2, from main, mode BOTH):
- **`ATLAS_FFN_NVFP4_MMQ=1`** — dense-FFN TC prefill, ~80 TFLOP/s (vs t_m128 ~51),
  accuracy-restored (#229). PRIMARY. A/B vs `ATLAS_BF16_TC_PREFILL=1` (bit-identical,
  safest) or `ATLAS_INT8_PREFILL=1` (cosine 0.99998).
- **`ATLAS_SSM_TAIL_MIDCHUNK=1`** — GDN tail capture (warm TTFT 2958→1784ms).
- `ATLAS_MTP_DRAFTER_PREFILL=1` — drafter context prefill (already used).
- MTP **K=1** (apples-to-apples ST-996 BFCL winner ~21.8 tok/s) or adaptive-K (`e6566c0b`).
- bf16 KV (pinned, reproducible).
- **Image**: `atlas-gb10:midchunk-adapk-ldmab` (`d24689b95bee`, from `e6566c0b`, already
  built 2026-07-19 ~15:00 — no rebuild needed).
- **Gate order**: coherence 14/14 + cold-TTFT probe FIRST (confirm TC prefill engages +
  TTFT drops), THEN ST-995+ST-996 mode BOTH.

Env vars confirmed in baseline `53543f55` (git grep): ATLAS_FFN_NVFP4_MMQ (3 files),
ATLAS_INT8_PREFILL, ATLAS_FP4_PREFILL, ATLAS_BF16_TC_PREFILL, ATLAS_FP8_M64_PREFILL.
ATLAS_SSM_TAIL_MIDCHUNK only on midchunk branch (0 files at 53543f55).

## RUN STATUS (2026-07-19 16:21 — launched, levers confirmed firing)
- **Image**: `atlas-gb10:midchunk-adapk-ldmab` (`d24689b95bee`) shipped to dgx2 (spark-43fa, 10.10.10.2).
- **Serve**: `atlas-lever-dgx2` on dgx2:8888, nvidia/Qwen3.6-27B-NVFP4, bf16 KV, util 0.70,
  `--num-drafts 3` (adaptive-K gate). Env: `ATLAS_FFN_NVFP4_MMQ=1` + `ATLAS_SSM_TAIL_MIDCHUNK=1`
  + `ATLAS_MTP_DRAFTER_PREFILL=1`. Weights at `/home/claude/.cache/huggingface` on dgx2.
- **Lever engagement (confirmed in serve log under real traffic)**:
  - `ATLAS_FFN_NVFP4_MMQ=1: dense-FFN gate/up prefill via vendored llama NVFP4 W4A4 MMQ (block-scale FP4 MMA, ~80 TFLOP/s vs t_m128 ~51)` ✅
  - `GDN prefill: FLA chunked path ACTIVE` ✅
  - `MTP drafter prefill: 1204 positions in 113.2 ms` ✅
  - Adaptive-K gate: `measured_effective=4.00 (mean_accepted=3.00) => ENABLED`, re-measures on
    depth-regime change, `mean_accepted=2.25 => ENABLED` ✅ (the "disable MTP when net-negative" behavior)
- **Coherence gate**: tool_ok=8/8, mangled=0/8, empty=0/8 ✅ (MMQ did NOT break tool calls).
- **Cold-TTFT probe (vs prior 53543f55 no-MMQ)**: cold_short 498→387ms (−22%), cold_long 514→352ms
  (−32%), decode ~20.4-20.8 tok/s (mid-20s held). MMQ lever is real.
- **E2e**: `inference-endpoint benchmark from-config --config run_lever_on_dgx2_20260719_161719.yaml`
  (both phases = mode BOTH) from dgx1 against dgx2:8888. Perf phase 2002 samples ~5.1s/it (~2h47m),
  then BFCL accuracy (~3h). Report dir: `results/lever_on_dgx2_20260719_161719/`.
  Log: `/workspace/e2e_lever_on_dgx2_<stamp>.log`. Background task `bvwmpn3jg`.
- **Apples-to-apples targets**: prior MTP K=1 ~21.8 tok/s on ST-996 BFCL; llama (wall 2h19m, TTFT 0.68s,
  TPOT 84ms, acc 86.55); vLLM 14.6 TPS e2e. This run = K=3+adaptive-K (gate auto-disables on low
  acceptance) + MMQ TC prefill + tail capture — the levers that were env-gated OFF before.

## RUN STATUS — w4a4 mlpinf levers-ON (2026-07-19 16:32, the scoped run)
Switched from nvidia NVFP4 to **centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf** (the scoped variant).
Same image `atlas-gb10:midchunk-adapk-ldmab` (`d24689b95bee`, e6566c0b) on dgx2:8888.
- **All levers ON**: `ATLAS_FFN_NVFP4_MMQ=1` (dense-FFN W4A4 MMQ ~80 TFLOP/s; centml is
  NVFP4 quant so MMQ applies) + `ATLAS_SSM_TAIL_MIDCHUNK=1` (warm TTFT) +
  `ATLAS_MTP_DRAFTER_PREFILL=1` + **grammar ON** (`--disable-tool-grammar false` →
  grammar-masked MTP) + `--num-drafts 1` (K=1, grammar-optimal + apples-to-apples vs
  prior 21.8) + adaptive-K gate + bf16 KV. Weights at `/workspace/.cache/huggingface` (centml).
- **Gate GREEN**: coherence 8/8 (mangled=0/empty=0); "Grammar constrained decoding
  active: parser=qwen3_xml, tools=2" firing under traffic; adaptive-K `measured_effective=2.00
  (mean_accepted=1.00) => ENABLED`; cold TTFT cold_short 384ms / cold_long 377ms (−27% vs
  no-MMQ), decode ~20-22 tok/s (at prior 21.8 apples-to-apples level).
- **E2e**: `inference-endpoint benchmark from-config --config run_lever_on_w4a4_<stamp>.yaml`
  (mode BOTH) from dgx1 → dgx2:8888. Perf 2002 samples ~4.1s/it (~2h14m) + BFCL accuracy (~3h).
  Report `results/lever_on_w4a4_<stamp>/`. Log `/workspace/e2e_lever_on_w4a4_<stamp>.log`.
  Background task `b7e4mql9k`.

## Iteration plan (watchers + never-stop-iterating)
- **dgx2**: w4a4 levers-ON e2e running (~5.5h). Watcher = e2e task notifies on completion/exit.
- **dgx3**: was busy serving Qwen3.6-35B-A3B-FP8 (another session, active traffic — NOT
  reclaimed). Watcher `bpaih2kzh` polls every 90s; notifies when dgx3 frees.
- **On dgx3-free** (next iterate): launch env-on A/B probes (no build needed) —
  `ATLAS_BF16_TC_PREFILL=1` (bit-identical, zero accuracy risk) vs `ATLAS_INT8_PREFILL=1`
  (cosine 0.999978) vs the running MMQ config; + `--kv-cache-dtype fp8` (2× KV capacity,
  now fixed). Coherence 8/8 + cold-TTFT A/B each. Picks the accuracy-safe FFN lever.
- **Grammar-multi-draft BUG#4 fix** (`ad73795b`, feat/grammar-masked-mtp-experimental):
  unlocks correct K=3+grammar (per-position DraftMaskProvider vs the image's held-fixed
  mask). Cherry-pick onto e6566c0b has 5 scheduler verify-file conflicts (real semantic
  merge: HEAD's `effective_drafts_under_grammar`+adaptive-K vs ad73795b's DraftMaskProvider).
  QUEUED — resolve carefully + cargo check + build, then test K=3+grammar acceptance on
  dgx3. Lower priority (K=1+grammar already works on dgx2).
- **Apples-to-apples targets**: prior MTP K=1 ~21.8 tok/s (ST-996 BFCL); llama (wall 2h19m,
  TTFT 0.68s, TPOT 84ms, acc 86.55); vLLM 14.6 TPS e2e.

## Cherry-pick attempt log (for the record)
- Created worktree `perf/lever-fold-all` off `perf/mtp-reverify-midchunk` (`e6566c0b`).
- `e780498f` (Dockerfile vendor COPY) = already in midchunk (empty cherry-pick, skipped).
- `7e1ffef9` (TC prefill) → 10 semantic conflicts in `dense_ffn.rs` (2 blocks 240-400
  lines) + 1 in `fp8_lut.rs`. Root cause: HEAD already absorbed the TC infra (kernel
  handles, `w8_gemm!` macro, transposed-weight scaffolding) and evolved past
  7e1ffef9's old `6ff19169` base; AND HEAD disabled 7e1ffef9's exact FP8 path for
  accuracy. **Aborted** — superseded, not needed.

## Gotchas (verified, don't re-derive)
- **Run from main, mode BOTH** — ST-995+ST-996 together, not separate `--6`.
- **bf16 KV intentionally pinned** (MODEL.toml `default_kv_dtype`; fp8 paged-prefill
  was 10% slower). fp8 KV now fixed (`53543f55`) + available via `--kv-cache-dtype fp8`
  for 2× KV capacity. e2e scripts pin bf16 for reproducibility.
- **Don't build from `6d514216`** (toolcall-hardening regresses tool calls 0/8). Base
  = `56aa136e` / `53543f55`.
- **Push method**: plain `git push` fails (bad creds + 6.3MB LFS). Use gh token +
  `x-access-token` URL + ≥180s timeout. See memory `atlas-git-push-method`.
- **Grammar disabled by default** (`--disable-tool-grammar true`). Enabling could
  raise 19.8% acceptance (testable), NOT cap thinking (thinking already off).
- **ST-996 = BFCL tool-calling, 19.8% MTP acceptance, 109/368 accept zero** — the
  decode killer on that half.
- **Multi-session**: dgx1 has a `perf_ab_base` run (16:02); dgx3 serving 35B. Use
  dgx2. Poll contention on Sonnet per CLAUDE.md protocol.

## Key files / refs
- Repo `/workspace/atlas`; worktrees: `atlas-mtp-reverify` (53543f55),
  `atlas-midchunk` (perf/mtp-reverify-midchunk), `atlas-opt-fp8kv`, `atlas-opt-inplace`.
- `/workspace/AGENTIC_2.5H_GOAL.md` — agentic gate + prefill root cause (FFN GEMM).
- `/workspace/DFLASH_VALIDATION_2026_07_16.md` — ST-996 = BFCL tool-calling, 19.8%
  acceptance, prior MTP K=1 21.8 tok/s.
- `/workspace/ATLAS_COMPETITIVE_AUDIT_2026_06_12.md` — 39-45 decode on 35B MoE; prefill ~2100 tok/s.
- `/workspace/OVERNIGHT_STATUS.md` — overnight K=3 results + fold state.
- `/workspace/st995_overnight.log` — ST-995 breakdown (decode p50=13.6, TTFT p50=0.86,
  decode 87% of per-sample time).
- `/workspace/llama_gb10_serve.log` — llama baseline.

## Memory
- `mtp-reverify-campaign` — campaign state + gotchas (predecessor).
- `atlas-lever-folding-2026-07-19` — this doc's summary (project memory).
- `atlas-git-push-method` — push auth.
