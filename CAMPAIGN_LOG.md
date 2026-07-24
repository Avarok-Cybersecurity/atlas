# Decode-gap campaign — running log (autonomous, goal-driven)

**Goal:** close the raw boosted-MTP decode gap vs vLLM on GB10. Deliver: completed conglomerate
e2e + pushed/confirmed folded wins (git-replicable + exact run cmds) + this running log + scoped
pieces from profiling. Started 2026-07-24 03:21 EDT, deadline +6h (~09:21).

## Target (validated basis — see DECODE_FOLD_LEDGER.md "RE-BASELINE")
Real vLLM (concise, confirmed by user): perf wall 5361s, qps 0.188, tps 14.6, IoU 0.6269;
accuracy 995 BFCL 86.43. Atlas already ≥ on tps/qps/wall/BFCL, ties IoU. **Pure decode gap:**
K=3 spec step ~112ms → 40ms/tok effective vs vLLM ~87ms → 31ms/tok. **~25ms/step to eliminate.**
Base non-spec decode ~63ms (memory floor, shared). Extending the lead, not surviving.

## DGX delegation (roles — do NOT cross-assign GPU work)
- **dgx1** (10.10.10.1): git FOLDING + VALIDATION (build, coherence+KL-drift+regression gate, A/B), coordination, qwen consults.
- **dgx2** (10.10.10.2): E2E runs (conglomerate + per-win confirmation). GPU-dedicated.
- **dgx3** (10.10.10.3): UTILIZATION / PROFILING (nsys phase-split, microbench). Respect any flagship.

## Gate (EVERY fold must pass, in order — no fold on plausibility)
1. Build clean (correct target: gb10, qwen3.6-27b, nvfp4; 157 kernels).
2. **Coherence / Gate-C2 NVFP4 smoke** — coherent English + valid tool call at temp 0.
3. **KL logit drift** — top-logprob KL(baseline‖candidate) on fixed prompts; PASS if mean KL < 1e-3
   (an output-neutral decode change → ~0; a numeric change is quantified here).
4. **Barebones regression** — BFCL subset (>=50) not below baseline; + measured TPOT A/B N>=3.
5. qwen adversarial review of raw diff + numbers before fold.
Win → commit immediately (tbraun96 author, Atlas co-author, no Claude attribution) → push.

## Exact reproduce commands
Build: `cd <worktree> && PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server --bin spark --features cuda`
Serve (frozen c2final, K=3): see DECODE_FOLD_LEDGER.md "Serve config".
Gate: `python3 kl_coherence_gate.py <baseline_port> <cand_port>` ; A/B: `bash draft_sweep.sh`-style.
e2e: endpoints-fresh edge-agentic-full-run config (temp0/seed42), 1007 perf + 995 BFCL.

## Iteration log (append-only)
- **03:21** Campaign start. State: L1 DEAD, L6 DEAD (both epilogue fusion, no TPOT win, byte-identical).
  dgx3 nsys phase-split RUNNING (port 8890). dgx2 e2e 985/2002. Branch @ a8fd2b52 (main + ledger).
  Pending: dgx3 phase split → localize the ~25ms/step → qwen ideate → build → gate → fold.

## Scoped pieces (from profiling — fill as results land)
- [pending dgx3] where the 25ms/step lives: drafter-propose (autoregressive, confirmed 2 serial
  forward_one passes: 1 MTP layer + lm_head M=1 each) vs M=3 verify efficiency vs launch bubbles.

## SCOPED LEVER (user-flagged 03:2x): W4A4 activation-quant NOT exploited in decode
FINDING: GB10 decode/verify uses `w4a16_gemv` / `w4a16_gemv_batchm` = W4 weights + **bf16 (A16)
activations** (quant_dispatch.rs:35,185; impl_a3.rs:169). Model is W4A4 (NVFP4 acts available), but
decode dequants acts to bf16. NO w4a4/w4a8/dp4a decode-GEMV kernel exists for gb10
(kernels/gb10/common: only w4a16_gemv, w8a16_gemv, dense_gemv_bf16/fp8w). strix banked W4A8 DP4A
(v_dot4 int8) = +25% MTP-verify GEMV. GB10 has native sm_121a FP4 MMA (~2x FP8, per fp4_mma_gb10)
UNUSED in decode. → candidate lever for qwen #2 (M=3 verify efficiency). Asking qwen GB10 vs gfx1151.

## 03:30 — ACTIVE COMPONENTS (all boxes working)
| box/agent | piece | status |
|---|---|---|
| dgx3 (agent) | nsys phase-split of K=3 step: drafter-propose vs M=3 verify vs bubbles; M=1-vs-M=3; non-spec T_no_spec | RUNNING (serve :8890 under nsys) |
| dgx2 | full MLCommons e2e on main 011bee65 (baseline confirm) | RUNNING ~1027/2002 |
| dgx1 (agent) | BUILD+microbench W4A4 verify GEMV (native NVFP4/E2M1 acts) vs w4a16(bf16 acts); microbench-first bandwidth gate | RUNNING (worktree .wt-w4a4) |
| dgx1 (qwen) | GB10 sm_121a FP4 vs gfx1151 int8 DP4A — activation-quant verdict | RUNNING (w4a4_consult.txt) |
| dgx1 (coord) | gate harness (kl_coherence_gate.py) + conglomerate launcher + this log | DONE, committed |

## CROSS-HARDWARE LEARNING (first-class theme — exploit base W4A4 weights + tricks everywhere)
The MLPerf checkpoint is NVFP4 **W4A4** — weights AND activations 4-bit — but GB10 decode only
exploits W4 (weights); activations run bf16. Bank of activation-quant tricks to port BOTH ways:
- **gfx1151 (strix, RDNA3.5):** no native FP4 MMA → **W4A8 int8 DP4A (v_dot4)** banked **+25% MTP-verify GEMV**.
- **GB10 (sm_121a, Blackwell):** has **native FP4 MMA (~2× FP8)** + int8 tensor cores — neither used in decode GEMV.
- OPEN QUESTIONS (qwen + microbench deciding): (1) at M<=3 the verify GEMV is weight-memory-bound
  (4-bit weights streamed regardless of act precision) → does act-quant help at all, or only the
  bf16-dequant *overhead*? (2) GB10 native-FP4 (W4A4) vs int8 (W4A8) — which wins, and does strix's
  +25% translate? (3) is FP4 MMA even usable at M=3 (tensor cores want M>=8) or must it be a GEMV?
- PRINCIPLE: verify by MEASUREMENT (microbench GB/s vs 273 peak). If w4a16 is already bandwidth-
  saturated at M=3, W4A4 cannot help and we've *confirmed we exploit W4A4 fully* — a valid result.
- Prior art to reuse: e2m1_branchless.cu, quantize_bf16_to_nvfp4.cu, dequant_nvfp4_bf16.cu,
  inferspark_prefill_paged_nvfp4.cu (FP4 MMA). Kernel toml/precedence: fp4_mma_gb10 memory.

## ★★★★★ 03:40 — dgx3 PHASE-SPLIT: THE GAP LOCALIZED (all 4 hypotheses REFUTED)
K=3 step 115ms, **96.2% GPU-busy** (idle 4.4ms). Breakdown:
| phase | ms | % |  |
|---|---|---|---|
| **verify projection GEMVs (M=3)** | **78.1** | **68%** | w4a16_dual_batch3 27%, **w8a16_batch4 24%(FP8!)**, w4a16_batch3 16%, qg 2% |
| full-attn (16 layers KV) | 14.6 | 12.7% | |
| MTP drafter propose (2 drafts) | 10.3 | 9.0% | ~1-layer head, NOT a 2nd model pass |
| lm_head verify | 3.1 | 2.7% | |
| GDN wy3 recurrence | 1.8 | 1.6% | batched, confirmed |
| GDN conv/norm epilogue | 1.0 | 0.9% | ← confirms L1/L6 rightly DEAD |
| D2D rollback + sampling + misc | 2.7 | 2.4% | |
| GPU idle | 4.4 | 3.8% | |
Verdicts: H1 drafter-serial REFUTED (10ms). H2 M=3-not-weight-once REFUTED (M3/M1=1.03-1.09×).
H3 bubbles REFUTED (96% busy). H4 rollback REFUTED (0.9ms). T_no_spec=91ms → spec HELPS ~1.8×.
**ROOT CAUSE: the projection GEMVs are weight-once but run at only ~60-65% of peak LPDDR5X BW
(~1.5-1.7× the bytes/BW floor). That inefficiency IS the 2× overhead.**
**LEVER #1 (the fix): bandwidth-tune w4a16_gemv_dual_batch3 + w8a16_gemv_batch4 + w4a16_gemv_batch3
(78ms) from 60-65% → 85-90% peak → 78→~56ms ≈ recover ~9ms/token.** Kernel access-pattern work
(coalescing, 16B vectorized weight loads, occupancy, unroll) — NOT act-quant (M=3 acts are tiny).
**LEVER #2 (W4A4-refined): w8a16_gemv_batch4 GDN in/out proj is FP8 = 2× NVFP4 bytes; W4 it → save
~12ms/step, BUT GDN FP8 is mandated (accuracy risk) → gate hard on KL/BFCL.**
Pivot: the W4A4 agent's act-quant premise is refuted; redirect to GEMV BANDWIDTH tuning (lever #1)
+ assess W4-for-FP8-GDN (lever #2). Artifacts dgx3:/workspace/decode_phasesplit_20260724_025300/.

## 03:45 — qwen W4A4 VERDICT (verifies "are we exploiting W4A4 fully": YES on weights)
qwen + phase-split AGREE: W4A4/W4A8 **activation-quant is NOT a lever at M<=3**. Per-projection at
M=3: weights ~9.44MB vs acts+out ~48KB → **acts are ~0.5% of traffic**. FP4 MMA is tile-based (pad
M=3→M=16 = 18.75% lane util); w4a16_gemv already reads weights ONCE amortized over M rows. Kernel is
memory-LATENCY bound (long-scoreboard stalls), not activation-precision bound. strix +25% W4A8-DP4A
does NOT transfer (that was a lightweight int dot, not FP4 MMA; GB10 baseline already weight-once).
→ **We exploit the 4-bit WEIGHTS fully; activations at M=3 are irrelevant.** Act-quant CLOSED.
Cross-hw learning banked: gfx1151 int-DP4A vs GB10 FP4-MMA differ fundamentally at low M; the
transferable trick is weight-once batched GEMV (both have it), NOT act-quant.

**Confirmed levers (weight-bandwidth, not act-precision):**
- LEVER #1 (primary): weight-load mainloop BW/latency tuning of w4a16_gemv_batch3/dual/qg + kernel
  occupancy → 60-65%→85-90% peak → ~+9ms/tok. Kernel agent on it; qwen BW-design in flight.
- LEVER #2: W4/NVFP4 the FP8 GDN in/out proj (w8a16_gemv_batch4, 24% step, 2× NVFP4 bytes) → halve
  traffic ~12ms/step. GDN-FP8 mandated → HARD KL/BFCL accuracy gate before fold.

## 03:35 — LEVER #2 REINSTATED (owner directive): W4A8 int8-act GEMV (strix trick, measured not dismissed)
Owner: the strix W4A8 (int8-act v_dot4 DP4A) verify-GEMV works there (+25%), so build it on GB10 and
let IoU+accuracy decide — do NOT dismiss on qwen's theory. Per heavylift disagreement protocol
(measurement decides). Mechanism at M=3: int8 acts don't cut DRAM traffic (weights dominate) but cut
mainloop compute/register pressure → better memory-LATENCY hiding → lift off the 60-65% BW ceiling
(qwen itself listed this as the only possible win). Kernel agent now has THREE candidates:
  (1) pure access-pattern BW tune of w4a16 (byte-identical, primary)
  (2) **W4A8 int8-act GEMV** (port strix w4a16_gemv_dp4a / quantize_act_int8_g16 → gb10) — KL-gated
  (3) [assess] W4/NVFP4 the FP8 w8a16 GDN proj
Each reported with M=3 GB/s + (byte-identical | KL vs bf16 ref). HARD gate before fold: faster at M=3
AND IoU/BFCL clear (int8 acts lose precision → full regression, not just microbench KL).

## WATCHERS (auto-notify → loop self-drives)
- qwen GEMV-BW-tuning consult (beu3jsnfa) — design for candidate #1.
- kernel agent (a877fb6b) — 3 candidates, dgx1 worktree .wt-w4a4.
- dgx2 baseline e2e completion (blz3264q2, polls 90s) — frees e2e box for conglomerate.
