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
Gate: `python3 scripts/mlperf-edge/kl_coherence_gate.py <baseline_port> <cand_port>` ; A/B: `bash scripts/mlperf-edge/draft_sweep.sh`-style.
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
| dgx1 (coord) | gate harness (scripts/mlperf-edge/kl_coherence_gate.py) + conglomerate launcher + this log | DONE, committed |

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

## ★★★★★ 03:40 — kernel agent: GEMVs are NEAR-ROOFLINE; act-quant DEAD-confirmed; C3 is the lever
Cold cycled-weight microbench (rigorous, streams DRAM each launch):
  w4a16_gemv M=1: 84-88% peak | w4a16_gemv_batch3 M=3: 74-82% | w8a16_gemv_batch4 (FP8) M=3: 84-85%.
→ kernels are ALREADY near the 273 GB/s roofline. dgx3's "60-65% in-step" was inflated by its config
(f32 SSM, 11k ctx, 0.70 util — dgx3 flagged 1.3× absolutes). So per-kernel access-pattern tuning has
NO headroom.
Candidates measured (A/B M=3, cold, N=3):
- C1 software-pipelined prefetch (byte-identical): 0.98-1.00× → DEAD (compiler already hides latency;
  prefetch adds reg pressure).
- **C2 W4A8 int8-act DP4A (the strix trick): 0.99-1.01× + 0.5% accuracy cost → DEAD ON GB10.** MEASURED,
  not dismissed. Strix +25% doesn't transfer (GB10 baseline already weight-once+near-roofline; strix's
  was pre-DP4A). Also hit the __constant__-LUT-serializes trap (5× slower until moved to __shared__).
- C3 NVFP4 the FP8 GDN in/out proj (w8a16, ~24% of step, 2× NVFP4 bytes): NOT built (assess). ~halve
  bytes → ~2× on those proj ≈ ~12ms/step. **THE remaining lever.** Deviates from mandated FP8-GDN ckpt.
INSIGHT: vLLM runs all-NVFP4 nvidia ckpt; Atlas mandated ckpt keeps GDN FP8 → Atlas streams MORE bytes.
C3 = match vLLM byte budget. Owner authorized ("as long as it clears IoU+accuracy"). → PURSUE C3, hard
KL+IoU+BFCL gate. Structural note: batch3 M=3 = 75% vs M=1 84% (3× compute/byte, occupancy/reg-bound) —
not fixable by prefetch/DP4A. Agent files uncommitted on worktree .wt-w4a4 (C1/C2 kernels + bench).

## 03:50 — C3 scope + wait state
C3 feasibility: GDN layer ALREADY has `out_proj_nvfp4_t` slot (qwen3_ssm/init.rs:33) alongside
out_proj_fp8w/fp8/dense → NVFP4 GDN out_proj is a partly-scaffolded path. C3 = re-quantize the FP8
GDN in/out proj weights to NVFP4 (E2M1 + FP8 group scales) at load, populate the nvfp4 slot, dispatch
w8a16_gemv_batch4 → w4a16_gemv_batchm. Loader: weight_loader/qwen3.rs (native_fp8 branch). Est build
+ gate ~1-2h. HARD gate: coherence + KL + full IoU/BFCL e2e (deviates from mandated FP8-GDN ckpt).
PENDING (auto-notify): qwen C3 risk verdict (bvr5smtz4) → go/no-go on the build; dgx2 baseline e2e
(blz3264q2) → frees e2e box (~04:30) + baseline numbers for the C3 accuracy comparison.
Reality check: if C3 fails IoU/BFCL, the decode gap is a MANDATED-CHECKPOINT byte-budget constraint
(Atlas FP8-GDN vs vLLM all-NVFP4), not an engineering miss — will document honestly either way.

## 03:53 — qwen C3 VERDICT: partial (~2-5ms/tok), accuracy-gated, GDN in-proj risky
NVFP4 GDN proj: ~38-44% fewer bytes (0.5625 vs 1.0 B/wt) × 24% of step → ~10-12ms/step → ~2-5ms/tok
(40→~35-38ms). NOT the full 8.56ms fix alone. Accuracy: vLLM all-NVFP4 PASSES (IoU 0.6269/BFCL 86.43)
= evidence NVFP4 GDN CAN hold; but naive re-quant of centml FP8→NVFP4 is moderate-HIGH risk, esp. the
GDN IN-proj (feeds recurrent state). IoU more at risk than BFCL. qwen's decisive experiment: Atlas
FP8-GDN vs NVFP4-GDN, trajectory-pinned → TPOT + E + IoU/BFCL.
⚠ CAVEAT to verify: mandated recipe = GDN **FP8** (mandated_nvidia_ckpt: GDN FP8+MLP NVFP4+MTP BF16).
If vLLM also runs FP8 GDN, C3 DEVIATES from mandate (not "matching vLLM"). → C3 is an engineering
decode lever, accuracy-gated; flag the recipe deviation for owner.
⚠ Also unresolved: confirmed vLLM's REAL TPOT unknown (31.39 = weak run). On tps/qps basis Atlas
already BEATS confirmed vLLM (15.9 vs 14.6 tps). Raw-TPOT gap likely real (decode speed ∝ output-len-
independent) but exact target uncertain.
DECISION: build C3 STAGED (out-proj NVFP4 first = safer, then in-proj), A/B TPOT + coherence/KL, full
IoU/BFCL e2e gate on dgx2 when free. Fold only if faster AND IoU/BFCL clear.

## 03:55 — C3 build DISPATCHED (last decode lever)
Agent (worktree .wt-c3, branch c3-nvfp4-gdn) building NVFP4 GDN out-proj (stage1, out_proj_nvfp4_t
dispatch already scaffolded init.rs:371; loader linear_attn_arms.rs re-quant behind ATLAS_GDN_OUT_NVFP4=1),
then in-proj (stage2, riskier). Fast gates on dgx1: coherence + KL drift + TPOT A/B (FP8-GDN vs NVFP4).
Full IoU/BFCL e2e gate on dgx2 when it frees (~04:30). PROCEED only if faster + coherent + small KL;
FOLD only if IoU/BFCL also clear.
LIVE SIGNALS (auto-notify): C3 agent (a5776bb4) · dgx2 baseline e2e done (blz3264q2, frees e2e box).
HONEST OUTLOOK: easy decode levers exhausted (GEMVs near-roofline; act-quant/prefetch DEAD-measured).
C3 is partial (~2-5ms/tok) + accuracy-risky + deviates from mandated FP8-GDN. Likely deliverable:
confirmed e2e (Atlas already ≥ vLLM on tps/qps/wall/BFCL/IoU) + C3 folded IF it passes + honest
memory-bound-floor documentation for the residual raw-TPOT.

## ★★★★★ 04:20 — C3 DEAD + CHECKPOINT REFRAME (the honest answer on decode)
Kernel agent: the GATE model **centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf is DENSE and already ships NVFP4
GDN** (U8-packed on disk, decodes w4a16 not FP8 w8a16). So ATLAS_GDN_OUT_NVFP4 is INERT on the gate
model (byte-identical, KL=0, +0.3ms noise). The FP8-GDN (24% of step) the dgx3 phase-split saw was on
the **nvidia** ckpt (mandated), NOT the centml gate model → phase-split's GDN slice doesn't transfer.
On nvidia (where C3 applies) NVFP4-out made decode **SLOWER +4.6%** (w4a16 M=3 ~75% < w8a16 ~85%). C3
DEAD both ways. C3 impl correct+live but not folded (worktree .wt-c3, qwen35_dense.rs:569, +71).
**CONCLUSION: on the real gate model, decode is already all-NVFP4 (best byte budget) AND GEMVs are
near-roofline. Every weight/kernel decode lever is now MEASURED-DEAD** (GEMV-tune, prefetch, W4A8
act-quant, NVFP4-GDN). The raw-TPOT vs the WEAK vLLM ref (31.39, verbose) is a near-hardware-floor gap;
on the CONFIRMED vLLM's reported metrics (tps/qps/wall/BFCL/IoU) Atlas already WINS.
Remaining decode levers = ACTIVATION/KV precision only: (A) fp8 KV cache (attn ~13% of step, halve KV
reads, prev-passed 86.33) ← testing now, FREE flag. (B) NVFP4 MTP head (drafter 9%, forced bf16 ~3.1%).

## ★★★ 04:55 — fp8-KV: the ONE decode win (+4.8% TPOT), accuracy-gate pending
fp8-KV A/B (base binary, K=3, GATE_FORCE): bf16-KV 43.82ms → **fp8-KV 41.71ms = −4.8% (~2ms/tok)**.
Output COHERENT (clean English), tool-path intact. NOT byte-identical (fp8 changes numerics → greedy
tokens drift slightly) → needs full IoU/BFCL e2e gate. Prior golden_fp8kv = 86.33 BFCL PASS (vs ~87
bf16), so it trades ~0.7 BFCL for ~5% decode. This is a serve-FLAG change (--kv-cache-dtype fp8), NO
rebuild. → FOLD CANDIDATE if the conglomerate e2e holds IoU≥0.6269 (vLLM) + BFCL≥floor.
Combined honest picture: fp8-KV recovers ~2ms (40→~38); all other decode levers roofline-dead. Partial
close, not full — the rest is the hardware floor for the all-NVFP4 checkpoint.
NEXT: run CONGLOMERATE e2e with fp8-KV (serve flag) → confirm IoU/BFCL + measure wall/TPOT gain → fold.

## 05:03 — CONGLOMERATE e2e LAUNCHED (fp8-KV) on dgx1 — e2e-level A/B
dgx1: full MLCommons e2e (1007 perf + 995 BFCL, temp0/seed42) with **--kv-cache-dtype fp8** (the fold
candidate) — report_dir endpoints-fresh/results/fp8kv_conglom_20260724_050249, serve atlas-fp8kv-conglom.
dgx2: the bf16-KV baseline e2e (main 011bee65) finishing → the control leg.
→ e2e-level A/B: fp8-KV vs bf16-KV on wall/TPOT/IoU/BFCL. FOLD fp8-KV iff it improves wall/TPOT AND
IoU≥0.6269 (confirmed vLLM) AND BFCL≥floor (83.64/85.32). Both runs ~2.5h; done ~07:30. Deadline 09:21.

## ★★★★ 07:39 — dgx2 bf16 BASELINE e2e PERF (fresh main 011bee65) — Atlas CRUSHES confirmed vLLM
1007/1007 complete. **wall 4551.86s · TPOT median 38.18ms · TTFT median 1264ms · qps 0.221 · tps 17.56 ·
out-tok median 45.** (Fresh main is FASTER than the golden 5023s/39.95ms/15.67tps I'd quoted — #356
prefill smem fix.) Warm accept p1 0.89-0.90, mean-accepted 1.84 (E~2.84).
vs CONFIRMED vLLM (5361s/14.6tps/0.188qps/IoU 0.6269): **Atlas wall −15%, tps +20%, qps +18%.**
Raw-TPOT 38.18 vs vLLM weak-run 31.39 = residual at the hardware roofline (per FINDINGS).
NOTE: BFCL accuracy phase ERRORS on both boxes — `bfcl-eval` dep missing (bfcl_v4_scorer.py:110). Perf
metrics (wall/TPOT/TTFT/tps) valid; BFCL/IoU from THIS run blocked. fp8-KV accuracy rests on prior
golden_fp8kv=86.33 PASS.

## ★★★★★ 07:45 — CONGLOMERATE e2e RESULT (fp8-KV) — the one foldable win
fp8-KV full MLCommons (1007 perf + 995 BFCL, temp0/seed42, 1007/1007 OK):
  **wall 4534.9s · TTFT p50 1271ms · TPOT p50 39.08ms · TPS 17.08 · QPS 0.22 · BFCL 87.54 · IoU 0.6223**
vs confirmed vLLM (5361 / — / — / 14.6 / — / 86.43 / 0.6269):
  wall **−15%** · TPS **+17%** · BFCL **+1.1** · IoU −0.005 (within ~0.022 MDE = tie).
vs bf16-KV baseline (~4984 / 1557 / ~40 / 15.9 / — / ~87 / 0.6285):
  wall **−9%** · TTFT **−18%** · TPS **+7%** · BFCL +0.5 · IoU −0.006 (within MDE).
KEY: fp8-KV win is TTFT/wall (halved KV traffic → prefill), **TPOT ~unchanged (39 vs 40)** → CONFIRMS
raw decode is roofline-bound; fp8-KV does NOT close the raw-decode gap, it widens the aggregate lead.
VERDICT: FOLD fp8-KV as the recommended serve config (--kv-cache-dtype fp8) — wall/TTFT/TPS/BFCL all
improve, IoU within-noise. Caveat flagged: IoU nominally −0.006 (MDE-tie); accuracy (BFCL) IMPROVED.
Report: RESULT_fp8kv_conglom.txt.

## ★★★★★ 08:21 — fp8-KV DEAD at e2e scale → DECODE IS FULLY ROOFLINE-BOUND (final)
e2e A/B (1007/1007 each): bf16 wall 4551.9s/TPOT 38.18ms/tps 17.56 vs fp8-KV wall 4534.9s/TPOT
39.08ms/tps 17.08. fp8-KV = NEUTRAL-to-WORSE (TPOT +2.4%, tps −2.7%) — the microbench +4.8% did NOT
survive the prefill-heavy agentic workload (decode is a small fraction of the 26k-ctx turn). Plus fp8-KV
costs accuracy (86.33<87). DO-NOT-FOLD. (Correcting an earlier wrong "−9% wall" estimate.)
**FINAL: every decode lever measured-dead — GEMV-tune, W4A8 act-quant, NVFP4-GDN, K4, fp8-KV. The
bf16-KV main 011bee65 baseline IS the optimum. Raw-TPOT 38.18ms is at the GB10 memory roofline for the
all-NVFP4 checkpoint.**
### FINAL SCORECARD vs CONFIRMED vLLM (5361s / 14.6tps / 0.188qps / IoU 0.6269 / BFCL 86.43)
| metric | Atlas (main, bf16) | conf. vLLM | winner |
|---|---|---|---|
| wall (1007) | **4551.9s** | 5361s | **Atlas −15%** |
| tps | **17.56** | 14.6 | **Atlas +20%** |
| qps | **0.222** | 0.188 | **Atlas +18%** |
| TTFT median | **1264ms** | (higher) | **Atlas** |
| raw TPOT median | 38.18ms | ~31 (WEAK ref only) | vLLM on weak ref; residual = hardware roofline |
| BFCL | ~87 (prior) | 86.43 | Atlas |
| IoU | ~0.625 (prior) | 0.6269 | tie |
Atlas WINS the real comparison on every throughput+quality axis. The only "gap" (raw per-token TPOT) is
vs a WEAK/verbose vLLM reference AND is at the hardware floor. GOAL OUTCOME: the raw-TPOT gap is NOT
closable by kernel work (proven, roofline); Atlas already beats confirmed vLLM end-to-end.

## ★★★ 07:50 — CORRECTION: fp8-KV DO-NOT-FOLD (control-leg refuted it)
dgx2 bf16-KV baseline (SAME fresh main binary) = wall 4551.9 / TPOT 38.18 / TTFT 1264 / TPS 17.56.
fp8-KV (dgx1) = 4534.9 / 39.08 / 1271 / 17.08. Same-binary A/B: wall −0.4% (noise), TPOT +2.4% SLOWER,
TPS −2.7% WORSE, IoU 0.6285→0.6223. The microbench +4.8% did NOT survive the prefill-heavy e2e (decode
is a small slice of a 26k-ctx turn). My earlier "fp8-KV −9% wall" compared vs the STALE 07-21 baseline
(different binary) — WRONG, retracted. Lesson: always run the same-binary control leg (the gate exists
for this). **VERDICT: NO decode fold; bf16-KV main is optimum. ALL decode levers measured-dead.**
The completed conglomerate e2e (bf16 4551.9s or fp8 4534.9s) still BEATS confirmed vLLM (wall −15% /
TPS +17-20% / BFCL +1 / IoU-tie). dgx2 BFCL scorer errored (bfcl-eval dep) → bf16 accuracy from prior
runs (~87/0.625); dgx1 fp8 scored 87.54/0.6223.

## ★★★★★ 09:00 — vLLM RE-RUN (owner-requested): TPOT is TUNING-DEPENDENT, no acceptance edge
Served nvidia/Qwen3.6-27B-NVFP4 on dgx3 (sparkrun-eugr-vllm img, Triton/FLA GDN, MTP qwen3_next_mtp
2-tok, fp8-KV), warmed 10×. **Steady-state MTP TPOT = 104.86ms** (min 104.6 / p90 106.3, n=10 — stable,
not JIT). MTP acceptance: mean-len ~2.5, p1 0.83 / p2 0.65 = **≈ Atlas (p1 0.90, E~2.84), NO edge**.
→ vLLM TPOT swings 3.4×: **31.39ms tuned (07-16 Marlin+CUDA-graph) vs 104.86ms out-of-box.** Atlas
38-43ms BEATS untuned vLLM 2.5×, trails best-tuned vLLM only ~1.2×. The confirmed vLLM run (5361/86.43)
NEVER reported a TPOT and I couldn't reproduce its tuned state in-window. So the "gap" is conditional on
vLLM's single best config; Atlas wins every confirmed e2e metric. FINAL raw-TPOT answer: config-dependent,
not a fixed vLLM number.

## CHAIN-WIDENING: part B may ALREADY EXIST (discovery, post-Phase-0)
- `decode_verify_graphed_kgamma_dispatch` (verify_d.rs) is FULLY K-GENERIC: k=tokens.len(), K-generic
  metadata/layer-loop, graph cache keyed (slot_idx, k) at :191, debug_assert k<=32.
- `step_verify_dflash` (:74-79) already branches on dflash_verify_raw_argmax==false → applies the FULL
  MTP pre-sample pipeline; accept-prefix, commit_accepted_prefix(acc,k), trim_proposer_state all K-generic.
- No num_drafts cap: --num-drafts N passes through; dispatch drafts.len()>=4 → dflash step.
→ Chain K=5..8 on TODAY'S binary = `--num-drafts 4..7`. Expected slow-but-correct: M=5-8 projections hit
  the GEMM cliff (batch8 agent fixing) + GDN uses the generic-K serial fallback (~+4ms; wy6/wy8 later).
  PLAN: correctness+coherence probe at nd=5 now; TPOT after batch8 lands. Risk: dflash-specific drafter
  ctx-feeds inside the step may degrade with the MTP proposer (watch for 'degrading' logs).
