# Submission handoff — chain-widened decode (K=4), GB10 dense-27B

**For:** Azeez (submission owner). **Code:** folded + pushed. **E2E:** golden config, full both-phase.

## 1. What changed (all folded, pushed, gated)

Branch `perf/decode-fold-2026-07-24` == `feat/tree-spec-decode` == **`1dcf2755`** on `avarok`
(fast-forward of `origin/main 011bee65` + this work — no rebase, no force-push).

| # | change | file(s) | why it matters |
|---|---|---|---|
| 1 | **`n==4` FFN arm** (the big one) | `layers/qwen3_attention/trait_impl/multi_seq/ffn.rs` | K=4 verify's dense FFN on the 16 full-attn layers was falling through to the MMQ prefill GEMM (`forward_prefill`) instead of the batched `forward_k4` GEMV. `forward_k4` existed and served the 48 GDN layers, but the attention path had no `n==4` arm. **This is the entire source of every historical "K=4 regresses" result** (54.8ms → ~31ms per the kernel's own docstring). |
| 2 | **`w4a16_gemv_batch8`** | `kernels/gb10/common/w4a16_gemv.cu` | New NVFP4 batched verify GEMV (E2M1 + FP8 group scales), instantiating the existing `w4a16_gemv_batchm_impl<8>`. Removes the M=5..8 tile-GEMM cliff. lm_head M=8: **19.4ms → 4.7ms**. |
| 3 | **M≤8 gate widening** | `model/impl_a3.rs` (lm_head), `multi_seq/qkv.rs`, `multi_seq/ffn.rs`, `layers/dense_ffn*.rs`, `qwen3_ssm/*` | Routes lm_head / QKV+O / dense-FFN / **GDN qkvz + out_proj** through batched GEMV at M≤8. Bonus find: the **GDN out_proj had no `==4` NVFP4 arm at all**. |
| 4 | **`wy5`–`wy8` batched GDN verify** | `kernels/gb10/common/gated_delta_rule_wyn.cu` (K-templated) + dispatch | Replaces the serial per-token GDN fallback at K=5..8. Kill-switch `ATLAS_GDN_WYN=0`. |
| 5 | shadow top-k instrumentation | `mtp_head/forward.rs`, `verify_k3/k4_step.rs`, `scripts/` | `ATLAS_MTP_SHADOW_TOPK=k`, observational only (default off). Produced the 19k-sample acceptance measurement that drove the design. |

**Not shipped (measured dead, documented in the ledger):** W4A8 int8-activation GEMV (the strix
trick — 0.99–1.01× on GB10 + 0.5% accuracy cost), native FP4-MMA verify (M≤8 too small for the
tiles), NVFP4-ing the FP8 GDN weights (inert on this checkpoint — its GDN is *already* NVFP4),
fp8-KV (neutral at e2e), tree drafting (chains beat trees by 17% on the measured acceptance).

## 2. Verification (what was gated, and how)

| gate | result |
|---|---|
| `w4a16_gemv_batch8` vs shipping `batch4` @ M∈{1,3,4} | **bit-exact** |
| `w4a16_gemv_batch8` vs `batch16` @ M∈{5,6,8} | **bit-exact** |
| `w4a16_gemv_batch8` @ M=8 vs CPU f64 dequant reference | worst case **0.19× of tolerance** |
| `wy5`–`wy8` vs serial per-token GDN reference (all layers) | **PASS**, `out_cos ≥ 0.9999984`, `state_cos = 1.0000000` |
| K=3 control on the new binary (regression guard) | **38.49ms == baseline** — no regression from any new arm (all `try_kernel`-gated) |
| Full e2e accuracy (IoU + BFCL) | **§4** |

## 3. Winner selection (agentic subset, 174 turns, identical seed/config, only `--num-drafts` varies)

| K | `--num-drafts` | TPOT | wall | TTFT | tps |
|---|---|---|---|---|---|
| 3 (control) | 2 | 38.49 ms | 835.9 s | 1301 ms | 17.64 |
| **4 (chosen)** | **3** | **31.69 ms** | 745.5 s | 1320 ms | 19.63 |
| 5 | 4 | 32.76 ms | **734.2 s** | **1212 ms** | **20.12** |
| 6 (wy6) | 5 | 32.29 ms | 735.1 s | 1225 ms | 18.71 |
| 8 | 7 | 34.64 ms | 786.2 s | 1274 ms | 18.61 |

**K=4 wins on TPOT (−17.7% vs control).** K=5 is the wall/tps/TTFT-optimal alternative within
noise — if the submission is scored on wall or throughput rather than per-token latency, K=5
(`--num-drafts 4`) is the better pick and is equally gated.

Why deeper K stops helping: depth-4+ conditional acceptance falls below the plateau and the
drafter's per-depth propose cost compounds (~3.4 ms/level); GDN is only 1.6% of the step, so
wy6-vs-serial is within noise.

## 4. E2E results (golden config, full MLCommons both-phase)

Run dir: `dgx1:/workspace/endpoints-fresh/results/chainK_golden_e2e_20260724_131209/`

<!-- FILLED 2026-07-24 -->
| metric | **this run (K=4)** | bf16 baseline (main) | confirmed vLLM | delta vs baseline / vLLM |
|---|---|---|---|---|
| perf wall (1007) | **4104.0 s** | 4551.9 s | 5361 s | **−9.8% / −23.4%** |
| TPOT median | **32.60 ms** | 38.18 ms | ~31.4 (tuned) / 104.9 (out-of-box) | **−14.6%** / +3.9% vs tuned |
| TTFT median | **1298 ms** | 1264 ms | — | +2.7% (noise) |
| qps | **0.245** | 0.221 | 0.188 | **+10.9% / +30.3%** |
| tps | **19.20** | 17.56 | 14.6 | **+9.3% / +31.5%** |
| IoU | **0.6231** | ~0.625 | 0.6269 | −0.002 / −0.004 (tie, MDE ≈0.022) |
| BFCL (995 ST) | **87.44** | ~87 | 86.43 | tie / **+1.01** |
| samples | 1007/1007, **0 failed** | 1007/1007 | 1006/1007 | — |

MLPerf accuracy floor (83.64 / 85.32): **BFCL 87.44 PASS**. IoU within measurement noise of both
the baseline and vLLM. Perf-phase duration 4104.0 s; BFCL-phase duration 4781.5 s.

## 5. Exact reproduce

```bash
# 1. code
git fetch avarok && git checkout 1dcf2755        # perf/decode-fold-2026-07-24

# 2. build (the ATLAS_TARGET_MODEL is load-bearing — without it you get a wrong binary)
PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b \
  cargo build --release -p spark-server --bin spark --features cuda
# expect: "compiled 157 kernels for target 0 (gb10, qwen3.6-27b, nvfp4)"

# 3. serve — GOLDEN CONFIG, frozen c2final env; ONLY --num-drafts differs from the golden run (2 -> 3)
sudo docker run -d --name atlas-golden-e2e --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "<worktree>/target/release/spark:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --host 0.0.0.0 --port 8888 --model-name centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking

# 4. e2e (config = the golden defaults_20260721 edge-agentic-full-run, temp 0.0 / seed 42)
cd /workspace/endpoints-fresh && \
  ./.venv/bin/inference-endpoint benchmark from-config -c <worktree>/golden_e2e.yaml --mode both -v
# or simply (the one reproduce entry point -- see REPLICATE.md):
#   ATLAS_BIN=$PWD/target/release/spark HARNESS_DIR=/workspace/endpoints-fresh \
#   BASE_CONFIG=/workspace/endpoints-fresh/results/defaults_20260721_173342/config.yaml \
#     ND=3 bash scripts/mlperf-edge/run_golden_e2e.sh
```

**Delta vs the golden serve line: `--num-drafts 2` → `3`. Everything else — every env flag, every
serve flag, model, harness config, seed — is byte-identical to the golden run.**

**As of this PR, K=4 is the shipped default**: `kernels/gb10/qwen3.6-27b/MODEL.toml`
`default_num_drafts = 1 → 3`, so serving without `--num-drafts` now selects K=4. Verified on a
clean serve: `num_drafts: using MODEL.toml default_num_drafts=3 (K=4)`, scheduler `num_drafts=3`,
coherent output, tool-call smoke PASS. `--num-drafts` still overrides.

## 6. Supporting artifacts (all on the branch)

`CAMPAIGN_LOG.md` (chronological log) · `DECODE_FOLD_LEDGER.md` (every lever + verdict) ·
`FINDINGS.md` (the roofline analysis) · `shadow_topk_stats.json` (19k-sample acceptance) ·
`scripts/parse_shadow_topk.py`, `scripts/tree_shape_search.py` · harness now under
`scripts/mlperf-edge/` (`run_golden_e2e.sh` is the entry point) and raw results under
`docs/campaigns/gb10-decode-fold-2026-07/raw/`
· `kn_ab_*.json` (per-K ladder results).
