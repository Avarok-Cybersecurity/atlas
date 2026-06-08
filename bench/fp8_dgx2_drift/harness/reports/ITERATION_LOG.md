# Atlas FP8 → vLLM-Parity Iteration Log

Goal: match vLLM's 10/10 cargo_valid on the opencode rust-axum harness for Qwen3.6-35B-A3B-FP8.

## Reference points

| Config | Score | Notes |
|---|---|---|
| vLLM **BF16** (separately-stored BF16 snapshot) | **10/10** | dgx1 earlier; full-precision ceiling |
| vLLM **FP8** (same on-disk Qwen3.6-A3B-FP8 file as Atlas) | **10/10** | dgx2; kv_cache_dtype=auto → BF16 KV; vLLM 0.20.2rc1 |
| Atlas FP8 historical baseline (sm1_a2ao_sc1) | 5/20 = 25% | older runs, sampler stack shipped during prior /loop |
| Atlas FP8 historical baseline (tierA raw) | 3/10 = 30% | pre-sampler-stack |

**Critical insight**: vLLM achieves 10/10 on the SAME FP8 checkpoint Atlas serves → 30% is NOT the FP8-quantization ceiling, it's an Atlas-side bug. Headroom: 70pp.

## Active session shipped fixes (atlas-gb10:b2-b4-b5)

| # | Fix | Status |
|---|---|---|
| B1 | Fused k_norm + RoPE + cache-write kernel (`fused_k_norm_rope_cache.cu`) + Rust wrappers + kernel handles | ✓ kernel compiled + registered; dispatch wiring DEFERRED (needs Q-only RoPE variant or raw-K scratch buffer) |
| B2 | Defensive `v_contiguous` memset before V-projection GEMM | ✓ shipped in `paged.rs:94` |
| B3 | GDN Frobenius clamp investigation | ✓ explored — clamp is symmetric (decode and prefill use identical kernel logic); NO bug. closed. |
| B4 | Dynamic chunk size: respect `--max-prefill-tokens N` when user-set (no 8192 hard cap) | ✓ shipped in `preflight.rs:40-56` |
| B5 | BC=32 → BC=64 in `prefill_paged_compute.cuh` | ✗ reverted. BC=64 compiled to PTX but PTX-JIT failed at runtime on `inferspark_prefill_paged_batched` (smem/register budget under PREFILL_BATCHED macro). BC=48 compiles but doesn't align to m16n8k16 MMA tiles. Deferred to a future smem redesign pass. |

Harness improvements (atlas-gb10:b2-b4-b5 + new run_tier.sh):
- A1 `--split-dgx`: parallel N/2 + N/2 split across dgx1 + dgx2
- A2 Warm-up "What is 2+2?" check — **HALTS run_tier.sh on failure** (catastrophic-regression guard)
- A3 Webserver test in score_run.py — builds + runs the Axum project, curls `/ping` on port 3001, asserts "pong" in body; new `webserver_ok` scoring field
- A4 `--cosine-mode` opt-in flag for per-layer drift diagnostic
- A5 Per-turn opencode breakdown (`tool_calls_per_turn[i]`)

## Atlas iterations this session

| # | Tier | Config delta vs prior baseline | N | Pass | % | Result class | Notes / next-step |
|---|---|---|---|---|---|---|---|
| 1 | fp8attntuned | + K_PROMOTE=64 (mathematically identical) + `__expf` → `expf` in silu/RMS (#5) + `inv_sqrt_d · log2(e)` softmax fold + `__expf` → `exp2f` (#3+#7) | 4 (killed) | 0/4 | 0% | regression | exp2f fold broke long-context attention; `_t` variants of moe_shared_expert_fused_fp8 weren't fully reverted at first |
| 2 | baseline_recheck | reverted attn kernels, kept K_PROMOTE + #5 | 5 (killed) | 0/5 | 0% | regression | `_t` variants still half-reverted (FP32 cast on BF16 scale buffer) |
| 3 | baseline2 | fully reverted `_t` variants. Only K_PROMOTE + #5 silu fix remain | 10 | 1/10 | 10% | within-noise vs 25% | Fisher exact p=0.40 vs sm1_a2ao_sc1 25%. Statistical noise. Confirmed reverts good. |
| 4 | bf16experts | FP8→BF16 dequant on load (256 routed + 1 shared per layer) + new `moe_bf16_grouped_gemm.cu` + `moe_shared_expert_fused_bf16.cu` (ATLAS_FP8_DEQUANT_MOE_TO_BF16=1) | 1 (killed) | 0/1 | 0% | rules out: BF16-dequant-on-load isn't vLLM-equivalent | model produced clean JSON content but reasoning collapses into syllable repetition. CPU dequant bakes scale into BF16 → loses precision vs FP8 native's deferred FP32 scale per K-block. Not actually equivalent to vLLM's true BF16-source loading. **Dead end**. |
| 5 | fp8cal256 | enable existing per-tensor calibration via `--fp8-kv-calibration-tokens 256` | 10 | 3/10 | 30% | **back at historical baseline** | Calibration ON brings 10% baseline2 back up to 30%. Confirms calibration is necessary; without it, scales stay at default=2.0. |
| 6 | atlas_fp8_bf16kv | `--kv-cache-dtype bf16` instead of FP8 (matching vLLM's auto-dtype=BF16) | 8 (killed) | 1/8 | ~12% | rules out: simple dtype switch | confirmed memory's documented L35-L39 BF16-KV cliff is still present. Need to find the bug. opencode prompts (9495 tok) auto-chunk at 4096 → 3 chunks → cross-chunk bug fires. |
| 7 | bf16kv_nochunk | BF16 KV + `--max-prefill-tokens 65536` (no `--enable-prefix-caching`) → 9495 tokens prefill in **single chunk** | in flight | — | — | bisecting chunked vs non-chunked path | If pass-rate jumps (≥80%), bug is in chunked-prefill resume path → SSM state precision or KV cache between-chunk corruption. If still bad, the bug is in non-chunked BF16 KV path itself. |
| 8 | b1_fused_bf16kv | Adds B1 dual-write fused k_norm+rope+cache-write kernel (`ATLAS_FUSED_KV=1`) on top of BF16 KV single-chunk path | 10 | 0/10 | 0% | rules out: K-side BF16 rounding fusion | Dispatch wired but webserver_ok=0/10. Confirms B1 was at the wrong layer of the chain. Don't ship this kernel for now. |
| 9 | nogrammar_bf16kv | New `--disable-tool-grammar=true` CLI flag (Phase 0 from 16-agent arch synthesis) — disables structural-tag XGrammar on `tool_choice="auto"` | 10 | 2/10 | 20% | **rules out: tool-grammar default theory; grammar is MASKING drift, not causing it** | 9/10 runs hit `empty_path>0` (model emits empty tool args without grammar enforcement). Class C from SYNTHESIS_ARCH_DIFF.md rejected. Grammar is load-bearing on FP8 — don't disable. Pivot to Class A FP8 numerical fixes. |
| 10 | moetopk_bf16kv | MoE topk deterministic tie-break (lower-index-wins) in `moe_topk.cu` warp + cross-warp reductions and `moe_topk_sigmoid.cu`. Synthesis fix #6. Image `atlas-gb10:moetopk`. | 10 | 3/10 | 30% | within-noise of baseline | Identical to historical 30% on cargo_valid; 0/10 webserver_ok. Fix is correct architecturally (matches vLLM) but doesn't single-handedly close the gap. Keep it shipped — it's still a real correctness fix. Stack next with `ATLAS_STRIP_REASONING_HISTORY=1` (no rebuild, runtime env). |
| 11 | moetopk_striprsn | moetopk image + `ATLAS_STRIP_REASONING_HISTORY=1` env (no rebuild). Tests synthesis phase #1 hypothesis that re-injecting prior `<think>` traces drives multi-turn collapse. | 10 | 5/10 | 50% | **+20pp signal, needs confirmation** | 5/10 cargo_valid (+20pp over baseline). 0/10 webserver_ok. Drift signatures: 2 runs path_drift (early warmup), 2 runs empty_path, 6 clean. Fisher exact p≈0.65 vs 3/10 baseline — NOT yet significant. Running striprsn_v2 N=10 to confirm; combined N=20 tightens p-value. |
| 12 | striprsn_v2 | identical to striprsn — re-run for statistical confirmation | in flight | — | — | confirmation | If 4+/10, combined N=20 ≥ 9/20 → p < 0.05 vs baseline; lever real. If 1-2/10, the 5/10 was noise. |

## Hypothesis ladder (active)

1. **Chunked prefill SSM-state precision drift between chunks** — most likely
   - SSM has recurrent state snapshotted between chunks. If snapshot has any precision loss, chunk N+1's hidden state diverges from single-shot. Attention's K/V at deep layers (largest magnitude) amplify the drift. BF16 KV stores the drift faithfully (FP8 KV rounds it out via per-tensor scale). Predicts: bf16kv_nochunk would lift significantly above 12%.
2. **Atlas BF16 paged-attention page-boundary read bug** — possible but less likely; would also affect FP8 KV at same magnitude
3. **Atlas attention softmax precision at large-magnitude K/V** — Atlas's softmax already uses `__expf`; vLLM uses FA3's `exp2(x · log2e - max_scaled)` fold which has fewer roundings. My earlier attempt to add this fold regressed.

## Bisection scoreboard (so far)

- ✅ vLLM-side FP8 quant on the same file: NOT the ceiling.
- ✅ Atlas-side sampler stack (SM1+WS1+AM1+SC1+A2-AO+B1): present and working; baseline is ~25-30%.
- ✅ Calibration: must be ON; default scales=2.0 stay at 10% (statistical noise vs 25%).
- ❌ FP8→BF16 dequant on load + custom BF16 grouped GEMM: dead end (worse than baseline because no per-K-block FP32 scale).
- ⏳ Single-chunk BF16 KV: in flight.

## Audit findings deferred (post-cliff fix)

From the 4-agent attention audit:
- Per-head FP8 KV scales (only matters if we go back to FP8 KV)
- BR/BC=128 tiles (perf at long context)
- FP32 scale storage (we tried, requires multi-site refactor)
- `__expf` → `exp2f` softmax fold (reverted — broke ctx >9k)

## 4-agent chunked-prefill audit (2026-05-27) — candidate fixes

Atlas-specific bug candidates for the BF16-KV chunked-prefill cliff (ranked by likelihood):

| # | Candidate | File | Estimated impact | Effort |
|---|---|---|---|---|
| 1 | **Double BF16 rounding in K/V norm + cache-write** — Atlas's `k_norm`/`v_norm` runs IN PLACE on `k_contiguous`/`v_contiguous` (BF16), then RoPE rotates (BF16), then `write_kv_cache` BF16-rounds again. vLLM keeps K/V in FP32 between norm and cache-write, BF16-rounds ONCE. At L35-L39 where K magnitudes peak, double-rounding compounds; FP8 KV's coarser quantization masks it. | `crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs:91-93, 188-222` | **HIGH** — direct match for L35-L39 cliff | ~30-50 LoC |
| 2 | **`v_contiguous` aliases K-projection buffer** — historical comment confirms prior "stale V on chunk-1+" regression. If GEMM/norm doesn't fully cover V region, chunk-1+ reads stale data. | `crates/spark-model/src/layers/qwen3_attention/prefill/paged.rs:91-93` | MEDIUM — chunked-prefill-specific | defensive memset, ~5 LoC |
| 3 | **GDN Frobenius clamp asymmetric application** between prefill vs decode paths. If the clamp fires in prefill but not decode (or vice versa), chunk-boundary hidden states differ from single-shot. | `kernels/gb10/common/gated_delta_rule.cu:159` | MEDIUM | audit + ~10 LoC |
| 4 | **No Mamba-block-aligned chunk split** — vLLM enforces `_mamba_block_aligned_split` so chunks land on block boundaries. Atlas chunks at 8192 unconditionally; can land mid-block and confuse cache layouts. | `crates/spark-server/src/main_modules/serve_phases/preflight.rs:40` | LOW-MEDIUM | scheduler change, ~20 LoC |
| 5 | **BC=32 → BC=64 in `prefill_paged_compute.cuh`** — reduces softmax-rescale step count from 591 → 296 for 19k context. Marginal precision/perf win. | `kernels/gb10/common/prefill_paged_compute.cuh` | LOW (marginal) | smem budget check, ~5 LoC |

Reference vLLM hybrid-model chunked-prefill bug history (confirms this is an active known issue class):
- vLLM #41726 (Nov 2025): TurboQuant + chunked continuation_prefill on Qwen3.5-9B hybrid
- vLLM #26201: prefix caching for hybrid models still tracked as unsupported
- vLLM #27264: cache malformation with SSM cache dtype float32 + wrap-around
- vLLM #13466: Mamba should return states in fp32 (Atlas already does)

## Notes for future iterations

- Whenever a tier shows 0/1, INVESTIGATE immediately; don't wait for N=10 to confirm.
- Use both DGXs in parallel (5/5 split) once a candidate fix is stable. SSH tunnel + harness on each side.
- Keep the `_t` variant kernels intact during sed-based rewrites (they have different field names from non-`_t`).
- Memory notes ARE point-in-time observations; verify against current code (e.g., bf16 KV cliff was documented but I had to re-verify it exists today).
