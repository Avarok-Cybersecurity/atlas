# arch_diff_06_flash_attn.md — FlashAttention compute: vLLM (FA3) vs Atlas

Sources
- vLLM FA3 SM90: `/home/nologik/vllm/vllm/.deps/vllm-flash-attn-src/hopper/softmax.h`, `flash_fwd_kernel_sm90.h`, `mainloop_fwd_sm90_tma_gmma_ws.hpp`, `tile_size.h`.
- Atlas SM121: `/workspace/atlas-mtp/kernels/gb10/common/prefill_paged_compute.cuh` (shared algorithm header), `inferspark_prefill_paged.cu` (BF16 KV), `inferspark_prefill_paged_fp8.cu` (FP8 KV).

## 1. Tile sizes BR × BC

Atlas: `BR=32, BC=32` / `BR64=64, BC=32` — `prefill_paged_compute.cuh:72-78`. B5 audit comment (2026-05-27): `BC=64` → PTX JIT fail under `PREFILL_BATCHED`; `BC=48` broke `m16n8k16` alignment. **BC=32 is a hardware-budget floor, not math choice.**

FA3 SM90 headdim=128 + causal + BF16: `{kBlockM=128, kBlockN=128, MmaPV_is_RS=true, IntraWGOverlap=true}` — `tile_size.h:40-44`. FP8 path similarly 128×128 (`tile_size.h:62-67`).

At 19k context: Atlas ~594 BC iterations/Q-row vs FA3 ~148. Each iteration is a potential `acc_o *= exp(m_prev - m_new)` rescale (Atlas conditional, line 289; FA3 unconditional `softmax.h:117-121`). Per-rescale `__expf`/`exp2f` is ≤2 ULP; compounded BF16 acc_o drift over 594 rescales ~7e-5 — small alone, **load-bearing when interacting with FP8 KV dequant noise (#5)**: each tile feeds a slightly-wrong rmax, shifting which elements survive `exp(s - max)` truncation.

## 2. softmax_scale precision

Atlas (`prefill_paged_compute.cuh:248-249, 306-307`):
```
acc_s[nt][k] *= inv_sqrt_d;                       // separate FP32 mul, two-pass
…
p = sw_exp(acc_s[nt][k] - m_r0);                  // sw_exp = __expf (line 68)
```
`inv_sqrt_d` is FP32; `sw_exp` is `__expf` by default (Phase 2b fix, line 67-68).

FA3 (`flash_fwd_kernel_sm90.h:585, 423-431`; `softmax.h:66-86`):
```
softmax_scale_log2 = softmax_scale * M_LOG2E;            // (1/sqrt(d)) * log2(e) folded
softmax_scale_log2 *= q_descale * k_descale;             // FP8: descale folded in too
…
max_scaled = max(mi) * scale - max_offset;
tensor(mi,ni) = exp2f(tensor(mi,ni) * scale - max_scaled);   // fused FFMA + exp2f
```
FA3 folds `1/sqrt(d) * log2(e)` (and FP8 q/k descales) into one scalar outside the loop, then one FFMA + `exp2f` per element. Atlas does MUL (line 249) then SUB then `__expf`. Mantissa accuracy is identical (`exp2f(x*log2(e)) ≡ __expf(x)`) but **Atlas materialises a scaled `acc_s` register tile, doubling rounding sites vs FA3's fused form**. Drift ≪1 ULP/element; compounds with #1.

## 3. acc_o dtype

Both FP32 (Atlas line 165; FA3 CUTLASS GMMA accumulator). **Match. No bug.**

## 4. P × V dtype

Atlas (Phase 2c, line 37-42, 314-316, 386-394): `smem_P` FP16, `smem_V` BF16 converted to FP16 in-register per-MMA via `bf16x2_to_f16x2_bits` (FP16 10-bit mantissa vs BF16's 7-bit — strict widening for `|v|<8`). MMA `m16n8k16.row.col.f32.f16.f16.f32`. Accumulator FP32.

FA3 BF16 KV: `gmma.bf16.bf16.f32`. FP8 KV: re-quantises P to FP8 E4M3 via `Max_offset=8` (`softmax.h:67-68, 147-149`), `gmma.f8.f8.f32`; `v_descale` in `finalize_dispatch` (`mainloop_fwd_sm90_tma_gmma_ws.hpp:1164`).

**Atlas is strictly more precise than FA3 on P** — no bug, deliberate. *Precision advantage* on P×V vs FA3 FP8, not a drift source.

## 5. FP8 KV dequant — CRITICAL

Atlas (`inferspark_prefill_paged_fp8.cu:13-16, 72-93`):
```
fp8_to_bf16(b, scale): float v = __half2float(__nv_cvt_fp8_to_halfraw(b, E4M3)) * scale; return __float2bfloat16(v);
LOAD_KV_TILE:  _sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale;
// per-element dequant into smem as BF16 → MMA is bf16.bf16.f32
```
Per-tensor `k_scale`, `v_scale` (single scalar each for the whole cache, `inferspark_prefill_paged_fp8.cu:47-50`). MMA is BF16, not FP8.

FA3 FP8: keeps FP8 in MMA; per-(batch, KV-head) `ptr_k_descale[bidb * stride + bidh_kv]`, `ptr_q_descale[…]` folded into `softmax_scale_log2 *= q_descale * k_descale` (line 427-429); per-head `v_descale` applied in finalize (`mainloop_fwd_sm90_tma_gmma_ws.hpp:1164`).

**Bug flags**:

(a) **F1 — High**: Atlas's per-tensor scale vs vLLM/FA3-style per-batch-per-KV-head scales. If the FP8 KV cache writer was calibrated per-block (vLLM convention) but Atlas reads it with one scalar `k_scale` covering all heads + all blocks, **deep layers / outlier heads will be miscalibrated by 10–50%**. *This is the most plausible single source of long-context FP8 drift.*

(b) **F2 — Med**: `fp8 → half → float → *scale → bf16(rne)` has two roundings (FP32 product → BF16). FA3 keeps MMA in FP8 — only `__expf`-path rounding. Atlas loses ~3 mantissa bits per K/V element relative to FA3 FP8-native MMA. (Note: Phase 2c P-precision win at #4 is on the *output* side and does not recover input-side K/V dequant error.)

(c) **F3 — Med**: `_sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale` at line 74 is a *runtime pointer equality*. If a future caller aliases K_cache==V_cache (e.g. unified KV pool, MQA fused cache), V loads silently pick `k_scale`. The original `KERNEL_PREAMBLE` attempt at line 52-61 is left in as commented-out dead code with a `FIXME` — the pointer-eq workaround is fragile.

## 6. Chunked-prefill `q_offset` + `kv_len`

Atlas (`prefill_paged_compute.cuh:108, 174-176, 252`):
```
num_kv_blocks = (kv_len + BC - 1) / BC;
mx = (q_offset + q_tile_end - 1) / BC; num_kv_blocks = min(num_kv_blocks, mx + 1);
qr = q_offset + q_start + row;     // causal mask kv_start+c > qr
```
Atlas threads `q_offset` from the chunked-prefill scheduler into both iteration bound and causal mask — correct when caller sets `q_offset` to absolute seq position. B4 cap bump (8192→65536) leaves this loop untouched. FA3 uses varlen+cum_seqlens (no `q_offset`).

**F4 — Low**: if a chunked-prefill caller passes `q_offset=0` for a non-first chunk, causal mask becomes lower-triangular w.r.t. chunk start instead of seq start — chunk-2 Q tokens self-mask attention to chunk-1 KV. *Add debug assert `q_offset + q_len <= kv_len`.*

## 7. Cross-warp m/l consistency

BR=32 (`prefill_paged_compute.cuh:159-160, 322-325, 333-348`): 4 warps, `pv_warp_m = (warp_id & 1) * 16` → {0,2} own rows 0-15, {1,3} own 16-31. QK^T only warps 0-1 (line 215). Warps 0,1 write `smem_ml`; 2,3 read after `__syncthreads()` line 330. Final-store splits `warp_id<2` (local) vs `>=2` (smem). **Consistent.**

BR=64 (`prefill_paged_compute.cuh:529-531, 683-715`): 8 warps, pairs (0,4)(1,5)(2,6)(3,7). QK^T warps 0-3, V-load warps 4-7. Warps 0-3 write `smem_ml64`; `__syncthreads()` line 697; warps 4-7 read line 702. Warps 4-7's `m_r`, `l_r` registers stay at init `-1e30f, 0.0f`; final-store uses `warp_id<4` to pick local-vs-smem. **Consistent.**

**F5 — Low**: line 688 empty `cp.async.commit_group;` from warps 0-3 to balance warps 4-7. PTX ISA says empty commit is a no-op; FA3 avoids this pattern. Likely safe on SM121, worth synthetic test.

## Bug-flag summary

| # | Severity | Site | Issue |
|---|----------|------|-------|
| F1 | **High** | `inferspark_prefill_paged_fp8.cu:47-50, 74` | Per-tensor `k_scale`/`v_scale` only; mismatch with per-block-quantised FP8 KV cache. **Most plausible FP8 long-context drift cause.** |
| F2 | Med | `inferspark_prefill_paged_fp8.cu:13-16` | `fp8→half→float→*scale→bf16` double rounding; FA3 keeps MMA in FP8. ~3 mantissa bits lost per K/V element. |
| F3 | Med | `inferspark_prefill_paged_fp8.cu:72-74` | Pointer-eq selects `_sc`; aliased K_cache==V_cache silently breaks V scale. Commented-out FIXME dead code remains. |
| F4 | Low | `prefill_paged_compute.cuh:175-176, 252` | Causal mask anchored on `q_offset` — caller must pass absolute seq pos. Add debug assert. |
| F5 | Low | `prefill_paged_compute.cuh:688` | Empty `cp.async.commit_group` on SM121 — FA3 avoids this pattern. |
| F6 | Info | `prefill_paged_compute.cuh:72-78, 248-249` | BC=32 vs FA3 BC=128 ⇒ ~4× rescale count + two-pass `inv_sqrt_d * acc_s` then `__expf(.-m)`. Bounded BF16 drift, compounds with F1/F2 under FP8 KV. |

**Recommendation order**: (1) audit FP8 KV writer for scale granularity vs reader's per-tensor assumption (F1); (2) prototype `BC=64` with reduced `N_TILES_PER_WARP` to fit smem (F6); (3) fix the pointer-eq scale dispatch (F3); (4) chunked-prefill `q_offset` debug assert (F4).
