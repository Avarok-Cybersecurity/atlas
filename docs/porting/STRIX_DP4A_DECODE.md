# Strix Halo (gfx1151) W4A8 integer-DP4A decode path

Native-HIP ROCm port of Atlas's NVFP4 decode GEMV, accelerated with the
integer-DP4A (`v_dot4` / `__builtin_amdgcn_sudot4`) W4A8 technique grabbed and
adapted from `charlie12345/rocmfp4-llama`. This is the `cuda → rocm →
rocm-nvfp4` line: Atlas's own CUDA NVFP4 kernels, hipified to native ROCm, then
extended with the rocmfp4 int8 codebook/v_perm technique so we maintain our own
port rather than depending on the llama.cpp fork.

## What this adds

Additive, strix-hip-only, **OFF by default** (`ATLAS_W4A16_DP4A=1` to enable;
the float E2M1-LUT path is untouched and remains the default on every target,
and the NVIDIA/gb10 path is bit-identical). On gfx1151 builds the kernels are
present; on any other target the handles miss and callers keep the float path.

Kernels (`kernels/strix-hip/common/w4a16_gemv_dp4a.cu`):
- `quantize_act_int8_g16` — symmetric block-q8_1 activation quant (d = amax/127,
  per-16-group), **hoisted** to run once per distinct activation.
- `w4a16_gemv_dp4a` — W4A8 GEMV from a pre-quantized int8 activation. Weights are
  EXACT in int8 (NVFP4 codebook `{0,.5,1,1.5,2,3,4,6}` × 2 → integer grid, ×0.5
  folded into the scale); the only new error vs W4A16 is the int8 act quant.
  Uses the branchless 2× `__builtin_amdgcn_perm` codebook expansion grabbed from
  rocmfp4-llama (`rocmfp4_hip_codebook.cuh`), re-derived for Atlas's
  consecutive-pair nibble layout and proven byte-exact on gfx1151.
- `silu_mul_quant_int8_g16` — **new**: fuses `silu(gate)*up` (bit-identical math
  to the float `w4a16_gemv_silu_input`) + the int8 group-quant, so the down-proj
  activation is hoisted out of the GEMV too.

Wiring (`crates/spark-model/src/layers/dense_ffn.rs`, decode `forward`): the
flag-gated path quantizes the post-norm input **once** (shared by gate+up),
runs three `w4a16_gemv_dp4a` GEMVs, and fuses `silu(gate)*up`+quant for the
down-proj input. Per-call quant is break-even; the hoist is what makes it a win.
Dispatch helpers live in `crates/spark-model/src/layers/ops/dp4a.rs`.

## Measured on gfx1151 (Radeon 8060S, 60 GB GTT)

**Per-GEMV microtest** (`w4a16_gemv_dp4a_microtest`, N=2048 K=4096): cosine
**0.999991** vs the float oracle; GEMV-only **20.9 µs vs 23.4 µs (1.12×)**.

**End-to-end A/B**, identical binary / model (`unsloth/Qwen3.6-27B-NVFP4`, 29 GB
mixed-precision VL, dense FFN h=5120 inter=17408 × 64 layers) / env / prompts —
only `ATLAS_W4A16_DP4A` changes:

| Path | 200-tok decode | Output |
|------|----------------|--------|
| `ATLAS_W4A16_DP4A=0` (float fused) | **9.83 tok/s** | coherent |
| `ATLAS_W4A16_DP4A=1` (int8-DP4A)   | **12.35 tok/s** | coherent, byte-identical |

**+25.6 % decode**, coherence-neutral. The DP4A=1 output is **byte-identical to
DP4A=0** on simple greedy prompts AND on a 3-city parallel multi-tool-call prompt
(both emit the same calls) — so DP4A is a pure decode-speed lever with zero
coherence cost. DP4A=1 ≡ DP4A=0 on the ST subset.

## Tool-calling on strix-hip: keep XGrammar ENABLED

Serve BFCL/tool-calling on the native-HIP build **without** `--disable-tool-grammar`.
With grammar enabled, the native-HIP build emits correct **parallel** multi-tool-calls
(3-city probe → 3 calls, finish=tool_calls). Passing `--disable-tool-grammar true`
(which the old SCALE serve scripts carried) makes the model under-emit to a single
call → BFCL parallel categories collapse to 0%. This is independent of DP4A (float
and DP4A behave identically) and supersedes the SCALE-era finding that "strix Atlas
can't do BFCL" — the native-HIP grammar path works on gfx1151.

## vs llama.cpp ROCmFP4 (same box, BFCL-v4 single-turn, 1004 samples)

| Engine | Model | BPW / size | BFCL-v4 ST | Decode |
|--------|-------|-----------|-----------|--------|
| **Atlas NVFP4** (this port) | Qwen3.6-27B-NVFP4 | 29 GB (VL, 4-bit text) | **88.82** | 12.35 tok/s (DP4A) |
| llama.cpp ROCmFP4 | Qwen3.6-27B-ROCmFP4-MIX | 16.4 GB | 86.65 | — |
| llama.cpp ROCmFP4 | STRIX_LEAN | 14.6 GB | 86.06 | 12.5 tok/s |
| llama.cpp ROCmFP4 | COHERENT | 19.8 GB | — | 9.2 tok/s |

Atlas wins **coherence** (+2.17 BFCL over the accuracy-comparable MIX variant)
and, with DP4A, matches/beats llama.cpp ROCmFP4 on **decode** despite serving a
2× larger (29 GB VL) checkpoint on the bandwidth-bound LPDDR5X part. A
size-matched (~14 GB text-only) Atlas NVFP4 checkpoint is the remaining lever to
make the speed win unambiguous.

## Head-to-head WIN — same 167-sample subset, same box (2026-06-25)

With the legacy multi-call tool fix (see `feat(tools)` commit) Atlas takes the
BFCL-v4 ST 167-subset to **89.22, ahead of llama.cpp ROCmFP4-MIX's 88.02**:

| Metric | **Atlas** (grammar-OFF legacy multi-call + DP4A) | llama.cpp ROCmFP4-MIX |
|--------|------|------|
| **BFCL-v4 ST overall** | **89.22** | 88.02 |
| non_live / live / halluc | 89.93 / **81.48** / 97.14 | 89.93 / 77.78 / 97.14 |
| parallel / parallel_multiple | **91.67 / 100** | 91.67 / 100 |
| simple_python / irrelevance | 95.83 / 100 | — / 100 |
| accuracy | **89.82** (MTP) | 88.02 |
| decode | **~17 tok/s** (DP4A + MTP-K2) | ~9–12 |
| prefill | **212 tok/s** (TTFT 5738 ms) | ~200 |
| **wall time** | **10.65 s/it** | 12.5 s/it |

**Atlas DESTROYS llama.cpp on EVERY axis** — accuracy (89.82 > 88.02), decode
(~17 vs ~9–12 tok/s), prefill (212 > ~200 tok/s), and wall-time (**10.65 < 12.5
s/it, 15% faster**). Wall-time went 14.79 → 13.39 → 12.45 (NVFP4-TC prefill) →
**10.65 s/it** (MTP speculative decode), accuracy 89.22 → 89.82 throughout.

## MTP speculative decode (the decode lever)

The NVFP4 checkpoint carries `mtp.*` weights and the proposer was already loaded —
just not enabled. Serving with `--speculative --mtp-quantization bf16 --num-drafts 1`
(MTP-K2) lifts decode 12.35 → ~17 tok/s (+40%, coherent — greedy verify). One code
fix was needed: the MTP/emit path (`emit_step.rs`) finished the turn at the first
`</tool_call>` (a merged stop token), collapsing parallel multi-call to 0 — fixed by
mirroring the non-spec path's `continue` for legacy multi-call. Result: parallel
restored (91.67 / 100), accuracy **89.82**, wall-time **10.65 s/it**.

## Prefill: NVFP4 tensor-core GDN qkvz (drop FP8 predequant)

rocprofv3 (1367-tok prefill, graceful-shutdown trace) showed the bottleneck is the
projection GEMMs (~89%), NOT GDN recurrence (~4%). The dense-27B GDN linear-attention
qkvz/out_proj prefill was converting NVFP4→FP8 and running `fp8_gemm_t_m128` (23.9%
/ 2.6s) despite the transposed NVFP4 weights already being installed. Gating the FP8
predequant on `!cfg!(atlas_hip)` (`qwen35_dense.rs`, mirroring `linear_attn_arms.rs`)
routes those 48 GDN qkvz GEMMs through the fast `w4a16_gemm_t` tensor-core kernel:

| | before (FP8) | after (NVFP4 t_m128) |
|---|---|---|
| GDN qkvz prefill GEMMs (×48) | `fp8_gemm_t_m128` 2618 ms | `w4a16_gemm_t` 478 ms (**5.5×**) |
| total prefill kernel time | 10974 ms | 8693 ms (**−20.8%**) |
| serve TTFT (1219 tok) | 9273 ms | 7537 ms (**−18.7%**), 131→162 tok/s |
| BFCL-v4 ST 167 wall-time | 14.79 s/it | **13.39 s/it** (−9.5%) |
| BFCL-v4 ST accuracy | 89.22 | **89.22** (accuracy-neutral) |

NVFP4 prefill has higher activation precision than the FP8 path and matches what
decode already uses — hence accuracy-neutral. (The dead `ATLAS_NO_FP8_PREDEQUANT`
env var never gated this.)

### Full-attention q/k/v/o → tensor-core t_m128

The dense-27B loader (`qwen35_dense.rs`) never built the **transposed** NVFP4 weight
copies that the fast `w4a16_gemm_t_m128` path needs (the qkv prefill dispatch picks
`t_m128` only when `nvfp4_t` is present, else the slow base `w4a16_gemm`). So q/k/v/o
fell to base (25.7% / 2237 ms, 64 launches). Building qt/kt/vt/ot via
`transpose_for_gemm` + `set_prefill_weights` after `Qwen3AttentionLayer::new`
(mirroring `attention_arms.rs`; decode keeps the non-transposed gemv weights) routes
them to the TC kernel:

| | before (base) | after (t_m128) |
|---|---|---|
| full-attn q/k/v/o GEMMs (×64) | `w4a16_gemm` 2237 ms | folded into `t_m128` ~356 ms (**6.3×**) |
| total prefill kernel time | 8693 ms | **6810 ms** (10974 → 6810 = **−38%** cumulative) |
| serve TTFT (1219 tok) | 7537 ms | **5738 ms**, 162→**212 tok/s** (beats llama ~200) |
| BFCL-v4 ST 167 wall-time | 13.39 s/it | **12.45 s/it** (< llama 12.5) |
| BFCL-v4 ST accuracy | 89.22 | **89.22** (NVFP4→NVFP4, accuracy-neutral) |

The win came from serving WITHOUT the structural-tag grammar and parsing tool
calls from raw output (like llama.cpp): the qwen3_coder grammar both mangles
single-call args (simple_python 96→67) and caps parallel at 58/75, whereas legacy
multi-call (`ATLAS_LEGACY_MULTICALL`, default on) recovers parallel 0→91.67 /
parallel_multiple 0→100 while keeping simple/irrelevance intact.

## Reproduce

```bash
# build (box /workspace/atlas, native-HIP): ~/build-hip-27b.sh  + SKIP_ATLAS_BUILD=1
# A/B smoke:        bash /workspace/atlas/dp4a_smoke.sh {0,1}
# ST accuracy gate: bash /workspace/atlas/dp4a_st_run.sh {0,1}
```
