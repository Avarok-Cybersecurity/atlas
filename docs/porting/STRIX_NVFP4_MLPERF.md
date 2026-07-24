# Strix Halo (gfx1151) — Qwen3.6-27B NVFP4: MLPerf-edge replication

Native-HIP Atlas (`spark`) serving **`nvidia/Qwen3.6-27B-NVFP4`** on a single AMD
Strix Halo desktop (Radeon 8060S, gfx1151/RDNA3.5, ROCm core-7.13), scored against
llama.cpp Q4_K_M on the same box and dataset.

> Supersedes the pre-submission revision of this doc, which described a different
> checkpoint (`unsloth/…`), the 60 GB laptop, `--num-drafts 1` at 32k, and decode
> ~16 tok/s. Those numbers predate K=3, drafter-refeed, the SSM snapshot fix and the
> M_TILE=64 prefill default; do not quote them.

## Submitted result (MLPerf v6.1, closed/edge, `Atlas_Inference_Inference_3`)

| metric | Atlas NVFP4 | llama.cpp Q4_K_M | |
|---|---|---|---|
| wall, 1007/1007 | **7108.6 s** | 10567.9 s | **1.49x** |
| QPS | **0.1417** | 0.0953 | **1.49x** |
| BFCL v4 overall / normalized | 86.23 / 87.76 | 87.04 / 89.35 | within draw noise |
| IoU (inline multiset) | 0.6272 | 0.6289 | tie (MDE ~0.02–0.04) |
| TTFT median | 2713 ms | — | |
| TPOT median | 47.28 ms | not recorded | |

llama.cpp's harness build left `tpot{}` and `output_sequence_lengths{}` empty, so its
decode speed is **not** directly comparable — see "Deriving llama decode" below before
quoting any head-to-head decode figure.

## Build

Needs `libibverbs-dev` plus the hip-port nvcc/libcuda shim (PR #326).

```bash
export ATLAS_TARGET_HW=strix-hip ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_TARGET_QUANT=nvfp4 \
       ATLAS_HIP_COMPAT_INCLUDE=$PWD/crates/atlas-kernels/hip/compat ATLAS_HIPCC=/opt/rocm/bin/hipcc \
       CUDARC_CUDA_VERSION=12080 PATH=$HOME/hip-port/fakebin:/opt/rocm/bin:$PATH \
       RUSTFLAGS="-L native=$HOME/hip-port/link" LIBRARY_PATH="$HOME/hip-port/link:/opt/rocm/core-7.13/lib"
cargo build --release -p spark-server --no-default-features --features cuda
```

`ATLAS_TARGET_MODEL` and `ATLAS_TARGET_QUANT` are **mandatory** — the default target is
a `qwen3-next-80b` kernel dir that does not exist under `strix-hip`, and omitting them
silently yields a binary identical to the previous one. Plain `cargo build` also pulls
in nccl and fails to link on a single-GPU box; `--no-default-features --features cuda`
is what avoids that.

## Serve (the `ATLAS_*` flags are load-bearing; keep `--speculative`)

```bash
export LD_LIBRARY_PATH=$HOME/hip-port/link:/opt/rocm/core-7.13/lib:/opt/rocm/lib
export ATLAS_W4A16_DP4A=1 ATLAS_FORCE_GLOBAL_GDN=1 ATLAS_W4A16_VARIANT=v1 \
       ATLAS_KV_EXTERNAL_RESERVE_GB=6 ATLAS_SSM_TAIL_MIDCHUNK=1 ATLAS_SSM_TAIL_PROTECT=1 \
       ATLAS_MTP_GATE_REPROBE=64 ATLAS_MTP_DRAFTER_PREFILL=1 ATLAS_MTP_CARRY_DRAFTER=1
spark serve $SNAP --model-name nvidia/Qwen3.6-27B-NVFP4 --host 0.0.0.0 --port 8081 \
  --max-seq-len 65536 --gpu-memory-utilization 0.35 --kv-cache-dtype bf16 --max-batch-size 1 \
  --speculative --num-drafts 2 --mtp-quantization bf16 --mtp-vocab 100000 \
  --disable-tool-grammar true --enable-prefix-caching \
  --ssm-cache-slots 64 --ssm-checkpoint-interval 16 --disable-thinking
```

- `--num-drafts 2` means **K=3** (drafts + 1). K=3 is the measured optimum on this box:
  warm 25.27 tok/s / TPOT 39.57 ms, vs K=4 at 17.16 tok/s / 58.29 ms (**32% slower**)
  at matched emitted-token counts. Caveat: that K=4 figure was taken with **no `n == 4`
  arm** in `layers/qwen3_attention/trait_impl/multi_seq/ffn.rs`, so n=4 falls into
  `forward_prefill` (GEMM tiles at M=4) instead of the batch-GEMV `forward_k4`. The
  gb10 tree fixed exactly this; until that arm is ported here, K=4 is **untested**, not
  disproven.
- **`--gpu-memory-utilization`**: the submission ran 0.40; on this box 0.40 now OOMs
  (`32 MB free / 62.5 GB`) even with no spark process, because the APU GTT reports ~93%
  allocated and does not release on `drop_caches`. **0.35 is the working ceiling.**
- `--ssm-checkpoint-interval 16` is required for warm restore: the default (256) only
  checkpoints SSM state every 4096 tokens, so a sub-4096 partial hit recomputes the
  whole SSM tail.

## Benchmark

Harness `Palanivelg/endpoints` @ `edc7ea0`, config `atlas_official_compliant.yaml`
(endpoint must be `http://localhost:8081` — the shipped edge-agentic yaml points at
`:8080`, which is llama's port and yields 1007 errors).

```bash
inference-endpoint benchmark from-config --config atlas_official_compliant.yaml --mode both
PYTHONPATH=src python scripts/check_compliance.py results/<run> \
  --ruleset mlperf-edge-current --model qwen3.6-27b
```

Seeds are mandated by `mlperf.conf` and recorded in the emitted `config.yaml`:
model **42**, scheduler **16159082839903944936**, dataloader **2747215439041700203**.
(A local `check_compliance` that asserts `seed == 42` for all three is stale.)

## Validation — what each change was gated on

Ordered by strength. `bench/strix_mlperf/` holds the probes.

1. **Prompt-logprob drift gate** (`logit_gate.py` + `logit_diff.py`) — `/v1/completions`
   with `echo=true, logprobs=5` returns prompt logprobs, produced purely by the prefill
   path. ~6.5k scored positions across short/code/2k/6k prompts. The M_TILE=64 prefill
   default returns **`max|dlogprob| = 0`, `max KL = 0`, all tokens identical**.
2. **Kernel byte-compare** (`examples/strix_ffn_bench`) — m128 vs M64 outputs compared
   byte-for-byte on all ten FFN shapes.
3. **Coherence / tool-call smoke before any long run.** NVFP4 can pass a timing A/B while
   emitting garbage, so smoke first (`348*27` → `9396`), then spend the hours.
4. **Cold + warm probes** (`cold_sweep.py`, `deep_probe.py`) for no TTFT regression, and
   `tpot_probe.py` for no decode regression.

Speculative decode is trajectory-dependent: always compare emitted token counts
alongside tok/s, or a "win" can be an artifact of a shorter answer.

## Current perf levers and where the wall is

- **SSM snapshot fix** (`cdd4de0`): `compute_session_hash` hashed a *growing* prompt
  prefix, so the session gate rejected every prior-turn anchor and each warm turn
  recomputed all KV. Fix = content-address non-tail snapshots (gate only `is_tail`) and
  capture on any warm continuation. Warm TTFT went flat: p90 6073 → 2348 ms, max 77560 →
  2607 ms.
- **M_TILE=64 prefill default** (`3209543`): `w4a16_gemm_t_m128` needs 34432 B LDS and
  256 VGPRs *with spill* → 3 waves/SIMD; the M64 `w4a16_gemm_t` needs 24192 B, no spill,
  and wins every shape 1.16–2.09x, bit-identically. `w4a16_gemm_t_k64` is 0.29–0.77x and
  is skipped on native-HIP. Cold prefill 256 → 276 tok/s; 8k cold TTFT 25.7 → 23.9 s.
- **Remaining gap is cold prefill.** ~34% MFU on gate/up against ~59 TFLOP/s bf16 peak.
  Next lever is LDS reduction on the M64 tiling (24192 → <21845 buys a 6th resident CTA),
  which needs an XOR swizzle rather than more padding — 16-byte alignment and an odd
  dword stride cannot both be had by padding alone. Note `int8_gemm_faith2` (+49% on
  GB10) is **not portable**: it is built on `ldmatrix` and `cp.async`, neither of which
  exists on gfx11.
- **Decode depth wall** is the GDN multi-token verify kernel, which serializes each draft
  position through 48 GDN layers with state checkpoint + rollback — not the FFN or
  lm_head GEMMs, both of which are already batched at M≤4.

## Deriving llama decode (and why the chart caveats it)

llama's summary has `ttft` and `latency` but empty `tpot{}`/`output_sequence_lengths{}`.
Decode speed can be estimated as `(latency.median − ttft.median) ÷ median output tokens`.
Applied to Atlas this gives 19.63 tok/s against its directly-measured 21.15 — agreeing
within 8%, so the formula is sound. Applied to llama.cpp on Strix it gives ~8.3 tok/s.

Two assumptions must travel with that number: llama's median output length is *assumed*
equal to Atlas's 44 tokens (llama's was never recorded), and llama's latency medians come
from its **agentic-2.5h run (963 samples)**, not the 1007-sample run behind the wall/QPS
figures. Compare like with like — derive both sides the same way, never Atlas's measured
TPOT against llama's derived value.
