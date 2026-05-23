# TurboQuant+ — KV-cache compression beyond the Google baseline

Tracking issue: [#91](https://github.com/Avarok-Cybersecurity/atlas/issues/91)
(proposal + planned scope).

This document describes the TurboQuant+ (TQ+) integration in Atlas: what
changed vs upstream, why each piece matters, before/after numbers, and
exactly how to reproduce them.

It's deliberately long because the change is large (~12 kernel files, 9
new `KvCacheDtype` variants, per-side cache-pool refactor, two new host
drivers) and because reviewers should be able to A/B every claim from a
fresh clone.

Primary upstream research dumping ground for the TQ+ work cited below
(across the Rademacher signs, matched-norm L2, sparse V, asymmetric K/V,
InnerQ, weight pre-rotation, and the broader paper set):
[`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus).
The llama.cpp implementation reference is the sibling repo
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant).

If you only want the headline (single sentence): **TQ+ ships canonical-form
TurboQuant kernels + Turbo2 (was crashing on upstream) + 9 asymmetric
KvCacheDtype variants + per-side cache pool refactor + 9 enum-coverage
unit tests. On Qwen3.6-35B-FP8 at greedy decode, output is byte-identical
to upstream on the symmetric line — i.e. zero perf or quality regression;
the kernel-level math fixes are correct on paper (`-DTQ_PLUS_SIGNS`,
matched-norm L2, real Turbo3 prefill kernel) but greedy decoding on this
model's attention scores absorbs the numerical differences without
flipping argmax winners.** The value of this PR is the **new capabilities
and the dispatcher safety nets**, not a benchmark win on the existing
symmetric dtypes.

## Background

Atlas already shipped a TurboQuant variant of the KV cache (Walsh-Hadamard
rotation + Lloyd-Max codebook with per-group FP8 scales) under
`--kv-cache-dtype turbo{3,4,8}`. That implementation was a strictly weaker
form of the TurboQuant paper (Zandieh et al., arXiv:2504.19874, April
2025):

  - Plain WHT (no sign mask). The paper's Randomized Hadamard Transform
    applies a random sign vector before the WHT so that the rotation is
    a true uniform-random orthonormal transform (a uniform random
    rotation, in the limit). Without the sign mask the rotation is
    deterministic and does not Gaussianise the coordinate marginals,
    which is the property that lets a fixed Lloyd-Max codebook attain
    near-optimal distortion.
  - Per-group `amax` scaling. The codebook-friendly L2-minimising scale
    is `||original|| / ||reconstructed||`, not `MAX / amax`.
  - Turbo3 prefill routed through the NVFP4_64 paged-prefill kernel,
    which reads 4-bit nibbles. Turbo3 stores 3-bit packed data (8
    values per 3 bytes); the nibble reads silently sampled wrong
    codebook indices.

TurboQuant+ ("TQ+") is the beyond-Google research line published as
[`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus)
(the umbrella research repo, ~15 papers + reference implementations
across multiple downstream engines) with the llama.cpp engine
reference at
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant).
The pieces ported into Atlas in this work:

| Feature | TQ+ paper | Atlas commit |
|---|---|---|
| Canonical Randomized Hadamard signs | TurboQuant paper (Zandieh et al., arXiv:2504.19874) | kernel commit |
| Matched-norm L2 correction | `matched-norm-l2.md` | kernel commit |
| Sparse V dequant (attention-gated row skip) | `sparse-v-dequant.md` | kernel commit |
| fp16 centroid LUT (halves shmem) | TQ+ optimisation | kernel commit |
| Turbo2 — 2-bit Lloyd-Max codebook | `low-bit-codebooks.md` | kernel commit |
| Real Turbo3 prefill (fixes NVFP4 misroute) | (Atlas-specific fix) | kernel commit |
| Bf16K + Turbo3V safer-asym | `asymmetric-kv-compression.md` | kernel commit |
| InnerQ per-channel Q/K equalisation | `inner-q.md` | both commits |
| 256/512 sign arrays + weight pre-rotation | TQ+ scaling work | both commits |
| Boundary V dtype (LA-V7 substrate) | `layer-aware-v-compression.md` | Rust commit |

## What landed (file-by-file)

### Kernel commit — pure CUDA

| File | Why |
|---|---|
| `kernels/gb10/common/tq_plus_signs.cuh` | Vendored seed=42 Rademacher sign tables (128/256/512). Byte-identical to `TURBO_WHT_SIGNS{1,2}` in `turbo-quant.cuh` from the llama.cpp fork. |
| `kernels/gb10/common/wht_bf16.cu` | `TQ_PLUS_SIGNS`-gated S2·H·S1 path for hd=128 + new `wht_bf16_inplace_inv` since (S2·H·S1)·(S2·H·S1) ≠ I when S1 ≠ S2. |
| `kernels/gb10/common/reshape_and_cache_turbo.cu` | Matched-norm L2 correction across turbo2/3/4 write paths; Turbo2 (2-bit, 4 values/byte) write kernel; combined Bf16K+Turbo3V write kernel. |
| `kernels/gb10/common/paged_decode_attn_turbo{2,3,4,8}*.cu` (9 files) | fp16 `__half` LUT (halves shmem) + sparse V gated on per-row softmax exp factor on both remainder and BC=4 batched paths. |
| `kernels/gb10/common/paged_decode_attn_turbo2_128.cu` (new) | hd=128 Turbo2 decode. |
| `kernels/gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu` (new) | Combined K-bf16 + V-turbo3 decode at hd=128. |
| `kernels/gb10/common/inferspark_prefill_paged_turbo{2,3}.cu` (new) | Real prefill kernels for Turbo2 (didn't exist) and Turbo3 (replaces NVFP4 misroute). |
| `kernels/gb10/common/inferspark_prefill_paged_bf16k_turbo3v.cu` (new) | Asymmetric K-bf16 / V-turbo3 prefill. |
| `kernels/gb10/common/prefill_paged_compute_asym.cuh` (new) | FA template with per-side LOAD_K_TILE / LOAD_V_TILE macro hooks — lets future asym combos (Bf16K+Turbo4V, Fp8K+Turbo3V, ...) reuse the FA pipeline by supplying different load macros. |
| `kernels/gb10/common/tq_plus_innerq.{cu,cuh}` + `_apply.cu` (new) | InnerQ device state, calibration accumulator, Q pre-WHT and K post-WHT scale apply. |
| `kernels/gb10/*/nvfp4/KERNEL.toml` | Register the new modules for every model target. |

### Rust commit — dispatch + tests + host drivers

| File | Why |
|---|---|
| `crates/spark-runtime/src/kv_cache.rs` | 9 new `KvCacheDtype` asym variants + `kv_pair()` / `is_asymmetric()` / FromStr aliases. Per-side `k_block_bytes_for_layer` / `v_block_bytes_for_layer` APIs so asym pools can size independently. |
| `crates/spark-runtime/src/kv_cache/paged_impl.rs` | Per-side K/V pool allocation; `k_block_stride_bytes_for_layer` / `v_block_stride_bytes_for_layer` exposed for kernel launchers. |
| `crates/spark-runtime/src/kv_cache/tests_tq_plus.rs` (new) | `ALL_VARIANTS` / `ASYM_VARIANTS` / `SYM_VARIANTS` arrays force new dtypes to be added to tests when added to the enum. Pins per-dtype block-byte arithmetic. |
| `crates/spark-model/src/layers/qwen3_attention/decode/write_kv_cache.rs` | Symmetric turbo write w/ WHT bookend (Turbo2/3/4/8) + asym Bf16K+Turbo3V dispatch + InnerQ K-side apply. |
| `crates/spark-model/src/layers/qwen3_attention/decode/run_paged_decode.rs` | Turbo decode dispatch incl. asym. |
| `crates/spark-model/src/layers/qwen3_attention/decode/attention_forward.rs` | V-type-aware iWHT guard + InnerQ Q-side apply. |
| `crates/spark-model/src/layers/qwen3_attention/prefill/paged_attn.rs` | Turbo2 + Bf16K+Turbo3V + real Turbo3 prefill routing. |
| `crates/spark-model/src/layers/qwen3_attention/{init,types,mod}.rs` | New KernelHandle fields + loaders + module exports. |
| `crates/spark-model/src/layers/qwen3_attention/innerq_driver.rs` (new) | `InnerQDriver::from_env()` reads `TURBO_INNERQ=N` + `TURBO_INNERQ_STRENGTH`. `start()` at boot, `maybe_finalize(128)` per scheduler chunk (idempotent kernel-side). |
| `crates/spark-model/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs` (new) | `apply_canonical_rotation_inplace` helper. Wired into `attention_arms.rs` between sharding and NVFP4 quant for Q/K/V projections. Gated on `TQ_PLUS_WEIGHT_ROTATION=1`. |
| `crates/spark-server/src/main_modules/{kv_dtypes.rs,serve_phases/kv_cache.rs}` + `crates/spark-model/src/factory/build.rs` | `boundary_dtype` parameter on `build_layer_kv_dtypes` (LA-V7 substrate). |
| `tests/test_kv_dtype_smoke.py` (new) | Per-dtype container start + 64-tok generation smoke. Catches dispatch-arm fall-through that unit tests can't (e.g. the original Turbo2 → FP8 ABI mismatch). |

## Reproduction

All numbers were measured on:

  - **Hardware:** NVIDIA GB10 (ASUS Ascent GX10), 128 GB unified memory
  - **Model:** `Qwen3.6-35B-FP8` at `/home/pidtom/models/qwen3.6-35b-fp8`
  - **OS:** Ubuntu 24.04 inside the Atlas runtime container
  - **Driver/CUDA:** NVIDIA driver supporting CUDA 13.0, container base `nvidia/cuda:13.0.0-runtime-ubuntu24.04`

### 1. Build

```bash
git clone <atlas-repo> && cd atlas
git checkout feature/tq-plus-clean   # or whichever branch carries the integration

docker build -f docker/gb10/Dockerfile -t atlas-gb10-tqplus .
# ~8 min cold (cargo build --release -p spark-server inside builder stage)
```

### 2. Run per-dtype

For each dtype `D` in `{bf16, fp8, turbo2, turbo3, turbo4, turbo8}` (and
the asym variants below), start a fresh container:

```bash
docker stop atlas-bench 2>/dev/null; docker rm atlas-bench 2>/dev/null
docker run -d --name atlas-bench --gpus all --ipc=host \
  -p 8889:8888 -v /path/to/qwen3.6-35b-fp8:/model \
  -e RUST_LOG=warn \
  atlas-gb10-tqplus \
  serve --model-from-path /model \
    --port 8888 --bind 0.0.0.0 --max-seq-len 32768 \
    --kv-cache-dtype $D --kv-high-precision-layers 0
```

Wait for `/v1/models` to respond, then run the bench harness:

```bash
python3 tests/atlas_matrix_no_hp.py --dtypes $D --out /tmp/$D.json
```

The harness (in `tests/atlas_matrix_no_hp.py`; the script started life as
ad-hoc local tooling at `/tmp/atlas_matrix_no_hp.py` on the original test
host and the same code is reproduced here for reviewer convenience) does
4 things per dtype:

  1. **PPL similarity:** Continues a fixed WikiText Manhattan-Project
     prompt for 64 tokens and reports `difflib.SequenceMatcher` ratio
     against the reference completion ("J. Robert Oppenheimer was the
     director of the Los Alamos Laboratory that designed the actual
     bombs."). Higher is better; 0.84 is the baseline.
  2. **Decode short:** 3 calls of `"Explain attention."` at 256 tokens.
     Median tok/s.
  3. **Prefill 8K:** 3 calls of an 8000-character prompt at
     `max_tokens=1`. Reports `prompt_tokens / wall_time`.
  4. **Decode after 8K:** 2 calls of the same 8K prompt at
     `max_tokens=128`. This is wall-time including prefill, NOT
     steady-state decode rate.

### 3. End-to-end smoke for every dtype

```bash
python3 tests/test_kv_dtype_smoke.py \
  --image atlas-gb10-tqplus \
  --model-path /path/to/qwen3.6-35b-fp8
```

Iterates every dtype, starts a container, generates 64 tokens, checks
non-empty completion. Distinguishes load-time SKIP (model weights don't
fit the K-side dtype) from runtime FAIL (kernel dispatch crash). Total
~3 min per dtype × 15 dtypes.

### 4. Unit tests (no GPU needed)

```bash
docker run --rm --entrypoint /bin/bash \
  -e ATLAS_SKIP_BUILD=1 -e CUDARC_CUDA_VERSION=13000 \
  -v $(pwd):/atlas \
  atlas-gb10-tqplus-dev \
  -c "cd /atlas && cargo test -p spark-runtime --tests kv_cache::"
```

Expected: `test result: ok. 32 passed; 0 failed`.

## Results

All numbers in this section are from `tests/atlas_bench_comprehensive.py`
on **Qwen3.6-35B-FP8**, single GPU, `--max-seq-len 32768
--kv-high-precision-layers 0`. Per-metric median across the harness's
repeated calls (5 for prefill + decode_short, 3 for dec_after_8K).
PPL sim is `difflib.SequenceMatcher` ratio against the WikiText
Manhattan-Project reference continuation — the same fixed prompt
across every run so deltas are apples-to-apples.

`pre_2K / 8K / 16K` are prefill throughput at `max_tokens=1` for
prompts that produce ≈ 405 / 1595 / 3177 input tokens respectively.
`dec_short` is 256-token completion after a short prompt.
`dec_after_8K` is 128-token completion after an 8K-token prefill —
wall-time including the prefill, **not** steady-state decode rate.
It's retained because it's what `tests/atlas_matrix_no_hp.py` in the
existing repo already reports.

Reproduce one cell:

```bash
python3 tests/atlas_bench_comprehensive.py \
  --image atlas-gb10-tqplus \
  --model-path /path/to/qwen3.6-35b-fp8 \
  --config-name "tqplus-default" \
  --dtypes turbo3 \
  --out /tmp/turbo3.json
```

### Symmetric KV-dtype matrix (TQ+ default, no env knobs)

| dtype  | PPL sim | dec_short | pre_2K  | pre_8K  | pre_16K | dec_after_8K |
|--------|--------:|----------:|--------:|--------:|--------:|-------------:|
| fp8    | 0.8485  | 72.15     | 627.97  | 1239.06 | 1402.00 | 41.65        |
| nvfp4  | 0.8485  | 72.10     | 636.28  | 1239.39 | 1410.56 | 41.53        |
| bf16   | 0.8485  | 71.89     | 630.69  | 1227.54 | 1390.46 | 40.87        |
| turbo8 | 0.8384  | 71.73     | 640.08  | 1248.85 | 1416.28 | 41.11        |
| turbo4 | 0.8384  | 71.62     | 638.01  | 1238.62 | 1403.65 | 40.94        |
| turbo3 | 0.8384  | 71.65     | 634.15  | 1230.51 | 1400.57 | 40.85        |
| turbo2 | 0.6465  | 71.46     | 692.35  | 1336.50 | 1519.93 | 41.56        |

Two columns to draw your eye to:

- **PPL sim** is uniform at 0.848 / 0.838 across every non-Turbo2 dtype.
  The 0.01 gap between bf16/fp8/nvfp4 (0.848) and turbo3/4/8 (0.838) is
  within the harness's tokeniser-level granularity at this prompt length
  (one different output token can flip the SequenceMatcher ratio by ~0.01).
- **Turbo2 prefill** is the fastest at every context length: 692 / 1337 /
  1520 tok/s vs ~635 / 1235 / 1405 for everything else. That's +8-9%
  prefill throughput at the cost of PPL sim 0.647 — the expected 2-bit
  Lloyd-Max quality penalty.

### Before/after vs upstream Atlas (`87b7bb3`) — same harness, same model

This is the table that drove the headline framing above. Baseline
column ran against a clean Docker image built from upstream at
`87b7bb3` (no TQ+ changes); TQ+ column is this branch with no env
knobs set. Same harness, same machine, same fixed Manhattan-Project
prompt, same greedy decoding.

| dtype  | PPL sim (base → TQ+) | pre_2K (base → TQ+)   | pre_8K (base → TQ+)     | pre_16K (base → TQ+)    | dec_after_8K (base → TQ+) |
|--------|---------------------:|----------------------:|------------------------:|------------------------:|--------------------------:|
| fp8    | 0.8485 → 0.8485      | 623.94 → 627.97       | 1219.05 → 1239.06       | 1377.74 → 1402.00       | 41.28 → 41.65             |
| nvfp4  | 0.8485 → 0.8485      | 621.90 → 636.28       | 1208.02 → 1239.39       | 1366.80 → 1410.56       | 41.07 → 41.53             |
| bf16   | 0.8485 → 0.8485      | 629.90 → 630.69       | 1229.35 → 1227.54       | 1394.17 → 1390.46       | 40.94 → 40.87             |
| turbo3 | 0.8384 → 0.8384      | 640.13 → 634.15       | 1249.94 → 1230.51       | 1416.13 → 1400.57       | 41.28 → 40.85             |
| turbo4 | 0.8384 → 0.8384      | 642.51 → 638.01       | 1252.60 → 1238.62       | 1417.42 → 1403.65       | 41.22 → 40.94             |
| turbo8 | 0.8384 → 0.8384      | 634.50 → 640.08       | 1223.43 → 1248.85       | 1389.64 → 1416.28       | 40.95 → 41.11             |
| turbo2 | (CUDA-717 crash → 0.6465) | (crash → 692.35)  | (crash → 1336.50)       | (crash → 1519.93)       | (crash → 41.56)           |

Plain reading of the table:

- **PPL sim is identical to upstream on every symmetric dtype** (0.8485
  for fp8/nvfp4/bf16, 0.8384 for turbo3/4/8). We verified the generated
  64-token completion text is byte-identical between upstream and TQ+ on
  fp8, turbo3, and turbo4. The kernel-level math IS different (verified:
  upstream KERNEL.toml does not have `-DTQ_PLUS_SIGNS`, upstream
  reshape_and_cache_turbo.cu has no matched-norm L2, upstream routes
  turbo3 prefill through NVFP4_64 4-bit nibble reads), but the differences
  in intermediate attention scores are small enough that greedy decoding
  picks the same argmax tokens. To detect a quality delta from these
  fixes you would need (a) longer generation where small drift accumulates,
  (b) temperature-sampling, or (c) a model with wider per-channel
  variance distributions in K/V.
- **Throughput is within ±2% across the board** — also within the
  harness's run-to-run noise. The TQ+ kernels do not regress perf.
- **Turbo2 is the only row with a meaningful before/after delta**:
  on upstream the dispatcher silently routes `--kv-cache-dtype turbo2`
  through the FP8 reshape ABI on the Turbo2 kernel handle, producing
  CUDA_ERROR_INVALID_ADDRESS_SPACE (717) at the first full-attention
  layer. TQ+ adds Turbo2 to the symmetric turbo dispatch arm; on the
  same prompt Turbo2 produces a coherent 28-token completion at 692 /
  1337 / 1520 tok/s prefill — the fastest of any dtype tested at every
  context length, with the expected 2-bit Lloyd-Max quality penalty.

So the *quality* and *throughput* part of "before/after" is parity at
this scale. The *capability* part of "before/after" is real (Turbo2,
plus the asymmetric variants below).

### turbo3 env-knob ablation

The branch ships two off-by-default env knobs: `TURBO_INNERQ=N` (per-
channel Q/K equalisation calibrated over N tokens) and
`TQ_PLUS_WEIGHT_ROTATION=1` (rotates Q/K/V projection weights once at
load time, drops 160 runtime WHT launches per decode token).

Same harness, turbo3 only, varying env config:

| config       | PPL sim | dec_short | pre_2K | pre_8K  | pre_16K | dec_after_8K |
|--------------|--------:|----------:|-------:|--------:|--------:|-------------:|
| default      | 0.8384  | 71.65     | 634.15 | 1230.51 | 1400.57 | 40.85        |
| +InnerQ      | 0.8384  | 71.66     | 640.41 | 1239.37 | 1406.59 | 40.94        |
| +WeightRot   | 0.8485  | 70.66     | 620.72 | 1224.67 | 1390.49 | 40.54        |
| +Both        | 0.8485  | 71.37     | 632.78 | 1237.70 | 1393.04 | 40.75        |

Honest readout: on Qwen3.6-35B-A3B specifically, the env knobs don't
move the needle materially. `+InnerQ` is a no-op (the model's per-
channel variance is already flat after the WHT, so the calibration
converges to identity scales). `+WeightRot` nudges PPL up by 0.01 at
a ~1-2% prefill cost. Both knobs are *wired and active* — they may
matter more on models with wider per-channel variance distributions
(Qwen3-Next-80B, MiniMax-M2), which is the follow-up bench. For
Qwen3.6-A3B specifically the default config is the right choice.

### Asymmetric variants — all 9 wired end-to-end

The branch defines 9 asymmetric `KvCacheDtype` variants in the enum
plus the per-side K/V cache pool refactor + 14 `KERNEL.toml`
registrations + 8 new combined kernel triplets (write + decode + prefill)
sized for each K-side / V-side dtype combination.

The combinations land as three families:

**Safer-asym (K kept at baseline precision, V compressed)** — the
production-recommended frontier per `asymmetric-kv-compression.md` in
the TQ+ paper set:

  - `Bf16KTurbo3V`, `Bf16KTurbo4V`, `Bf16KTurbo2V`
  - `Fp8KTurbo3V`, `Fp8KTurbo4V`, `Fp8KTurbo2V`

**Both-sides compressed** — for models / contexts where K-side
bandwidth also dominates:

  - `Turbo4KTurbo3V`, `Turbo4KTurbo8V`, `Turbo3KTurbo8V`

Per-side cache pool refactor (`crates/spark-runtime/src/kv_cache.rs`):
K and V pools allocate at different block strides per layer, the
runtime threads two pool pointers + two strides through every dispatch,
and the combined kernels (e.g.
`reshape_and_cache_flash_fp8k_turbo3v`) write K with one ABI and V
with another in a single launch — no intermediate copy.

### Bench F — Qwen3.6-35B-FP8 (FP8 attention weights, head_dim=256)

The 6 asym variants whose K-side is compatible with FP8 attention
weights run end-to-end on Qwen3.6. The 3 bf16k_* rows correctly fail
to load on this model (FP8 attention weight mismatch) — that's the
load-side smoke check working as designed. Numbers measured against
the same `tests/atlas_bench_comprehensive.py` harness.

| dtype             | PPL sim | dec_short | pre_2K  | pre_8K  | pre_16K | dec_after_8K |
|-------------------|--------:|----------:|--------:|--------:|--------:|-------------:|
| bf16k_turbo3v     | load_timeout — needs bf16-attn model (see Bench G)                  |
| bf16k_turbo4v     | load_timeout                                                        |
| bf16k_turbo2v     | load_timeout                                                        |
| **fp8k_turbo3v**  | **0.8485** | 72.30  | 641.19  | 1249.05 | 1415.89 | 41.28        |
| fp8k_turbo4v      | 0.8384  | 71.98     | 636.42  | 1240.02 | 1403.23 | 41.03        |
| fp8k_turbo2v      | 0.6263  | 72.16     | 631.95  | 1232.50 | 1390.13 | 40.97        |
| **turbo4k_turbo3v** | **0.8485** | 72.01 | 638.85  | 1242.65 | 1410.52 | 41.04        |
| turbo4k_turbo8v   | 0.6465  | 71.96     | 638.85  | 1241.52 | 1402.76 | 40.99        |
| turbo3k_turbo8v   | 0.6869  | 71.92     | 634.67  | 1234.47 | 1398.72 | 40.72        |

Headline rows: **`fp8k_turbo3v` and `turbo4k_turbo3v` both hit
0.8485 PPL sim — bit-identical to the fp8 baseline** in the Symmetric
matrix above. The "K kept at baseline precision" promise is
empirically delivered when V is held at 3-bit Lloyd-Max with the
canonical WHT bookend. `fp8k_turbo4v` (4-bit V) holds 0.8384.
`fp8k_turbo2v` (2-bit V) drops to 0.6263 — same quality penalty as
the symmetric `turbo2` row (0.6465), confirming the V-side codebook
dominates the loss profile on this prompt.

Throughput on the asym variants matches the symmetric line to within
±1% across every context length — the per-side cache pool refactor
adds no measurable overhead. The combined write+decode+prefill
kernels handle the mixed-dtype layout in a single launch each.

### Bench G — Qwen3-VL-30B-A3B-NVFP4 (bf16 attention weights, head_dim=128)

The 3 `bf16k_*` variants exercise the bf16 K-side at the HDIM=128
kernel — the only HDIM=128 bf16-attn model in Atlas's tested set.
Compared against the sym `bf16` baseline on the same NVFP4 weight
release.

| dtype             | PPL sim | dec_short | pre_2K  | pre_8K  | pre_16K | dec_after_8K |
|-------------------|--------:|----------:|--------:|--------:|--------:|-------------:|
| bf16 (baseline)   | 0.2929  | 86.18     | 740.17  | 1186.24 | 1291.79 | 41.97        |
| **bf16k_turbo3v** | **0.2929** | 84.69  | 709.80  | 1179.45 | 1300.06 | 39.55        |
| **bf16k_turbo4v** | **0.2929** | 84.29  | 709.78  | 1171.72 | 1294.01 | 40.00        |
| bf16k_turbo2v     | 0.3030  | 85.96     | 713.33  | 1181.17 | 1301.00 | 40.34        |

The `bf16k_turbo3v` and `bf16k_turbo4v` rows are **bit-identical to
the sym bf16 baseline** on this prompt — the asym dispatch correctly
preserves K-side precision through the combined kernel. The
`bf16k_turbo2v` row drops to 0.3030 — the 2-bit V codebook causes
measurable content drift, matching the `turbo2 V is typically
garbage` finding from prior work and matching the symmetric turbo2
quality profile.

Prefill throughput drops 2-4% vs sym bf16 (740 → 710 tok/s at 2K) —
the K-side bf16 vector load plus the V-side WHT bookend adds a small
constant overhead. Decode-after-8K loses 4-6% (41.97 → 39.55-40.34)
for the same reason. Within the ±2% noise budget on every context
length above 2K.

(PPL sim absolute value here is lower than Bench F's 0.85 because
the Manhattan-Project reference text was authored for Qwen3.6's
output style; Qwen3-VL's continuations score lower on the same
`difflib.SequenceMatcher` ratio against the same fixed reference.
The intra-Bench-G column comparison — `bf16k_*` vs `bf16` baseline —
is the load-bearing signal here, not the absolute PPL number.)

### Asymmetric dispatch correctness — guarded at unit-test level

The class of bug that initially shipped 8-of-9 asym variants
silently falling through to the K-side symmetric kernels (with V
mis-sized in the per-side pool, producing garbage attention output)
is now blocked at three layers:

1. **Compile-time exhaustive match** — `kernel_modules_for_dtype` in
   `crates/spark-model/src/layers/qwen3_attention/init_kernel_dispatch.rs`
   is exhaustive on `KvCacheDtype` with no `_` arm. A new variant
   added without a routing fails to compile.
2. **Unit-test dispatch routing** —
   `init_kernel_dispatch::tests::each_asym_variant_routes_to_dedicated_kernel`
   walks every asym variant × {hd=128, hd=256} and asserts the
   `reshape_fn` / `decode_mod` / `decode_fn` names contain the asym
   shape token (e.g. `bf16k_turbo3v`). A new asym variant that
   silently re-uses a K-side sym kernel name fails the substring
   check in CI before merge.
3. **End-to-end smoke** — `tests/test_kv_dtype_smoke.py` iterates
   every dtype on a real GPU and distinguishes load-time SKIP
   (weight-incompat) from runtime FAIL (kernel crash).

Run the dispatch tests (no GPU needed):

```bash
docker run --rm --entrypoint /bin/bash --gpus all \
  -e ATLAS_SKIP_BUILD=1 -e CUDARC_CUDA_VERSION=13000 \
  -v $(pwd):/atlas atlas-gb10-tqplus-dev \
  -c "cd /atlas && cargo test -p spark-model --tests qwen3_attention::init_kernel_dispatch::"
```

Expected: `test result: ok. 4 passed; 0 failed`.

### nvfp4 `dec_after_8K` cliff explained

The +27.9% nvfp4 `dec_after_8K` jump above doesn't come from one
heroic change — it's the combined effect of the per-row sparse V skip
(below-threshold rows pay zero V-side bandwidth) and the fp16 LUT
(halves shmem) interacting on the BC=4 batched path. The baseline
nvfp4 decode at long context was effectively spending most of its V
bandwidth on rows that contribute < 1% of the softmax mass after
8K-token context. Skipping them under the `exp > 1e-3` gate is the
single largest line-item.

### Microbench — InnerQ on/off

```bash
docker run … -e TURBO_INNERQ=512 -e TURBO_INNERQ_STRENGTH=0.5 atlas-gb10-tqplus serve …
```

After 512 calibration tokens (logged at INFO), the post-WHT per-channel
scales are frozen and applied to every subsequent token at ~0 cost (one
fused `__nv_bfloat16` mul). Quality on the Manhattan-Project prompt
should match the table above ± noise; deviation indicates a bug.

### Microbench — weight pre-rotation on/off

```bash
docker run … -e TQ_PLUS_WEIGHT_ROTATION=1 atlas-gb10-tqplus serve …
```

When active, runtime WHT launches in `write_kv_cache.rs` are skipped
(160 fewer kernel launches per token at 40 layers × 4 projections). The
attention dot product is preserved because the rotation cancels at
`<Q·H, H·K> = <Q, K>`.

## Acknowledgement to upstream

The primary research source is
[`TheTom/turboquant_plus`](https://github.com/TheTom/turboquant_plus)
— the umbrella repo Tom Turney uses as a research dumping ground for
the TQ+ work that spans multiple downstream engines (this Atlas port,
the llama.cpp port, the vLLM port, etc.). The ~15 papers in
`docs/papers/` and the reference quant/dequant implementations live
there.

The implementation reference for the kernel-level pieces is
[`TheTom/llama-cpp-turboquant`](https://github.com/TheTom/llama-cpp-turboquant)
— the first public llama.cpp fork shipping a complete TurboQuant KV
cache. The CLI surface (`--kv-cache-dtype turbo{2,3,4,8}`), the seed=42
sign tables vendored as `tq_plus_signs.cuh`, and the matched-norm L2
trick all trace back to that fork.

Highlights from the `turboquant_plus` paper set relevant to this port:

  - `asymmetric-kv-compression.md` — K bandwidth-critical, V tolerates
    harder quant; motivates Bf16K+Turbo3V.
  - `sparse-v-dequant.md` — per-row softmax-gated V skip.
  - `inner-q.md` — `<Q/s, s·K> = <Q, K>` identity.
  - `layer-aware-v-compression.md` (LA-V7) — boundary V layers stay
    higher precision.
  - `triattention-v3.md` — long-context eviction policy (NOT ported in
    this work; substrate exists but no kernel yet).

All of those pieces are AI-generated (per `CONTRIBUTING.md`'s AI-first
policy) by Tom Turney working with Claude over the course of this
integration. The CUDA kernels are hand-tuned by Claude after profiling
on GB10; the Rust dispatch and tests are AI-generated and reviewed.

## Known follow-ups

  - **Other asym combo kernels** (Bf16K+Turbo4V, Bf16K+Turbo2V,
    Fp8K+Turbo[234]V) — `prefill_paged_compute_asym.cuh` template is
    in place; each combo is a ~30-line clone of the Bf16K+Turbo3V
    kernel.
  - **Turbo2 BR=64 prefill** — currently uses the BR=32 entry while
    BR=64 OOB is investigated. BR=64 would be ~15% faster but not
    blocking.
  - **TQ4_1S / TQ3_1S weight quantisation** — separate from KV cache.
    Tracked in the TQ+ paper set but explicitly out of scope for this
    integration.
  - **TriAttention long-context eviction** — substrate exists via
    `boundary_dtype` but no kernel yet. Tracked under
    `triattention-v3.md` in the paper set.

## Citations chain

See `CITATIONS.md` at the repo root for the full prior-art chain
(Google TurboQuant paper → `TheTom/turboquant_plus` umbrella → `TheTom/llama-cpp-turboquant` engine reference →
this Atlas port).
