# Ternary-Bonsai-27B on Avarok — Serving Receipt

The validated, reproducible configuration for serving `prism-ml/Ternary-Bonsai-27B`
(ternary Q2_0 Qwen3.6-27B GDN-hybrid + vision) on a single NVIDIA GB10 (sm_121, ~120 GB
unified). All numbers below are measured on GB10, 2026-07-15. Branch: `feat/ternary-bonsai`.

---

## TL;DR — the production stack

**native-Q2 (Tier-1/2/3) + GDN-FlashInfer prefill.** Versus the BF16-dequant baseline on
the same model/prompts/hardware:

| Axis | BF16 dequant | native-Q2 + GDN-FI | Win |
|---|---|---|---|
| Resident weights | 50.96 GB | **14.10 GB** | **3.6× leaner** |
| Prefill (1449-tok, C=1) | 554 tok/s | **943 tok/s** | **1.70×** (grows with context) |
| Decode (C=1) | 14.3 tok/s | **21.1 tok/s** | **1.48×** |
| Concurrent decode (C≥2) | — | works | Tier-3 fixed the hard crash |

Coherent throughout. Both wins are orthogonal: native-Q2 cuts **weight bandwidth**
(decode + memory), GDN-FI cuts **GDN compute** (prefill; GDN = 48 of 64 layers).

---

## 1. Environment

| Component | Value |
|---|---|
| Container | `avarok-gb10:b12x-ready` (cargo/rustc, CUDA 13.2, `libcute_dsl_runtime.so`, `CUTE_DSL_ARCH=sm_121a`, FlashInfer at `/opt/flashinfer`) |
| Binary | Host-built `spark` **runs as-is in the container** (glibc 2.39 both, cudart.so.13 soname-compat). No in-container rebuild needed for the production stack. |
| GDN-FI AOT lib | `3rdparty_patches/gdn_aot/libatlasgdn.so` (prebuilt; syms `avarok_gdn_load`, `avarok_gdn_prefill_packed_managed`; transpose handoff fixed) |
| Model files | `/tank/hf/hub/models--prism-ml--Ternary-Bonsai-27B-gguf/.../` (Q2_0 backbone + mmproj) |

Build (only needed if changing code): all targets, no nccl —
```
AVAROK_TARGET_MODEL='*' cargo build -p spark-server --release --bin spark \
  --no-default-features --features cuda
```

## 2. Model directory

`--model-from-path <dir>` where `<dir>` contains:
- `config.json` = nvidia/Qwen3.6-27B-NVFP4 config with **`quantization_config` stripped**,
  **`vision_config` kept**, `deepstack_visual_indexes=[]`, `image_token_id=248056`.
- `Ternary-Bonsai-27B-Q2_0.gguf` (symlink; the 2-bit backbone).
- `Ternary-Bonsai-27B-mmproj-Q8_0.gguf` (symlink; vision tower, auto-detected sidecar).
- `tokenizer.json`, `generation_config.json`.

Text-in-image OCR is weak (2-bit degrades fine glyph discrimination); use a Q2_g64 / Q4_1 /
bf16 backbone if OCR matters. Vision tower is unaffected (grounding/layout are correct).

## 3. Environment flags — THE RECEIPT

| Flag | Value | What it does | Status |
|---|---|---|---|
| `AVAROK_GGUF_NATIVE_Q2` | `1` | Keep id-42 weights **2-bit packed** (Tier-1 decode GEMV + the 22 GB memory win) | **REQUIRED** |
| `AVAROK_GGUF_NATIVE_Q2_MMQ` | `1` | **Tier-2** native packed tensor-core MMQ prefill (no BF16 dequant tax, no co-dispatch race) | **REQUIRED** |
| _(Tier-3 batched decode)_ | — | Always-on code (no flag) — C≥2 concurrent decode via `q2_0_gemv_vec_batchm` | auto |
| `AVAROK_GDN_FLASHINFER` | `1` | GDN-FI **prefill** (48 GDN layers); prefill-only, decode stays FLA (transpose handoff fixed → coherent) | **RECOMMENDED** (container-only) |
| `AVAROK_GDN_LIB` | `.../gdn_aot/libatlasgdn.so` | dlopen path for the GDN-FI AOT lib | with GDN-FI |
| `CUTE_DSL_ARCH` | `sm_121a` | cute-DSL kernel arch (GDN-FI) | with GDN-FI |
| `AVAROK_PREFILL_CODISPATCH` | `1` | Fuse concurrent prefills within a window | **RECOMMENDED** |
| `AVAROK_PREFILL_CODISPATCH_WINDOW_MS` | `80` | Co-dispatch fusion window | with codispatch |
| `AVAROK_KV_OVERCOMMIT` | `1` | On-demand paged KV instead of a hard "pool fits N seqs" error / 400s at long ctx or many seqs | **RECOMMENDED** |
| `LD_LIBRARY_PATH` | `…/cuda-13.2/compat:…/gdn_aot:/usr/local/lib:/usr/local/cuda/lib64` | compat driver (13.2) + GDN-FI lib + cute runtime | with GDN-FI |

### Do NOT set (attn-FlashInfer — not usable for Bonsai)
`AVAROK_FLASHINFER_PREFILL`, `AVAROK_PREFILL_VARLEN`, `AVAROK_Q12_BATCHED_FIRST_CHUNK`.
attn-FI's FA2 kernel compiles for sm_121f and dispatches correctly on tiny batches
(verified: batch=2, 82 tokens), but its only hook is the **chunk-0 batched attention path**,
which is numerically off + collapses at scale (measured C=8: TTFT 15.4 s, decode ~0). It is
gated off by default for that reason (`prefill_inner.rs:128`). Would need that path fixed
first; only worthwhile for long-context (8K+) serving. Not part of the production stack.

## 4. Serve flags

```
spark serve --model-from-path <dir> --bind 127.0.0.1 --port <p> \
  --scheduling-policy slai --tbt-deadline-ms 100 \
  --max-batch-size 8 --max-num-seqs 8 \
  --max-prefill-tokens 16384 --max-seq-len <ctx> \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8            # bf16 for long-context quality
```

## 5. Full copy-paste run (container)

```bash
docker run --rm --gpus all --network host \
  -v /home/ms/avarok/.claude/worktrees/ternary-bonsai:/work \
  -v /tank:/tank -v <modeldir-parent>:/models \
  -e LD_LIBRARY_PATH=/usr/local/cuda-13.2/compat:/work/3rdparty_patches/gdn_aot:/usr/local/lib:/usr/local/cuda/lib64 \
  -e CUTE_DSL_ARCH=sm_121a \
  -e AVAROK_GDN_LIB=/work/3rdparty_patches/gdn_aot/libatlasgdn.so \
  -e AVAROK_GGUF_NATIVE_Q2=1 -e AVAROK_GGUF_NATIVE_Q2_MMQ=1 \
  -e AVAROK_GDN_FLASHINFER=1 \
  -e AVAROK_PREFILL_CODISPATCH=1 -e AVAROK_PREFILL_CODISPATCH_WINDOW_MS=80 \
  -e AVAROK_KV_OVERCOMMIT=1 \
  --entrypoint bash avarok-gb10:b12x-ready -c \
  '/work/target/release/spark serve --model-from-path /models/bonsai-vision \
     --bind 127.0.0.1 --port 8880 --scheduling-policy slai --tbt-deadline-ms 100 \
     --max-batch-size 8 --max-num-seqs 8 --max-prefill-tokens 16384 \
     --max-seq-len 8192 --gpu-memory-utilization 0.65'
```

### Health checks (in the serve log)
- `AVAROK_GGUF_NATIVE_Q2=1: keeping id-42 FFN projections packed (group 128)` — native-Q2 on.
- `AVAROK_GDN_FLASHINFER: FlashInfer GDN kernel loaded (opt-in)` — GDN-FI engaged (fires on first prefill).
- Coherence probe returns `Paris`.

## 6. Measured performance (single GB10)

**BF16-dequant vs native-Q2, identical fixed prompts:**
| prompt | metric | BF16 | native-Q2 (FLA) | native-Q2 + GDN-FI |
|---|---|---|---|---|
| 585 tok | prefill tok/s (C=1) | 469 | 636 | 748 |
| 1449 tok | prefill tok/s (C=1) | 554 | 781 | **943** |
| — | decode tok/s (C=1) | 14.3 | 21.1 | 21.0 |
| — | resident weights | 50.96 GB | 14.10 GB | 14.10 GB |

GDN-FI prefill speedup grows with context (1.15× @585 → 1.21× @1449). Decode is
bandwidth-bound and flat across concurrency (GDN serializes) but its floor is higher with Q2.
Image prefill (1541 patch tokens): ~280 tok/s agg, decode ~21 tok/s, C=1.

## 7. Native-Q2 kernel tiers (all shipped, GPU-validated)

- **Tier-1** decode GEMV `q2_0_gemv_vec` — 99% BW, 1.52× whole-model decode, 22 GB reclaimed, byte-exact.
- **Tier-2** MMQ prefill `q2_0_mmq.cu` (`AVAROK_GGUF_NATIVE_Q2_MMQ=1`) — keep-packed int8 tensor-core, kills the ~2 s dequant tax + the co-dispatch race.
- **Tier-3** batched decode — wire `q2_0_gemv_vec_batchm` into FFN `forward_k2/k3`; C≥2 concurrent decode no longer hard-crashes.
