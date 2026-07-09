# Atlas on DGX Spark GB10 — Deployment & Compatibility Guide

New to Atlas on a Spark? Start here. Read this **before** you pick a model: which
model and quant fit your box and your goal, what to do when it OOMs, and the
gotchas that cost an evening — then it points you at the exact copy-paste recipe.

Selection + troubleshooting only. It doesn't repeat runtime mechanics documented
elsewhere:

| For… | Read |
|------|------|
| Copy-paste per-model `docker run` recipes | [`QUICKSTART.md`](../QUICKSTART.md) · the `@atlas` [recipe registry](https://github.com/Avarok-Cybersecurity/atlas-recipes) |
| Deployment *modes* (single-GPU, EP=2/TP=2, NVMe swap) | [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) |
| The native binary (no Docker) | [`docs/DISTRIBUTION.md`](DISTRIBUTION.md) |
| Adding a new model/hardware target | [`docs/HARDWARE.md`](HARDWARE.md) · [`AGENTS.md`](../AGENTS.md) |

**Serve config SSOT:** the `defaults:` block of the matching
[`atlas-recipes`](https://github.com/Avarok-Cybersecurity/atlas-recipes) recipe is
the authoritative launch config for each model — continuously tuned, pinning the
flags that hold the quality gates. **If a flag here and a recipe disagree, the
recipe wins.** This guide is the *why*; the recipe is the exact *what*.

---

## 1. The box

| | |
|---|---|
| GPU | NVIDIA GB10 (Grace-Blackwell), compute capability **sm_121** (12.1) |
| GPU memory | **119.7 GB** unified (LPDDR5X, ~273 GB/s) |
| CPU arch | aarch64 |
| **Min driver** | **580** (CUDA 13.0). The engine embeds PTX and re-JITs to your SM at launch — no `nvcc` at runtime, but the driver floor is hard. |
| Multi-node | 2× DGX Spark over RoCEv2 (`enp1s0f0np0`) for EP=2 |
| Image | `avarok/atlas-gb10:latest` — one **multi-model** binary; the right kernel set is auto-selected at startup from the model's `config.json` |

**Prerequisites**, in order:
1. NVIDIA driver ≥ 580 — verify: `nvidia-smi` shows CUDA 13.0+. (Native binary
   fails fast below the floor; bypass at your own risk with `ATLAS_SKIP_DRIVER_CHECK=1`.)
2. NVIDIA Container Toolkit (for the Docker path).
3. A clean GPU before launch: `nvidia-smi` should show no other process holding
   VRAM. Atlas sizes its KV cache from *free* memory at boot.
4. HuggingFace cache mounted (`-v ~/.cache/huggingface:/root/.cache/huggingface`).
   Weights download on first run — **plan disk**: checkpoints range ~15 GB (27–35B
   NVFP4) to ~81 GB (122B). See §5 for the one download gotcha.
5. EP=2 only: passwordless SSH between the two nodes and the RoCE NIC up (§7).

---

## 2. Model × quant compatibility matrix

Every model below runs on the **same** `avarok/atlas-gb10:latest` image. "Quant"
is the **weight** format of the HuggingFace checkpoint you point `serve` at; the
nvfp4 kernel bundle carries native FP8 and BF16 paths too, so an FP8 checkpoint
serves correctly on the same image (runtime gate:
`nvfp4↔nvfp4 · nvfp4→fp8 · nvfp4→bf16`).

`tok/s` is a single-stream p50 approximation (ISL≈128, conc=1) from the recipe
catalogue — a feel, not a benchmark. "Ctx" is a **practical** single-Spark ceiling
at conc=1; it trades against batch size and KV dtype (see §4).

### Single-node (fits one GB10)

| Model | HuggingFace id | Params (total/active) | Weight quant | Architecture | ~tok/s | Practical ctx | Notes |
|-------|----------------|----------------------|--------------|--------------|--------|---------------|-------|
| **Qwen3.6-35B-A3B** ⭐ | `Qwen/Qwen3.6-35B-A3B-FP8` | 35B / 3B | FP8 | hybrid SSM (GDN) + attn + MoE | ~157 (nvfp4) | **64K** | **Daily-driver / agentic coding.** MTP K=2, DFlash drafter `z-lab/…-DFlash`, live tool-call streaming |
| **Qwen3.6-27B** | `Qwen/Qwen3.6-27B-FP8` | 27B dense | FP8 | hybrid attn + GDN + dense FFN | ~15 | 24K | Dense reasoning; MTP off; DFlash `z-lab/Qwen3.6-27B-DFlash` |
| **Qwen3.5-35B-A3B** | `Sehyo/Qwen3.5-35B-A3B-NVFP4` | 35B / 3B | NVFP4 | hybrid GDN + attn + MoE | ~131 | 24K | MTP K=2. (⚠️ HF-id drift — see below) |
| **Qwen3.5-27B** | `Kbenkhaled/Qwen3.5-27B-NVFP4` | 27B dense | NVFP4 | hybrid attn + SSM, dense FFN | ~14 | 24K | Dense; MTP off |
| **Qwen3-Next-80B-A3B** | `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | 80B / 3B | NVFP4 | hybrid SSM + MoE (512 experts) | ~74–104 | 8K | MTP; NVIDIA reference checkpoint |
| **Qwen3-Coder-Next** | `Qwen/Qwen3-Coder-Next-FP8` | 80B / 3B | FP8 | hybrid GDN + attn + MoE (same as Qwen3-Next) | ~58 | 16K | **Coding-specialized.** `--kv-cache-dtype bf16` (FP8 KV incompatible), `--ssm-cache-slots 0`, `--oom-guard-mb 1024`, `qwen3_coder` parser. Agentic quality degrades after ~5–8 tool turns (model limitation) |
| **Qwen3-VL-30B-A3B** | `ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4` | 30B / 3B | NVFP4 | **vision** + full-attn MoE | ~97 | 32K | Image input; no thinking mode; MTP off |
| **Nemotron-3-Nano-30B-A3B** | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4` | 30B / 3.5B | NVFP4 (mixed) | mamba2 + MoE + attn | ~88 | 16K | ModelOpt mixed-precision (nvfp4 + few FP8) |
| **Nemotron-3-Super-120B-A12B** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 120B / 12B | NVFP4 (mixed) | mamba2 + latent-MoE + attn | ~24 | 64K | LatentMoE; loop-watchdog on. Fits one GB10 |
| **Mistral-Small-4-119B** | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 119B / 6.5B | NVFP4 | MLA + MoE | ~33 | 8K | **`--kv-cache-dtype bf16` required** (FP8/NVFP4 KV breaks the MLA latent); tool parser is a known gap; registry-only |
| **Gemma-4-31B** | `nvidia/Gemma-4-31B-IT-NVFP4` | 31B dense | NVFP4 | dense, sliding+full attn | ~9 | 16K | Vision; `gemma4` tool parser; registry-only (no per-model image) |
| **Gemma-4-26B-A4B** | `bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16` | 26B / 4B | **NVFP4A16** | MoE GeGLU | ~67 | 16K | 4-bit weights / 16-bit activations; registry-only |

### Large models (single-node tight, or scale out to EP=2 / EP=4)

| Model | HuggingFace id | Params (total/active) | Quant | Topology | ~tok/s | Notes |
|-------|----------------|----------------------|-------|----------|--------|-------|
| **Qwen3.5-122B-A10B** | `Sehyo/Qwen3.5-122B-A10B-NVFP4` | 122B / 10B | NVFP4 | single (tight) **or** EP=2 | ~33 single / ~51 EP=2 | 256 experts; MTP K=2. Single-node needs the tight-KV recipe (§4) |
| **MiniMax-M2.7** | `lukealonso/MiniMax-M2.7-NVFP4` | 229B / ~10B | NVFP4 | **EP=2** (2 nodes, TP=1) | — | 256-expert sigmoid MoE; BF16 KV bring-up, no MTP; registry-only |
| **Qwen3.5-397B-A17B** | `nvidia/Qwen3.5-397B-A17B-NVFP4` | 397B / 17B | NVFP4 | **EP=4 (4 nodes) only** | — | ~200 GB weights; single/2-node OOM at preflight |

**"registry-only"** = the kernels are in the multi-model image and it serves fine,
but there's no turnkey per-model `docker/gb10/<m>/` Dockerfile — run it against
`avarok/atlas-gb10:latest` with the recipe flags.

**⚠️ SSOT drifts to be aware of** (registry `MODEL.toml` vs the deployable recipe
id; tracked for reconciliation — the recipe id is what to pull):
- **Qwen3.5-35B:** registry says `Sehyo/…`, one Dockerfile example says
  `Kbenkhaled/…`. Both are NVFP4 35B-A3B builds; prefer the recipe's id — if one
  404s, try the other.
- **Qwen3.6-27B:** registry `MODEL.toml` says `Qwen/Qwen3.6-27B`, but the recipe
  and Dockerfile serve `Qwen/Qwen3.6-27B-FP8` — pull the `-FP8` checkpoint.
- **MiniMax-M2.7:** registry `MODEL.toml` (`minimax-m2-229b`) points at the base
  `MiniMaxAI/MiniMax-M2.7`, while the recipe uses the quantized
  `lukealonso/MiniMax-M2.7-NVFP4` — use the recipe's NVFP4 id.
- **`qwen3.6-*` folders say `nvfp4/` but ship FP8 checkpoints** — harmless (the
  nvfp4 bundle runs FP8), but don't let the folder name mislead you.

**"Practical ctx"** is a conc=1 comfort ceiling — deliberately different from
[`QUICKSTART.md`](../QUICKSTART.md)'s conservative maxima and from the higher
`--max-seq-len` some recipes set (the flagship pushes 64K). Gauge headroom with it;
launch with the recipe's `max_model_len`; trade context against batch/KV via §4.

---

## 3. Pick your recipe (goal → model)

| Your goal | Start with | Why |
|-----------|-----------|-----|
| **Agentic coding, tool use, daily driver** | `Qwen3.6-35B-A3B-FP8` | 64K ctx, MTP K=2, live tool-call streaming, `qwen3_coder` parser — the tuned flagship |
| **Max single-stream throughput** | `Qwen3.6-35B-A3B` (nvfp4, ~157) or `Qwen3.5-35B-A3B-NVFP4` (~131) | NVFP4 weights hit GB10's native FP4 MMA; MTP amplifies |
| **Vision / image input** | `Qwen3-VL-30B-A3B-Instruct-NVFP4` | The vision model; ~97 tok/s |
| **Biggest quality you can host** | `Qwen3.5-122B-A10B-NVFP4` on **EP=2** (2 Sparks) | 122B/10B across two nodes, ~51 tok/s |
| **Long context on one box** | `Qwen3.6-35B-A3B-FP8` @ 64K, add NVMe swap for more | FP8 KV + `--kv-high-precision-layers auto` fits 64K; swap extends it (§4) |
| **Smallest / dense reasoning** | `Qwen3.5-27B-NVFP4` or `Qwen3.6-27B-FP8` | Dense hybrids; ~14–15 tok/s, low VRAM |

Then copy that model's recipe from [`QUICKSTART.md`](../QUICKSTART.md) or run
`sparkrun run @atlas/<recipe-stem>`. Deviate from the flagship only with a reason —
the recipe defaults encode gate-passing choices.

---

## 4. Quant selection & the OOM / context ladder

### Weight quant — which checkpoint to pull
| Weight quant | What it is | Choose when |
|--------------|-----------|-------------|
| **NVFP4** | 4-bit (E2M1) weights + FP8 block scales; native GB10 FP4 MMA | Default. Fastest decode; smallest weights. Most models ship NVFP4. |
| **FP8** | 8-bit weights, native on GB10 | Higher-precision path; the `qwen3.6` flagship line. Slightly larger, very stable. |
| **NVFP4A16** | 4-bit weights, **16-bit activations** | Only where the checkpoint ships it (Gemma-4-26B). Higher activation precision, a touch slower. |
| **BF16** | reference | Rarely — largest footprint; a numeric reference, not a deployment target. |

You don't choose weight quant with a flag — you choose it by which **HuggingFace
checkpoint** you point `serve` at (the id in §2). The image serves all three.

### KV-cache dtype — `--kv-cache-dtype` (independent of weight quant)
KV precision trades context length / batch against fidelity. Six options:

| `--kv-cache-dtype` | Bits/elem | Notes |
|--------------------|-----------|-------|
| `bf16` | 16 | Reference precision; most KV memory. Bring-up default for new models. |
| `fp8` | 8 | The workhorse. Pair with `--kv-high-precision-layers auto` to keep the sensitive layers in bf16. Flagship uses this at 64K. |
| `turbo8` | 8 | FP8 variant with a different scale layout. |
| `nvfp4` | 4 | Half the memory of fp8 — the throughput recipes (35B) use this. |
| `turbo4` / `turbo3` | 4 / ~3 | Aggressive low-bit KV; most context per GB, most fidelity risk. |

Rule of thumb: **`fp8` + `--kv-high-precision-layers auto`** for quality-sensitive
/ long-context work; **`nvfp4`** when you're chasing tok/s or context and the model
tolerates it (the 35B NVFP4 recipes do). New/unproven model → start `bf16`, then
tighten.

### If you OOM — the lever ladder (apply in order)
A GB10 has ~120 GB. Budget ≈ *weights + CUDA/NCCL workspace + dequant scratch + KV
(grows with `max-seq-len × max-num-seqs`)*. When boot fails to allocate or you see
**`KV cache can hold at most 0 concurrent sequence(s)`**, walk this ladder:

1. **Lower `--gpu-memory-utilization`** first if it fails *during* boot (workspace
   starve): `0.85` default → `0.70`. If you have headroom and nothing else on the
   GPU, raise toward `0.92` for more KV.
2. **Cut `--max-seq-len`** — KV scales linearly with it (64K → 16K is a 4× KV cut).
3. **Cut `--max-num-seqs`** — fewer concurrent sequences = less KV. On the tight
   122B single-node recipe this is `--max-num-seqs 4`.
4. **Drop KV to a lower-bit dtype** — `fp8` → `nvfp4` halves KV memory.
5. **Free SSM slots** — `--ssm-cache-slots 0` reclaims the Mamba/GDN state pool on
   hybrid models when you need the last GB (tight 122B recipe uses this).
6. **`--oom-guard-mb 1024`** — reserves a guard band so the allocator fails clean
   instead of the driver killing the process mid-run.
7. **Still short? Go multi-node (EP=2)** for the 120B-class models, or **NVMe
   high-speed swap** to spill cold KV to disk (§`DEPLOYMENT.md` §3 — remember the
   container flags in §5 below).

---

## 5. Known issues & workarounds (consolidated)

The gotchas that cost people an evening, in one place:

1. **"No MTP weights found" / missing `extra_weights.safetensors` (35B under a
   mounted HF cache).** Volume-mounting `~/.cache/huggingface` can break the
   symlinked extra-weights file. **Fix:** download to a real dir and serve from a
   path:
   ```bash
   huggingface-cli download Sehyo/Qwen3.5-35B-A3B-NVFP4 --local-dir /models/qwen3.5-35b
   docker run ... -v /models/qwen3.5-35b:/model avarok/atlas-gb10:latest \
     serve --model-from-path /model --speculative --num-drafts 1
   ```
2. **High-speed swap silently does nothing / permission errors.** `io_uring`
   needs relaxed container security. Add **`--security-opt seccomp=unconfined
   --ulimit memlock=-1`** to the `docker run` (this is required for
   `--high-speed-swap`, and is easy to miss — it's not in `DEPLOYMENT.md §3`).
3. **EP=2 crashes with an SSM intermediate-buffer error.** Rank 0 and rank 1 must
   launch with **identical** `--speculative` / `--mtp-quantization` / `--num-drafts`
   flags — otherwise rank 0's verify lands on a layer rank 1 never allocated for.
   `scripts/start-ep2.sh` mirrors them; if you hand-write two `docker run`s, copy
   the spec flags verbatim. (See §7.)
4. **Native binary: "driver too old" / glibc errors.** The tarball fails fast below
   driver 580 (CUDA 13.0) and is built against glibc 2.39 (Ubuntu 24.04). Older
   distro → use the Docker image. Emergency bypass: `ATLAS_SKIP_DRIVER_CHECK=1`.
5. **First request hangs for 5–30 s.** That's cold-start CUDA-graph capture +
   autotuner + prefix-cache init, not a hang. Eliminate it with `--warmup-prompt`.
6. **Garbage / incoherent output.** Two usual causes: (a) **tool-parser mismatch** —
   the parser must match the model (`qwen3_coder` for Qwen3.5/3.6, `hermes` for
   some, `gemma4` for Gemma-4); pass `--tool-call-parser` explicitly in production
   rather than relying on auto-resolution. (b) an aggressive **KV dtype** the model
   doesn't tolerate — step it up (`turbo3`→`nvfp4`→`fp8`). If neither, it's a bug —
   **never assume the model is at fault; file it** (`AGENTS.md` failure-modes).
7. **`KV cache can hold at most 0 concurrent sequence(s)`.** You asked for more KV
   than fits — walk the §4 ladder (this fires most on single-node 122B).

---

## 6. Health, observability & a working smoke test

```bash
curl http://localhost:8888/v1/models        # loaded model — the liveness check the test suite uses
curl http://localhost:8888/health           # liveness
curl http://localhost:8888/metrics          # Prometheus exposition
# Coherence smoke — deterministic, should print exactly "4":
curl -s http://localhost:8888/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"atlas","messages":[{"role":"user","content":"What is 2+2? Reply with just the number."}],"max_tokens":16,"temperature":0}'
```
Logs go to stdout (`docker logs <container>`); `RUST_LOG` controls verbosity
(`info` default, `debug` for kernel traces).

---

## 7. EP=2 multi-node troubleshooting

(This section is the target of the `DEPLOYMENT.md` cross-reference.) Two GB10
Sparks, one OpenAI endpoint on rank 0. Launch via the canonical launcher:

```bash
HEAD_IP=10.10.10.1 WORKER_IP=10.10.10.2 \
  bash scripts/start-ep2.sh Sehyo/Qwen3.5-122B-A10B-NVFP4
```

What it sets (and you must replicate if you hand-roll `docker run`s):
- `NCCL_SOCKET_IFNAME=enp1s0f0np0` — the GB10 RoCE NIC. **Change this for
  non-DGX-Spark hardware** or NCCL binds the wrong interface and hangs.
- `NCCL_NVLS_ENABLE=0` (no NVLink), `NCCL_NET_GDR_LEVEL=0` / `NCCL_NET_GDR_C2C=0` /
  `NCCL_DMABUF_ENABLE=0` (GB10 has no GDS), `NCCL_PROTO=Simple`, `NCCL_ALGO=Ring`.
- Rank 0 → `HEAD_IP:8888`, rank 1 → `WORKER_IP:8889`, both `--master-addr HEAD_IP
  --master-port 29500`.

Failure modes:
| Symptom | Cause / fix |
|---------|-------------|
| Boot hangs at "waiting for worker" | Passwordless SSH between nodes not set up, or worker container never started. Check `docker logs` on **both** ranks. |
| NCCL init timeout / `unhandled system error` | `NCCL_SOCKET_IFNAME` wrong for your NIC, or the RoCE fabric down (`ibstat`, `ip link`). Confirm both nodes see `10.10.10.x`. |
| SSM intermediate-buffer / verify-layer error | Spec-decode flag asymmetry between ranks — see Known Issue #3. |
| Rank 1 serves nothing on 8889 | Expected — the API is served **only** from rank 0. Point clients at `HEAD_IP:8888`. |
| Port already in use | A stale container/`spark serve` on 8888/8889/29500. `docker ps`, stop it, retry. |

Topology quick pick: **EP=2** (`--ep-size 2 --tp-size 1`) for MoE expert sharding
across two nodes — this is what the 122B and MiniMax-M2.7 recipes use. **TP+EP**
(`--tp-size 2 --ep-size 2`, a 4-rank layout) is only for a model that must shard
*both* attention and experts; none of the current 2-node recipes need it. Pure TP=2
is rare and OOM-prone for these models. The 397B needs **EP=4** (4 nodes).

---

## 8. What "verified" means (so you can trust an image)

An Atlas image is only cut after it passes the **serve matrix** — every model×quant
in §2 boots, stays coherent, and holds four quality signals. If you're evaluating
Atlas or reproducing a claim, these are the signals and where they live:

| Signal | What it proves | How it's checked |
|--------|----------------|------------------|
| **Coherence** | No CJK/tag leakage, greedy determinism, tool reliability ≥8/10, valid streaming | `scripts/test_coherence.py`, `scripts/dev/coherence_test.py` |
| **Agentic** | Real multi-turn tool loops build & pass unit tests | `scripts/dev/agentic_test.py`, `scripts/test_multiturn_agentic.py` |
| **KL-divergence** | Logits stay faithful to a BF16 reference (argmax agreement, KL nats) | `bench/fp8_dgx2_drift/` + the `/coherence-parity` skill |
| **ST-subset accuracy** | Tool-calling accuracy on a BFCL single-turn subset | external gorilla / `spark-arena` harness (see note) |

The full gate spec — exactly what each script asserts, and how the release
pipeline turns it into a pass/fail before an image ships — is in the maintainer
`atlas-release` skill (`references/verify-matrix.md`). *Note:* the ST-subset /
BFCL accuracy harness currently runs outside this repo; reproducing that specific
number needs the external gorilla setup. Everything else in the table runs from a
clean checkout against a running server.

---

## See also
- [`QUICKSTART.md`](../QUICKSTART.md) — the copy-paste recipes this guide routes to.
- [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) — deployment modes + NVMe swap internals.
- [`docs/DISTRIBUTION.md`](DISTRIBUTION.md) — the native (no-Docker) binary.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) · [`AGENTS.md`](../AGENTS.md) — building & contributing.
- [`atlas-recipes`](https://github.com/Avarok-Cybersecurity/atlas-recipes) — the serve-config SSOT (`sparkrun run @atlas/<recipe>`).
