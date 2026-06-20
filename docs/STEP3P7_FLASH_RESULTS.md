# Step 3.7 Flash NVFP4 — Integration Report

**Date**: 2026-06-03 → 2026-06-05 (3-day integration sprint)
**Node**: DGX Spark (GB10, 121.7 GB unified memory)
**Checkpoint**: `stepfun-ai/Step-3.7-Flash-NVFP4` (121 GB, 14 safetensors)
**PR**: [#119](https://github.com/Avarok-Cybersecurity/atlas/pull/119)
**Branch**: `feat/step3p7-flash`

---

## Summary

| Stage | Status | Notes |
|-------|--------|-------|
| Config parsing | **PASS** | Nested `text_config` → `ModelConfig` |
| Kernel target | **PASS** | `(sm_121, step3p7-flash, nvfp4)` — 90 PTX modules |
| Weight detection | **PASS** | ModelOpt NVFP4, 138 ignore modules |
| EP topology | **PASS** | Local experts [0, 144) for rank 0 of 2 |
| Preprocessing script | **PASS** | 504 fused → 146,587 per-expert tensors, 3 min |
| Single-GPU serve (fused) | **BLOCKED** | OOM: 120 GB on-disk > ~115 GB free after init |
| EP=2 serve (split) | **PARTIAL** | Server starts, CUDA 700 during inference |
| MTP speculative | **DEFERRED** | Loader returns empty; use `--speculative 0` |

---

## Model Architecture

Step 3.7 Flash is a **196B MoE** with **~11B active** per forward pass.
Architecture is a hybrid of patterns Atlas already supports:

| Property | Step 3.7 | Nearest Atlas Model |
|----------|----------|-------------------|
| Expert routing | Sigmoid, 288 top-8 | MiniMax M2 (sigmoid, 256 top-8) |
| Shared expert | 1280 intermediate | Qwen 3.5 (shared expert) |
| Partial RoPE | 0.5 (64/128 dims) | MiniMax M2 (0.5) |
| Head dim | 128 | MiniMax M2 (128) |
| Attention gate | g_proj (head-wise) | Qwen 3.5 (g_proj) |
| Mixed attention | 12 full + 36 sliding | Gemma-4 (mixed types) |
| MTP draft | 3 modules | MiniMax M2 (3 modules) |
| Sliding window | 512 tokens | Gemma-4 |

### Layer pattern

48 layers total: 12 full-attention + 36 sliding-attention in a repeating
`[full, sliding, sliding, sliding]` pattern. The first 3 layers (0–2) use
dense FFN; layers 3–44 are MoE; layers 45–47 are MTP draft modules.

### Per-layer `rope_theta`

This is the most architecturally distinctive feature. Step 3.7 uses
**per-layer rope_theta** — not a single scalar:

- Full-attention layers: `rope_theta = 5,000,000`
- Sliding-attention layers: `rope_theta = 10,000`

This is a 500× difference. The two theta values correspond 1:1 with the
layer type: every full-attention layer gets the large theta (long-range
positional encoding), every sliding-attention layer gets the small theta
(local positional encoding within the 512-token window).

> **Known limitation**: Atlas `ModelConfig` stores a single `rope_theta`
> scalar. The current parser takes element [0] (5,000,000 — the
> full-attention value). Sliding-attention layers receive the wrong theta.
> Correct fix requires per-layer theta in `ModelConfig` or a dispatch
> lookup keyed on `layer_types`. Quality impact is bounded because sliding
> layers only attend within a 512-token window where the positional
> difference between theta=10K and theta=5M is small — but it *will*
> matter at the margin.

### Per-layer `partial_rotary_factor`

Similarly, `config.json` provides a per-layer array. All 48 values are
`0.5`, so the current single-scalar approach is correct. Documented here
in case a future Step variant changes this.

---

## Weight Format: Fused vs Per-Expert Tensors

The HuggingFace checkpoint (`stepfun-ai/Step-3.7-Flash-NVFP4`) stores
expert weights in **fused tensors** — all 288 experts packed into single
tensors per projection type:

```
model.layers.3.moe.gate_proj.weight          shape: [288, 1280, 2048] (NVFP4)
model.layers.3.moe.gate_proj.weight_scale    shape: [288, 1280, 128]  (FP8)
model.layers.3.moe.gate_proj.weight_scale_2  shape: [288]             (FP32)
model.layers.3.moe.gate_proj.input_scale     shape: [288]             (FP32)
```

This creates two problems for Atlas:

1. **OOM on single-Spark**: The fused checkpoint is 121 GB on disk. After
   CUDA init + 90 PTX modules, ~115 GB remains — not enough.
2. **EP incompatibility**: Atlas EP filtering relies on
   `parse_expert_index()` finding `.experts.N.` in tensor names. Fused
   tensor names have no per-expert index, so EP can't skip remote experts.

### Solution: Preprocessing Script

`scripts/preprocess_step3p7_experts.py` splits fused tensors into
per-expert format:

```bash
python3 scripts/preprocess_step3p7_experts.py \
  --input /path/to/Step-3.7-Flash-NVFP4 \
  --output /path/to/Step-3.7-Flash-NVFP4-split
```

**Results on DGX Spark NVMe:**

| Metric | Value |
|--------|-------|
| Input tensors | 504 fused |
| Output tensors | 146,587 (145,152 per-expert + 1,435 pass-through) |
| Output shards | 12 safetensors |
| Output size | 120.37 GB |
| Processing time | ~3 minutes |
| Dependencies | Python 3, `safetensors` (no numpy/torch) |

The script uses raw byte slicing — no numpy or torch dependency for
tensor operations. Only the `safetensors` library is needed for header
parsing and serialization. This is deliberate: DGX Spark Ubuntu
environments may not have PyTorch installed.

**Per-expert tensor format after split:**
```
model.layers.3.moe.experts.0.gate_proj.weight          shape: [1280, 2048]
model.layers.3.moe.experts.0.gate_proj.weight_scale    shape: [1280, 128]
model.layers.3.moe.experts.0.gate_proj.weight_scale_2  shape: []  (scalar)
model.layers.3.moe.experts.0.gate_proj.input_scale     shape: []  (scalar)
```

### Dual-Path Weight Loader

The Rust weight loader (`weight_loader/step3p7.rs`) auto-detects the
checkpoint format:

```rust
fn has_per_expert_tensors(store: &TensorStore) -> bool {
    store.get("model.layers.3.moe.experts.0.gate_proj.weight").is_some()
}
```

- **Per-expert path** (split checkpoint): Loads individual expert tensors,
  compatible with EP filtering. Recommended for production.
- **Fused path** (HuggingFace original): GPU pointer arithmetic via
  `slice_fused_experts()`. Single-GPU only, no EP support.

---

## Build Verification

Six iterative builds to reach clean compilation:

| Build | Issue | Fix |
|-------|-------|-----|
| 1 | Unused import `parse_step3p7` | Fixed re-export chain in `config.rs` |
| 2 | Unused `Context` import | Removed from `step3p7.rs` |
| 3 | `eos_token_id` is JSON array, not scalar | Array-aware parser |
| 4 | `sliding_attention` unknown `LayerType` | Pre-process `layer_types` → `FullAttention` + set `sliding_window` |
| 5 | Layer count mismatch (48 vs 45, MTP included) | Fixed `layer_types` to exclude MTP entries |
| 6 | `tracing` crate unavailable in `atlas-core` | Removed tracing call |

**Final build**: Clean. 90 kernel modules compiled. Docker image
`atlas-step3p7:latest` (2.79 GB).

---

## Runtime Verification (Single-GPU, Fused Checkpoint)

### Launch Command
```bash
sudo docker run --name atlas-step3p7 --gpus all --ipc=host --network host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  atlas-step3p7:latest serve stepfun-ai/Step-3.7-Flash-NVFP4 \
    --port 8888 --kv-cache-dtype fp8 --kv-high-precision-layers auto \
    --gpu-memory-utilization 0.92 --scheduling-policy slai \
    --max-seq-len 8192 --speculative 0 --ssm-cache-slots 0
```

### Results

| Stage | Result | Details |
|-------|--------|---------|
| Config parse | **PASS** | `Step3p7ForConditionalGeneration` → `step3p7` model type |
| Kernel target | **PASS** | `(sm_121, step3p7-flash, nvfp4)` selected, 90 PTX modules loaded |
| GPU backend | **PASS** | `AtlasCudaBackend` initialized |
| Weight format | **PASS** | ModelOpt NVFP4 detected, 138 non-weight tensors ignored |
| EP topology | **PASS** | EP=1: local experts [0, 288), all experts assigned |
| OOM pre-flight | **FAIL** | 120 GB on-disk exceeds ~115 GB available after CUDA init |

**Root cause**: The fused HuggingFace checkpoint is 121 GB. After
allocating CUDA context + 90 PTX modules + buffer arena, only ~115 GB
remains. This is a fundamental size constraint — the model is 6 GB
larger than Qwen3.5-122B (90 GB) and 26 GB larger than Nemotron-Super
(94 GB), the two largest models Atlas currently serves on single-GPU.

**Path forward**: EP=2 with the split checkpoint. Each rank loads 144 of
288 experts, reducing per-node weight footprint to ~61 GB (estimated).
This is well within single-Spark capacity and matches how the model was
designed to be served.

---

## Bug Sweep (Opus 4.8 Code Review)

A complete review of the 5,816-line diff was performed using Claude Code
with Opus 4.8. Nine findings ranked by severity; six addressed in commit
[`c78f626`](https://github.com/marksunner/atlas/commit/c78f626a93654680649991316ba0d36d66808762):

### Bugs Fixed

| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | Critical | CUDA `silu_down` kernel: `s_act[1024]` overflows for `moe_intermediate_size=1280` | `s_act[1280]` |
| 2 | Critical | `rope_theta` default `500000` wrong by 10×; actual first element is `5000000` | Fixed default + array handling |
| 3 | High | Per-expert `weight_scale_2` (per-tensor scalar) not split by preprocessing; loader expected per-expert key | Fallback to global tensor |
| 4 | High | `eos_token_id` took `[0]` = 1 (BOS); should take `[-1]` = 128007 (chat stop token) | Takes last element |
| 5 | Medium | Preprocessing script `total_size` always 0 (empty list comprehension) | Accumulates actual byte count |
| 6 | Medium | `MODEL.toml` layer counts wrong (45 full ≠ 12+36=48; "4:1 ratio" ≠ 12:36=1:3) | Corrected counts and ratio |

### Known Limitations (Documented, Not Bugs)

| # | Issue | Impact | Notes |
|---|-------|--------|-------|
| 7 | Per-layer `rope_theta` collapsed to scalar | Quality degradation at margin | Requires `ModelConfig` schema change |
| 8 | `layer_types` distinction (full vs sliding) discarded | All layers get `sliding_window=512` | Correct for sliding layers; full-attention layers shouldn't window but Atlas applies globally |
| 9 | `shared_expert_gate` dummy (2 bytes) vs blend kernel expectation | Potential OOB read if blend kernel fires | Step 3.7 shared expert is unconditional — verify blend kernel not on dispatch path |

---

## Memory Budget (Estimated, EP=2)

Based on Qwen3.5-122B single-GPU precedent and Step 3.7 architecture:

| Component | Estimate | Basis |
|-----------|----------|-------|
| Expert weights (144 of 288) | ~59 GB | Half of 118 GB expert tensors |
| Shared weights (embed, norm, attn, shared expert) | ~2 GB | 1,435 non-expert tensors |
| Buffer arena | ~2.5 GB | Similar to 122B (8192-token chunks) |
| KV cache (FP8, 12 attn layers) | ~0.8 GB | Same as 122B (12 attn layers, FP8) |
| OOM guard | 4 GB | Standard |
| **Total per node** | **~68 GB** | **Fits comfortably in 121.7 GB** |
| Headroom | ~54 GB | Available for larger `max_seq_len` |

---

## Preprocessing Script Technical Details

### What gets split

Only MoE expert projection tensors with `shape[0] == num_experts` (288):
- `gate_proj.weight` — `[288, 1280, 2048]` → 288 × `[1280, 2048]`
- `gate_proj.weight_scale` — `[288, 1280, 128]` → 288 × `[1280, 128]`
- `up_proj.*` — same shapes as `gate_proj`
- `down_proj.weight` — `[288, 2048, 1280]` → 288 × `[2048, 1280]` (transposed)
- `down_proj.weight_scale` — `[288, 2048, 80]` → 288 × `[2048, 80]`

### What passes through unchanged

- Embedding + LM head (~1 GB)
- Attention projections (q/k/v/o_proj, g_proj) per layer
- Shared expert projections per MoE layer
- Layer norms (RMS norm weights)
- Router weights + bias per MoE layer
- Vision model tensors (~1.5 GB, not used for text inference)

### Scale tensor handling

`weight_scale_2` and `input_scale` are per-expert scalars in the fused
format (`shape: [288]`). The script splits them into individual scalar
tensors (`shape: []`) per expert. The Rust loader's fallback logic handles
both cases: if a per-expert `weight_scale_2` exists it uses it; if not, it
falls back to the fused tensor and indexes by expert ID.

---

## Files Changed

### New files (4)
- `crates/atlas-core/src/config/parsers/step3p7.rs` — config parser (287 lines)
- `crates/spark-model/src/weight_loader/step3p7.rs` — weight loader (439 lines)
- `kernels/gb10/step3p7-flash/MODEL.toml` — kernel target + sampling defaults
- `scripts/preprocess_step3p7_experts.py` — fused→per-expert tensor splitter (318 lines)

### Modified files (4)
- `crates/atlas-core/src/config.rs` — re-export `parse_step3p7`
- `crates/atlas-core/src/config/dispatch.rs` — `"step3p7"` dispatch arm
- `crates/atlas-core/src/config/parsers/mod.rs` — module + re-export
- `crates/spark-model/src/factory.rs` — `Step3p7WeightLoader` registration

### Kernel reuse
No new CUDA kernels. Step 3.7 reuses `kernels/gb10/common/` with
`-DHDIM=128` (same head dimension as MiniMax M2). The sigmoid routing
kernel (`moe_topk_sigmoid`) supports up to 512 experts; Step 3.7's 288
is well within range.

---

## Next Steps

1. **[P0] EP=2 runtime validation**: Run Atlas with the split checkpoint
   across two DGX Sparks. Verify weight loading, expert routing, and
   inference quality. Hardware is available (Spark 4 + Spark 6, QSFP
   200 Gbps interconnect verified).
2. **[P1] Per-layer rope_theta**: Propose `ModelConfig` schema extension
   or per-layer override mechanism. This is the single largest quality
   gap — 500× theta difference between full and sliding layers.
3. **[P1] Sliding vs full attention dispatch**: Investigate whether Atlas
   runtime can conditionally apply `sliding_window` per layer based on
   `layer_types`. Currently all layers receive `sliding_window=512`.
4. **[P2] MTP support**: Step 3.7 has 3 MTP draft modules (layers 45–47).
   Loader currently returns empty for MTP tensors. Enabling speculative
   decoding requires mapping these to Atlas's MTP framework.
5. **[P2] Benchmark**: Once serving, run the standard test suite
   (coherence, tool calls, TPS, long context) and add results to
   `tests/SINGLE_GPU_RESULTS.md`.

---

## EP=2 Runtime Results (2026-06-06)

**Nodes:** Spark 4 (10.0.0.4, rank 0) + Spark 6 (10.0.0.6, rank 1)
**Interconnect:** QSFP 200 Gbps RoCE v2, HCA `roceP2p1s0f1`

| Stage | Result | Details |
|-------|--------|--------|
| Fast weight load | **PASS** | 74,011 tensors/rank, 70.5 GB/rank, ~30s |
| EP topology | **PASS** | Rank 0: experts [0, 144), Rank 1: [144, 288) |
| NCCL init | **PASS** | RoCE v2, ring, sub-second |
| Model build | **PASS** | 45 layers, per-expert format detected |
| Server listen | **PASS** | `127.0.0.1:8888` |
| **Inference** | **FAIL** | `CUDA_ERROR_ILLEGAL_ADDRESS (700)` during prefill |

### Memory (Actual)
| Metric | Value |
|--------|-------|
| Weights per rank | 70.53 GB |
| GPU used after load | 73.24 GB |
| GPU free | ~44 GB |

### Crash Analysis: q_proj ≠ hidden_size

Step 3.7's Q projection output (8192 = 64 heads × 128 dim) is **2×**
the hidden dimension (4096). Every other Atlas model has
`q_heads * head_dim == hidden_size`. If attention kernels size buffers
by `hidden_size`, the Q projection overflows.

| Model | hidden | q_heads×head_dim | Match? |
|-------|--------|-----------------|--------|
| Qwen 3.5-122B | 5120 | 40×128=5120 | ✅ |
| MiniMax M2 | 6144 | 48×128=6144 | ✅ |
| Step 3.7 | 4096 | 64×128=8192 | ❌ (2×) |

This requires attention kernel adaptation.

### Additional Bugs Fixed During EP Sprint

| # | Bug | Fix |
|---|-----|-----|
| 7 | `has_per_expert_tensors()` probed expert 0 (not on all EP ranks) | Scan store keys for any expert pattern |
| 8 | `shared_expert_gate` was 2 bytes; blend kernel reads `[hidden]` vector | Full `h*2` byte zero buffer |
| 9 | Bugs #1-#4 not in Spark source (only on GitHub fork) | Applied locally |
