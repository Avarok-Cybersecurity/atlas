# Atlas NVFP4 Quantizer — Pure Rust Implementation Plan

**Date**: 2026-03-17
**Status**: Planning (not started)
**Goal**: Ship `spark quantize` CLI that converts BF16/FP8 → NVFP4 checkpoints with SOTA quality

---

## Why

Current NVFP4 checkpoints come from Python toolchains (llm-compressor, AutoFP4). Quality varies by calibration: the Sehyo 122B has coherence issues in tool-use scenarios that may stem from suboptimal calibration. A built-in quantizer gives us:

1. **Quality control** — Apply best-known techniques (MR-GPTQ, Four-over-Six, SmoothQuant)
2. **Zero Python dependency** — Single `spark quantize` command, no conda/pip
3. **MoE-aware calibration** — Expert-balanced sampling for even expert coverage
4. **Format compatibility** — Output is loadable by Atlas and vLLM

---

## NVFP4 Format Recap

- **Data**: E2M1 (4-bit), packed 2 per byte. 16 representable values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
- **Block scale**: FP8 E4M3, one per 16 elements (group_size=16)
- **Global scale**: FP32 per tensor. `scale2 = max(abs(tensor)) / (6.0 × 448.0)`
- **Dequant**: `value = E2M1_LUT[nibble] × block_scale_fp8 × global_scale_fp32`

---

## Tiered Implementation

### Tier 1: RTN + Four-over-Six (MVP) — ~2 weeks

Pure arithmetic, no calibration data needed. Competitive with llm-compressor default.

| Component | Effort | Notes |
|-----------|--------|-------|
| E2M1 codec (encode/decode, LUT) | 1 day | 16-value lookup, bit packing |
| FP8 E4M3 codec | 1 day | Already exists in `atlas-quant/src/fp8.rs` |
| Safetensors reader | 0 days | Already exists (`safetensors` crate v0.5) |
| Safetensors writer | 2 days | Use `safetensors` crate's `serialize` API |
| RTN quantizer | 2 days | Per-tensor global scale → per-block FP8 scale → round to nearest E2M1 |
| Four-over-Six | 1 day | Quantize each block at scale-to-4 AND scale-to-6, pick lower MSE |
| Ignore list handling | 1 day | Skip layers from `quantization_config.ignore` (gates, linear_attn, lm_head) |
| CompressedTensors output naming | 1 day | `weight_packed`, `weight_scale`, `weight_global_scale` |
| CLI: `spark quantize` subcommand | 1 day | `--input`, `--output`, `--method rtn` |
| Parallel tensor processing | 1 day | Rayon for CPU parallelism across tensors |
| **Total** | **~11 days** | |

**Quality expected**: ~95-96% of BF16 (comparable to llm-compressor RTN baseline).

**Key algorithm (RTN)**:
```
for each weight tensor W[N, K]:
  global_scale = max(abs(W)) / (6.0 * 448.0)
  W_scaled = W * (1.0 / global_scale)
  for each block of 16 values in W_scaled:
    block_max = max(abs(block))
    block_scale = cast_fp8_e4m3(block_max / 6.0)
    for each value in block:
      fp4 = round_to_nearest_e2m1(value / block_scale)
    // Four-over-Six: also try scale-to-4, keep if lower MSE
```

**References**:
- [Four Over Six: Adaptive Block Scaling](https://arxiv.org/abs/2512.02010) — MIT Han Lab
- [NVIDIA NVFP4 Format Blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

---

### Tier 2: SmoothQuant + Hadamard (+ calibration) — ~2 weeks additional

Requires pre-computed activation statistics (from a calibration run). Significantly improves quality for smaller models and MoE.

| Component | Effort | Notes |
|-----------|--------|-------|
| Calibration data loader | 2 days | Read tokenized calibration dataset (pre-tokenized JSONL) |
| Forward pass for statistics | 3 days | Run BF16 model through calibration samples, collect per-channel absmax. Reuse Atlas's existing inference pipeline |
| SmoothQuant migration | 2 days | Per-channel scaling: `W_smooth = W * diag(s)`, `X_smooth = X * diag(1/s)` where `s = absmax(X)^α / absmax(W)^(1-α)` |
| Walsh-Hadamard transform | 1 day | Butterfly algorithm, block_size=16 (matching NVFP4 group_size) |
| Static activation scales | 1 day | Per-tensor FP32 global scale from calibration absmax |
| **Total** | **~9 days** | |

**Quality expected**: ~97% of BF16 (comparable to llm-compressor with calibration).

**References**:
- [MR-GPTQ: Bridging the Gap for Microscaling FP4](https://arxiv.org/abs/2509.23202) — ICLR 2026
- SmoothQuant (Xiao et al., 2023)

---

### Tier 3: MR-GPTQ (SOTA quality) — ~3 weeks additional

Full GPTQ adapted for FP4's non-uniform grid. Best published PTQ method for NVFP4.

| Component | Effort | Notes |
|-----------|--------|-------|
| Hessian computation | 3 days | X^T X from calibration activations per layer. Needs `faer` crate for LAPACK |
| Cholesky factorization | 2 days | `faer` crate has efficient Cholesky. Need dampening (diag += λ) |
| GPTQ column-wise update loop | 3 days | Sequential column quantization with error compensation. Must adapt for E2M1 grid (not uniform INT) |
| MSE-optimized scale search | 2 days | Alternating optimization: fix fp4 codes → optimize scales, fix scales → optimize fp4 codes |
| Block-wise Hadamard rotation | 1 day | Rotate weight+Hessian jointly before GPTQ (block_size=16) |
| Layer-by-layer pipeline | 2 days | Process one layer at a time to bound memory. Free layer N before processing N+1 |
| MoE expert-balanced sampling | 2 days | Route calibration tokens, ensure all 256 experts get ≥N samples |
| **Total** | **~15 days** | |

**Quality expected**: 98-99% of BF16 (SOTA for PTQ NVFP4).

**Rust dependencies**:
- `faer` — Linear algebra (Cholesky, matrix multiply). Pure Rust, no LAPACK/BLAS needed.
- `rayon` — Parallel tensor processing.
- `safetensors` — Read/write checkpoints.

**References**:
- [MR-GPTQ](https://arxiv.org/abs/2509.23202) — Full algorithm with FP4 grid adaptation
- [Geometry of LLM Quantization](https://arxiv.org/abs/2507.18553) — ICLR 2026, theoretical grounding
- [MoEQuant: Expert-Balanced Quantization](https://arxiv.org/abs/2505.03804)
- [EAQuant: Expert-Aware PTQ for MoE](https://arxiv.org/abs/2506.13329)

---

### Tier 4: Advanced recovery (optional) — research territory

| Technique | Effort | Quality Gain | Feasibility |
|-----------|--------|-------------|-------------|
| RaZeR (redundant zero remapping) | 3 days | Moderate | Requires custom dequant kernels |
| ARCQuant (augmented residual channels) | 1 week | Large | Requires modified GEMM dispatch |
| Logit distillation (Dropbox) | 1 week | Large (99%+) | Needs inference for logit collection |
| QAD (quantization-aware distillation) | N/A | Near-BF16 | **Not feasible** — requires training loop |

---

## Architecture

```
crates/
  atlas-quant/           # Existing crate, extend with quantizer
    src/
      fp8.rs             # FP8 E4M3 LUT (exists)
      e2m1.rs            # NEW: E2M1 encode/decode, LUT, packing
      nvfp4.rs           # NEW: RTN + Four-over-Six + SmoothQuant quantizer
      gptq.rs            # NEW (Tier 3): GPTQ with FP4 grid adaptation
      calibrate.rs       # NEW (Tier 2): Calibration data collection
      smooth.rs          # NEW (Tier 2): SmoothQuant migration factors
      hadamard.rs        # NEW (Tier 2): Walsh-Hadamard butterfly
      writer.rs          # NEW: Safetensor checkpoint writer
      lib.rs             # Public API

  spark-server/
    src/
      cli.rs             # Add `quantize` subcommand
      main.rs            # Route to quantize entrypoint
```

### CLI Design

```bash
# Tier 1: No calibration needed
spark quantize \
  --input Qwen/Qwen3.5-122B-A10B \
  --output ./Qwen3.5-122B-A10B-NVFP4 \
  --method rtn \
  --four-over-six

# Tier 2: With calibration
spark quantize \
  --input Qwen/Qwen3.5-122B-A10B \
  --output ./Qwen3.5-122B-A10B-NVFP4 \
  --method smooth-rtn \
  --calibration-data wikitext2 \
  --calibration-samples 128

# Tier 3: GPTQ quality
spark quantize \
  --input Qwen/Qwen3.5-122B-A10B \
  --output ./Qwen3.5-122B-A10B-NVFP4 \
  --method gptq \
  --calibration-data c4 \
  --calibration-samples 512 \
  --hadamard \
  --expert-balanced
```

---

## Effort Summary

| Tier | Method | Effort | Quality (vs BF16) | Dependencies |
|------|--------|--------|-------------------|-------------|
| 1 | RTN + Four-over-Six | **~2 weeks** | ~95-96% | safetensors, rayon |
| 2 | + SmoothQuant + Hadamard | **+2 weeks** | ~97% | + calibration data |
| 3 | + MR-GPTQ | **+3 weeks** | ~98-99% | + faer (linear algebra) |
| 4 | + Advanced recovery | **+2-4 weeks** | ~99%+ | Research territory |

**Recommended MVP**: Tier 1 (~2 weeks). Gives immediate value with zero calibration.
**Recommended target**: Tier 2 (~4 weeks total). Matches llm-compressor quality.
**Stretch goal**: Tier 3 (~7 weeks total). Exceeds all existing open-source quantizers.

---

## What We Skip (and why)

- **QAD / fine-tuning** — Requires backpropagation through full model. Not feasible without training infrastructure.
- **AutoRound** — Requires gradient descent with learned rounding. Too complex for pure Rust quantizer.
- **INT4 / GPTQ-classic** — We only need NVFP4 (E2M1), not integer quantization.
- **Activation quantization** — Dynamic per-token activation quantization already happens at inference time in Atlas's CUDA kernels. The quantizer only handles weights.

---

## Key Research Papers

| Paper | arXiv | Year | Key Contribution |
|-------|-------|------|-----------------|
| MR-GPTQ | 2509.23202 | ICLR 2026 | GPTQ adapted for FP4 non-uniform grid |
| Four Over Six | 2512.02010 | 2025 | Adaptive scale-to-4/6 block scaling |
| RaZeR | 2501.04052 | 2025 | Redundant zero remapping for E2M1 |
| ARCQuant | 2601.07475 | 2026 | Augmented residual channels for FP4 |
| QAD (NVIDIA) | 2601.20088 | 2026 | Quantization-aware distillation |
| Pretraining with NVFP4 | 2509.25149 | 2025 | Validates E2M1+E4M3 at 10T tokens |
| FP4 All the Way | 2505.19115 | 2025 | Fully quantized training with FP4 |
| MoEQuant | 2505.03804 | 2025 | Expert-balanced calibration for MoE |
| EAQuant | 2506.13329 | 2025 | Expert-aware PTQ for MoE |
| Geometry of Quantization | 2507.18553 | ICLR 2026 | GPTQ = Babai's nearest plane algorithm |
| MXFP4 Error Reduction | 2603.08713 | 2026 | OAS+MBS close MXFP4-NVFP4 gap |
| Quartet II | 2601.22813 | 2026 | MS-EDEN + Four-over-Six for training |
| Dropbox Logit Distillation | blog | 2025 | Per-channel linear correction (99%+ recovery) |
